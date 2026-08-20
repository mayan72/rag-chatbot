"""
History Service

Responsibilities
----------------
1. Read RAG CSV logs.
2. Read JSONL logs (optional).
3. Return history records.
4. Calculate dashboard KPIs.
5. Calculate analytics metrics.

This service contains NO FastAPI code.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from config import LOG_DIR


# ==========================================================
# CSV Columns
# ==========================================================

COL_TIMESTAMP = "timestamp"
COL_STATUS = "status"
COL_REQUEST_ID = "request_id"

COL_PROVIDER = "provider"
COL_MODEL = "model"

COL_CONFIDENCE = "confidence"
COL_MAX_SIMILARITY = "max_similarity"

COL_TOTAL_TIME = "total_time_ms"
COL_RETRIEVAL_TIME = "retrieval_time_ms"
COL_LLM_TIME = "llm_time_ms"

COL_PROVIDER_LATENCY = "llm_provider_latency_ms"

COL_CONTEXT_LENGTH = "context_length"
COL_CHUNKS = "chunks_retrieved"

COL_RETRIEVAL_THRESHOLD = "retrieval_threshold"
COL_TOP_K = "top_k"
COL_TEMPERATURE = "temperature"

COL_INPUT_TOKENS = "input_tokens"
COL_OUTPUT_TOKENS = "output_tokens"
COL_TOTAL_TOKENS = "total_tokens"

COL_INPUT_COST = "input_cost"
COL_OUTPUT_COST = "output_cost"
COL_EMBEDDING_COST = "embedding_cost"
COL_TOTAL_COST = "total_cost"


logger = logging.getLogger(__name__)


CSV_LOG = Path(LOG_DIR) / "rag_logs.csv"


class HistoryService:

    def __init__(self):

        logger.info(
            "HistoryService initialized."
        )

    # ==========================================================
    # Private
    # ==========================================================

    def _load_dataframe(self) -> pd.DataFrame:
        """
        Load RAG history CSV.

        Returns
        -------
        pandas.DataFrame
            Empty dataframe if file does not exist
            or cannot be read.
        """

        if not CSV_LOG.exists():

            logger.warning(
                "History CSV not found: %s",
                CSV_LOG,
            )

            return pd.DataFrame()

        try:

            df = pd.read_csv(
                CSV_LOG
            )

            logger.info(
                "Loaded %d history records.",
                len(df),
            )

            return df

        except Exception:

            logger.exception(
                "Unable to read history CSV."
            )

            return pd.DataFrame()

    # ==========================================================
    # Helpers
    # ==========================================================

    @staticmethod
    def _numeric_series(
        df: pd.DataFrame,
        column: str,
    ) -> pd.Series:
        """
        Safely return a numeric Series.

        Missing columns become zeros instead of
        causing the dashboard to fail.
        """

        if column not in df.columns:

            return pd.Series(
                0,
                index=df.index,
                dtype="float64",
            )

        return pd.to_numeric(
            df[column],
            errors="coerce",
        ).fillna(0)

    # ==========================================================
    # History
    # ==========================================================

    def get_history(
        self,
        limit: int | None = None,
    ) -> List[Dict]:
        """
        Return latest RAG executions.
        """

        df = self._load_dataframe()

        if df.empty:

            return []

        # ------------------------------------------------------
        # Sort newest first
        # ------------------------------------------------------

        if COL_TIMESTAMP in df.columns:

            df[COL_TIMESTAMP] = pd.to_datetime(
                df[COL_TIMESTAMP],
                errors="coerce",
            )

            df = df.sort_values(
                COL_TIMESTAMP,
                ascending=False,
                na_position="last",
            )

        # ------------------------------------------------------
        # Limit
        # ------------------------------------------------------

        if limit is not None:

            limit = max(
                1,
                int(limit),
            )

            df = df.head(limit)

        # ------------------------------------------------------
        # Convert timestamp back to string
        # ------------------------------------------------------

        if COL_TIMESTAMP in df.columns:

            df[COL_TIMESTAMP] = (
                df[COL_TIMESTAMP]
                .dt.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )

        # ------------------------------------------------------
        # Replace NaN values
        # ------------------------------------------------------

        df = df.fillna("")

        return df.to_dict(
            orient="records"
        )

    # ==========================================================
    # Dashboard KPIs
    # ==========================================================

    def get_summary(self) -> Dict:
        """
        Calculate dashboard KPIs.

        questions_today:
            Number of executions recorded today.

        avg_confidence:
            Average retrieval confidence today.

        avg_response_time:
            Average total execution time today.

        today_cost:
            Total LLM cost today.

        success_rate:
            Successful execution percentage today.
        """

        df = self._load_dataframe()

        if df.empty:

            return {

                "questions_today": 0,

                "avg_confidence": 0,

                "avg_response_time": 0,

                "today_cost": 0,

                "success_rate": 0,

            }

        # ------------------------------------------------------
        # Convert timestamp
        # ------------------------------------------------------

        if COL_TIMESTAMP in df.columns:

            df[COL_TIMESTAMP] = pd.to_datetime(
                df[COL_TIMESTAMP],
                errors="coerce",
            )

            today = pd.Timestamp.now().normalize()

            df = df[
                df[COL_TIMESTAMP] >= today
            ]

        # No records today
        if df.empty:

            return {

                "questions_today": 0,

                "avg_confidence": 0,

                "avg_response_time": 0,

                "today_cost": 0,

                "success_rate": 0,

            }

        # ------------------------------------------------------
        # Numeric fields
        # ------------------------------------------------------

        confidence = self._numeric_series(
            df,
            COL_CONFIDENCE,
        )

        response_time = self._numeric_series(
            df,
            COL_TOTAL_TIME,
        )

        total_cost = self._numeric_series(
            df,
            COL_TOTAL_COST,
        )

        # ------------------------------------------------------
        # Status
        # ------------------------------------------------------

        if COL_STATUS in df.columns:

            status = (
                df[COL_STATUS]
                .fillna("")
                .astype(str)
                .str.upper()
            )

        else:

            status = pd.Series(
                "",
                index=df.index,
            )

        success_count = (
            status == "SUCCESS"
        ).sum()

        total_count = len(df)

        success_rate = (

            success_count /
            total_count *
            100

        ) if total_count else 0

        # ------------------------------------------------------
        # Summary
        # ------------------------------------------------------

        return {

            "questions_today":
                total_count,

            "avg_confidence":
                round(
                    confidence.mean(),
                    3,
                ),

            "avg_response_time":
                round(
                    response_time.mean(),
                    2,
                ),

            "today_cost":
                round(
                    total_cost.sum(),
                    6,
                ),

            "success_rate":
                round(
                    success_rate,
                    2,
                ),

        }

    # ==========================================================
    # Analytics
    # ==========================================================

    def get_analytics(self) -> Dict:
        """
        Calculate analytics metrics across
        the available history records.
        """

        df = self._load_dataframe()

        if df.empty:

            return {

                "response_times": [],

                "confidence": [],

                "costs": [],

                "providers": {},

                "avg_retrieval_time": 0,

                "avg_provider_latency": 0,

                "avg_context_length": 0,

                "avg_chunks": 0,

                "retrieval_times": [],

                "llm_times": [],

                "total_tokens": [],

            }

        # ------------------------------------------------------
        # Core analytics
        # ------------------------------------------------------

        response_times = (
            self._numeric_series(
                df,
                COL_TOTAL_TIME,
            )
            .tolist()
        )

        confidence = (
            self._numeric_series(
                df,
                COL_CONFIDENCE,
            )
            .tolist()
        )

        costs = (
            self._numeric_series(
                df,
                COL_TOTAL_COST,
            )
            .tolist()
        )

        # ------------------------------------------------------
        # Provider usage
        # ------------------------------------------------------

        providers = {}

        if COL_PROVIDER in df.columns:

            providers = (

                df[COL_PROVIDER]

                .fillna("Unknown")

                .astype(str)

                .replace(
                    "",
                    "Unknown",
                )

                .value_counts()

                .to_dict()

            )

        # ------------------------------------------------------
        # Provider latency
        # ------------------------------------------------------

        provider_latency = (
            self._numeric_series(
                df,
                COL_PROVIDER_LATENCY,
            )
        )

        # ------------------------------------------------------
        # Context length
        # ------------------------------------------------------

        context_length = (
            self._numeric_series(
                df,
                COL_CONTEXT_LENGTH,
            )
        )

        # ------------------------------------------------------
        # Chunks
        # ------------------------------------------------------

        chunks = (
            self._numeric_series(
                df,
                COL_CHUNKS,
            )
        )

        # ------------------------------------------------------
        # Retrieval time
        # ------------------------------------------------------

        retrieval_time = (
            self._numeric_series(
                df,
                COL_RETRIEVAL_TIME,
            )
        )

        avg_retrieval_time = (
            retrieval_time.mean()
        )

        retrieval_times = (
            retrieval_time
            .tolist()
        )

        # ------------------------------------------------------
        # LLM time
        # ------------------------------------------------------

        llm_times = (
            self._numeric_series(
                df,
                COL_LLM_TIME,
            )
            .tolist()
        )

        # ------------------------------------------------------
        # Total tokens
        # ------------------------------------------------------

        total_tokens = (
            self._numeric_series(
                df,
                COL_TOTAL_TOKENS,
            )
            .tolist()
        )

        # ------------------------------------------------------
        # Return
        # ------------------------------------------------------

        return {

            "response_times":
                response_times,

            "confidence":
                confidence,

            "costs":
                costs,

            "providers":
                providers,

            "avg_retrieval_time":
                round(
                    avg_retrieval_time,
                    2,
                ),

            "avg_provider_latency":
                round(
                    provider_latency.mean(),
                    2,
                ),

            "avg_context_length":
                round(
                    context_length.mean(),
                    0,
                ),

            "avg_chunks":
                round(
                    chunks.mean(),
                    2,
                ),

            "retrieval_times":
                retrieval_times,

            "llm_times":
                llm_times,

            "total_tokens":
                total_tokens,

        }

    # ==========================================================
    # Stats
    # ==========================================================

    def get_total_questions(self) -> int:

        return len(
            self._load_dataframe()
        )

    # ==========================================================

    def get_success_count(self) -> int:

        df = self._load_dataframe()

        if df.empty:

            return 0

        if COL_STATUS not in df.columns:

            return 0

        return int(
            (
                df[COL_STATUS]
                .fillna("")
                .astype(str)
                .str.upper()
                == "SUCCESS"
            ).sum()
        )

    # ==========================================================

    def get_failure_count(self) -> int:

        df = self._load_dataframe()

        if df.empty:

            return 0

        if COL_STATUS not in df.columns:

            return 0

        return int(
            (
                df[COL_STATUS]
                .fillna("")
                .astype(str)
                .str.upper()
                == "FAILED"
            ).sum()
        )