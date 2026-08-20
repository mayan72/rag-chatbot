"""
Question / Answer Export Service

Reads the existing rag_runs.jsonl file
and creates an Excel file containing only:

    - Question
    - Answer
"""

import json
import logging
from pathlib import Path
from typing import List, Dict

import pandas as pd

from config import LOG_DIR


logger = logging.getLogger(__name__)


JSONL_LOG = Path(LOG_DIR) / "rag_runs.jsonl"


class QAExportService:

    # ==========================================================
    # Read JSONL
    # ==========================================================

    def get_question_answers(self) -> List[Dict]:

        records = []

        if not JSONL_LOG.exists():

            logger.warning(
                "RAG JSONL log not found: %s",
                JSONL_LOG,
            )

            return records

        try:

            with open(
                JSONL_LOG,
                "r",
                encoding="utf-8",
            ) as file:

                for line_number, line in enumerate(
                    file,
                    start=1,
                ):

                    line = line.strip()

                    if not line:
                        continue

                    try:

                        data = json.loads(line)

                    except json.JSONDecodeError:

                        logger.warning(
                            "Skipping invalid JSONL line: %d",
                            line_number,
                        )

                        continue

                    question = data.get(
                        "question"
                    )

                    answer = data.get(
                        "answer"
                    )

                    # --------------------------------------------------
                    # Only records containing Q&A
                    # --------------------------------------------------

                    if not question:
                        continue

                    if answer is None:
                        continue

                    records.append(
                        {
                            "Question": str(
                                question
                            ),

                            "Answer": str(
                                answer
                            ),
                        }
                    )

        except Exception:

            logger.exception(
                "Failed to read RAG JSONL log."
            )

            raise

        logger.info(
            "Loaded %d question/answer records.",
            len(records),
        )

        return records

    # ==========================================================
    # Create Excel
    # ==========================================================

    def create_excel(self):

        records = self.get_question_answers()

        df = pd.DataFrame(
            records,
            columns=[
                "Question",
                "Answer",
            ],
        )

        return df