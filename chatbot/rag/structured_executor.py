"""
Safely execute a QueryPlan against stored tables.

No raw SQL is generated. Only allow-listed aggregations and filters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from config import MAX_VALUE_MATCH_CANDIDATES
from rag.query_planner import QueryFilter, QueryPlan
from rag.table_store import INTERNAL_COLUMNS, TableStore
from rag.text_normalize import best_value_match, normalize_text
from debug_trace import dbg

logger = logging.getLogger(__name__)


@dataclass
class StructuredResult:
    matched: bool
    answer: str = ""
    value: Any = None
    operation: str = ""
    row_count: int = 0
    filters: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    table_id: str = ""
    document_name: str = ""


class StructuredExecutor:

    def __init__(self, table_store: Optional[TableStore] = None):
        self.table_store = table_store or TableStore()

    def execute(
        self,
        plan: QueryPlan,
        schemas: List[dict],
    ) -> StructuredResult:
        if plan.mode != "aggregate":
            return StructuredResult(matched=False)

        tables = self._select_tables(plan, schemas)
        if not tables:
            return StructuredResult(
                matched=True,
                answer="I don't have enough information in my knowledge base.",
            )

        best_result = None
        for schema, frame in tables:
            result = self._execute_on_table(plan, schema, frame)
            if best_result is None:
                best_result = result
                continue
            if result.row_count > best_result.row_count:
                best_result = result
            elif (
                result.row_count == best_result.row_count
                and result.value is not None
                and best_result.value is None
            ):
                best_result = result

        return best_result or StructuredResult(
            matched=True,
            answer="I don't have enough information in my knowledge base.",
        )

    def _select_tables(
        self,
        plan: QueryPlan,
        schemas: List[dict],
    ) -> List[tuple]:
        selected = []
        needed = {item.column for item in plan.filters}
        if plan.target_column:
            needed.add(plan.target_column)
        if plan.second_column:
            needed.add(plan.second_column)

        for schema in schemas:
            document_id = schema.get("document_id")
            if plan.table_id and document_id != plan.table_id:
                continue
            frame = self.table_store.load_dataframe(document_id)
            if frame.empty:
                continue
            if needed and not needed.issubset(set(frame.columns)):
                continue
            selected.append((schema, frame))
        return selected

    def _execute_on_table(
        self,
        plan: QueryPlan,
        schema: dict,
        frame: pd.DataFrame,
    ) -> StructuredResult:
        filtered = frame.copy()
        applied = []

        for query_filter in plan.filters:
            filtered, applied_filter = self._apply_filter(filtered, query_filter)
            applied.append(applied_filter)

        dbg(
            "EXECUTOR_FILTERED",
            document_name=schema.get("document_name"),
            document_id=schema.get("document_id"),
            rows_before=len(frame),
            rows_after=len(filtered),
            operation=plan.operation,
            target_column=plan.target_column,
            second_column=plan.second_column,
            applied=applied,
            sample_after=filtered.head(3).astype(str).to_dict(orient="records"),
        )

        value = self._aggregate(filtered, plan)
        row_count = int(len(filtered))
        if plan.operation == "correlation" and plan.target_column and plan.second_column:
            if plan.target_column in filtered.columns and plan.second_column in filtered.columns:
                paired = pd.DataFrame(
                    {
                        "left": pd.to_numeric(
                            filtered[plan.target_column], errors="coerce"
                        ),
                        "right": pd.to_numeric(
                            filtered[plan.second_column], errors="coerce"
                        ),
                    }
                ).dropna()
                row_count = int(len(paired))
        dbg(
            "EXECUTOR_VALUE",
            document_name=schema.get("document_name"),
            value=value,
            row_count=row_count,
        )
        sources = self._sample_sources(filtered, schema)

        if value is None:
            answer = "I don't have enough information in my knowledge base."
        elif plan.operation == "count":
            answer = str(int(value))
        elif isinstance(value, float):
            answer = str(round(value, 4))
        else:
            answer = str(value)

        return StructuredResult(
            matched=True,
            answer=answer,
            value=value,
            operation=plan.operation,
            row_count=row_count,
            filters=applied,
            sources=sources,
            table_id=schema.get("document_id", ""),
            document_name=schema.get("document_name", ""),
        )

    def _apply_filter(
        self,
        frame: pd.DataFrame,
        query_filter: QueryFilter,
    ) -> tuple:
        column = query_filter.column
        if column not in frame.columns:
            return frame.iloc[0:0], {
                "column": column,
                "op": query_filter.op,
                "value": query_filter.value,
                "matched": False,
            }

        series = frame[column]
        op = query_filter.op
        raw_value = query_filter.value

        if op in {"gt", "gte", "lt", "lte"}:
            numeric = pd.to_numeric(series, errors="coerce")
            try:
                target = float(raw_value)
            except (TypeError, ValueError):
                return frame.iloc[0:0], {
                    "column": column,
                    "op": op,
                    "value": raw_value,
                    "matched": False,
                }
            comparators = {
                "gt": numeric > target,
                "gte": numeric >= target,
                "lt": numeric < target,
                "lte": numeric <= target,
            }
            mask = comparators[op].fillna(False)
            return frame[mask], {
                "column": column,
                "op": op,
                "value": raw_value,
                "matched": True,
            }

        candidates = [
            value
            for value in series.dropna().unique()[:MAX_VALUE_MATCH_CANDIDATES]
        ]
        resolved = best_value_match(raw_value, candidates, min_score=0.68)
        compare_value = str(resolved[0]) if resolved else str(raw_value)
        wanted = normalize_text(compare_value)
        normalized = series.astype(str).map(normalize_text)

        if op == "ne":
            mask = normalized != wanted
        elif op == "contains":
            mask = normalized.str.contains(wanted, regex=False) | (
                normalized == wanted
            )
            if resolved:
                mask = mask | (normalized == normalize_text(resolved[0]))
        else:
            mask = normalized == wanted
            if resolved and resolved[1] >= 0.72:
                mask = mask | normalized.str.contains(
                    normalize_text(resolved[0]),
                    regex=False,
                )

        return frame[mask.fillna(False)], {
            "column": column,
            "op": op,
            "requested_value": raw_value,
            "matched_value": compare_value,
            "matched": True,
        }

    def _aggregate(self, frame: pd.DataFrame, plan: QueryPlan):
        if plan.operation == "count":
            return int(len(frame))

        if plan.operation == "correlation":
            return self._correlation(frame, plan.target_column, plan.second_column)

        column = plan.target_column
        if not column or column not in frame.columns:
            return None

        if plan.operation == "distinct_count":
            return int(frame[column].nunique(dropna=True))

        numeric = pd.to_numeric(frame[column], errors="coerce")
        numeric = numeric.dropna()
        if numeric.empty:
            return None

        if plan.operation == "sum":
            return float(numeric.sum())
        if plan.operation == "avg":
            return float(numeric.mean())
        if plan.operation == "min":
            return numeric.min()
        if plan.operation == "max":
            return numeric.max()
        return None

    def _correlation(
        self,
        frame: pd.DataFrame,
        left: Optional[str],
        right: Optional[str],
    ):
        if not left or not right:
            return None
        if left not in frame.columns or right not in frame.columns:
            return None
        paired = pd.DataFrame(
            {
                "left": pd.to_numeric(frame[left], errors="coerce"),
                "right": pd.to_numeric(frame[right], errors="coerce"),
            }
        ).dropna()
        if len(paired) < 2:
            return None
        if paired["left"].nunique() < 2 or paired["right"].nunique() < 2:
            return None
        value = paired["left"].corr(paired["right"], method="pearson")
        if pd.isna(value):
            return None
        return float(value)

    def _sample_sources(
        self,
        frame: pd.DataFrame,
        schema: dict,
    ) -> List[Dict[str, Any]]:
        samples = []
        preview_columns = [
            column["name"]
            for column in schema.get("columns", [])
            if column.get("name") not in INTERNAL_COLUMNS
        ][:8]

        for _, row in frame.head(5).iterrows():
            item = {
                "document_name": schema.get("document_name"),
                "document_id": schema.get("document_id"),
                "row_number": int(row.get("__row_number", 0) or 0),
            }
            for column in preview_columns:
                if column in row:
                    item[column] = None if pd.isna(row[column]) else str(row[column])
            samples.append(item)
        return samples
