"""
Turn a natural-language question into a structured plan.

The plan only uses columns discovered from uploaded tables.
Nothing is hardcoded to a specific spreadsheet.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rag.table_store import INTERNAL_COLUMNS
from rag.text_normalize import (
    best_column_match,
    best_value_match,
    normalize_text,
    token_fuzzy_score,
)

logger = logging.getLogger(__name__)

ALLOWED_OPERATIONS = {
    "count",
    "sum",
    "avg",
    "min",
    "max",
    "distinct_count",
    "correlation",
}

ALLOWED_FILTER_OPS = {
    "eq",
    "ne",
    "contains",
    "gt",
    "gte",
    "lt",
    "lte",
}

AGGREGATE_HINTS = (
    "count",
    "how many",
    "how much",
    "number of",
    "total number",
    "total ",
    "sum of",
    "sum ",
    "average",
    "avg ",
    "mean ",
    "minimum",
    "maximum",
    "min ",
    "max ",
    "distinct",
    "unique",
    "correlation",
    "correlated",
    "corr(",
    "pearson",
)

PLANNER_SYSTEM_PROMPT = """
You convert a user question into a JSON query plan over uploaded tables.

Use ONLY column names from the provided schemas.
Never invent columns.
If the question is not a table aggregation (count/sum/avg/min/max/distinct count/correlation),
return {"mode":"semantic"}.

Return JSON only with this shape:
{
  "mode": "aggregate" | "semantic",
  "operation": "count" | "sum" | "avg" | "min" | "max" | "distinct_count" | "correlation",
  "target_column": null or column name,
  "second_column": null or column name (required for correlation),
  "table_id": null or document_id,
  "filters": [
    {"column": "exact column name", "op": "eq|contains|gt|gte|lt|lte|ne", "value": "..."}
  ]
}
"""


@dataclass
class QueryFilter:
    column: str
    op: str
    value: str
    score: float = 1.0


@dataclass
class QueryPlan:
    mode: str
    operation: str = "count"
    target_column: Optional[str] = None
    second_column: Optional[str] = None
    table_id: Optional[str] = None
    filters: List[QueryFilter] = field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""


class QueryPlanner:

    def plan(
        self,
        question: str,
        schemas: List[dict],
        llm: Any = None,
    ) -> QueryPlan:
        if not schemas:
            return QueryPlan(mode="semantic", reason="no tables")

        heuristic = self._heuristic_plan(question, schemas)
        if heuristic.mode == "aggregate" and heuristic.confidence >= 0.78:
            return heuristic

        looks_structured = self._looks_like_aggregate(
            question.casefold(),
            self._public_columns(schemas),
        )

        if llm is not None and looks_structured:
            llm_plan = self._llm_plan(question, schemas, llm)
            if llm_plan is not None and llm_plan.mode == "aggregate":
                return llm_plan

        return heuristic

    def _heuristic_plan(self, question: str, schemas: List[dict]) -> QueryPlan:
        lowered = question.casefold()
        columns = self._public_columns(schemas)
        if not columns:
            return QueryPlan(mode="semantic", reason="tables have no columns")

        if not self._looks_like_aggregate(lowered, columns):
            return QueryPlan(mode="semantic", reason="not an aggregation question")

        correlation_plan = self._correlation_plan(question, schemas, columns)
        if correlation_plan is not None:
            return correlation_plan
        if self._detect_operation(lowered) == "correlation":
            return QueryPlan(
                mode="aggregate",
                operation="correlation",
                confidence=0.8,
                reason="correlation columns not found",
            )

        operation = self._detect_operation(lowered)
        filters: List[QueryFilter] = []
        used_spans = []

        where_matches = list(
            re.finditer(
                r"(?:where|with|whose)\s+(.+?)\s+(?:is|equals|=|==)\s+['\"]?([^,'\".]+?)['\"]?(?=\s+(?:and|for|in|on)\b|[?.]|$)",
                question,
                flags=re.IGNORECASE,
            )
        )

        for match in where_matches:
            column_hit = best_column_match(match.group(1), columns)
            if not column_hit:
                continue
            filters.append(
                QueryFilter(
                    column=column_hit[0],
                    op="eq",
                    value=match.group(2).strip(),
                    score=column_hit[1],
                )
            )
            used_spans.append((match.start(), match.end()))

        region_match = re.search(
            r"(?i)\b(?:in|for)\s+(?:the\s+)?([A-Za-z0-9_ -]+?)\s+region\b",
            question,
        )
        if region_match:
            region_column = best_column_match("region", columns, min_score=0.6)
            if region_column:
                filters.append(
                    QueryFilter(
                        column=region_column[0],
                        op="eq",
                        value=region_match.group(1).strip(),
                        score=region_column[1],
                    )
                )
                used_spans.append((region_match.start(), region_match.end()))

        remainder = question
        for start, end in reversed(used_spans):
            remainder = remainder[:start] + " " + remainder[end:]

        remainder = re.sub(
            r"(?i)\b(give me|tell me|what|the|sum of|average of|avg of|count of rows|count of|how many rows|number of rows|total|rows where|where|is|are|for|with|equals|please|region)\b",
            " ",
            remainder,
        )
        remainder = remainder.strip(" .?")

        target_column = None
        metric_match = re.search(
            r"(?i)\b(?:sum|total|average|avg|min|max|mean|distinct count|unique count)\s+(?:of\s+)?([A-Za-z0-9_ ]+?)(?=\s+(?:where|for|with|in)\b|$)",
            question,
        )
        if metric_match:
            metric_hit = best_column_match(
                metric_match.group(1),
                columns,
                min_score=0.55,
            )
            if metric_hit:
                target_column = metric_hit[0]

        if target_column is None and operation in {
            "sum",
            "avg",
            "min",
            "max",
            "distinct_count",
        }:
            target_hit = best_column_match(remainder, columns, min_score=0.6)
            if target_hit:
                target_column = target_hit[0]

        extra_hint = remainder
        if extra_hint:
            value_filter = self._infer_value_filter(extra_hint, schemas, columns)
            if value_filter is not None:
                already = {item.column for item in filters}
                if value_filter.column not in already:
                    filters.append(value_filter)

        confidence = 0.55
        if filters:
            confidence = 0.82 + min(0.1, 0.04 * len(filters))
            confidence = min(0.95, confidence)

        if operation == "count" and not filters:
            confidence = 0.8

        needed = {item.column for item in filters}
        if target_column:
            needed.add(target_column)
        table_id = self._choose_table_id(schemas, needed)

        return QueryPlan(
            mode="aggregate",
            operation=operation,
            target_column=target_column,
            table_id=table_id,
            filters=filters,
            confidence=confidence,
            reason="heuristic",
        )

    def _correlation_plan(
        self,
        question: str,
        schemas: List[dict],
        columns: List[str],
    ) -> Optional[QueryPlan]:
        pair = self._parse_correlation_pair(question, columns)
        if pair is None:
            return None
        left, right = pair
        table_id = self._choose_table_id(schemas, {left, right})
        return QueryPlan(
            mode="aggregate",
            operation="correlation",
            target_column=left,
            second_column=right,
            table_id=table_id,
            filters=[],
            confidence=0.93,
            reason="heuristic-correlation",
        )

    def _parse_correlation_pair(
        self,
        question: str,
        columns: List[str],
    ) -> Optional[tuple]:
        patterns = (
            r"(?i)\b(?:pearson\s+)?corr(?:elation)?(?:\s+coefficient)?\s+(?:between|of)\s+(.+?)\s+and\s+(.+?)\s*$",
            r"(?i)\bhow\s+(?:strongly|well)\s+(?:are|is)\s+(.+?)\s+(?:and|&)\s+(.+?)\s+correlated",
            r"(?i)\bcorr\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)",
        )
        match = None
        for pattern in patterns:
            match = re.search(pattern, question.strip().rstrip("?"))
            if match:
                break
        if not match:
            return None
        left_hit = best_column_match(match.group(1).strip(), columns, min_score=0.55)
        right_hit = best_column_match(match.group(2).strip(), columns, min_score=0.55)
        if not left_hit or not right_hit:
            return None
        if left_hit[0] == right_hit[0]:
            return None
        return left_hit[0], right_hit[0]

    def _looks_like_aggregate(self, lowered: str, columns: List[str]) -> bool:
        if any(hint in lowered for hint in AGGREGATE_HINTS):
            return True
        if re.search(r"\b(total|sum|average|avg)\b", lowered):
            for column in columns:
                if normalize_text(column) and normalize_text(column) in normalize_text(lowered):
                    return True
        return False

    def _choose_table_id(
        self,
        schemas: List[dict],
        needed: set,
    ) -> Optional[str]:
        if not needed:
            return None
        for schema in schemas:
            names = {
                column.get("name")
                for column in schema.get("columns", [])
                if column.get("name")
            }
            if needed.issubset(names):
                return schema.get("document_id")
        return None

    def _infer_value_filter(
        self,
        hint: str,
        schemas: List[dict],
        columns: List[str],
    ) -> Optional[QueryFilter]:
        hint_norm = normalize_text(hint)
        if not hint_norm:
            return None

        best: Optional[QueryFilter] = None

        for schema in schemas:
            for column in schema.get("columns", []):
                name = column.get("name")
                if not name or name in INTERNAL_COLUMNS:
                    continue
                for sample in column.get("sample_values", []):
                    score = token_fuzzy_score(hint, sample)
                    if best is None or score > best.score:
                        best = QueryFilter(
                            column=name,
                            op="contains",
                            value=hint,
                            score=score,
                        )

        column_hit = best_column_match(hint, columns, min_score=0.85)
        if column_hit and (best is None or column_hit[1] > best.score):
            return None

        if best is None or best.score < 0.72:
            return None

        return best

    def _detect_operation(self, lowered: str) -> str:
        if any(
            token in lowered
            for token in ("correlation", "correlated", "pearson", "corr(")
        ):
            return "correlation"
        if any(token in lowered for token in ("average", "avg ", "mean ")):
            return "avg"
        if re.search(r"\btotal\s+(number|count)\s+of\s+rows\b", lowered):
            return "count"
        if re.search(r"\b(sum|total|how much)\b", lowered):
            return "sum"
        if "sum of" in lowered or lowered.startswith("sum "):
            return "sum"
        if "distinct" in lowered or "unique" in lowered:
            return "distinct_count"
        if "minimum" in lowered or re.search(r"\bmin\b", lowered):
            return "min"
        if "maximum" in lowered or re.search(r"\bmax\b", lowered):
            return "max"
        return "count"

    def _public_columns(self, schemas: List[dict]) -> List[str]:
        names = []
        seen = set()
        for schema in schemas:
            for column in schema.get("columns", []):
                name = column.get("name")
                if not name or name in INTERNAL_COLUMNS or name in seen:
                    continue
                seen.add(name)
                names.append(name)
        return names

    def _llm_plan(
        self,
        question: str,
        schemas: List[dict],
        llm: Any,
    ) -> Optional[QueryPlan]:
        compact_schemas = []
        for schema in schemas:
            compact_schemas.append(
                {
                    "document_id": schema.get("document_id"),
                    "document_name": schema.get("document_name"),
                    "row_count": schema.get("row_count"),
                    "columns": [
                        {
                            "name": col["name"],
                            "sample_values": col.get("sample_values", [])[:8],
                        }
                        for col in schema.get("columns", [])
                        if col.get("name") not in INTERNAL_COLUMNS
                    ],
                }
            )

        messages = [
            {
                "role": "system",
                "content": PLANNER_SYSTEM_PROMPT.strip(),
            },
            {
                "role": "user",
                "content": (
                    "SCHEMAS:\n"
                    f"{json.dumps(compact_schemas, ensure_ascii=False)}\n\n"
                    f"QUESTION:\n{question}\n"
                ),
            },
        ]

        try:
            result = llm.generate(messages)
            parsed = self._parse_json(result.get("answer", ""))
        except Exception:
            logger.exception("LLM query planner failed.")
            return None

        if not parsed:
            return None

        mode = str(parsed.get("mode", "semantic")).lower()
        if mode != "aggregate":
            return QueryPlan(mode="semantic", reason="llm-semantic")

        operation = str(parsed.get("operation", "count")).lower()
        if operation not in ALLOWED_OPERATIONS:
            operation = "count"

        public_columns = self._public_columns(schemas)
        filters = []
        for item in parsed.get("filters") or []:
            column = str(item.get("column", "")).strip()
            resolved = best_column_match(column, public_columns, min_score=0.6)
            op = str(item.get("op", "eq")).lower()
            if op not in ALLOWED_FILTER_OPS:
                op = "eq"
            if not resolved:
                continue
            filters.append(
                QueryFilter(
                    column=resolved[0],
                    op=op,
                    value=str(item.get("value", "")).strip(),
                    score=resolved[1],
                )
            )

        target_column = parsed.get("target_column")
        if target_column:
            target_hit = best_column_match(
                str(target_column),
                public_columns,
                min_score=0.6,
            )
            target_column = target_hit[0] if target_hit else None

        second_column = parsed.get("second_column")
        if second_column:
            second_hit = best_column_match(
                str(second_column),
                public_columns,
                min_score=0.6,
            )
            second_column = second_hit[0] if second_hit else None

        table_id = parsed.get("table_id")
        valid_ids = {schema.get("document_id") for schema in schemas}
        if table_id not in valid_ids:
            table_id = None

        needed = {item.column for item in filters}
        if target_column:
            needed.add(target_column)
        if second_column:
            needed.add(second_column)
        if needed and table_id is None:
            table_id = self._choose_table_id(schemas, needed)

        return QueryPlan(
            mode="aggregate",
            operation=operation,
            target_column=target_column,
            second_column=second_column,
            table_id=table_id,
            filters=filters,
            confidence=0.9,
            reason="llm",
        )

    def _parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                return None
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
