"""
Hybrid question answering.

1. If uploaded tables can answer an aggregation exactly, do that.
2. Otherwise fall back to semantic RAG.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from rag.query_planner import QueryPlanner
from rag.structured_executor import StructuredExecutor, StructuredResult
from rag.table_store import TableStore

logger = logging.getLogger(__name__)


class HybridQAEngine:

    def __init__(
        self,
        table_store: Optional[TableStore] = None,
        planner: Optional[QueryPlanner] = None,
        executor: Optional[StructuredExecutor] = None,
        llm: Any = None,
    ):
        self.table_store = table_store or TableStore()
        self.planner = planner or QueryPlanner()
        self.executor = executor or StructuredExecutor(self.table_store)
        self.llm = llm
        self.table_store.sync_from_data_dir()

    def answer(self, question: str) -> Optional[StructuredResult]:
        schemas = self.table_store.list_schemas()
        if not schemas:
            self.table_store.sync_from_data_dir()
            schemas = self.table_store.list_schemas()

        if not schemas:
            return None

        plan = self.planner.plan(
            question=question,
            schemas=schemas,
            llm=self.llm,
        )

        logger.info(
            "Query plan | mode=%s | op=%s | filters=%s | confidence=%.2f | reason=%s",
            plan.mode,
            plan.operation,
            [(item.column, item.op, item.value) for item in plan.filters],
            plan.confidence,
            plan.reason,
        )

        if plan.mode != "aggregate":
            return None

        return self.executor.execute(plan, schemas)
