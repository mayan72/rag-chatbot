"""
Production RAG Chatbot

Responsibilities
----------------
1. Receive user question
2. Retrieve relevant documents
3. Build prompt
4. Call Grok API
5. Return structured response
6. Log everything

This class NEVER directly interacts with Chroma.
All retrieval happens through SemanticRetriever.
"""

from __future__ import annotations
from logger.run_logger import RunLogger
import logging
import time
from datetime import datetime
import uuid
from typing import Dict, Any
from logger.cost_calculator import CostCalculator
from logger.run_logger import RunLogger
from config import (
    TOP_K_RESULTS,
    SIMILARITY_THRESHOLD,
    LLM_TEMPERATURE,
)
from llm.llm_factory import LLMFactory
from rag.retriever import SemanticRetriever
from rag.prompt_builder import PromptBuilder
from rag.hybrid_qa import HybridQAEngine

logger = logging.getLogger(__name__)


class RAGChatbot:
    """
    Production-ready RAG chatbot.
    """

    def __init__(self):

        logger.info("Initializing RAG Chatbot...")

        # -------------------------------------------------------
        # Retriever
        # -------------------------------------------------------

        self.retriever = SemanticRetriever()

        # -------------------------------------------------------
        # Prompt Builder
        # -------------------------------------------------------

        self.prompt_builder = PromptBuilder()
        self.run_logger = RunLogger()
        self.hybrid_qa = None

        # -------------------------------------------------------
        # Grok Client
        # -------------------------------------------------------

        # self.client = OpenAI(
        #     api_key=GROK_API_KEY,
        #     base_url=GROK_BASE_URL,
        # )
        # self.client = genai.Client(
        #     api_key=GOOGLE_API_KEY
        # )
        self.llm = LLMFactory.create()
        self.hybrid_qa = HybridQAEngine(llm=self.llm)

        self.cost_calculator = CostCalculator()

        logger.info("Grok client initialized.")

        logger.info("RAG Chatbot Ready.")

        # ============================================================
    # Private Methods
    # ============================================================

    
        # ============================================================

    def ask(
        self,
        question: str,
    ) -> Dict[str, Any]:
        """
        Execute the complete RAG pipeline.
        """

        logger.info("=" * 80)
        logger.info("New Question: %s", question)

        overall_start = time.perf_counter()

        request_id = str(uuid.uuid4())

        # -------------------------------------------------------
        # Step 0 : Structured table QA (counts / sums / filters)
        # -------------------------------------------------------

        structured = self.hybrid_qa.answer(question)

        if structured and structured.matched:

            total_time = (
                time.perf_counter() - overall_start
            ) * 1000

            log_payload = {
                "status": "SUCCESS",
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id,
                "provider": "structured",
                "model": "table-engine",
                "question": question,
                "answer": structured.answer,
                "confidence": 1.0,
                "max_similarity": 1.0,
                "should_answer": True,
                "chunks_retrieved": len(structured.sources or []),
                "context_length": 0,
                "retrieval_time_ms": 0,
                "llm_time_ms": 0,
                "llm_provider_latency_ms": 0,
                "total_time_ms": round(total_time, 2),
                "retrieval_threshold": SIMILARITY_THRESHOLD,
                "top_k": TOP_K_RESULTS,
                "temperature": LLM_TEMPERATURE,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "input_cost": 0,
                "output_cost": 0,
                "embedding_cost": 0,
                "total_cost": 0,
                "sources": structured.sources or [],
                "error": "",
            }

            self.run_logger.log_success(log_payload)

            return {
                "answer": structured.answer,
                "confidence": 1.0,
                "provider": "structured",
                "model": "table-engine",
                "sources": structured.sources or [],
                "retrieval_time_ms": 0,
                "llm_time_ms": 0,
                "llm_provider_latency_ms": 0,
                "total_time_ms": round(total_time, 2),
                "chunks_retrieved": len(structured.sources or []),
                "context_length": 0,
                "retrieval_threshold": SIMILARITY_THRESHOLD,
                "top_k": TOP_K_RESULTS,
                "temperature": LLM_TEMPERATURE,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost": 0,
            }

        # -------------------------------------------------------
        # Step 1 : Retrieve
        # -------------------------------------------------------

        retrieval = self.retriever.retrieve(question)

        logger.info(
            "Retrieval completed | "
            "confidence=%.4f | "
            "max_similarity=%.4f",
            retrieval.confidence,
            retrieval.max_similarity,
        )

        # -------------------------------------------------------
        # Step 2 : Similarity Check
        # -------------------------------------------------------

        if not retrieval.should_answer:

            logger.warning(
                "Similarity below threshold. "
                "Skipping LLM."
            )

            total_time = (
                time.perf_counter() - overall_start
            ) * 1000

            answer = (
                "I don't have enough information "
                "in my knowledge base."
            )

            log_payload = {

                "status": "SUCCESS",

                "timestamp": datetime.now().isoformat(),

                "request_id": request_id,

                "provider": self.llm.provider,

                "model": self.llm.model,

                "question": question,

                "answer": answer,

                "confidence": retrieval.confidence,

                "max_similarity":
                    retrieval.max_similarity,

                "should_answer":
                    retrieval.should_answer,

                "chunks_retrieved":
                    len(retrieval.chunks),

                "context_length":
                    len(retrieval.context),

                "retrieval_time_ms":
                    retrieval.retrieval_time_ms,

                "llm_time_ms": 0,

                "llm_provider_latency_ms": 0,

                "total_time_ms":
                    round(total_time, 2),

                "retrieval_threshold":
                    SIMILARITY_THRESHOLD,

                "top_k":
                    TOP_K_RESULTS,

                "temperature":
                    LLM_TEMPERATURE,

                "input_tokens": 0,

                "output_tokens": 0,

                "total_tokens": 0,

                "input_cost": 0,

                "output_cost": 0,

                "embedding_cost": 0,

                "total_cost": 0,

                "sources":
                    retrieval.sources,

                "error": "",
            }

            self.run_logger.log_success(
                log_payload
            )

            return {

                "answer": answer,

                "confidence":
                    retrieval.confidence,

                "provider":
                    self.llm.provider,

                "model":
                    self.llm.model,

                "retrieval_time_ms":
                    retrieval.retrieval_time_ms,

                "llm_time_ms": 0,

                "llm_provider_latency_ms": 0,

                "total_time_ms":
                    round(total_time, 2),

                "chunks_retrieved":
                    len(retrieval.chunks),

                "context_length":
                    len(retrieval.context),

                "retrieval_threshold":
                    SIMILARITY_THRESHOLD,

                "top_k":
                    TOP_K_RESULTS,

                "temperature":
                    LLM_TEMPERATURE,

                "input_tokens": 0,

                "output_tokens": 0,

                "total_tokens": 0,

                "cost": 0,

            }

        # -------------------------------------------------------
        # Step 3 : Prompt
        # -------------------------------------------------------

        messages = self.prompt_builder.build(
            question=question,
            context=retrieval.context,
        )

        # -------------------------------------------------------
        # Step 4 : LLM
        # -------------------------------------------------------

        logger.info("=" * 80)
        logger.info(
            "QUESTION:\n%s",
            question,
        )
        logger.info("=" * 80)
        logger.info(
            "CONTEXT SENT TO LLM:\n%s",
            retrieval.context,
        )
        logger.info("=" * 80)

        provider_start = time.perf_counter()

        llm_result = self.llm.generate(
            messages
        )

        provider_latency = (
            time.perf_counter() - provider_start
        ) * 1000

        # -------------------------------------------------------
        # Step 5 : Cost
        # -------------------------------------------------------

        cost = self.cost_calculator.calculate(

            provider=self.llm.provider,

            input_tokens=
                llm_result["input_tokens"],

            output_tokens=
                llm_result["output_tokens"],

            embedding_tokens=0,

        )

        # -------------------------------------------------------
        # Step 6 : Total Time
        # -------------------------------------------------------

        total_time = (
            time.perf_counter() - overall_start
        ) * 1000

        logger.info(
            "Pipeline Finished | Total Time=%.2f ms",
            total_time,
        )

        # -------------------------------------------------------
        # Step 7 : Logging
        # -------------------------------------------------------

        log_payload = {

            "status": "SUCCESS",

            "timestamp":
                datetime.now().isoformat(),

            "request_id":
                request_id,

            "provider":
                self.llm.provider,

            "model":
                self.llm.model,

            "question":
                question,

            "answer":
                llm_result["answer"],

            "confidence":
                retrieval.confidence,

            "max_similarity":
                retrieval.max_similarity,

            "should_answer":
                retrieval.should_answer,

            "chunks_retrieved":
                len(retrieval.chunks),

            "context_length":
                len(retrieval.context),

            "retrieval_time_ms":
                retrieval.retrieval_time_ms,

            "llm_time_ms":
                llm_result["llm_time_ms"],

            "llm_provider_latency_ms":
                round(
                    provider_latency,
                    2,
                ),

            "total_time_ms":
                round(
                    total_time,
                    2,
                ),

            "retrieval_threshold":
                SIMILARITY_THRESHOLD,

            "top_k":
                TOP_K_RESULTS,

            "temperature":
                LLM_TEMPERATURE,

            "input_tokens":
                llm_result["input_tokens"],

            "output_tokens":
                llm_result["output_tokens"],

            "total_tokens":
                llm_result["total_tokens"],

            "input_cost":
                cost["input_cost"],

            "output_cost":
                cost["output_cost"],

            "embedding_cost":
                cost["embedding_cost"],

            "total_cost":
                cost["total_cost"],

            "sources":
                retrieval.sources,

            "error": "",
        }

        self.run_logger.log_success(
            log_payload
        )

        # -------------------------------------------------------
        # Step 8 : Return
        # -------------------------------------------------------

        return {

            "answer":
                llm_result["answer"],

            "confidence":
                retrieval.confidence,

            "provider":
                self.llm.provider,

            "model":
                self.llm.model,

            "sources":
                retrieval.sources,

            "retrieval_time_ms":
                retrieval.retrieval_time_ms,

            "llm_time_ms":
                llm_result["llm_time_ms"],

            "llm_provider_latency_ms":
                round(
                    provider_latency,
                    2,
                ),

            "total_time_ms":
                round(
                    total_time,
                    2,
                ),

            "chunks_retrieved":
                len(retrieval.chunks),

            "context_length":
                len(retrieval.context),

            "retrieval_threshold":
                SIMILARITY_THRESHOLD,

            "top_k":
                TOP_K_RESULTS,

            "temperature":
                LLM_TEMPERATURE,

            "input_tokens":
                llm_result["input_tokens"],

            "output_tokens":
                llm_result["output_tokens"],

            "total_tokens":
                llm_result["total_tokens"],

            "cost":
                cost["total_cost"],

            "cost_breakdown":
                cost,

        }