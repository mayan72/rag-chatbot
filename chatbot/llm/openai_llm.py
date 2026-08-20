"""
OpenAI LLM
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Any, List

from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
)

from llm.base_llm import BaseLLM

logger = logging.getLogger(__name__)


class OpenAILLM(BaseLLM):

    provider = "openai"

    model = LLM_MODEL

    input_cost_per_million = 0.00

    output_cost_per_million = 0.00

    def __init__(self):

        logger.info(
            "Initializing OpenAI..."
        )

        self.client = OpenAI(
            api_key=OPENAI_API_KEY,
        )

        logger.info(
            "OpenAI Ready."
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
    ) -> Dict[str, Any]:

        logger.info(
            "Calling OpenAI..."
        )

        start = time.perf_counter()

        try:

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=LLM_TEMPERATURE,
            )

            llm_time = (
                time.perf_counter() - start
            ) * 1000

            answer = (
                response.choices[0]
                .message
                .content
                .strip()
            )

            usage = getattr(
                response,
                "usage",
                None,
            )

            input_tokens = (
                usage.prompt_tokens
                if usage
                else 0
            )

            output_tokens = (
                usage.completion_tokens
                if usage
                else 0
            )

            total_tokens = (
                usage.total_tokens
                if usage
                else (
                    input_tokens
                    + output_tokens
                )
            )

            logger.info(
                "OpenAI completed | "
                "Input=%d | "
                "Output=%d | "
                "Total=%d | "
                "Time=%.2f ms",
                input_tokens,
                output_tokens,
                total_tokens,
                llm_time,
            )

            return {

                "answer": answer,

                "input_tokens": input_tokens,

                "output_tokens": output_tokens,

                "total_tokens": total_tokens,

                "llm_time_ms": round(
                    llm_time,
                    2,
                ),
            }

        except Exception:

            logger.exception(
                "OpenAI request failed."
            )

            raise