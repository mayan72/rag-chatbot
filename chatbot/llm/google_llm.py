"""
Google Gemini LLM
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Any, List

from google import genai
from google.genai import types

from config import (
    GOOGLE_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    MAX_OUTPUT_TOKENS,
)

from llm.base_llm import BaseLLM

logger = logging.getLogger(__name__)


class GoogleLLM(BaseLLM):

    provider = "google"

    model = LLM_MODEL

    input_cost_per_million = 0.00

    output_cost_per_million = 0.00

    def __init__(self):

        logger.info(
            "Initializing Google Gemini..."
        )

        self.client = genai.Client(
            api_key=GOOGLE_API_KEY,
        )

        logger.info(
            "Google Gemini Ready."
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Generate response using Gemini.
        """

        logger.info(
            "Calling Gemini API..."
        )

        start = time.perf_counter()

        try:

            system_text = ""
            user_parts = []

            for message in messages:
                role = message.get("role", "")
                content = message.get("content", "")
                if role == "system":
                    system_text += content + "\n"
                else:
                    user_parts.append(content)

            response = self.client.models.generate_content(
                model=self.model,
                contents="\n\n".join(user_parts).strip(),
                config=types.GenerateContentConfig(
                    system_instruction=system_text.strip() or None,
                    temperature=LLM_TEMPERATURE,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                ),
            )

            llm_time = (
                time.perf_counter() - start
            ) * 1000

            answer = (
                response.text.strip()
            )

            usage = getattr(
                response,
                "usage_metadata",
                None,
            )

            input_tokens = (
                usage.prompt_token_count
                if usage
                else 0
            )

            output_tokens = (
                usage.candidates_token_count
                if usage
                else 0
            )

            total_tokens = (
                usage.total_token_count
                if usage
                else (
                    input_tokens
                    + output_tokens
                )
            )

            logger.info(
                "Gemini completed | "
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
                "Gemini request failed."
            )

            raise