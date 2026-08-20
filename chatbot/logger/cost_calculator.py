"""
Cost Calculator

Responsible for estimating LLM cost.

Update only this file when model pricing changes.
"""

from __future__ import annotations

import logging
from typing import Dict

from config import (
    MODEL_PRICING,
    EMBEDDING_COST_PER_MILLION,
)

logger = logging.getLogger(__name__)




class CostCalculator:

    def __init__(self):
        logger.info("CostCalculator initialized.")

    def calculate(
    self,
    provider: str,
    input_tokens: int,
    output_tokens: int,
    embedding_tokens: int = 0,
) -> Dict[str, float]:
        """
        Calculate total request cost.

        Returns
        -------
        {
            "input_tokens": ...,
            "output_tokens": ...,
            "embedding_tokens": ...,

            "input_cost": ...,
            "output_cost": ...,
            "embedding_cost": ...,

            "total_tokens": ...,
            "total_cost": ...
        }
        """
        pricing = MODEL_PRICING.get(provider)

        if pricing is None:
            raise ValueError(
                f"Unsupported provider: {provider}"
            )

        input_cost_per_million = pricing["input_per_million"]

        output_cost_per_million = pricing["output_per_million"]

        input_cost = (
            input_tokens / 1_000_000
        ) * input_cost_per_million

        output_cost = (
            output_tokens / 1_000_000
        ) * output_cost_per_million

        embedding_cost = (
            embedding_tokens / 1_000_000
        ) * EMBEDDING_COST_PER_MILLION

        total_tokens = (
            input_tokens
            + output_tokens
            + embedding_tokens
        )

        total_cost = (
            input_cost
            + output_cost
            + embedding_cost
        )

        logger.info(
            "Cost Calculation | "
            "Provider=%s | "
            "Input Tokens=%d | "
            "Output Tokens=%d | "
            "Embedding Tokens=%d | "
            "Total Tokens=%d | "
            "Total Cost=$%.8f",
            provider,
            input_tokens,
            output_tokens,
            embedding_tokens,
            total_tokens,
            total_cost,
        )

        return {
            "provider": provider,

            "input_tokens": input_tokens,

            "output_tokens": output_tokens,

            "embedding_tokens": embedding_tokens,

            "total_tokens": total_tokens,

            "input_cost": round(input_cost, 8),

            "output_cost": round(output_cost, 8),

            "embedding_cost": round(embedding_cost, 8),

            "total_cost": round(total_cost, 8),

        }