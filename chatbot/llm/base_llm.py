"""
Base LLM Interface

Every LLM provider must inherit from this class.

Responsibilities
----------------
1. Provide a common interface.
2. Standardize responses.
3. Make chatbot.py provider-independent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseLLM(ABC):
    """
    Base class for all LLM providers.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        input_cost_per_million: float,
        output_cost_per_million: float,
    ):
        self.provider = provider
        self.model = model
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million

    input_cost_per_million: float = 0.0

    output_cost_per_million: float = 0.0

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Generate an answer from the LLM.

        Must return:

        {
            "answer": "...",

            "input_tokens": int,

            "output_tokens": int,

            "total_tokens": int,

            "llm_time_ms": float,
        }
        """
        raise NotImplementedError