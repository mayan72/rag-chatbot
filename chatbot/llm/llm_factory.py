"""
LLM Factory

Creates the appropriate LLM based on configuration.
"""

from __future__ import annotations

import logging

from config import (
    LLM_PROVIDER,
)

from llm.google_llm import GoogleLLM
from llm.openai_llm import OpenAILLM

logger = logging.getLogger(__name__)


class LLMFactory:

    @staticmethod
    def create():

        logger.info(
            "Selected LLM Provider: %s",
            LLM_PROVIDER,
        )

        if LLM_PROVIDER == "google":

            return GoogleLLM()

        if LLM_PROVIDER == "openai":

            return OpenAILLM()

        raise ValueError(
            f"Unsupported LLM Provider: {LLM_PROVIDER}"
        )