"""
Settings Service

Reads runtime configuration
for the currently active LLM.
"""

from __future__ import annotations

import logging
from typing import Dict

from config import (
    LLM_PROVIDER,
    LLM_MODEL,
    LLM_TEMPERATURE,
    MAX_OUTPUT_TOKENS,
    TOP_K_RESULTS,
    SIMILARITY_THRESHOLD,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

logger = logging.getLogger(__name__)


class SettingsService:

    def __init__(self):

        logger.info(
            "SettingsService initialized."
        )

    def get_summary(self) -> Dict:

        return {

            "provider": LLM_PROVIDER,

            "model": LLM_MODEL,

            "temperature": LLM_TEMPERATURE,

            "max_output_tokens": MAX_OUTPUT_TOKENS,

            "top_k": TOP_K_RESULTS,

            "similarity_threshold": SIMILARITY_THRESHOLD,

            "chunk_size": CHUNK_SIZE,

            "chunk_overlap": CHUNK_OVERLAP,

        }