"""
Metrics Service

Aggregates data from all services.

This is the ONLY service that Dashboard APIs should call.
"""

from __future__ import annotations

import logging
from typing import Dict

from services.history_service import HistoryService
from services.knowledge_service import KnowledgeService
from services.settings_service import SettingsService

logger = logging.getLogger(__name__)


class MetricsService:

    def __init__(self):

        self.history = HistoryService()

        self.knowledge = KnowledgeService()

        self.settings = SettingsService()

        logger.info("MetricsService initialized.")

    # ==========================================================
    # Dashboard Summary
    # ==========================================================

    def get_dashboard_summary(self) -> Dict:

        history = self.history.get_summary()

        knowledge = self.knowledge.get_summary()

        settings = self.settings.get_summary()

        return {

            "status": "Ready",

            **history,

            **knowledge,

            **settings,

        }

    # ==========================================================
    # History
    # ==========================================================

    def get_history(self):

        return self.history.get_history()

    # ==========================================================
    # Analytics
    # ==========================================================

    def get_analytics(self):

        return self.history.get_analytics()

    # ==========================================================
    # Knowledge
    # ==========================================================

    def get_knowledge(self):

        return self.knowledge.get_summary()

    # ==========================================================
    # Settings
    # ==========================================================

    def get_settings(self):

        return self.settings.get_summary()

    