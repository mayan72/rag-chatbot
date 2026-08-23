"""
Temporary RAG debugger.

Writes each pipeline step to:
  - the server terminal
  - chatbot/logs/rag_debug.log

Disable with:  RAG_DEBUG=0
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

from config import LOG_DIR

logger = logging.getLogger("RAG_DEBUG")

ENABLED = os.getenv("RAG_DEBUG", "1").strip().lower() not in {
    "0",
    "false",
    "off",
    "no",
}

DEBUG_LOG = LOG_DIR / "rag_debug.log"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def dbg(step: str, **data: Any) -> None:
    if not ENABLED:
        return

    payload = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "step": step,
        **data,
    }
    line = json.dumps(payload, default=str, ensure_ascii=False)
    logger.warning("RAG_DEBUG | %s", line)
    with DEBUG_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print("\n========== RAG_DEBUG |", step, "==========")
    print(json.dumps(payload, indent=2, default=str, ensure_ascii=False))
    print("====================================\n", flush=True)
