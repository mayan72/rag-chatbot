"""
Generic string matching for any tabular schema.

No file-specific column names or commodity names.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable, Optional, Tuple


def normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.casefold().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_fuzzy_score(query: str, candidate: str) -> float:
    q = normalize_text(query)
    c = normalize_text(candidate)

    if not q or not c:
        return 0.0

    if q == c:
        return 1.0

    if q in c or c in q:
        shorter = min(len(q), len(c))
        longer = max(len(q), len(c))
        return 0.82 + 0.18 * (shorter / longer)

    q_tokens = q.split()
    c_tokens = c.split()
    c_token_set = set(c_tokens)

    token_scores = []
    for token in q_tokens:
        if token in c_token_set:
            token_scores.append(1.0)
            continue
        best = max(
            SequenceMatcher(None, token, other).ratio()
            for other in c_tokens
        )
        token_scores.append(best)

    token_avg = sum(token_scores) / len(token_scores)
    seq = SequenceMatcher(None, q, c).ratio()
    return max(token_avg, seq)


def best_column_match(
    query: str,
    columns: Iterable[str],
    min_score: float = 0.72,
) -> Optional[Tuple[str, float]]:
    best_name = None
    best_score = 0.0

    for column in columns:
        score = token_fuzzy_score(query, column)
        if score > best_score:
            best_name = column
            best_score = score

    if best_name is None or best_score < min_score:
        return None

    return best_name, best_score


def best_value_match(
    query: str,
    values: Iterable[object],
    min_score: float = 0.72,
) -> Optional[Tuple[object, float]]:
    best_value = None
    best_score = 0.0

    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text.lower() == "nan":
            continue
        score = token_fuzzy_score(query, text)
        if score > best_score:
            best_value = value
            best_score = score

    if best_value is None or best_score < min_score:
        return None

    return best_value, best_score
