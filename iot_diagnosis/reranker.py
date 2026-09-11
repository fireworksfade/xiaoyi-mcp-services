from __future__ import annotations

import math
import json
import os
import re
from collections import Counter
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _tokens(text: str) -> list[str]:
    lowered = text.lower()
    words = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", lowered)
    compact = re.sub(r"\s+", "", lowered)
    return words + [compact[index : index + 3] for index in range(max(0, len(compact) - 2))]


def lexical_similarity(query: str, content: str) -> float:
    left = Counter(_tokens(query))
    right = Counter(_tokens(content))
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(token, 0) for token, value in left.items())
    norm_left = math.sqrt(sum(value * value for value in left.values()))
    norm_right = math.sqrt(sum(value * value for value in right.values()))
    return dot / (norm_left * norm_right) if norm_left and norm_right else 0.0


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        expected_source: str,
        top_k: int,
    ) -> list[dict[str, Any]]: ...


class WeightedReranker:
    """Deterministic default that can be replaced by a learned reranker."""

    name = "weighted"
    used_fallback = False

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        expected_source: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        ranked = []
        for candidate in candidates:
            retrieval_score = min(max(float(candidate.get("retrieval_score") or 0.0), 0.0), 1.0)
            lexical_score = lexical_similarity(query, str(candidate.get("content") or ""))
            source_bonus = 0.05 if candidate.get("source") == expected_source else 0.0
            verified_case_bonus = 0.03 if candidate.get("source") == "fault_cases" else 0.0
            score = min(
                1.0,
                0.72 * retrieval_score + 0.20 * lexical_score + source_bonus + verified_case_bonus,
            )
            item = {**candidate, "score": round(score, 4)}
            item.pop("retrieval_score", None)
            ranked.append(item)
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[:top_k]


class RemoteReranker:
    """Cross-encoder reranker served by the local Qwen model service."""

    name = "qwen3_remote"

    def __init__(self, url: str, timeout_seconds: float = 30):
        self.url = url
        self.timeout_seconds = timeout_seconds
        self.fallback = WeightedReranker()
        self.used_fallback = False

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        expected_source: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        payload = json.dumps(
            {
                "query": query,
                "documents": [str(item.get("content") or "") for item in candidates],
                "top_n": min(top_k, len(candidates)),
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = Request(
            self.url,
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
            ranked = []
            for item in result["results"]:
                candidate = dict(candidates[int(item["index"])])
                candidate["score"] = round(float(item["score"]), 4)
                candidate.pop("retrieval_score", None)
                ranked.append(candidate)
            return ranked[:top_k]
        except (
            HTTPError,
            URLError,
            TimeoutError,
            ValueError,
            KeyError,
            IndexError,
            TypeError,
        ):
            self.used_fallback = True
            return self.fallback.rerank(
                query,
                candidates,
                expected_source=expected_source,
                top_k=top_k,
            )


def reranker_from_env() -> Reranker:
    provider = os.getenv("DIAGNOSIS_RERANKER_PROVIDER", "weighted").strip().lower()
    if provider == "weighted":
        return WeightedReranker()
    if provider in {"remote", "qwen3"}:
        url = os.getenv("DIAGNOSIS_RERANKER_URL", "").strip()
        if not url:
            raise ValueError("RERANKER_PROVIDER_NOT_CONFIGURED")
        return RemoteReranker(
            url,
            float(os.getenv("DIAGNOSIS_RERANKER_TIMEOUT_SECONDS", "30")),
        )
    raise ValueError("RERANKER_PROVIDER_INVALID")
