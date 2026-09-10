from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from iot_diagnosis.repository import DiagnosisRepository
from iot_diagnosis.router import infer_fault_type, route_query, validate_sources


def _tokens(text: str) -> list[str]:
    lowered = text.lower()
    words = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", lowered)
    compact = re.sub(r"\s+", "", lowered)
    trigrams = [compact[index : index + 3] for index in range(max(0, len(compact) - 2))]
    return words + trigrams


def similarity(query: str, content: str) -> float:
    left = Counter(_tokens(query))
    right = Counter(_tokens(content))
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(token, 0) for token, value in left.items())
    norm_left = math.sqrt(sum(value * value for value in left.values()))
    norm_right = math.sqrt(sum(value * value for value in right.values()))
    return dot / (norm_left * norm_right) if norm_left and norm_right else 0.0


def rewrite_query(
    query: str,
    state: dict[str, Any] | None,
    logs: list[str] | None,
) -> str:
    context = [query]
    if state:
        context.extend(
            [
                f"WiFi={state.get('wifi_status')}",
                f"RSSI={state.get('rssi')}",
                f"MQTT={state.get('mqtt_status')}",
                f"温度={state.get('temperature')}",
            ]
        )
    context.extend((logs or [])[:10])
    return "；".join(str(item) for item in context if item)


def search_knowledge(
    repository: DiagnosisRepository,
    query: str,
    sources: list[str] | None = None,
    top_k: int = 5,
    *,
    state: dict[str, Any] | None = None,
    logs: list[str] | None = None,
) -> dict[str, Any]:
    selected = validate_sources(sources or route_query(query, state, logs).sources)
    rewritten = rewrite_query(query, state, logs)
    candidates: list[dict[str, Any]] = []

    if "fault_cases" in selected:
        for case in repository.fault_cases():
            content = "；".join(
                [
                    case["fault_name"],
                    *case["symptoms"],
                    *case["logs"],
                    case["cause"],
                    case["solution"],
                ]
            )
            candidates.append(
                {
                    "source": "fault_cases",
                    "id": case["fault_id"],
                    "title": case["fault_name"],
                    "content": content,
                    "retrieval_score": similarity(rewritten, content),
                }
            )

    for document in repository.knowledge_documents(selected):
        candidates.append(
            {
                "source": document["source"],
                "id": document["source_id"],
                "title": document["title"],
                "content": document["content"],
                "retrieval_score": similarity(rewritten, document["content"]),
            }
        )

    if "realtime_db" in selected and state:
        content = (
            f"设备在线={state.get('online')}；WiFi={state.get('wifi_status')}；"
            f"RSSI={state.get('rssi')}；MQTT={state.get('mqtt_status')}；"
            f"温度={state.get('temperature')}"
        )
        candidates.append(
            {
                "source": "realtime_db",
                "id": str(state["device_id"]),
                "title": "当前设备状态",
                "content": content,
                "retrieval_score": similarity(rewritten, content),
            }
        )

    for item in repository.vector_search(rewritten, selected, max(top_k * 3, 10)):
        if not all(key in item for key in ("source", "id", "title", "content")):
            continue
        candidates.append(
            {
                "source": item["source"],
                "id": item["id"],
                "title": item["title"],
                "content": item["content"],
                "retrieval_score": float(item.get("score") or 0.0),
            }
        )

    deduplicated: dict[tuple[str, str], dict[str, Any]] = {}
    for item in candidates:
        key = (item["source"], item["id"])
        current = deduplicated.get(key)
        if not current or item["retrieval_score"] > current["retrieval_score"]:
            deduplicated[key] = item

    fault_type = infer_fault_type(" ".join([query, *(logs or [])]))
    expected_source = f"{fault_type}_docs"
    reranked = []
    for item in deduplicated.values():
        source_bonus = 0.08 if item["source"] in {"fault_cases", expected_source} else 0.0
        item = {**item, "score": round(min(1.0, item["retrieval_score"] + source_bonus), 4)}
        item.pop("retrieval_score")
        reranked.append(item)
    reranked.sort(key=lambda item: item["score"], reverse=True)
    return {
        "query": query,
        "rewritten_query": rewritten,
        "selected_sources": selected,
        "candidate_count": len(candidates),
        "results": reranked[:top_k],
    }


def search_fault_cases(
    repository: DiagnosisRepository,
    query: str,
    device_type: str | None,
    fault_type: str | None,
    top_k: int,
) -> list[dict[str, Any]]:
    results = []
    for case in repository.fault_cases(device_type, fault_type):
        content = "；".join(
            [case["fault_name"], *case["symptoms"], *case["logs"], case["cause"], case["solution"]]
        )
        results.append(
            {
                "fault_id": case["fault_id"],
                "fault_name": case["fault_name"],
                "symptoms": case["symptoms"],
                "cause": case["cause"],
                "solution": case["solution"],
                "verified": True,
                "similarity": round(similarity(query, content), 4),
            }
        )
    return sorted(results, key=lambda item: item["similarity"], reverse=True)[:top_k]
