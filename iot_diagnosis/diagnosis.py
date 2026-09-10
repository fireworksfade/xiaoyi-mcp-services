from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any

from iot_diagnosis.repository import DiagnosisRepository
from iot_diagnosis.llm import DiagnosisLLMClient, LLMClientError
from iot_diagnosis.retrieval import search_knowledge
from iot_diagnosis.router import RouteDecision, route_query


def _profile(text: str, state: dict[str, Any]) -> dict[str, Any]:
    lowered = text.lower()
    temperature = state.get("temperature")
    if "keep alive" in lowered or "mqtt" in lowered or state.get("mqtt_status") == "disconnected":
        return {
            "fault_type": "mqtt_connection",
            "fault_name": "MQTT Keep Alive Timeout",
            "cause": "MQTT Keep Alive 参数、Broker 超时配置或客户端心跳可能异常",
            "solutions": [
                "检查 MQTT Keep Alive 参数",
                "检查 Broker 连接超时配置",
                "确认客户端心跳是否正常发送",
            ],
        }
    if "wifi" in lowered or "rssi" in lowered or (state.get("rssi") or 0) < -75:
        return {
            "fault_type": "wifi_connection",
            "fault_name": "WiFi 信号或连接异常",
            "cause": "无线信号弱、信道拥塞或接入点距离可能导致连接不稳定",
            "solutions": ["检查天线与供电", "缩短接入点距离", "检查信道拥塞与重连日志"],
        }
    if "sensor" in lowered or "传感器" in lowered or (temperature is not None and temperature >= 70):
        return {
            "fault_type": "sensor_anomaly",
            "fault_name": "传感器读数异常",
            "cause": "传感器接线、供电、量程或环境温度可能异常",
            "solutions": ["核对接线和供电", "检查量程与采样周期", "使用参考设备交叉验证"],
        }
    return {
        "fault_type": "device_runtime",
        "fault_name": "设备运行状态异常",
        "cause": "当前证据不足，可能与固件、内存或网络状态有关",
        "solutions": ["收集更完整日志", "检查固件版本与复位原因", "安排人工现场检查"],
    }


def diagnose(
    repository: DiagnosisRepository,
    device_id: str,
    query: str,
    supplied_logs: list[str] | None,
    use_realtime_state: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    llm_client = DiagnosisLLMClient.from_env()
    state = repository.get_device_status(device_id)
    if not state:
        raise LookupError("DEVICE_NOT_FOUND")
    log_records = repository.get_device_logs(device_id, 50) or []
    logs = supplied_logs or [str(item["message"]) for item in log_records]
    route: RouteDecision = route_query(
        query,
        state if use_realtime_state else None,
        logs,
        llm_client,
    )

    retrieval_started = time.perf_counter()
    if route.need_retrieval:
        retrieval = search_knowledge(
            repository,
            query,
            route.sources,
            route.top_k,
            state=state if use_realtime_state else None,
            logs=logs,
        )
    else:
        retrieval = {
            "candidate_count": 0,
            "results": [],
            "selected_sources": route.sources,
            "rewritten_query": query,
        }
    retrieval_latency_ms = round((time.perf_counter() - retrieval_started) * 1000, 3)

    combined = " ".join([query, *logs, *(item["content"] for item in retrieval["results"])])
    profile = _profile(combined, state)
    llm_latency_ms = route.llm_latency_ms
    input_tokens = route.input_tokens
    output_tokens = route.output_tokens
    llm_fallback_reason = None if llm_client.available else "LLM_NOT_CONFIGURED"
    if llm_client.available:
        try:
            llm_response = llm_client.diagnose(query, state, logs, retrieval["results"])
            candidate = llm_response.data
            solutions = candidate.get("solutions")
            required = ("fault_type", "fault_name", "cause")
            if not all(isinstance(candidate.get(key), str) and candidate[key] for key in required):
                raise LLMClientError("LLM_RESPONSE_INVALID")
            if not isinstance(solutions, list) or not solutions:
                raise LLMClientError("LLM_RESPONSE_INVALID")
            profile = {
                "fault_type": candidate["fault_type"],
                "fault_name": candidate["fault_name"],
                "cause": candidate["cause"],
                "solutions": [str(item) for item in solutions],
                "severity": str(candidate.get("severity") or "medium"),
                "confidence": float(candidate.get("confidence") or 0.5),
                "requires_manual_inspection": bool(
                    candidate.get("requires_manual_inspection", False)
                ),
            }
            llm_latency_ms += llm_response.latency_ms
            input_tokens += llm_response.input_tokens
            output_tokens += llm_response.output_tokens
        except (LLMClientError, TypeError, ValueError):
            llm_fallback_reason = "LLM_DIAGNOSIS_FAILED"
    evidence = []
    if use_realtime_state:
        evidence.extend(
            [
                f"WiFi={state['wifi_status']}",
                f"RSSI={state['rssi']} dBm",
                f"MQTT={state['mqtt_status']}",
                f"temperature={state['temperature']} °C",
            ]
        )
    evidence.extend(logs[:5])

    scores = [float(item["score"]) for item in retrieval["results"]]
    retrieval_score = max(scores, default=0.8 if not route.need_retrieval else 0.0)
    case_similarity = max(
        (float(item["score"]) for item in retrieval["results"] if item["source"] == "fault_cases"),
        default=0.0,
    )
    evidence_score = min(1.0, len(evidence) / 5)
    llm_score = min(max(float(profile.get("confidence", 0.82)), 0.0), 1.0)
    if not llm_client.available and profile["fault_type"] == "device_runtime":
        llm_score = 0.55
    confidence = round(
        0.35 * retrieval_score + 0.25 * evidence_score + 0.20 * case_similarity + 0.20 * llm_score,
        4,
    )
    severity = profile.get("severity") or (
        "high" if not state["online"] or (state.get("temperature") or 0) >= 70 else "medium"
    )
    elapsed = round((time.perf_counter() - started) * 1000, 3)
    sources = [
        {"source_type": item["source"], "source_id": item["id"], "score": item["score"]}
        for item in retrieval["results"][:4]
    ]
    result = {
        "diagnosis_id": f"DIA_{datetime.now(timezone.utc):%Y%m%d}_{uuid.uuid4().hex[:8].upper()}",
        "device_id": device_id,
        "fault_type": profile["fault_type"],
        "fault_name": profile["fault_name"],
        "severity": severity,
        "confidence": confidence,
        "cause": profile["cause"],
        "evidence": list(dict.fromkeys(evidence)),
        "solutions": profile["solutions"],
        "requires_manual_inspection": bool(
            profile.get("requires_manual_inspection", confidence < 0.65)
        ),
        "route": {
            "router": route.router,
            "retrieval_required": route.need_retrieval,
            "selected_sources": route.sources,
        },
        "sources": sources,
        "observability": {
            "request_id": str(uuid.uuid4()),
            "retrieval_count": int(retrieval["candidate_count"]),
            "rerank_count": len(retrieval["results"]),
            "retrieval_latency_ms": retrieval_latency_ms,
            "llm_latency_ms": round(llm_latency_ms, 3),
            "total_latency_ms": elapsed,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "llm_fallback_reason": llm_fallback_reason,
            "error": None,
        },
    }
    repository.save_diagnosis({**result, "request_id": result["observability"]["request_id"], "query": query})
    return result
