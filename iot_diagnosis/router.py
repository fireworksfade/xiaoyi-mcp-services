from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from iot_diagnosis.llm import DiagnosisLLMClient, LLMClientError


ALLOWED_SOURCES = {
    "fault_cases",
    "mqtt_docs",
    "wifi_docs",
    "sensor_docs",
    "device_docs",
    "realtime_db",
}


@dataclass(frozen=True)
class RouteDecision:
    fault_type: str
    need_retrieval: bool
    sources: list[str]
    top_k: int
    router: str
    llm_latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def infer_fault_type(text: str) -> str:
    lowered = text.lower()
    if "mqtt" in lowered or "broker" in lowered or "keep alive" in lowered:
        return "mqtt"
    if "wifi" in lowered or "rssi" in lowered or "无线" in lowered:
        return "wifi"
    if "温度" in lowered or "sensor" in lowered or "传感器" in lowered:
        return "sensor"
    if "内存" in lowered or "heap" in lowered or "重启" in lowered:
        return "device"
    return "unknown"


def route_query(
    query: str,
    state: dict[str, Any] | None = None,
    logs: list[str] | None = None,
    llm_client: DiagnosisLLMClient | None = None,
) -> RouteDecision:
    normalized = query.lower().replace(" ", "")
    realtime_patterns = (
        "当前rssi",
        "现在rssi",
        "当前温度",
        "设备在线",
        "是否在线",
        "mqtt状态",
        "当前状态",
    )
    diagnostic_markers = ("为什么", "原因", "故障", "异常", "断开", "超时", "失败")
    fault_type = infer_fault_type(" ".join([query, *(logs or [])]))

    if any(pattern in normalized for pattern in realtime_patterns) and not any(
        marker in normalized for marker in diagnostic_markers
    ):
        return RouteDecision(
            fault_type=fault_type,
            need_retrieval=False,
            sources=["realtime_db"],
            top_k=1,
            router="rule",
        )

    client = llm_client or DiagnosisLLMClient.from_env()
    if client.available:
        try:
            response = client.route(query, state, logs or [])
            data = response.data
            sources = validate_sources(list(data.get("sources") or []))
            if not sources:
                raise ValueError("INVALID_REQUEST")
            top_k = min(max(int(data.get("top_k") or 5), 1), 20)
            return RouteDecision(
                fault_type=str(data.get("fault_type") or fault_type),
                need_retrieval=bool(data.get("need_retrieval", True)),
                sources=sources,
                top_k=top_k,
                router="llm",
                llm_latency_ms=response.latency_ms,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )
        except (LLMClientError, TypeError, ValueError):
            pass

    source_map = {
        "mqtt": ["fault_cases", "mqtt_docs", "realtime_db"],
        "wifi": ["fault_cases", "wifi_docs", "realtime_db"],
        "sensor": ["fault_cases", "sensor_docs", "realtime_db"],
        "device": ["fault_cases", "device_docs", "realtime_db"],
        "unknown": ["fault_cases", "device_docs", "realtime_db"],
    }
    sources = source_map[fault_type]
    if state and state.get("mqtt_status") == "disconnected" and "mqtt_docs" not in sources:
        sources = ["fault_cases", "mqtt_docs", *sources[1:]]
    return RouteDecision(
        fault_type=fault_type,
        need_retrieval=True,
        sources=list(dict.fromkeys(sources)),
        top_k=5,
        router="heuristic_fallback",
    )


def validate_sources(sources: list[str]) -> list[str]:
    invalid = set(sources) - ALLOWED_SOURCES
    if invalid:
        raise ValueError("INVALID_REQUEST")
    return list(dict.fromkeys(sources))
