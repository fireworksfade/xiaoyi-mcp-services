import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from mcp.client import Client

from iot_diagnosis import server
from iot_diagnosis.external import QdrantVectorStore
from iot_diagnosis.llm import LLMResponse
from iot_diagnosis.mqtt import MQTTIngestor
from iot_diagnosis.repository import DiagnosisRepository, iso
from iot_diagnosis.retrieval import search_knowledge


async def test_core_tools_are_discoverable_and_callable(tmp_path, monkeypatch) -> None:
    repository = DiagnosisRepository(str(tmp_path / "diagnosis.db"))
    monkeypatch.setattr(server, "repository", repository)

    async with Client(server.mcp) as client:
        tools = await client.list_tools()
        listed = tools.tools if hasattr(tools, "tools") else tools["tools"]
        assert {
            tool[0]
            if isinstance(tool, tuple)
            else tool["name"]
            if isinstance(tool, dict)
            else tool.name
            for tool in listed
        } == {
            "diagnose_fault",
            "get_device_status",
            "get_device_logs",
            "search_knowledge",
            "search_fault_cases",
            "add_verified_fault_case",
        }

        status = await client.call_tool("get_device_status", {"device_id": "ESP32_05"})
        assert status.structured_content["data"]["rssi"] == -47

        logs = await client.call_tool(
            "get_device_logs", {"device_id": "ESP32_05", "level": "ERROR"}
        )
        assert "keep alive timeout" in logs.structured_content["data"]["logs"][0]["message"]


async def test_router_retrieval_diagnosis_and_traceability(tmp_path, monkeypatch) -> None:
    repository = DiagnosisRepository(str(tmp_path / "diagnosis.db"))
    monkeypatch.setattr(server, "repository", repository)

    async with Client(server.mcp) as client:
        realtime = await client.call_tool(
            "diagnose_fault",
            {"device_id": "ESP32_05", "query": "设备当前 RSSI 是多少？"},
        )
        realtime_data = realtime.structured_content["data"]
        assert realtime_data["route"]["router"] == "rule"
        assert realtime_data["route"]["retrieval_required"] is False

        diagnosis = await client.call_tool(
            "diagnose_fault",
            {
                "device_id": "ESP32_05",
                "query": "设备为什么一直断开 MQTT 连接？",
                "logs": ["MQTT disconnected", "MQTT keep alive timeout"],
            },
        )
        item = diagnosis.structured_content["data"]
        assert item["fault_type"] == "mqtt_connection"
        assert item["route"]["router"] == "heuristic_fallback"
        assert {source["source_type"] for source in item["sources"]} >= {
            "fault_cases",
            "mqtt_docs",
        }
        assert item["observability"]["retrieval_count"] >= 2


async def test_verified_case_gate_and_dynamic_retrieval(tmp_path, monkeypatch) -> None:
    repository = DiagnosisRepository(str(tmp_path / "diagnosis.db"))
    monkeypatch.setattr(server, "repository", repository)
    payload = {
        "device_id": "ESP32_05",
        "fault_type": "wifi",
        "fault_name": "WiFi 弱信号",
        "symptoms": ["频繁重连"],
        "logs": ["RSSI -85 dBm"],
        "cause": "接入点距离过远",
        "solution": "调整天线并缩短距离",
    }

    async with Client(server.mcp) as client:
        denied = await client.call_tool(
            "add_verified_fault_case",
            {**payload, "verified": False, "verified_by": "operator"},
        )
        assert denied.structured_content["error"]["code"] == "CASE_NOT_VERIFIED"

        added = await client.call_tool(
            "add_verified_fault_case",
            {**payload, "verified": True, "verified_by": "operator"},
        )
        assert added.structured_content["data"]["indexed"] is True

        cases = await client.call_tool(
            "search_fault_cases",
            {"query": "WiFi RSSI -85", "device_type": "ESP32", "fault_type": "wifi"},
        )
        assert cases.structured_content["data"]["results"][0]["fault_name"] == "WiFi 弱信号"


def test_vector_search_without_realtime_state_does_not_fail(tmp_path, monkeypatch) -> None:
    repository = DiagnosisRepository(str(tmp_path / "diagnosis.db"))
    monkeypatch.setattr(
        repository,
        "vector_search",
        lambda *_args: [
            {
                "source": "mqtt_docs",
                "id": "VECTOR_MQTT_01",
                "title": "MQTT vector result",
                "content": "MQTT keep alive timeout",
                "score": 0.9,
            }
        ],
    )

    result = search_knowledge(repository, "MQTT keep alive timeout", sources=[])

    assert result["results"]
    assert all(item["source"] != "realtime_db" for item in result["results"])


def test_qdrant_existing_collection_is_not_recreated(monkeypatch) -> None:
    requests = []

    def fake_request(self, method, path, payload=None):
        requests.append((method, path, payload))
        return {"status": "ok"}

    monkeypatch.setattr(QdrantVectorStore, "_request", fake_request)

    QdrantVectorStore("http://qdrant:6333", "iot_diagnosis_knowledge")

    assert requests == [("GET", "/collections/iot_diagnosis_knowledge", None)]


def test_mqtt_style_status_and_log_updates(tmp_path) -> None:
    repository = DiagnosisRepository(str(tmp_path / "diagnosis.db"))
    repository.upsert_status(
        "ESP32_05",
        {
            "timestamp": iso(),
            "wifi": "connected",
            "mqtt": "connected",
            "rssi": -51,
            "temperature": 28.2,
            "uptime": 90000,
        },
    )
    repository.add_log(
        "ESP32_05",
        {"level": "INFO", "module": "mqtt", "message": "MQTT connected"},
    )
    assert repository.get_device_status("ESP32_05")["mqtt_status"] == "connected"
    assert repository.get_device_logs("ESP32_05", level="INFO")[0]["message"] == "MQTT connected"


def test_fault_topic_is_ingested_and_stale_heartbeat_marks_device_offline(tmp_path) -> None:
    repository = DiagnosisRepository(str(tmp_path / "diagnosis.db"), offline_after_seconds=30)
    stale = (datetime.now(timezone.utc) - timedelta(seconds=31)).isoformat()
    repository.upsert_status(
        "ESP32_STALE",
        {"timestamp": stale, "online": True, "mqtt": "connected", "wifi": "connected"},
    )
    assert repository.get_device_status("ESP32_STALE")["online"] is False
    assert repository.get_device_status("ESP32_STALE")["reported_online"] is True

    ingestor = MQTTIngestor(repository)
    ingestor._on_message(
        None,
        None,
        SimpleNamespace(
            topic="iot/ESP32_STALE/fault",
            payload=json.dumps(
                {
                    "fault_type": "mqtt_timeout",
                    "level": "ERROR",
                    "message": "MQTT keep alive timeout",
                }
            ).encode("utf-8"),
        ),
    )
    fault = repository.get_device_logs("ESP32_STALE", level="ERROR")[0]
    assert fault["module"] == "fault:mqtt_timeout"
    assert fault["message"] == "MQTT keep alive timeout"


async def test_configured_llm_drives_router_and_diagnosis(tmp_path, monkeypatch) -> None:
    repository = DiagnosisRepository(str(tmp_path / "diagnosis.db"))
    monkeypatch.setattr(server, "repository", repository)

    class FakeLLM:
        available = True

        def route(self, _query, _state, _logs):
            return LLMResponse(
                data={
                    "fault_type": "mqtt",
                    "need_retrieval": True,
                    "sources": ["fault_cases", "mqtt_docs", "realtime_db"],
                    "top_k": 4,
                },
                latency_ms=12.5,
                input_tokens=80,
                output_tokens=20,
            )

        def diagnose(self, _query, _state, _logs, _contexts):
            return LLMResponse(
                data={
                    "fault_type": "mqtt_connection",
                    "fault_name": "LLM 诊断的 MQTT 超时",
                    "cause": "客户端心跳间隔大于 Broker 超时",
                    "solutions": ["缩短心跳间隔", "检查 Broker 超时"],
                    "severity": "high",
                    "confidence": 0.91,
                    "requires_manual_inspection": False,
                },
                latency_ms=25.0,
                input_tokens=160,
                output_tokens=60,
            )

    monkeypatch.setattr(
        "iot_diagnosis.diagnosis.DiagnosisLLMClient.from_env",
        lambda: FakeLLM(),
    )
    async with Client(server.mcp) as client:
        result = await client.call_tool(
            "diagnose_fault",
            {"device_id": "ESP32_05", "query": "MQTT 为什么反复超时？"},
        )
    item = result.structured_content["data"]
    assert item["route"]["router"] == "llm"
    assert item["fault_name"] == "LLM 诊断的 MQTT 超时"
    assert item["observability"]["llm_latency_ms"] == 37.5
    assert item["observability"]["input_tokens"] == 240
    assert item["observability"]["output_tokens"] == 80
