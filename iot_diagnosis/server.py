from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp_types import ToolAnnotations
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse

from common.results import failure, success
from iot_diagnosis.diagnosis import diagnose
from iot_diagnosis.mqtt import MQTTIngestor
from iot_diagnosis.repository import DiagnosisRepository
from iot_diagnosis.retrieval import search_fault_cases as retrieve_fault_cases
from iot_diagnosis.retrieval import search_knowledge as retrieve_knowledge


repository = DiagnosisRepository(
    os.getenv("DIAGNOSIS_DATABASE_PATH", "data/iot_diagnosis.db"),
    int(os.getenv("DIAGNOSIS_OFFLINE_AFTER_SECONDS", "120")),
)


@asynccontextmanager
async def service_lifespan(_server):
    ingestor = None
    if os.getenv("MQTT_ENABLED", "false").lower() == "true":
        ingestor = MQTTIngestor(repository)
        ingestor.start()
    try:
        yield {"mqtt_ingestor": ingestor}
    finally:
        if ingestor:
            ingestor.stop()


mcp = MCPServer(
    "iot-diagnosis",
    title="IoT Diagnosis MCP Server",
    description="Hybrid Adaptive Multi-Source RAG for ESP32 fault diagnosis",
    version="1.0.0",
    lifespan=service_lifespan,
)
read_only = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)


@mcp.tool(annotations=read_only)
def diagnose_fault(
    device_id: Annotated[str, Field(min_length=1, max_length=120)],
    query: Annotated[str, Field(min_length=1, max_length=2000)],
    logs: list[str] | None = None,
    use_realtime_state: bool = True,
) -> dict[str, Any]:
    """执行设备状态、日志、自适应多源检索、重排和结构化故障诊断。"""
    try:
        return success(diagnose(repository, device_id, query, logs, use_realtime_state))
    except LookupError:
        return failure("DEVICE_NOT_FOUND", f"Device {device_id} does not exist")
    except Exception:
        return failure("LLM_ERROR", "诊断流程执行失败", retryable=True)


@mcp.tool(annotations=read_only)
def get_device_status(device_id: str) -> dict[str, Any]:
    """查询设备在线状态、WiFi、RSSI、MQTT、温度、uptime 和最近上报时间。"""
    item = repository.get_device_status(device_id)
    return success(item) if item else failure("DEVICE_NOT_FOUND", f"Device {device_id} does not exist")


@mcp.tool(annotations=read_only)
def get_device_logs(
    device_id: str,
    limit: Annotated[int, Field(ge=1, le=500)] = 50,
    level: Annotated[str | None, Field(pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")] = None,
) -> dict[str, Any]:
    """读取指定设备最近日志，可按日志级别过滤。"""
    items = repository.get_device_logs(device_id, limit, level)
    if items is None:
        return failure("DEVICE_NOT_FOUND", f"Device {device_id} does not exist")
    return success({"device_id": device_id, "logs": items})


@mcp.tool(annotations=read_only)
def search_knowledge(
    query: Annotated[str, Field(min_length=1, max_length=2000)],
    sources: list[str] | None = None,
    top_k: Annotated[int, Field(ge=1, le=20)] = 5,
) -> dict[str, Any]:
    """按指定来源或 Adaptive Router 的选择执行多源知识检索和统一重排。"""
    try:
        return success(retrieve_knowledge(repository, query, sources, top_k))
    except ValueError:
        return failure("INVALID_REQUEST", "包含不支持的知识源")
    except Exception:
        return failure("RETRIEVAL_FAILED", "知识检索失败", retryable=True)


@mcp.tool(annotations=read_only)
def search_fault_cases(
    query: Annotated[str, Field(min_length=1, max_length=2000)],
    device_type: str | None = "ESP32",
    fault_type: str | None = None,
    top_k: Annotated[int, Field(ge=1, le=20)] = 5,
) -> dict[str, Any]:
    """检索已由人工确认的历史故障案例。"""
    try:
        results = retrieve_fault_cases(repository, query, device_type, fault_type, top_k)
        return success({"results": results})
    except Exception:
        return failure("RETRIEVAL_FAILED", "故障案例检索失败", retryable=True)


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    )
)
def add_verified_fault_case(
    device_id: str,
    fault_type: str,
    fault_name: str,
    symptoms: list[str],
    logs: list[str],
    cause: str,
    solution: str,
    verified: bool,
    verified_by: str | None = None,
) -> dict[str, Any]:
    """仅在人工确认信息完整时写入故障案例并加入检索索引。"""
    if not verified or not verified_by or not verified_by.strip():
        return failure("CASE_NOT_VERIFIED", "故障案例必须经过人工确认")
    try:
        return success(
            repository.add_verified_fault_case(
                {
                    "device_id": device_id,
                    "fault_type": fault_type,
                    "fault_name": fault_name,
                    "symptoms": symptoms,
                    "logs": logs,
                    "cause": cause,
                    "solution": solution,
                    "verified_by": verified_by.strip(),
                }
            )
        )
    except Exception:
        return failure("DATABASE_ERROR", "故障案例写入失败", retryable=True)


@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request) -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "service": "iot-diagnosis-mcp",
            "version": "1.0.0",
            "storage": repository.storage_status(),
        }
    )


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host=os.getenv("DIAGNOSIS_HOST", "127.0.0.1"),
        port=int(os.getenv("DIAGNOSIS_PORT", "9001")),
    )
