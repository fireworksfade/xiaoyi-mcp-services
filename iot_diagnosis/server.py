from __future__ import annotations

import os
import asyncio
from contextlib import asynccontextmanager
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp_types import ToolAnnotations
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse

from common.results import failure, success
from iot_diagnosis.auth import auth_configuration
from iot_diagnosis.diagnosis import diagnose
from iot_diagnosis.embeddings import retrieval_model_status
from iot_diagnosis.ingestion import ingest_text
from iot_diagnosis.mqtt import MQTTIngestor
from iot_diagnosis.repository import DiagnosisRepository
from iot_diagnosis.retrieval import search_fault_cases as retrieve_fault_cases
from iot_diagnosis.retrieval import search_knowledge as retrieve_knowledge


repository = DiagnosisRepository(
    os.getenv("DIAGNOSIS_DATABASE_PATH", "data/iot_diagnosis.db"),
    int(os.getenv("DIAGNOSIS_OFFLINE_AFTER_SECONDS", "120")),
)
auth_settings, token_verifier = auth_configuration()


@asynccontextmanager
async def service_lifespan(_server):
    ingestor = None
    sync_task = None
    if os.getenv("MQTT_ENABLED", "false").lower() == "true":
        ingestor = MQTTIngestor(repository)
        ingestor.start()
    sync_interval = max(0.0, float(os.getenv("DIAGNOSIS_SYNC_RETRY_SECONDS", "30")))
    if sync_interval:
        async def retry_external_writes() -> None:
            while True:
                await asyncio.sleep(sync_interval)
                await asyncio.to_thread(repository.retry_external_sync)

        sync_task = asyncio.create_task(retry_external_writes())
    try:
        yield {"mqtt_ingestor": ingestor}
    finally:
        if sync_task:
            sync_task.cancel()
            try:
                await sync_task
            except asyncio.CancelledError:
                pass
        if ingestor:
            ingestor.stop()


mcp = MCPServer(
    "iot-diagnosis",
    title="IoT Diagnosis MCP Server",
    description="Hybrid Adaptive Multi-Source RAG for ESP32 fault diagnosis",
    version="1.3.0",
    lifespan=service_lifespan,
    auth=auth_settings,
    token_verifier=token_verifier,
)
read_only = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)
DeviceId = Annotated[str, Field(min_length=1, max_length=120)]
ShortText = Annotated[str, Field(min_length=1, max_length=300)]
LongText = Annotated[str, Field(min_length=1, max_length=4000)]
LogText = Annotated[str, Field(min_length=1, max_length=2000)]
LogList = Annotated[list[LogText], Field(max_length=50)]


def _record_diagnosis_failure(device_id: str, query: str, code: str, message: str) -> dict[str, str]:
    try:
        return repository.save_diagnosis_error(device_id, query, code, message)
    except Exception:
        return {}


@mcp.tool(annotations=read_only)
def diagnose_fault(
    device_id: DeviceId,
    query: Annotated[str, Field(min_length=1, max_length=2000)],
    logs: LogList | None = None,
    use_realtime_state: bool = True,
) -> dict[str, Any]:
    """执行设备状态、日志、自适应多源检索、重排和结构化故障诊断。"""
    try:
        return success(diagnose(repository, device_id, query, logs, use_realtime_state))
    except LookupError:
        message = f"Device {device_id} does not exist"
        trace = _record_diagnosis_failure(device_id, query, "DEVICE_NOT_FOUND", message)
        return failure("DEVICE_NOT_FOUND", message, details=trace)
    except Exception:
        message = "诊断流程执行失败"
        trace = _record_diagnosis_failure(device_id, query, "LLM_ERROR", message)
        return failure("LLM_ERROR", message, retryable=True, details=trace)


@mcp.tool(annotations=read_only)
def get_diagnosis_trace(
    diagnosis_id: Annotated[str, Field(min_length=1, max_length=120)],
) -> dict[str, Any]:
    """按诊断 ID 查询路由、检索上下文、耗时、Token 和错误记录。"""
    item = repository.get_diagnosis_trace(diagnosis_id)
    return (
        success(item)
        if item
        else failure("DIAGNOSIS_NOT_FOUND", f"Diagnosis {diagnosis_id} does not exist")
    )


@mcp.tool(annotations=read_only)
def get_device_status(device_id: DeviceId) -> dict[str, Any]:
    """查询设备在线状态、WiFi、RSSI、MQTT、温度、uptime 和最近上报时间。"""
    item = repository.get_device_status(device_id)
    return success(item) if item else failure("DEVICE_NOT_FOUND", f"Device {device_id} does not exist")


@mcp.tool(annotations=read_only)
def list_devices(
    device_type: Annotated[str | None, Field(max_length=120)] = None,
    online: bool | None = None,
    limit: Annotated[int, Field(ge=1, le=200)] = 100,
    offset: Annotated[int, Field(ge=0, le=100_000)] = 0,
) -> dict[str, Any]:
    """列出已发现设备及其最新状态，可按设备类型和当前在线状态过滤。"""
    return success(repository.list_devices(device_type, online, limit, offset))


@mcp.tool(annotations=read_only)
def get_device_logs(
    device_id: DeviceId,
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
def list_knowledge_documents(
    source: Annotated[
        str | None,
        Field(pattern="^(mqtt_docs|wifi_docs|sensor_docs|device_docs)$"),
    ] = None,
    device_type: Annotated[str | None, Field(max_length=120)] = None,
    limit: Annotated[int, Field(ge=1, le=200)] = 100,
    offset: Annotated[int, Field(ge=0, le=100_000)] = 0,
) -> dict[str, Any]:
    """列出已摄取知识文档及分块数量，不返回大段正文。"""
    return success(
        repository.list_knowledge_documents(source, device_type, limit, offset)
    )


@mcp.tool(annotations=read_only)
def list_fault_cases(
    device_type: Annotated[str | None, Field(max_length=120)] = None,
    limit: Annotated[int, Field(ge=1, le=200)] = 50,
    offset: Annotated[int, Field(ge=0, le=100_000)] = 0,
) -> dict[str, Any]:
    """分页列出已验证故障案例（含沉淀来源与根因摘要，不含逐条日志）。"""
    return success(repository.list_fault_cases(device_type, limit, offset))


@mcp.tool(annotations=read_only)
def search_fault_cases(
    query: Annotated[str, Field(min_length=1, max_length=2000)],
    device_type: Annotated[str | None, Field(max_length=120)] = "ESP32",
    fault_type: Annotated[str | None, Field(max_length=120)] = None,
    top_k: Annotated[int, Field(ge=1, le=20)] = 5,
) -> dict[str, Any]:
    """检索已由人工确认的历史故障案例。"""
    try:
        results = retrieve_fault_cases(repository, query, device_type, fault_type, top_k)
        return success({"results": results})
    except Exception:
        return failure("RETRIEVAL_FAILED", "故障案例检索失败", retryable=True)


@mcp.tool(annotations=read_only)
def list_diagnoses(
    device_id: Annotated[str | None, Field(max_length=120)] = None,
    fault_type: Annotated[str | None, Field(max_length=120)] = None,
    status: Annotated[str, Field(pattern="^(all|succeeded|failed)$")] = "all",
    limit: Annotated[int, Field(ge=1, le=200)] = 50,
    offset: Annotated[int, Field(ge=0, le=100_000)] = 0,
) -> dict[str, Any]:
    """列出诊断历史摘要，可过滤设备、故障类型及成功或失败状态。"""
    return success(
        repository.list_diagnoses(device_id, fault_type, status, limit, offset)
    )


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    )
)
def add_verified_fault_case(
    device_id: DeviceId,
    fault_type: Annotated[str, Field(min_length=1, max_length=120)],
    fault_name: ShortText,
    symptoms: Annotated[list[ShortText], Field(min_length=1, max_length=20)],
    logs: Annotated[list[LogText], Field(min_length=1, max_length=50)],
    cause: LongText,
    solution: LongText,
    verified: bool,
    verified_by: Annotated[str | None, Field(max_length=160)] = None,
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


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
def ingest_knowledge_text(
    source: str,
    document_id: Annotated[str, Field(min_length=1, max_length=120)],
    title: Annotated[str, Field(min_length=1, max_length=300)],
    content: Annotated[str, Field(min_length=1, max_length=200_000)],
    device_type: Annotated[str | None, Field(max_length=120)] = "ESP32",
    chunk_size: Annotated[int, Field(ge=300, le=4000)] = 1200,
    overlap: Annotated[int, Field(ge=0, le=500)] = 120,
) -> dict[str, Any]:
    """摄取已提取的文本或 Markdown，分块后写入知识库和向量索引。"""
    try:
        return success(
            ingest_text(
                repository,
                source=source,
                document_id=document_id,
                title=title,
                content=content,
                device_type=device_type,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )
    except ValueError as exc:
        return failure(str(exc), "知识文档参数无效")
    except Exception:
        return failure("DATABASE_ERROR", "知识文档写入失败", retryable=True)


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=True,
        openWorldHint=False,
    )
)
def delete_knowledge_document(
    source: str,
    document_id: Annotated[str, Field(min_length=1, max_length=120)],
) -> dict[str, Any]:
    """从 SQLite、MySQL 镜像和 Qdrant 向量索引中删除整个知识文档。"""
    try:
        return success(
            repository.delete_knowledge_document(source=source, document_id=document_id)
        )
    except ValueError as exc:
        return failure(str(exc), "知识文档参数无效")
    except Exception:
        return failure("DATABASE_ERROR", "知识文档删除失败", retryable=True)


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=True,
        openWorldHint=False,
    )
)
def delete_fault_case(
    fault_id: Annotated[str, Field(min_length=2, max_length=32)],
) -> dict[str, Any]:
    """删除一条已验证故障案例，并同步清理 MySQL 镜像与 Qdrant 向量。"""
    try:
        return success(repository.delete_fault_case(fault_id))
    except ValueError:
        return failure("FAULT_ID_INVALID", "案例编号无效")
    except Exception:
        return failure("DATABASE_ERROR", "故障案例删除失败", retryable=True)


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
def rebuild_vector_index(
    sources: Annotated[list[str] | None, Field(max_length=5)] = None,
) -> dict[str, Any]:
    """以 SQLite 中的知识分块和已确认案例为准，批量重建 Qdrant 向量索引。"""
    try:
        return success(repository.rebuild_vector_index(sources))
    except ValueError:
        return failure("INVALID_REQUEST", "包含不支持的知识源")
    except Exception:
        return failure("VECTOR_REBUILD_FAILED", "向量索引重建失败", retryable=True)


@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request) -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "service": "iot-diagnosis-mcp",
            "version": "1.3.0",
            "storage": repository.storage_status(),
        }
    )


@mcp.custom_route("/ready", methods=["GET"])
async def ready(_: Request) -> JSONResponse:
    storage = repository.storage_status()
    retrieval_models = await asyncio.to_thread(retrieval_model_status)
    issues = []
    if os.getenv("DIAGNOSIS_MYSQL_DSN", "").strip() and storage["mysql"] != "connected":
        issues.append("mysql")
    if os.getenv("DIAGNOSIS_QDRANT_URL", "").strip() and storage["qdrant"] != "connected":
        issues.append("qdrant")
    backlog_limit = max(0, int(os.getenv("DIAGNOSIS_READY_MAX_PENDING_SYNC", "1000")))
    if storage["outbox"]["pending"] > backlog_limit:
        issues.append("outbox")
    if retrieval_models["status"] == "unavailable":
        issues.append("retrieval_models")
    payload = {
        "status": "ready" if not issues else "not_ready",
        "service": "iot-diagnosis-mcp",
        "version": "1.3.0",
        "storage": storage,
        "retrieval_models": retrieval_models,
        "issues": issues,
    }
    return JSONResponse(payload, status_code=200 if not issues else 503)


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host=os.getenv("DIAGNOSIS_HOST", "127.0.0.1"),
        port=int(os.getenv("DIAGNOSIS_PORT", "9001")),
    )
