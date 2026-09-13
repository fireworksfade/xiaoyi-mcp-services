from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp_types import ToolAnnotations
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse

from common.results import failure, success
from iot_control import actions, remediation_events
from iot_control.auth import auth_configuration
from iot_control.mqtt import ControlMQTT
from iot_control.repository import ControlRepository

logger = logging.getLogger("xiaoyi.iot_control.server")

repository = ControlRepository(
    os.getenv("CONTROL_DATABASE_PATH", "data/iot_control.db"),
    command_timeout_seconds=int(os.getenv("CONTROL_COMMAND_TIMEOUT_SECONDS", "30")),
    verify_window_seconds=int(os.getenv("CONTROL_VERIFY_WINDOW_SECONDS", "60")),
    proposal_ttl_minutes=int(os.getenv("CONTROL_PROPOSAL_TTL_MINUTES", "30")),
)
auth_settings, token_verifier = auth_configuration()


@asynccontextmanager
async def service_lifespan(_server):
    global _channel
    channel: ControlMQTT | None = None
    watchdog = None
    if os.getenv("MQTT_ENABLED", "false").lower() == "true":
        channel = ControlMQTT(repository)
        channel.start()

        async def background_worker() -> None:
            while True:
                await asyncio.sleep(2)
                try:
                    finalized = await asyncio.to_thread(repository.process_timeouts)
                except Exception:
                    logger.exception("Timeout processing failed")
                    continue
                # 修复完成事件交给诊断服务消费（案例沉淀归诊断服务所有）
                if finalized and channel is not None:
                    try:
                        remediation_events.publish_completed_events(channel, repository, finalized)
                    except Exception:
                        logger.exception("Remediation event publishing failed")

        watchdog = asyncio.create_task(background_worker())
    _channel = channel
    try:
        yield {"mqtt_channel": channel}
    finally:
        _channel = None
        if watchdog:
            watchdog.cancel()
            try:
                await watchdog
            except asyncio.CancelledError:
                pass
        if channel:
            channel.stop()


_channel: ControlMQTT | None = None


mcp = MCPServer(
    "iot-control",
    title="IoT Control MCP Server",
    description="Device remediation commands, approval proposals, recovery verification and automatic case archiving for ESP32",
    version="1.1.0",
    lifespan=service_lifespan,
    auth=auth_settings,
    token_verifier=token_verifier,
)
read_only = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)
DeviceId = Annotated[str, Field(min_length=1, max_length=120)]
ShortText = Annotated[str, Field(min_length=1, max_length=300)]
LongText = Annotated[str, Field(min_length=1, max_length=4000)]
JsonParameters = Annotated[dict[str, Any], Field(max_properties=10)]


def _mqtt_channel() -> ControlMQTT | None:
    return _channel


@mcp.tool(annotations=read_only)
def list_device_actions(
    device_id: DeviceId | None = None,
) -> dict[str, Any]:
    """列出可下发的设备修复动作、风险级别、参数与适用故障类型。"""
    return success({"actions": actions.action_catalog()})


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    )
)
def execute_device_action(
    device_id: DeviceId,
    action: Annotated[str, Field(min_length=1, max_length=120)],
    reason: Annotated[str, Field(min_length=1, max_length=2000)],
    parameters: JsonParameters | None = None,
    issued_by: Annotated[str, Field(max_length=160)] = "agent",
    diagnosis_id: Annotated[str | None, Field(max_length=120)] = None,
) -> dict[str, Any]:
    """下发低风险修复动作（重启/改配置等高风险动作会被拒绝并要求走提案审批）。

    若本次运行已产出诊断结果，必须传入 diagnosis_id，恢复成功后将自动沉淀为故障案例。
    """
    item = actions.get_action(action)
    if not item:
        return failure("UNKNOWN_ACTION", f"Action {action} is not supported")
    if item["risk_level"] != actions.LOW_RISK:
        return failure(
            "ACTION_REQUIRES_APPROVAL",
            "高风险动作需要人工批准，请改用 create_remediation_proposal 创建修复提案",
            details={"action": action, "risk_level": item["risk_level"]},
        )
    if error_code := actions.validate_parameters(action, parameters):
        return failure(error_code, "动作参数无效")
    command = repository.create_command(
        device_id=device_id,
        action=action,
        risk_level=item["risk_level"],
        parameters=parameters,
        reason=reason,
        issued_by=issued_by,
        diagnosis_id=diagnosis_id,
    )
    channel = _mqtt_channel()
    if channel is None:
        return failure("MQTT_UNAVAILABLE", "设备控制通道未启用", retryable=True)
    delivered = channel.send_command(device_id, command)
    return success({**command, "delivered": delivered})


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    )
)
def create_remediation_proposal(
    device_id: DeviceId,
    action: Annotated[str, Field(min_length=1, max_length=120)],
    reason: LongText,
    impact: LongText,
    parameters: JsonParameters | None = None,
    diagnosis_id: Annotated[str | None, Field(max_length=120)] = None,
) -> dict[str, Any]:
    """为高风险动作创建修复提案，等待人工批准后才会执行。

    若本次运行已产出诊断结果，必须传入 diagnosis_id，恢复成功后将自动沉淀为故障案例。
    """
    item = actions.get_action(action)
    if not item:
        return failure("UNKNOWN_ACTION", f"Action {action} is not supported")
    if item["risk_level"] != actions.HIGH_RISK:
        return failure(
            "ACTION_NOT_HIGH_RISK",
            "低风险动作可直接执行，请改用 execute_device_action",
            details={"action": action, "risk_level": item["risk_level"]},
        )
    if error_code := actions.validate_parameters(action, parameters):
        return failure(error_code, "动作参数无效")
    proposal = repository.create_proposal(
        device_id=device_id,
        action=action,
        parameters=parameters,
        reason=reason,
        impact=impact,
        diagnosis_id=diagnosis_id,
    )
    return success(proposal)


@mcp.tool(annotations=read_only)
def get_action_result(
    command_id: Annotated[str, Field(min_length=1, max_length=120)] | None = None,
    proposal_id: Annotated[str, Field(min_length=1, max_length=120)] | None = None,
) -> dict[str, Any]:
    """按命令 ID 或提案 ID 查询执行与恢复验证状态。"""
    if not command_id and not proposal_id:
        return failure("INVALID_REQUEST", "command_id 与 proposal_id 至少提供一个")
    if command_id:
        command = repository.get_command(command_id)
        if command is None:
            return failure("COMMAND_NOT_FOUND", f"Command {command_id} does not exist")
        return success({"command": command})
    proposal = repository.get_proposal(proposal_id)
    if proposal is None:
        return failure("PROPOSAL_NOT_FOUND", f"Proposal {proposal_id} does not exist")
    command = repository.get_command(proposal["command_id"]) if proposal["command_id"] else None
    return success({"proposal": proposal, "command": command})


@mcp.tool(annotations=read_only)
def list_remediation_proposals(
    status: Annotated[str | None, Field(pattern="^(pending|approved|rejected|expired)$")] = None,
    limit: Annotated[int, Field(ge=1, le=200)] = 50,
    offset: Annotated[int, Field(ge=0, le=100_000)] = 0,
) -> dict[str, Any]:
    """列出修复提案，可按状态过滤并分页；过期待提案会自动标记为 expired。"""
    return success(repository.list_proposals(status, limit, offset))


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    )
)
def decide_remediation_proposal(
    proposal_id: Annotated[str, Field(min_length=1, max_length=120)],
    decision: Annotated[str, Field(pattern="^(approved|rejected)$")],
    decided_by: Annotated[str, Field(min_length=1, max_length=160)],
    expected_version: Annotated[int, Field(ge=1)],
) -> dict[str, Any]:
    """人工决策修复提案；批准后立即下发命令并进入恢复验证（仅限已授权后端调用）。"""
    try:
        proposal, command = repository.decide_proposal(
            proposal_id,
            decision,
            decided_by,
            expected_version,
            risk_level_of=actions.risk_level,
        )
    except LookupError:
        return failure("PROPOSAL_NOT_FOUND", f"Proposal {proposal_id} does not exist")
    except ValueError as exc:
        return failure(str(exc), "提案决策被拒绝")
    delivered = None
    if command is not None:
        channel = _mqtt_channel()
        if channel is None:
            return failure("MQTT_UNAVAILABLE", "设备控制通道未启用", retryable=True)
        delivered = channel.send_command(proposal["device_id"], command)
    return success({**proposal, "command": command, "delivered": delivered})


@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "iot-control", "version": "1.1.0"})


@mcp.custom_route("/ready", methods=["GET"])
async def ready(_: Request) -> JSONResponse:
    channel = _mqtt_channel()
    problems: list[str] = []
    if channel is None:
        problems.append("mqtt_channel")
    elif not channel.client.is_connected():
        problems.append("mqtt_disconnected")
    if problems:
        return JSONResponse({"status": "degraded", "problems": problems}, status_code=503)
    return JSONResponse({"status": "ready", "service": "iot-control", "version": "1.1.0"})


def main() -> None:
    mcp.run(
        transport="streamable-http",
        host=os.getenv("CONTROL_HOST", "127.0.0.1"),
        port=int(os.getenv("CONTROL_PORT", "9002")),
    )


if __name__ == "__main__":
    main()
