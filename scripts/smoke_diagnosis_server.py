"""Exercise the six v1.0 tools through the live Streamable HTTP endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

import httpx2
from agents.mcp import MCPServerStreamableHttp


def structured(result) -> dict:
    value = getattr(result, "structured_content", None)
    if not isinstance(value, dict):
        raise AssertionError("MCP_RESULT_INVALID")
    return value


def direct_http_client(
    headers: dict[str, str] | None = None,
    timeout: Any = None,
    auth: Any = None,
) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(
        headers=headers,
        timeout=timeout,
        auth=auth,
        follow_redirects=False,
        trust_env=False,
    )


async def run(url: str, timeout: float) -> None:
    server = MCPServerStreamableHttp(
        name="iot-diagnosis-smoke",
        params={"url": url, "timeout": timeout, "httpx_client_factory": direct_http_client},
        use_structured_content=True,
        client_session_timeout_seconds=timeout + 5,
    )
    async with server:
        tools = await server.list_tools()
        names = {tool.name for tool in tools}
        expected = {
            "diagnose_fault",
            "get_device_status",
            "get_device_logs",
            "search_knowledge",
            "search_fault_cases",
            "add_verified_fault_case",
        }
        assert names == expected
        status = structured(await server.call_tool("get_device_status", {"device_id": "ESP32_05"}))
        logs = structured(
            await server.call_tool(
                "get_device_logs", {"device_id": "ESP32_05", "limit": 10, "level": "ERROR"}
            )
        )
        knowledge = structured(
            await server.call_tool(
                "search_knowledge",
                {"query": "MQTT keep alive timeout", "sources": [], "top_k": 5},
            )
        )
        cases = structured(
            await server.call_tool(
                "search_fault_cases",
                {"query": "MQTT 频繁断开", "device_type": "ESP32", "fault_type": "mqtt"},
            )
        )
        diagnosis = structured(
            await server.call_tool(
                "diagnose_fault",
                {
                    "device_id": "ESP32_05",
                    "query": "设备为什么一直断开 MQTT 连接？",
                    "logs": ["MQTT disconnected", "MQTT keep alive timeout"],
                },
            )
        )
        rejected = structured(
            await server.call_tool(
                "add_verified_fault_case",
                {
                    "device_id": "ESP32_05",
                    "fault_type": "mqtt",
                    "fault_name": "未确认案例",
                    "symptoms": ["断开"],
                    "logs": ["timeout"],
                    "cause": "待确认",
                    "solution": "待确认",
                    "verified": False,
                    "verified_by": "operator",
                },
            )
        )

    assert status["ok"] is True and status["data"]["device_id"] == "ESP32_05"
    assert logs["ok"] is True and logs["data"]["logs"]
    assert knowledge["ok"] is True and knowledge["data"]["results"]
    assert cases["ok"] is True and cases["data"]["results"]
    assert diagnosis["ok"] is True and diagnosis["data"]["sources"]
    assert rejected["error"]["code"] == "CASE_NOT_VERIFIED"
    print(
        json.dumps(
            {
                "tools": sorted(names),
                "device": status["data"]["device_id"],
                "router": diagnosis["data"]["route"]["router"],
                "sources": diagnosis["data"]["route"]["selected_sources"],
                "fault_type": diagnosis["data"]["fault_type"],
                "verified_case_gate": "ok",
            },
            ensure_ascii=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:9001/mcp")
    parser.add_argument("--timeout", type=float, default=90)
    args = parser.parse_args()
    asyncio.run(run(args.url, args.timeout))


if __name__ == "__main__":
    main()
