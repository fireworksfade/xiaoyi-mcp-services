import os

import pytest

from agents.mcp import MCPServerStreamableHttp

from app.mcp_http import mcp_httpx_client_factory


async def test_streamable_http_discovery() -> None:
    url = os.getenv("MCP_INTEGRATION_URL", "").strip()
    if not url:
        pytest.skip("set MCP_INTEGRATION_URL to run the live transport test")
    token = os.getenv("MCP_INTEGRATION_TOKEN", "").strip()
    client = MCPServerStreamableHttp(
        name="iot-diagnosis-test",
        params={
            "url": url,
            "headers": {"Authorization": f"Bearer {token}"} if token else None,
            "timeout": 10,
            "httpx_client_factory": mcp_httpx_client_factory,
        },
        client_session_timeout_seconds=15,
    )
    async with client:
        tools = await client.list_tools()
    assert {tool.name for tool in tools} == {
        "diagnose_fault",
        "list_devices",
        "get_device_status",
        "get_device_logs",
        "get_diagnosis_trace",
        "list_diagnoses",
        "list_knowledge_documents",
        "ingest_knowledge_text",
        "search_knowledge",
        "search_fault_cases",
        "add_verified_fault_case",
        "rebuild_vector_index",
    }
