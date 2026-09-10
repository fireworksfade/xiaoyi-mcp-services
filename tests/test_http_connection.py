from agents.mcp import MCPServerStreamableHttp

from app.mcp_http import mcp_httpx_client_factory


async def test_streamable_http_discovery() -> None:
    client = MCPServerStreamableHttp(
        name="iot-diagnosis-test",
        params={
            "url": "http://127.0.0.1:9001/mcp",
            "timeout": 10,
            "httpx_client_factory": mcp_httpx_client_factory,
        },
        client_session_timeout_seconds=15,
    )
    async with client:
        tools = await client.list_tools()
    assert {tool.name for tool in tools} == {
        "diagnose_fault",
        "get_device_status",
        "get_device_logs",
        "search_knowledge",
        "search_fault_cases",
        "add_verified_fault_case",
    }
