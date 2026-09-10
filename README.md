# IoT Diagnosis MCP Server

按照《IoT Diagnosis MCP Server Specification v1.0》实现的统一诊断 MCP，使用 MCP Python SDK 2.x 的 Streamable HTTP 传输。

```powershell
python -m iot_diagnosis.server
```

- MCP：`http://127.0.0.1:9001/mcp`
- 健康检查：`http://127.0.0.1:9001/health`
- 数据库：默认 `data/iot_diagnosis.db`，可通过 `DIAGNOSIS_DATABASE_PATH` 修改

核心工具为 `diagnose_fault`、`get_device_status`、`get_device_logs`、`search_knowledge`、`search_fault_cases` 和 `add_verified_fault_case`。最后一个工具只有在 `verified=true` 且 `verified_by` 非空时才会写入案例库。

设置 `DIAGNOSIS_LLM_API_KEY` 与 `DIAGNOSIS_LLM_MODEL` 后，复杂问题会调用兼容 Chat Completions 的真实 LLM Router 和 Diagnosis，并记录耗时与 Token 数量；未配置或调用失败时使用启发式回退，结果中的 `route.router` 和 `observability.llm_fallback_reason` 会明确标识。

设置 `MQTT_ENABLED=true` 后，服务会订阅 `iot/{device_id}/status`、`iot/{device_id}/telemetry`、`iot/{device_id}/logs`、`iot/{device_id}/fault` 和 `iot/{device_id}/heartbeat`。也可以单独运行模拟器：

```powershell
python -m iot_diagnosis.simulator --device-id ESP32_05 --scenario mqtt_timeout
```

运行在线协议验收：

```powershell
..\backend\.venv\Scripts\python.exe scripts\smoke_diagnosis_server.py
```

使用响应较慢的外部模型时，可提高单次 MCP 工具调用的客户端超时：

```powershell
..\backend\.venv\Scripts\python.exe scripts\smoke_diagnosis_server.py --timeout 180
```
