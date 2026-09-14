# IoT Diagnosis MCP Server

按照 v1.0 基线、v1.1 completion、v1.2 core-model 和 v1.3 discovery 规格实现的统一诊断 MCP，使用 MCP Python SDK 2.x 的 Streamable HTTP 传输。

```powershell
python -m iot_diagnosis.server
```

- MCP：`http://127.0.0.1:9001/mcp`
- 健康检查：`http://127.0.0.1:9001/health`
- 就绪检查：`http://127.0.0.1:9001/ready`
- 数据库：默认 `data/iot_diagnosis.db`，可通过 `DIAGNOSIS_DATABASE_PATH` 修改

保留六个 v1.0 核心工具，并新增 `get_diagnosis_trace`、`ingest_knowledge_text`、`rebuild_vector_index`、`list_devices`、`list_diagnoses`、`list_knowledge_documents`、`list_fault_cases` 和 `delete_fault_case`，共 14 个工具。列表工具支持过滤和 `limit`/`offset` 分页，使 Agent 无需预先知道设备、诊断、文档或案例 ID。写入型工具需要在主后端中显式启用；人工案例只有在 `verified=true` 且 `verified_by` 非空时才会写入。`delete_fault_case` 会同步清理 MySQL 镜像与 Qdrant 向量（按案例 payload 的 `document_id` 过滤），要求先执行 `rebuild_vector_index(["fault_cases"])` 让存量案例向量携带该字段。向量重建以 SQLite 为事实源，可按来源批量重建 Qdrant 索引。

SQLite 是本地事实源。MySQL/Qdrant 写入失败会进入持久化 outbox，后台按 `DIAGNOSIS_SYNC_RETRY_SECONDS` 重试，健康结果中的 `storage.outbox` 会显示积压。

Diagnosis Repository 的公开入口保持为 `iot_diagnosis.repository.DiagnosisRepository`；
内部已按 `repositories/device_state.py`、`logs.py`、`knowledge_cases.py`、
`diagnosis_records.py` 和 `external_sync.py` 拆分。schema 版本由 `common/migrations.py`
与各服务的 `migrations/` 目录独立管理，Repository 构造期间不再执行全库外部同步。

## 开发验证

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m ruff check common iot_diagnosis iot_control model_service scripts tests
.venv\Scripts\python -m ruff format --check common iot_diagnosis iot_control model_service scripts tests
.venv\Scripts\python -m mypy common iot_diagnosis iot_control model_service
.venv\Scripts\python -m pytest -q tests
```

设置 `DIAGNOSIS_LLM_API_KEY` 与 `DIAGNOSIS_LLM_MODEL` 后，复杂问题会调用兼容 Chat Completions 的真实 LLM Router 和 Diagnosis，并记录耗时与 Token 数量；未配置或调用失败时使用启发式回退，结果中的 `route.router` 和 `observability.llm_fallback_reason` 会明确标识。

RSSI、温度、在线状态、WiFi/MQTT 连接状态等实时问题由 Rule Router 直接回答，响应包含 `answer` 和 `realtime_state`，不会调用 LLM。启发式诊断覆盖 WiFi 弱信号/断开、MQTT Broker 不可达/认证失败/Keep Alive 超时、传感器读取失败/数据异常、设备重启/内存不足和网络延迟/丢包。`get_diagnosis_trace` 会返回完整诊断结果快照及最终上下文。

直接运行 MCP 时默认使用无需网络的 hash embedding 和加权 reranker。Docker Compose 默认部署 `Qwen/Qwen3-Embedding-0.6B` 与 `Qwen/Qwen3-Reranker-0.6B` GPU 服务，使用 1024 维 `iot_diagnosis_qwen3` 集合；首次启动会下载模型到 `retrieval-model-cache` 卷。文档摄取和索引重建会批量请求 Embedding 并批量写入 Qdrant。检索响应中的 `embedding_provider`、`reranker.provider` 和 `reranker.fallback` 可确认实际模型与回退状态。已有 Qdrant collection 的维度不匹配时服务会拒绝使用，避免写入损坏。

模型服务：

```text
GET  http://127.0.0.1:9010/live            # 进程存活（始终 200）
GET  http://127.0.0.1:9010/ready           # 两模型加载完成才 200；加载中 503 status=loading
GET  http://127.0.0.1:9010/health          # 兼容旧探针，语义与 /ready 相同
POST http://127.0.0.1:9010/v1/embeddings
POST http://127.0.0.1:9010/rerank
```

模型缓存模式由 `MODEL_CACHE_MODE` 控制（默认 `download`：允许联网下载缺失文件到持久卷；
`offline`：禁止下载，缓存不完整时 `/ready` 返回 503 与 `MODEL_CACHE_INCOMPLETE`）。
缓存可用性可用 `python -m scripts.check_model_cache` 预先校验（退出码 0 表示完整）。
GPU 档位要求约 4 GiB 空闲显存、约 2.5 GiB 首次下载磁盘与 5–20 分钟冷启动时间。

## 文档摄取

```powershell
..\backend\.venv\Scripts\python.exe -m scripts.ingest_documents .\docs\mqtt.md --source mqtt_docs --document-id mqtt-guide --title "MQTT Guide"
```

支持 `.txt`、`.md`、`.markdown` 和 `.pdf`。相同文档 ID 会替换旧分块。

仓库在 `knowledge/` 中附带十份整理后的 ESP32/MQTT 诊断资料，可一次性重复摄取：

```powershell
..\backend\.venv\Scripts\python.exe -m scripts.ingest_recommended_documents
```

这组资料覆盖 ESP-MQTT 错误、MQTT Session/QoS、Mosquitto、WiFi 断线、致命错误与重启、Watchdog、Core Dump、内存问题、I2C 和 ADC。文档 ID 稳定，重复执行会替换原有分块。

## 评测

```powershell
..\backend\.venv\Scripts\python.exe -m scripts.evaluate_rag --database .\data\iot_diagnosis_eval.db
```

报告 Recall@K、Precision@K、MRR、Hit Rate、Router/Source/Diagnosis/Diagnosis Name Accuracy、延迟与 Token 使用。

Compose 内运行真实 Qwen3 检索评测：

```powershell
docker exec last-work-iot-diagnosis-mcp-1 python /app/scripts/evaluate_rag.py --profile live-retrieval --database /app/data/iot_diagnosis.db
```

该 profile 保留真实 Qdrant、Embedding 和 Reranker，但关闭外部诊断 LLM；报告会明确显示 provider、fallback、平均延迟、P50 和 P95。可用 `--output` 将报告保存为独立 JSON 文件。

## 可选认证

设置 `DIAGNOSIS_MCP_BEARER_TOKEN` 后，`/mcp` 要求 Bearer Token；同时将 `DIAGNOSIS_MCP_PUBLIC_URL` 设置为客户端可识别的服务 URL。`/health` 用于 liveness，`/ready` 会检查存储、outbox 和 `DIAGNOSIS_RETRIEVAL_MODEL_HEALTH_URL`，任一已配置核心依赖不可用时返回 503。

设置 `MQTT_ENABLED=true` 后，服务会订阅 `iot/{device_id}/status`、`iot/{device_id}/telemetry`、`iot/{device_id}/logs`、`iot/{device_id}/fault` 和 `iot/{device_id}/heartbeat`。也可以单独运行模拟器：

```powershell
python -m iot_diagnosis.simulator --device-id ESP32_05 --scenario mqtt_timeout
```

模拟器支持机群模式：一个进程内以线程模拟整个 ESP32 机群，拓扑由 `iot_diagnosis/fleet.json` 定义（12 台设备：8 台 normal、1 台 mqtt_timeout、1 台 wifi_weak、1 台 sensor_error、1 台 unstable 随机掉线恢复）。每台设备带部署位置命名、温度/RSSI 基线、固件版本与上报间隔，首次上报即自动注册进设备表：

```powershell
python -m iot_diagnosis.simulator --host 127.0.0.1 --fleet iot_diagnosis/fleet.json
```

在 `fleet.json` 中增删设备或调整 `temperature_base`/`rssi_base`/`interval` 即可改变机群规模与行为；`defaults` 段提供各设备的缺省值。注意：普通场景的 `temperature_base` 需低于 70 °C，否则会触发传感器异常误报（加载时校验）。单设备模式保留 `--device-id/--scenario/--interval` 原有用法；两种模式下 `docker compose stop` 都会让设备主动发布 offline 状态（`offline_reason=graceful_shutdown`），进程被强杀时由遗嘱代发 `client_lost`。

模拟器额外支持 `inject_fault` 下行动作（`iot/{device_id}/cmd`，参数 `scenario` 取 SCENARIOS 之一，`normal` 表示清除全部故障标志），可对任意节点动态注入/解除故障，配合批量案例脚本复用整个机群。在诊断容器内运行 `python scripts/generate_fault_cases.py [--rounds 2]` 可批量执行「注入故障 → 诊断 → 修复闭环 → 自动沉淀」生成真实案例（低风险直执行，高风险走提案审批后自动批准，结束时机群恢复健康）；`python scripts/purge_fault_cases.py` 逐条清空案例库并同步清理 MySQL 镜像与 Qdrant 向量。

运行在线协议验收：

```powershell
..\backend\.venv\Scripts\python.exe scripts\smoke_diagnosis_server.py
```

使用响应较慢的外部模型时，可提高单次 MCP 工具调用的客户端超时：

```powershell
..\backend\.venv\Scripts\python.exe scripts\smoke_diagnosis_server.py --timeout 180
```
