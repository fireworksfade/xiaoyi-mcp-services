from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from iot_diagnosis.external import ExternalStores


logger = logging.getLogger("xiaoyi.iot_diagnosis.repository")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: datetime | None = None) -> str:
    return (value or utc_now()).isoformat()


class DiagnosisRepository:
    def __init__(self, path: str, offline_after_seconds: int = 120):
        self.path = path
        self.offline_after_seconds = max(1, offline_after_seconds)
        self._lock = threading.RLock()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()
        self.external = ExternalStores()
        self._sync_external_snapshot()

    def _external_call(self, component: str, method: str, *args) -> bool:
        target = getattr(self.external, component, None)
        if not target:
            return False
        try:
            result = getattr(target, method)(*args)
            self.external.errors.pop(component, None)
            return result is not False
        except Exception as exc:
            self.external.errors[component] = type(exc).__name__
            logger.exception("External %s operation %s failed", component, method)
            return False

    @staticmethod
    def _case_document(item: dict[str, Any]) -> dict[str, Any]:
        content = "；".join(
            [
                item["fault_name"],
                *item["symptoms"],
                *item["logs"],
                item["cause"],
                item["solution"],
            ]
        )
        return {
            "source": "fault_cases",
            "id": item["fault_id"],
            "title": item["fault_name"],
            "content": content,
            "device_type": item["device_type"],
        }

    def _sync_external_snapshot(self) -> None:
        with self._connect() as db:
            devices = {row["device_id"]: dict(row) for row in db.execute("SELECT * FROM device")}
            statuses = [dict(row) for row in db.execute("SELECT * FROM device_status")]
            logs = [dict(row) for row in db.execute("SELECT * FROM device_log")]
            documents = [dict(row) for row in db.execute("SELECT * FROM knowledge_document")]
        cases = self.fault_cases()
        self._external_call("mysql", "upsert_device_statuses", list(devices.values()), statuses)
        self._external_call("mysql", "add_logs", logs)
        for item in documents:
            self._external_call("mysql", "upsert_document", item)
            self._external_call(
                "qdrant",
                "upsert",
                {
                    "source": item["source"],
                    "id": item["source_id"],
                    "title": item["title"],
                    "content": item["content"],
                    "device_type": item.get("device_type"),
                },
            )
        for item in cases:
            self._external_call("mysql", "upsert_fault_case", item)
            self._external_call("qdrant", "upsert", self._case_document(item))

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS device (
                    device_id TEXT PRIMARY KEY,
                    device_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    firmware_version TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS device_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_id TEXT NOT NULL REFERENCES device(device_id),
                    online INTEGER NOT NULL,
                    wifi_status TEXT NOT NULL,
                    rssi INTEGER,
                    mqtt_status TEXT NOT NULL,
                    temperature REAL,
                    uptime INTEGER,
                    timestamp TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_device_status_latest
                    ON device_status(device_id, timestamp DESC);
                CREATE TABLE IF NOT EXISTS device_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_id TEXT NOT NULL REFERENCES device(device_id),
                    level TEXT NOT NULL,
                    module TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_device_log_latest
                    ON device_log(device_id, timestamp DESC);
                CREATE TABLE IF NOT EXISTS knowledge_document (
                    source TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    device_type TEXT,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(source, source_id)
                );
                CREATE TABLE IF NOT EXISTS fault_case (
                    fault_id TEXT PRIMARY KEY,
                    device_id TEXT,
                    device_type TEXT NOT NULL,
                    fault_type TEXT NOT NULL,
                    fault_name TEXT NOT NULL,
                    symptoms_json TEXT NOT NULL,
                    logs_json TEXT NOT NULL,
                    cause TEXT NOT NULL,
                    solution TEXT NOT NULL,
                    verified INTEGER NOT NULL,
                    verified_by TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS diagnosis_record (
                    diagnosis_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    device_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    fault_type TEXT NOT NULL,
                    fault_name TEXT NOT NULL,
                    cause TEXT NOT NULL,
                    solutions_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    router_type TEXT NOT NULL,
                    selected_sources_json TEXT NOT NULL,
                    retrieved_documents_json TEXT NOT NULL,
                    retrieval_count INTEGER NOT NULL,
                    rerank_count INTEGER NOT NULL,
                    retrieval_latency_ms REAL NOT NULL DEFAULT 0,
                    llm_latency_ms REAL NOT NULL DEFAULT 0,
                    total_latency_ms REAL NOT NULL,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )
            diagnosis_columns = {
                row[1] for row in db.execute("PRAGMA table_info(diagnosis_record)").fetchall()
            }
            for name, definition in {
                "retrieval_latency_ms": "REAL NOT NULL DEFAULT 0",
                "llm_latency_ms": "REAL NOT NULL DEFAULT 0",
                "input_tokens": "INTEGER NOT NULL DEFAULT 0",
                "output_tokens": "INTEGER NOT NULL DEFAULT 0",
                "error": "TEXT",
            }.items():
                if name not in diagnosis_columns:
                    db.execute(f"ALTER TABLE diagnosis_record ADD COLUMN {name} {definition}")
            db.execute("PRAGMA optimize")
            self._seed(db)

    def _seed(self, db: sqlite3.Connection) -> None:
        now = iso()
        db.execute(
            "INSERT OR IGNORE INTO device VALUES (?, ?, ?, ?, ?)",
            ("ESP32_05", "ESP32", "实验室节点 05", "1.2.0", now),
        )
        if not db.execute("SELECT 1 FROM device_status LIMIT 1").fetchone():
            db.execute(
                """INSERT INTO device_status
                (device_id, online, wifi_status, rssi, mqtt_status, temperature, uptime, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                ("ESP32_05", 1, "connected", -47, "disconnected", 26.3, 86400, now),
            )
            db.execute(
                "INSERT INTO device_log(device_id, level, module, message, timestamp) VALUES (?, ?, ?, ?, ?)",
                ("ESP32_05", "ERROR", "mqtt", "MQTT keep alive timeout", now),
            )
        documents = (
            (
                "mqtt_docs",
                "MQTT_DOC_03",
                "MQTT Keep Alive 与连接超时",
                "客户端必须在 Keep Alive 窗口内发送控制报文。频繁出现 keep alive timeout 时，应检查客户端心跳、Broker 超时和网络延迟。",
            ),
            (
                "wifi_docs",
                "WIFI_DOC_02",
                "ESP32 WiFi 弱信号诊断",
                "RSSI 低于 -75 dBm 时连接稳定性会明显下降。检查天线、供电、信道拥塞与接入点距离。",
            ),
            (
                "sensor_docs",
                "SENSOR_DOC_01",
                "传感器异常值排查",
                "传感器读数异常时先检查接线、供电、采样周期和量程，并与已知参考值交叉验证。",
            ),
            (
                "device_docs",
                "ESP32_DOC_01",
                "ESP32 运行状态检查",
                "设备重启、内存不足或长时间运行异常时，应核对 uptime、复位原因、空闲堆和固件版本。",
            ),
        )
        for source, source_id, title, content in documents:
            db.execute(
                "INSERT OR IGNORE INTO knowledge_document VALUES (?, ?, ?, ?, ?, ?)",
                (source, source_id, title, content, "ESP32", now),
            )
        db.execute(
            """INSERT OR IGNORE INTO fault_case VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "F105",
                "ESP32_05",
                "ESP32",
                "mqtt",
                "MQTT Keep Alive异常",
                json.dumps(["MQTT频繁掉线"], ensure_ascii=False),
                json.dumps(["MQTT keep alive timeout"], ensure_ascii=False),
                "Keep Alive参数或 Broker 超时设置异常",
                "检查客户端心跳并将 Keep Alive 调整到合理区间",
                1,
                "seed",
                "built_in",
                now,
                now,
            ),
        )

    def get_device_status(self, device_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            device = db.execute(
                "SELECT * FROM device WHERE device_id = ?", (device_id,)
            ).fetchone()
            if not device:
                return None
            status = db.execute(
                "SELECT * FROM device_status WHERE device_id = ? ORDER BY timestamp DESC, id DESC LIMIT 1",
                (device_id,),
            ).fetchone()
        if not status:
            return None
        last_seen = datetime.fromisoformat(str(status["timestamp"]).replace("Z", "+00:00"))
        if last_seen.tzinfo is None:
            last_seen = last_seen.replace(tzinfo=timezone.utc)
        heartbeat_fresh = utc_now() - last_seen <= timedelta(seconds=self.offline_after_seconds)
        reported_online = bool(status["online"])
        return {
            "device_id": device_id,
            "device_type": device["device_type"],
            "name": device["name"],
            "firmware_version": device["firmware_version"],
            "online": reported_online and heartbeat_fresh,
            "reported_online": reported_online,
            "heartbeat_fresh": heartbeat_fresh,
            "wifi_status": status["wifi_status"],
            "rssi": status["rssi"],
            "mqtt_status": status["mqtt_status"],
            "temperature": status["temperature"],
            "uptime": status["uptime"],
            "last_seen": status["timestamp"],
        }

    def get_device_logs(
        self, device_id: str, limit: int = 50, level: str | None = None
    ) -> list[dict[str, Any]] | None:
        if not self.get_device_status(device_id):
            return None
        query = "SELECT timestamp, level, module, message FROM device_log WHERE device_id = ?"
        params: list[Any] = [device_id]
        if level:
            query += " AND level = ?"
            params.append(level.upper())
        query += " ORDER BY timestamp DESC, id DESC LIMIT ?"
        params.append(limit)
        with self._connect() as db:
            return [dict(row) for row in db.execute(query, params).fetchall()]

    def upsert_status(self, device_id: str, payload: dict[str, Any]) -> None:
        timestamp = str(payload.get("timestamp") or iso())
        previous = self.get_device_status(device_id)
        with self._lock, self._connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO device VALUES (?, ?, ?, ?, ?)",
                (
                    device_id,
                    str(payload.get("device_type") or "ESP32"),
                    str(payload.get("name") or device_id),
                    str(payload.get("firmware_version") or "unknown"),
                    timestamp,
                ),
            )
            db.execute(
                """INSERT INTO device_status
                (device_id, online, wifi_status, rssi, mqtt_status, temperature, uptime, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    device_id,
                    int(bool(payload.get("online", True))),
                    str(
                        payload.get(
                            "wifi_status",
                            payload.get(
                                "wifi", previous and previous["wifi_status"] or "unknown"
                            ),
                        )
                    ),
                    payload.get("rssi", previous and previous["rssi"]),
                    str(
                        payload.get(
                            "mqtt_status",
                            payload.get(
                                "mqtt", previous and previous["mqtt_status"] or "unknown"
                            ),
                        )
                    ),
                    payload.get("temperature", previous and previous["temperature"]),
                    payload.get("uptime", previous and previous["uptime"]),
                    timestamp,
                ),
            )
        device = {
            "device_id": device_id,
            "device_type": str(payload.get("device_type") or "ESP32"),
            "name": str(payload.get("name") or device_id),
            "firmware_version": str(payload.get("firmware_version") or "unknown"),
            "created_at": timestamp,
        }
        current = self.get_device_status(device_id)
        if current:
            status = {
                "device_id": device_id,
                "online": current["reported_online"],
                "wifi_status": current["wifi_status"],
                "rssi": current["rssi"],
                "mqtt_status": current["mqtt_status"],
                "temperature": current["temperature"],
                "uptime": current["uptime"],
                "timestamp": current["last_seen"],
            }
            self._external_call("mysql", "upsert_device_status", device, status)

    def add_log(self, device_id: str, payload: dict[str, Any]) -> None:
        if not self.get_device_status(device_id):
            self.upsert_status(device_id, payload)
        item = {
            "device_id": device_id,
            "level": str(payload.get("level") or "INFO").upper(),
            "module": str(payload.get("module") or "device"),
            "message": str(payload.get("message") or ""),
            "timestamp": str(payload.get("timestamp") or iso()),
        }
        with self._lock, self._connect() as db:
            db.execute(
                "INSERT INTO device_log(device_id, level, module, message, timestamp) VALUES (?, ?, ?, ?, ?)",
                (
                    item["device_id"],
                    item["level"],
                    item["module"],
                    item["message"],
                    item["timestamp"],
                ),
            )
        self._external_call("mysql", "add_log", item)

    def add_fault(self, device_id: str, payload: dict[str, Any]) -> None:
        fault_type = str(payload.get("fault_type") or payload.get("type") or "unknown")
        message = str(payload.get("message") or payload.get("description") or fault_type)
        self.add_log(
            device_id,
            {
                "timestamp": payload.get("timestamp") or iso(),
                "level": payload.get("level") or "ERROR",
                "module": f"fault:{fault_type}",
                "message": message,
            },
        )

    def knowledge_documents(self, sources: list[str]) -> list[dict[str, Any]]:
        selected = [item for item in sources if item != "realtime_db"]
        if not selected:
            return []
        placeholders = ",".join("?" for _ in selected)
        with self._connect() as db:
            rows = db.execute(
                f"SELECT * FROM knowledge_document WHERE source IN ({placeholders})",
                selected,
            ).fetchall()
        return [dict(row) for row in rows]

    def fault_cases(
        self, device_type: str | None = None, fault_type: str | None = None
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM fault_case WHERE verified = 1"
        params: list[Any] = []
        if device_type:
            query += " AND device_type = ?"
            params.append(device_type)
        if fault_type:
            query += " AND fault_type LIKE ?"
            params.append(f"%{fault_type}%")
        with self._connect() as db:
            rows = db.execute(query, params).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["symptoms"] = json.loads(item.pop("symptoms_json"))
            item["logs"] = json.loads(item.pop("logs_json"))
            item["verified"] = bool(item["verified"])
            result.append(item)
        return result

    def add_verified_fault_case(self, payload: dict[str, Any]) -> dict[str, Any]:
        fault_id = f"F{uuid.uuid4().hex[:8].upper()}"
        now = iso()
        with self._lock, self._connect() as db:
            status = self.get_device_status(str(payload["device_id"]))
            db.execute(
                """INSERT INTO fault_case VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    fault_id,
                    payload["device_id"],
                    status["device_type"] if status else "ESP32",
                    payload["fault_type"],
                    payload["fault_name"],
                    json.dumps(payload["symptoms"], ensure_ascii=False),
                    json.dumps(payload["logs"], ensure_ascii=False),
                    payload["cause"],
                    payload["solution"],
                    1,
                    payload["verified_by"],
                    "human_verified",
                    now,
                    now,
                ),
            )
        item = {
            "fault_id": fault_id,
            "device_id": payload["device_id"],
            "device_type": status["device_type"] if status else "ESP32",
            "fault_type": payload["fault_type"],
            "fault_name": payload["fault_name"],
            "symptoms": payload["symptoms"],
            "logs": payload["logs"],
            "cause": payload["cause"],
            "solution": payload["solution"],
            "verified": True,
            "verified_by": payload["verified_by"],
            "source": "human_verified",
            "created_at": now,
            "updated_at": now,
        }
        mysql_saved = self._external_call("mysql", "upsert_fault_case", item)
        vector_indexed = self._external_call("qdrant", "upsert", self._case_document(item))
        return {
            "fault_id": fault_id,
            "verified": True,
            "indexed": True,
            "mysql_saved": mysql_saved,
            "vector_indexed": vector_indexed,
            "created_at": now,
        }

    def save_diagnosis(self, record: dict[str, Any]) -> None:
        created_at = iso()
        with self._lock, self._connect() as db:
            db.execute(
                """INSERT INTO diagnosis_record
                (diagnosis_id, request_id, device_id, query, fault_type, fault_name, cause,
                 solutions_json, confidence, router_type, selected_sources_json,
                 retrieved_documents_json, retrieval_count, rerank_count,
                 retrieval_latency_ms, llm_latency_ms, total_latency_ms, input_tokens,
                 output_tokens, error, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record["diagnosis_id"],
                    record["request_id"],
                    record["device_id"],
                    record["query"],
                    record["fault_type"],
                    record["fault_name"],
                    record["cause"],
                    json.dumps(record["solutions"], ensure_ascii=False),
                    record["confidence"],
                    record["route"]["router"],
                    json.dumps(record["route"]["selected_sources"], ensure_ascii=False),
                    json.dumps(record["sources"], ensure_ascii=False),
                    record["observability"]["retrieval_count"],
                    record["observability"]["rerank_count"],
                    record["observability"]["retrieval_latency_ms"],
                    record["observability"]["llm_latency_ms"],
                    record["observability"]["total_latency_ms"],
                    record["observability"]["input_tokens"],
                    record["observability"]["output_tokens"],
                    record["observability"]["error"],
                    created_at,
                ),
            )
        self._external_call(
            "mysql",
            "upsert_diagnosis",
            {**record, "created_at": created_at},
        )

    def vector_search(
        self, query: str, sources: list[str], top_k: int
    ) -> list[dict[str, Any]]:
        target = self.external.qdrant
        if not target:
            return []
        try:
            return target.search(query, sources, top_k)
        except Exception as exc:
            self.external.errors["qdrant"] = type(exc).__name__
            logger.exception("Qdrant search failed")
            return []

    def storage_status(self) -> dict[str, Any]:
        return self.external.status()
