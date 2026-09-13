from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import sqlite3
import threading
import time
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
        self.retry_external_sync()
        self._sync_external_snapshot()

    def _external_call(self, component: str, method: str, *args) -> bool:
        self.external.ensure_connected(component)
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

    def _dispatch_external(self, component: str, operation: str, payload: dict[str, Any]) -> bool:
        if operation == "upsert_device_status":
            return self._external_call(
                component,
                operation,
                payload["device"],
                payload["status"],
            )
        return self._external_call(component, operation, payload)

    def _enqueue_external_sync(
        self,
        component: str,
        operation: str,
        payload: dict[str, Any],
    ) -> None:
        payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        dedupe_key = hashlib.sha256(
            f"{component}:{operation}:{payload_json}".encode("utf-8")
        ).hexdigest()
        now = iso()
        with self._lock, self._connect() as db:
            db.execute(
                """INSERT INTO external_sync_outbox
                (id, dedupe_key, component, operation, payload_json, attempts,
                 last_error, created_at, updated_at, completed_at)
                VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?, NULL)
                ON CONFLICT(dedupe_key) DO UPDATE SET
                    payload_json=excluded.payload_json,
                    last_error=excluded.last_error,
                    updated_at=excluded.updated_at,
                    completed_at=NULL,
                    claimed_at=NULL""",
                (
                    str(uuid.uuid4()),
                    dedupe_key,
                    component,
                    operation,
                    payload_json,
                    self.external.errors.get(component, "UNAVAILABLE"),
                    now,
                    now,
                ),
            )

    def _external_write(
        self,
        component: str,
        operation: str,
        payload: dict[str, Any],
    ) -> bool:
        if not self.external.is_configured(component):
            return False
        delivered = self._dispatch_external(component, operation, payload)
        if not delivered:
            self._enqueue_external_sync(component, operation, payload)
        return delivered

    def retry_external_sync(self, limit: int = 100, lease_seconds: int = 300) -> dict[str, int]:
        # 单条 UPDATE 原子认领（带租约）：多个进程/线程同时重试时不会重复
        # 派发同一批待同步记录；崩溃进程的租约过期后可被其他进程接管。
        now = iso()
        lease_deadline = iso(utc_now() - timedelta(seconds=lease_seconds))
        with self._lock, self._connect() as db:
            rows = db.execute(
                """UPDATE external_sync_outbox
                SET claimed_at=?, attempts=attempts+1, updated_at=?
                WHERE id IN (
                    SELECT id FROM external_sync_outbox
                    WHERE completed_at IS NULL
                      AND (claimed_at IS NULL OR claimed_at < ?)
                    ORDER BY created_at, id
                    LIMIT ?
                )
                RETURNING id, component, operation, payload_json""",
                (now, now, lease_deadline, max(1, min(limit, 1000))),
            ).fetchall()

        delivered = 0
        failed = 0
        for row in rows:
            payload = json.loads(row["payload_json"])
            success = self._dispatch_external(row["component"], row["operation"], payload)
            finished = iso()
            with self._lock, self._connect() as db:
                if success:
                    delivered += 1
                    db.execute(
                        """UPDATE external_sync_outbox
                        SET last_error=NULL, updated_at=?, completed_at=?, claimed_at=NULL
                        WHERE id=?""",
                        (finished, finished, row["id"]),
                    )
                else:
                    failed += 1
                    db.execute(
                        """UPDATE external_sync_outbox
                        SET last_error=?, updated_at=?, claimed_at=NULL
                        WHERE id=?""",
                        (
                            self.external.errors.get(row["component"], "UNAVAILABLE"),
                            finished,
                            row["id"],
                        ),
                    )
        return {"processed": len(rows), "delivered": delivered, "failed": failed}

    def external_sync_status(self) -> dict[str, Any]:
        with self._connect() as db:
            pending = db.execute(
                """SELECT component, COUNT(*) AS count, MAX(attempts) AS max_attempts
                FROM external_sync_outbox
                WHERE completed_at IS NULL
                GROUP BY component"""
            ).fetchall()
        by_component = {row["component"]: row["count"] for row in pending}
        return {
            "pending": sum(by_component.values()),
            "by_component": by_component,
            "max_attempts": max((row["max_attempts"] for row in pending), default=0),
        }

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
            # Qdrant 删除按 payload 的 document_id 过滤，案例删除链路依赖该字段
            "document_id": item["fault_id"],
            "title": item["fault_name"],
            "content": content,
            "device_type": item["device_type"],
        }

    @staticmethod
    def _knowledge_vector_document(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "source": item["source"],
            "id": item["source_id"],
            "document_id": item.get("document_id")
            or item["source_id"].split("#", 1)[0],
            "chunk_index": int(item.get("chunk_index") or 0),
            "title": item["title"],
            "content": item["content"],
            "device_type": item.get("device_type"),
        }

    def _qdrant_write_many(self, items: list[dict[str, Any]]) -> int:
        if not items or not self.external.is_configured("qdrant"):
            return 0
        batch_size = max(
            1, min(int(os.getenv("DIAGNOSIS_VECTOR_BATCH_SIZE", "32")), 100)
        )
        indexed = 0
        for offset in range(0, len(items), batch_size):
            batch = items[offset : offset + batch_size]
            if self._external_call("qdrant", "upsert_many", batch):
                indexed += len(batch)
                continue
            for item in batch:
                self._enqueue_external_sync("qdrant", "upsert", item)
        return indexed

    def _sync_external_snapshot(self) -> None:
        with self._connect() as db:
            devices = {row["device_id"]: dict(row) for row in db.execute("SELECT * FROM device")}
            statuses = [dict(row) for row in db.execute("SELECT * FROM device_status")]
            logs = [dict(row) for row in db.execute("SELECT * FROM device_log")]
            documents = [dict(row) for row in db.execute("SELECT * FROM knowledge_document")]
            diagnosis_rows = [dict(row) for row in db.execute("SELECT * FROM diagnosis_record")]
        cases = self.fault_cases()
        if self.external.is_configured("mysql"):
            if not self._external_call(
                "mysql", "upsert_device_statuses", list(devices.values()), statuses
            ):
                for status in statuses:
                    device = devices.get(status["device_id"])
                    if device:
                        self._enqueue_external_sync(
                            "mysql",
                            "upsert_device_status",
                            {"device": device, "status": status},
                        )
            if not self._external_call("mysql", "add_logs", logs):
                for item in logs:
                    self._enqueue_external_sync("mysql", "add_log", item)
        vector_documents = []
        for item in documents:
            self._external_write("mysql", "upsert_document", item)
            vector_documents.append(self._knowledge_vector_document(item))
        for item in cases:
            self._external_write("mysql", "upsert_fault_case", item)
            vector_documents.append(self._case_document(item))
        self._qdrant_write_many(vector_documents)
        diagnoses = [self._diagnosis_record_from_row(item) for item in diagnosis_rows]
        if diagnoses and self.external.is_configured("mysql"):
            if not self._external_call("mysql", "upsert_diagnoses", diagnoses):
                for item in diagnoses:
                    self._enqueue_external_sync("mysql", "upsert_diagnosis", item)

    @staticmethod
    def _diagnosis_record_from_row(item: dict[str, Any]) -> dict[str, Any]:
        contexts = json.loads(item["retrieved_documents_json"])
        record = {
            "diagnosis_id": item["diagnosis_id"],
            "request_id": item["request_id"],
            "device_id": item["device_id"],
            "query": item["query"],
            "fault_type": item["fault_type"],
            "fault_name": item["fault_name"],
            "cause": item["cause"],
            "solutions": json.loads(item["solutions_json"]),
            "confidence": item["confidence"],
            "route": {
                "router": item["router_type"],
                "selected_sources": json.loads(item["selected_sources_json"]),
            },
            "sources": contexts,
            "trace_contexts": contexts,
            "observability": {
                "retrieval_count": item["retrieval_count"],
                "rerank_count": item["rerank_count"],
                "retrieval_latency_ms": item["retrieval_latency_ms"],
                "llm_latency_ms": item["llm_latency_ms"],
                "total_latency_ms": item["total_latency_ms"],
                "input_tokens": item["input_tokens"],
                "output_tokens": item["output_tokens"],
                "error": item["error"],
            },
            "created_at": item["created_at"],
        }
        result_json = item.get("result_json")
        if result_json:
            record["result"] = json.loads(result_json)
        else:
            record["result"] = {
                key: record[key]
                for key in (
                    "diagnosis_id",
                    "device_id",
                    "fault_type",
                    "fault_name",
                    "cause",
                    "solutions",
                    "confidence",
                    "route",
                    "sources",
                    "observability",
                )
            }
        return record

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
                    result_json TEXT,
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
                CREATE TABLE IF NOT EXISTS external_sync_outbox (
                    id TEXT PRIMARY KEY,
                    dedupe_key TEXT NOT NULL UNIQUE,
                    component TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT,
                    claimed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS ix_external_sync_pending
                    ON external_sync_outbox(completed_at, created_at);
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
                "result_json": "TEXT",
            }.items():
                if name not in diagnosis_columns:
                    db.execute(f"ALTER TABLE diagnosis_record ADD COLUMN {name} {definition}")
            knowledge_columns = {
                row[1] for row in db.execute("PRAGMA table_info(knowledge_document)").fetchall()
            }
            for name, definition in {
                "document_id": "TEXT",
                "chunk_index": "INTEGER NOT NULL DEFAULT 0",
            }.items():
                if name not in knowledge_columns:
                    db.execute(f"ALTER TABLE knowledge_document ADD COLUMN {name} {definition}")
            outbox_columns = {
                row[1] for row in db.execute("PRAGMA table_info(external_sync_outbox)").fetchall()
            }
            if "claimed_at" not in outbox_columns:
                db.execute("ALTER TABLE external_sync_outbox ADD COLUMN claimed_at TEXT")
            db.execute(
                """UPDATE knowledge_document
                SET document_id = CASE
                    WHEN instr(source_id, '#') > 0
                    THEN substr(source_id, 1, instr(source_id, '#') - 1)
                    ELSE source_id
                END
                WHERE document_id IS NULL OR document_id = ''"""
            )
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
                """INSERT OR IGNORE INTO knowledge_document
                (source, source_id, title, content, device_type, created_at, document_id, chunk_index)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (source, source_id, title, content, "ESP32", now, source_id, 0),
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

    def list_devices(
        self,
        device_type: str | None = None,
        online: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        query = "SELECT * FROM device"
        params: list[Any] = []
        if device_type:
            query += " WHERE device_type = ?"
            params.append(device_type)
        query += " ORDER BY device_id"
        with self._connect() as db:
            devices = [dict(row) for row in db.execute(query, params).fetchall()]

        items = []
        for device in devices:
            status = self.get_device_status(device["device_id"])
            if status:
                item = status
            else:
                item = {
                    **device,
                    "online": False,
                    "reported_online": None,
                    "heartbeat_fresh": False,
                    "wifi_status": None,
                    "rssi": None,
                    "mqtt_status": None,
                    "temperature": None,
                    "uptime": None,
                    "last_seen": None,
                }
            if online is None or item["online"] is online:
                items.append(item)

        total = len(items)
        return {
            "items": items[offset : offset + limit],
            "total": total,
            "limit": limit,
            "offset": offset,
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
            self._external_write(
                "mysql",
                "upsert_device_status",
                {"device": device, "status": status},
            )

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
        self._external_write("mysql", "add_log", item)

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

    def list_knowledge_documents(
        self,
        source: str | None = None,
        device_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        query = "SELECT * FROM knowledge_document"
        clauses: list[str] = []
        params: list[Any] = []
        if source:
            clauses.append("source = ?")
            params.append(source)
        if device_type:
            clauses.append("device_type = ?")
            params.append(device_type)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY source, document_id, chunk_index, source_id"
        with self._connect() as db:
            rows = [dict(row) for row in db.execute(query, params).fetchall()]

        documents: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            document_id = row.get("document_id") or row["source_id"].split("#", 1)[0]
            key = (row["source"], document_id)
            if key not in documents:
                documents[key] = {
                    "source": row["source"],
                    "document_id": document_id,
                    "title": re.sub(r" \(\d+/\d+\)$", "", row["title"]),
                    "device_type": row.get("device_type"),
                    "chunk_count": 0,
                    "content_chars": 0,
                    "created_at": row["created_at"],
                }
            item = documents[key]
            item["chunk_count"] += 1
            item["content_chars"] += len(row["content"])
            if row["created_at"] > item["created_at"]:
                item["created_at"] = row["created_at"]

        items = list(documents.values())
        total = len(items)
        return {
            "items": items[offset : offset + limit],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    def replace_knowledge_document(
        self,
        *,
        source: str,
        document_id: str,
        title: str,
        chunks: list[str],
        device_type: str | None,
    ) -> dict[str, Any]:
        created_at = iso()
        items = [
            {
                "source": source,
                "source_id": f"{document_id}#{index:04d}",
                "document_id": document_id,
                "chunk_index": index,
                "title": title if len(chunks) == 1 else f"{title} ({index + 1}/{len(chunks)})",
                "content": content,
                "device_type": device_type,
                "created_at": created_at,
            }
            for index, content in enumerate(chunks)
        ]
        with self._lock, self._connect() as db:
            db.execute(
                """DELETE FROM knowledge_document
                WHERE source = ? AND (
                    document_id = ? OR source_id = ? OR source_id LIKE ?
                )""",
                (source, document_id, document_id, f"{document_id}#%"),
            )
            db.executemany(
                """INSERT INTO knowledge_document
                (source, source_id, title, content, device_type, created_at, document_id, chunk_index)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        item["source"],
                        item["source_id"],
                        item["title"],
                        item["content"],
                        item["device_type"],
                        item["created_at"],
                        item["document_id"],
                        item["chunk_index"],
                    )
                    for item in items
                ],
            )

        delete_payload = {"source": source, "document_id": document_id}
        mysql_results = [self._external_write("mysql", "delete_document", delete_payload)]
        vector_delete = self._external_write("qdrant", "delete_document", delete_payload)
        for item in items:
            mysql_results.append(self._external_write("mysql", "upsert_document", item))
        vector_items = [self._knowledge_vector_document(item) for item in items]
        indexed_count = self._qdrant_write_many(vector_items)
        mysql_saved = all(mysql_results)
        vector_indexed = vector_delete and indexed_count == len(vector_items)
        sync_status = "complete" if mysql_saved and vector_indexed else (
            "pending" if self.external_sync_status()["pending"] else "local_only"
        )
        return {
            "source": source,
            "document_id": document_id,
            "chunk_count": len(items),
            "chunk_ids": [item["source_id"] for item in items],
            "mysql_saved": mysql_saved,
            "vector_indexed": vector_indexed,
            "sync_status": sync_status,
        }

    def delete_knowledge_document(
        self,
        *,
        source: str,
        document_id: str,
    ) -> dict[str, Any]:
        allowed_sources = {"mqtt_docs", "wifi_docs", "sensor_docs", "device_docs"}
        if source not in allowed_sources:
            raise ValueError("INVALID_DOCUMENT_SOURCE")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}", document_id):
            raise ValueError("DOCUMENT_ID_INVALID")
        with self._lock, self._connect() as db:
            cursor = db.execute(
                """DELETE FROM knowledge_document
                WHERE source = ? AND (
                    document_id = ? OR source_id = ? OR source_id LIKE ?
                )""",
                (source, document_id, document_id, f"{document_id}#%"),
            )
            deleted = cursor.rowcount
        delete_payload = {"source": source, "document_id": document_id}
        mysql_saved = self._external_write("mysql", "delete_document", delete_payload)
        vector_deleted = self._external_write("qdrant", "delete_document", delete_payload)
        sync_status = "complete" if mysql_saved and vector_deleted else (
            "pending" if self.external_sync_status()["pending"] else "local_only"
        )
        return {
            "source": source,
            "document_id": document_id,
            "deleted_chunks": max(deleted, 0),
            "mysql_saved": mysql_saved,
            "vector_deleted": vector_deleted,
            "sync_status": sync_status,
        }

    def rebuild_vector_index(self, sources: list[str] | None = None) -> dict[str, Any]:
        allowed_sources = {
            "fault_cases",
            "mqtt_docs",
            "wifi_docs",
            "sensor_docs",
            "device_docs",
        }
        selected = list(dict.fromkeys(sources or sorted(allowed_sources)))
        if set(selected) - allowed_sources:
            raise ValueError("INVALID_REQUEST")
        started = time.perf_counter()
        document_sources = [source for source in selected if source != "fault_cases"]
        documents = [
            self._knowledge_vector_document(item)
            for item in self.knowledge_documents(document_sources)
        ]
        cases = (
            [self._case_document(item) for item in self.fault_cases()]
            if "fault_cases" in selected
            else []
        )
        items = documents + cases
        indexed = self._qdrant_write_many(items)
        pending = self.external_sync_status()["by_component"].get("qdrant", 0)
        target = self.external.qdrant
        provider = getattr(getattr(target, "embedding_provider", None), "name", "disabled")
        return {
            "sources": selected,
            "attempted": len(items),
            "indexed": indexed,
            "pending": pending,
            "embedding_provider": provider,
            "collection": getattr(target, "collection", None),
            "dimensions": getattr(target, "dimensions", None),
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            "sync_status": (
                "complete"
                if indexed == len(items) and target is not None
                else "pending"
                if pending
                else "local_only"
            ),
        }

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

    def list_fault_cases(
        self,
        device_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """分页列出已验证案例（不含逐条日志正文），供案例库浏览。"""
        query = "SELECT * FROM fault_case WHERE verified = 1"
        params: list[Any] = []
        if device_type:
            query += " AND device_type = ?"
            params.append(device_type)
        count_query = f"SELECT COUNT(*) FROM ({query})"
        query += " ORDER BY created_at DESC, fault_id DESC LIMIT ? OFFSET ?"
        with self._connect() as db:
            total = int(db.execute(count_query, params).fetchone()[0])
            rows = db.execute(query, [*params, limit, offset]).fetchall()
        return {
            "items": [
                {
                    "fault_id": row["fault_id"],
                    "device_id": row["device_id"],
                    "device_type": row["device_type"],
                    "fault_type": row["fault_type"],
                    "fault_name": row["fault_name"],
                    "symptoms": json.loads(row["symptoms_json"]),
                    "cause": row["cause"],
                    "solution": row["solution"],
                    "verified_by": row["verified_by"],
                    "source": row["source"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

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
        mysql_saved = self._external_write("mysql", "upsert_fault_case", item)
        vector_indexed = self._external_write("qdrant", "upsert", self._case_document(item))
        sync_status = "complete" if mysql_saved and vector_indexed else (
            "pending" if self.external_sync_status()["pending"] else "local_only"
        )
        return {
            "fault_id": fault_id,
            "verified": True,
            "indexed": vector_indexed,
            "mysql_saved": mysql_saved,
            "vector_indexed": vector_indexed,
            "sync_status": sync_status,
            "created_at": now,
        }

    def delete_fault_case(self, fault_id: str) -> dict[str, Any]:
        """删除一条已验证案例，并同步清理 MySQL 镜像与 Qdrant 向量。"""
        if not re.fullmatch(r"F[A-Za-z0-9]{1,31}", fault_id or ""):
            raise ValueError("FAULT_ID_INVALID")
        with self._lock, self._connect() as db:
            cursor = db.execute("DELETE FROM fault_case WHERE fault_id = ?", (fault_id,))
            deleted = cursor.rowcount
        mysql_payload = {"fault_id": fault_id}
        vector_payload = {"source": "fault_cases", "document_id": fault_id}
        mysql_saved = self._external_write("mysql", "delete_fault_case", mysql_payload)
        vector_deleted = self._external_write("qdrant", "delete_document", vector_payload)
        sync_status = "complete" if mysql_saved and vector_deleted else (
            "pending" if self.external_sync_status()["pending"] else "local_only"
        )
        return {
            "fault_id": fault_id,
            "deleted": bool(deleted),
            "mysql_saved": mysql_saved,
            "vector_deleted": vector_deleted,
            "sync_status": sync_status,
        }

    def save_diagnosis(self, record: dict[str, Any]) -> None:
        created_at = iso()
        with self._lock, self._connect() as db:
            db.execute(
                """INSERT INTO diagnosis_record
                (diagnosis_id, request_id, device_id, query, fault_type, fault_name, cause,
                 solutions_json, confidence, router_type, selected_sources_json,
                 retrieved_documents_json, result_json, retrieval_count, rerank_count,
                 retrieval_latency_ms, llm_latency_ms, total_latency_ms, input_tokens,
                 output_tokens, error, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                    json.dumps(
                        record.get("trace_contexts", record["sources"]),
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        {
                            key: value
                            for key, value in record.items()
                            if key not in {"trace_contexts", "request_id"}
                        },
                        ensure_ascii=False,
                    ),
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
        self._external_write(
            "mysql",
            "upsert_diagnosis",
            {**record, "created_at": created_at},
        )

    def save_diagnosis_error(
        self,
        device_id: str,
        query: str,
        code: str,
        message: str,
    ) -> dict[str, str]:
        diagnosis_id = f"DIA_{utc_now():%Y%m%d}_{uuid.uuid4().hex[:8].upper()}"
        request_id = str(uuid.uuid4())
        record = {
            "diagnosis_id": diagnosis_id,
            "request_id": request_id,
            "device_id": device_id,
            "query": query,
            "fault_type": "unknown",
            "fault_name": "Diagnosis failed",
            "cause": message,
            "solutions": [],
            "confidence": 0.0,
            "route": {"router": "failed", "selected_sources": []},
            "sources": [],
            "trace_contexts": [],
            "observability": {
                "retrieval_count": 0,
                "rerank_count": 0,
                "retrieval_latency_ms": 0.0,
                "llm_latency_ms": 0.0,
                "total_latency_ms": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "error": code,
            },
        }
        self.save_diagnosis(record)
        return {"diagnosis_id": diagnosis_id, "request_id": request_id}

    def get_diagnosis_trace(self, diagnosis_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM diagnosis_record WHERE diagnosis_id = ?",
                (diagnosis_id,),
            ).fetchone()
        if not row:
            return None
        item = self._diagnosis_record_from_row(dict(row))
        item["contexts"] = item.pop("trace_contexts")
        item.pop("sources")
        return item

    def latest_remediation_diagnosis(
        self, device_id: str, within_minutes: int = 60
    ) -> dict[str, Any] | None:
        """按设备找最近一次可用于案例沉淀的成功诊断（修复事件未带 diagnosis_id 时兜底）。"""
        cutoff = iso(utc_now() - timedelta(minutes=within_minutes))
        with self._connect() as db:
            row = db.execute(
                """SELECT * FROM diagnosis_record
                WHERE device_id = ? AND error IS NULL AND fault_type != 'realtime_state'
                  AND created_at >= ?
                ORDER BY created_at DESC, diagnosis_id DESC LIMIT 1""",
                (device_id, cutoff),
            ).fetchone()
        if not row:
            return None
        return self._diagnosis_record_from_row(dict(row))

    def list_diagnoses(
        self,
        device_id: str | None = None,
        fault_type: str | None = None,
        status: str = "all",
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        query = "SELECT * FROM diagnosis_record"
        clauses: list[str] = []
        params: list[Any] = []
        if device_id:
            clauses.append("device_id = ?")
            params.append(device_id)
        if fault_type:
            clauses.append("fault_type = ?")
            params.append(fault_type)
        if status == "succeeded":
            clauses.append("error IS NULL")
        elif status == "failed":
            clauses.append("error IS NOT NULL")
        if clauses:
            query += " WHERE " + " AND ".join(clauses)

        count_query = f"SELECT COUNT(*) FROM ({query})"
        query += " ORDER BY created_at DESC, diagnosis_id DESC LIMIT ? OFFSET ?"
        with self._connect() as db:
            total = int(db.execute(count_query, params).fetchone()[0])
            rows = db.execute(query, [*params, limit, offset]).fetchall()

        return {
            "items": [
                {
                    "diagnosis_id": row["diagnosis_id"],
                    "device_id": row["device_id"],
                    "query": row["query"],
                    "fault_type": row["fault_type"],
                    "fault_name": row["fault_name"],
                    "confidence": row["confidence"],
                    "router": row["router_type"],
                    "selected_sources": json.loads(row["selected_sources_json"]),
                    "status": "failed" if row["error"] else "succeeded",
                    "error": row["error"],
                    "total_latency_ms": row["total_latency_ms"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

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
        return {**self.external.status(), "outbox": self.external_sync_status()}
