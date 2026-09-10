from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import uuid
from typing import Any
from urllib.error import HTTPError
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen


logger = logging.getLogger("xiaoyi.iot_diagnosis.external")


def feature_vector(text: str, dimensions: int = 384) -> list[float]:
    tokens = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text.lower())
    compact = re.sub(r"\s+", "", text.lower())
    tokens.extend(compact[index : index + 3] for index in range(max(0, len(compact) - 2)))
    vector = [0.0] * dimensions
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[bucket] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


class MySQLMirror:
    def __init__(self, dsn: str):
        parsed = urlsplit(dsn)
        if parsed.scheme != "mysql" or not parsed.hostname or not parsed.path.strip("/"):
            raise ValueError("DIAGNOSIS_MYSQL_DSN_INVALID")
        self.params = {
            "host": parsed.hostname,
            "port": parsed.port or 3306,
            "user": unquote(parsed.username or ""),
            "password": unquote(parsed.password or ""),
            "database": parsed.path.strip("/"),
            "charset": "utf8mb4",
            "autocommit": True,
        }
        self.ensure_schema()

    def _connect(self):
        import pymysql

        return pymysql.connect(**self.params)

    def ensure_schema(self) -> None:
        statements = (
            """CREATE TABLE IF NOT EXISTS device (
                device_id VARCHAR(120) PRIMARY KEY, device_type VARCHAR(120) NOT NULL,
                name VARCHAR(200) NOT NULL, firmware_version VARCHAR(120), created_at VARCHAR(64) NOT NULL
            ) CHARACTER SET utf8mb4""",
            """CREATE TABLE IF NOT EXISTS device_status (
                id BIGINT AUTO_INCREMENT PRIMARY KEY, device_id VARCHAR(120) NOT NULL,
                online BOOLEAN NOT NULL, wifi_status VARCHAR(80) NOT NULL, rssi INT,
                mqtt_status VARCHAR(80) NOT NULL, temperature DOUBLE, uptime BIGINT,
                timestamp VARCHAR(64) NOT NULL, UNIQUE KEY uq_device_status_time(device_id, timestamp),
                INDEX idx_device_status_latest(device_id, timestamp)
            ) CHARACTER SET utf8mb4""",
            """CREATE TABLE IF NOT EXISTS device_log (
                id BIGINT AUTO_INCREMENT PRIMARY KEY, device_id VARCHAR(120) NOT NULL,
                level VARCHAR(32) NOT NULL, module VARCHAR(120) NOT NULL, message TEXT NOT NULL,
                timestamp VARCHAR(64) NOT NULL,
                UNIQUE KEY uq_device_log_entry(device_id, timestamp, module, message(191)),
                INDEX idx_device_log_latest(device_id, timestamp)
            ) CHARACTER SET utf8mb4""",
            """CREATE TABLE IF NOT EXISTS knowledge_document (
                source VARCHAR(120) NOT NULL, source_id VARCHAR(160) NOT NULL,
                title VARCHAR(300) NOT NULL, content TEXT NOT NULL, device_type VARCHAR(120),
                created_at VARCHAR(64) NOT NULL, PRIMARY KEY(source, source_id)
            ) CHARACTER SET utf8mb4""",
            """CREATE TABLE IF NOT EXISTS fault_case (
                fault_id VARCHAR(120) PRIMARY KEY, device_id VARCHAR(120), device_type VARCHAR(120) NOT NULL,
                fault_type VARCHAR(120) NOT NULL, fault_name VARCHAR(300) NOT NULL,
                symptoms_json JSON NOT NULL, logs_json JSON NOT NULL, cause TEXT NOT NULL,
                solution TEXT NOT NULL, verified BOOLEAN NOT NULL, verified_by VARCHAR(160) NOT NULL,
                source VARCHAR(120) NOT NULL, created_at VARCHAR(64) NOT NULL, updated_at VARCHAR(64) NOT NULL
            ) CHARACTER SET utf8mb4""",
            """CREATE TABLE IF NOT EXISTS diagnosis_record (
                diagnosis_id VARCHAR(120) PRIMARY KEY, request_id VARCHAR(120) NOT NULL,
                device_id VARCHAR(120) NOT NULL, query TEXT NOT NULL, fault_type VARCHAR(120) NOT NULL,
                fault_name VARCHAR(300) NOT NULL, cause TEXT NOT NULL, solutions_json JSON NOT NULL,
                confidence DOUBLE NOT NULL, router_type VARCHAR(80) NOT NULL,
                selected_sources_json JSON NOT NULL, retrieved_documents_json JSON NOT NULL,
                observability_json JSON NOT NULL, created_at VARCHAR(64) NOT NULL
            ) CHARACTER SET utf8mb4""",
        )
        with self._connect() as db:
            with db.cursor() as cursor:
                for statement in statements:
                    cursor.execute(statement)

    def upsert_device_status(self, device: dict[str, Any], status: dict[str, Any]) -> None:
        with self._connect() as db, db.cursor() as cursor:
            cursor.execute(
                """INSERT INTO device(device_id, device_type, name, firmware_version, created_at)
                VALUES (%s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE
                device_type=VALUES(device_type), name=VALUES(name), firmware_version=VALUES(firmware_version)""",
                (
                    device["device_id"], device["device_type"], device["name"],
                    device.get("firmware_version"), device.get("created_at") or status["timestamp"],
                ),
            )
            cursor.execute(
                """INSERT INTO device_status
                (device_id, online, wifi_status, rssi, mqtt_status, temperature, uptime, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE online=VALUES(online), wifi_status=VALUES(wifi_status),
                rssi=VALUES(rssi), mqtt_status=VALUES(mqtt_status), temperature=VALUES(temperature),
                uptime=VALUES(uptime)""",
                (
                    status["device_id"], status["online"], status["wifi_status"], status.get("rssi"),
                    status["mqtt_status"], status.get("temperature"), status.get("uptime"), status["timestamp"],
                ),
            )

    def upsert_device_statuses(
        self, devices: list[dict[str, Any]], statuses: list[dict[str, Any]]
    ) -> None:
        if not devices and not statuses:
            return
        with self._connect() as db, db.cursor() as cursor:
            if devices:
                cursor.executemany(
                    """INSERT INTO device(device_id, device_type, name, firmware_version, created_at)
                    VALUES (%s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE
                    device_type=VALUES(device_type), name=VALUES(name),
                    firmware_version=VALUES(firmware_version)""",
                    [
                        (
                            item["device_id"], item["device_type"], item["name"],
                            item.get("firmware_version"), item["created_at"],
                        )
                        for item in devices
                    ],
                )
            if statuses:
                cursor.executemany(
                    """INSERT INTO device_status
                    (device_id, online, wifi_status, rssi, mqtt_status, temperature, uptime, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE online=VALUES(online), wifi_status=VALUES(wifi_status),
                    rssi=VALUES(rssi), mqtt_status=VALUES(mqtt_status),
                    temperature=VALUES(temperature), uptime=VALUES(uptime)""",
                    [
                        (
                            item["device_id"], item["online"], item["wifi_status"], item.get("rssi"),
                            item["mqtt_status"], item.get("temperature"), item.get("uptime"),
                            item["timestamp"],
                        )
                        for item in statuses
                    ],
                )
    def add_log(self, item: dict[str, Any]) -> None:
        with self._connect() as db, db.cursor() as cursor:
            cursor.execute(
                """INSERT IGNORE INTO device_log(device_id, level, module, message, timestamp)
                VALUES (%s, %s, %s, %s, %s)""",
                (item["device_id"], item["level"], item["module"], item["message"], item["timestamp"]),
            )

    def add_logs(self, items: list[dict[str, Any]]) -> None:
        if not items:
            return
        with self._connect() as db, db.cursor() as cursor:
            cursor.executemany(
                """INSERT IGNORE INTO device_log(device_id, level, module, message, timestamp)
                VALUES (%s, %s, %s, %s, %s)""",
                [
                    (item["device_id"], item["level"], item["module"], item["message"], item["timestamp"])
                    for item in items
                ],
            )

    def upsert_document(self, item: dict[str, Any]) -> None:
        with self._connect() as db, db.cursor() as cursor:
            cursor.execute(
                """INSERT INTO knowledge_document(source, source_id, title, content, device_type, created_at)
                VALUES (%s, %s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE
                title=VALUES(title), content=VALUES(content), device_type=VALUES(device_type)""",
                (
                    item["source"], item["source_id"], item["title"], item["content"],
                    item.get("device_type"), item["created_at"],
                ),
            )

    def upsert_fault_case(self, item: dict[str, Any]) -> None:
        with self._connect() as db, db.cursor() as cursor:
            cursor.execute(
                """INSERT INTO fault_case
                (fault_id, device_id, device_type, fault_type, fault_name, symptoms_json, logs_json,
                 cause, solution, verified, verified_by, source, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE fault_name=VALUES(fault_name), symptoms_json=VALUES(symptoms_json),
                logs_json=VALUES(logs_json), cause=VALUES(cause), solution=VALUES(solution),
                verified=VALUES(verified), verified_by=VALUES(verified_by), updated_at=VALUES(updated_at)""",
                (
                    item["fault_id"], item.get("device_id"), item["device_type"], item["fault_type"],
                    item["fault_name"], json.dumps(item["symptoms"], ensure_ascii=False),
                    json.dumps(item["logs"], ensure_ascii=False), item["cause"], item["solution"],
                    item["verified"], item["verified_by"], item["source"], item["created_at"], item["updated_at"],
                ),
            )

    def upsert_diagnosis(self, item: dict[str, Any]) -> None:
        with self._connect() as db, db.cursor() as cursor:
            cursor.execute(
                """INSERT INTO diagnosis_record
                (diagnosis_id, request_id, device_id, query, fault_type, fault_name, cause,
                 solutions_json, confidence, router_type, selected_sources_json,
                 retrieved_documents_json, observability_json, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE observability_json=VALUES(observability_json)""",
                (
                    item["diagnosis_id"], item["request_id"], item["device_id"], item["query"],
                    item["fault_type"], item["fault_name"], item["cause"],
                    json.dumps(item["solutions"], ensure_ascii=False), item["confidence"],
                    item["route"]["router"],
                    json.dumps(item["route"]["selected_sources"], ensure_ascii=False),
                    json.dumps(item["sources"], ensure_ascii=False),
                    json.dumps(item["observability"], ensure_ascii=False), item["created_at"],
                ),
            )

    def ping(self) -> bool:
        with self._connect() as db, db.cursor() as cursor:
            cursor.execute("SELECT 1")
            return cursor.fetchone()[0] == 1


class QdrantVectorStore:
    dimensions = 384

    def __init__(self, url: str, collection: str):
        self.url = url.rstrip("/")
        self.collection = collection
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        path = f"/collections/{self.collection}"
        try:
            self._request("GET", path)
            return
        except HTTPError as exc:
            if exc.code != 404:
                raise
        self._request(
            "PUT",
            path,
            {"vectors": {"size": self.dimensions, "distance": "Cosine"}},
        )

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(
            f"{self.url}{path}",
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))

    def upsert(self, item: dict[str, Any]) -> bool:
        source = str(item["source"])
        source_id = str(item["id"])
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{source_id}"))
        self._request(
            "PUT",
            f"/collections/{self.collection}/points?wait=true",
            {
                "points": [
                    {
                        "id": point_id,
                        "vector": feature_vector(str(item["content"]), self.dimensions),
                        "payload": item,
                    }
                ]
            },
        )
        return True

    def search(self, query: str, sources: list[str], top_k: int) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "query": feature_vector(query, self.dimensions),
            "limit": top_k,
            "with_payload": True,
        }
        if sources:
            payload["filter"] = {"must": [{"key": "source", "match": {"any": sources}}]}
        response = self._request(
            "POST", f"/collections/{self.collection}/points/query", payload
        )
        points = (response.get("result") or {}).get("points") or []
        return [
            {**dict(point.get("payload") or {}), "score": float(point.get("score") or 0.0)}
            for point in points
        ]

    def ping(self) -> bool:
        response = self._request("GET", f"/collections/{self.collection}")
        return response.get("status") == "ok"


class ExternalStores:
    def __init__(self):
        self.mysql: MySQLMirror | None = None
        self.qdrant: QdrantVectorStore | None = None
        self.errors: dict[str, str] = {}
        mysql_dsn = os.getenv("DIAGNOSIS_MYSQL_DSN", "").strip()
        qdrant_url = os.getenv("DIAGNOSIS_QDRANT_URL", "").strip()
        if mysql_dsn:
            try:
                self.mysql = MySQLMirror(mysql_dsn)
            except Exception as exc:
                self.errors["mysql"] = type(exc).__name__
                logger.exception("MySQL mirror unavailable")
        if qdrant_url:
            try:
                self.qdrant = QdrantVectorStore(
                    qdrant_url,
                    os.getenv("DIAGNOSIS_QDRANT_COLLECTION", "iot_diagnosis_knowledge"),
                )
            except Exception as exc:
                self.errors["qdrant"] = type(exc).__name__
                logger.exception("Qdrant vector store unavailable")

    def status(self) -> dict[str, Any]:
        mysql_status = "disabled"
        qdrant_status = "disabled"
        if self.mysql:
            try:
                mysql_status = "connected" if self.mysql.ping() else "fallback"
            except Exception as exc:
                mysql_status = "fallback"
                self.errors["mysql"] = type(exc).__name__
        if self.qdrant:
            try:
                qdrant_status = "connected" if self.qdrant.ping() else "fallback"
            except Exception as exc:
                qdrant_status = "fallback"
                self.errors["qdrant"] = type(exc).__name__
        return {
            "sqlite": "connected",
            "mysql": mysql_status,
            "qdrant": qdrant_status,
            "errors": self.errors,
        }
