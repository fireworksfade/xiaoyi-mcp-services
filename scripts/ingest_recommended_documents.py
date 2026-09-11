from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from iot_diagnosis.ingestion import ingest_text
from iot_diagnosis.repository import DiagnosisRepository


DOCUMENTS = (
    (
        "esp_mqtt_error_diagnosis.md",
        "mqtt_docs",
        "ESP_MQTT_ERROR_DIAG",
        "ESP-MQTT 错误事件与连接故障诊断",
    ),
    (
        "esp_wifi_disconnect_diagnosis.md",
        "wifi_docs",
        "ESP_WIFI_DISCONNECT_DIAG",
        "ESP32 WiFi 断线原因与重连诊断",
    ),
    (
        "esp_fatal_reset_diagnosis.md",
        "device_docs",
        "ESP_FATAL_RESET_DIAG",
        "ESP32 致命错误、重启原因与 Backtrace 诊断",
    ),
    (
        "esp_heap_memory_diagnosis.md",
        "device_docs",
        "ESP_HEAP_MEMORY_DIAG",
        "ESP32 内存泄漏、碎片与低内存诊断",
    ),
    (
        "esp_i2c_sensor_diagnosis.md",
        "sensor_docs",
        "ESP_I2C_SENSOR_DIAG",
        "ESP32 I2C 传感器通信故障诊断",
    ),
    (
        "mqtt_session_qos_diagnosis.md",
        "mqtt_docs",
        "MQTT_SESSION_QOS_DIAG",
        "MQTT Keep Alive、Session 与 QoS 故障诊断",
    ),
    (
        "esp_watchdog_task_diagnosis.md",
        "device_docs",
        "ESP_WATCHDOG_TASK_DIAG",
        "ESP32 Watchdog 与任务阻塞诊断",
    ),
    (
        "esp_core_dump_backtrace_diagnosis.md",
        "device_docs",
        "ESP_CORE_DUMP_DIAG",
        "ESP32 Core Dump 与 Backtrace 分析",
    ),
    (
        "esp_adc_calibration_diagnosis.md",
        "sensor_docs",
        "ESP_ADC_CALIBRATION_DIAG",
        "ESP32 ADC 校准与传感器漂移诊断",
    ),
    (
        "mosquitto_broker_diagnosis.md",
        "mqtt_docs",
        "MOSQUITTO_BROKER_DIAG",
        "Mosquitto Broker 配置与日志诊断",
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest the curated ESP32 diagnosis documents shipped with the service"
    )
    parser.add_argument(
        "--database",
        default=os.getenv("DIAGNOSIS_DATABASE_PATH", "data/iot_diagnosis.db"),
    )
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=120)
    args = parser.parse_args()

    knowledge_dir = Path(__file__).resolve().parents[1] / "knowledge"
    repository = DiagnosisRepository(args.database)
    results = []
    for filename, source, document_id, title in DOCUMENTS:
        results.append(
            ingest_text(
                repository,
                source=source,
                document_id=document_id,
                title=title,
                content=(knowledge_dir / filename).read_text(encoding="utf-8"),
                device_type="ESP32",
                chunk_size=args.chunk_size,
                overlap=args.overlap,
            )
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
