from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone

import paho.mqtt.client as mqtt


def publish(client: mqtt.Client, topic: str, payload: dict) -> None:
    client.publish(topic, json.dumps(payload, ensure_ascii=False), qos=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--device-id", default="ESP32_05")
    parser.add_argument(
        "--scenario",
        choices=["normal", "mqtt_timeout", "wifi_weak", "sensor_error"],
        default="mqtt_timeout",
    )
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()

    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"simulator-{args.device_id}",
    )
    client.connect(args.host, args.port, keepalive=60)
    client.loop_start()
    uptime = 86400
    try:
        while True:
            timestamp = datetime.now(timezone.utc).isoformat()
            rssi = -82 if args.scenario == "wifi_weak" else random.randint(-55, -42)
            mqtt_status = "disconnected" if args.scenario == "mqtt_timeout" else "connected"
            temperature = 78.0 if args.scenario == "sensor_error" else round(random.uniform(25, 29), 2)
            common = {
                "timestamp": timestamp,
                "online": True,
                "wifi_status": "connected",
                "rssi": rssi,
                "mqtt_status": mqtt_status,
                "temperature": temperature,
                "uptime": uptime,
                "device_type": "ESP32",
                "firmware_version": "1.2.0",
            }
            publish(client, f"iot/{args.device_id}/status", common)
            publish(client, f"iot/{args.device_id}/telemetry", common)
            publish(
                client,
                f"iot/{args.device_id}/heartbeat",
                {"timestamp": timestamp, "online": True, "uptime": uptime},
            )
            if args.scenario == "mqtt_timeout":
                publish(
                    client,
                    f"iot/{args.device_id}/logs",
                    {
                        "timestamp": timestamp,
                        "level": "ERROR",
                        "module": "mqtt",
                        "message": "MQTT keep alive timeout",
                    },
                )
                publish(
                    client,
                    f"iot/{args.device_id}/fault",
                    {
                        "timestamp": timestamp,
                        "fault_type": "mqtt_timeout",
                        "level": "ERROR",
                        "message": "MQTT keep alive timeout",
                    },
                )
            uptime += int(args.interval)
            time.sleep(args.interval)
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
