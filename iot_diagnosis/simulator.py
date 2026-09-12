from __future__ import annotations

import argparse
import json
import random
import threading
import time
from datetime import datetime, timezone

import paho.mqtt.client as mqtt


def publish(client: mqtt.Client, topic: str, payload: dict) -> None:
    client.publish(topic, json.dumps(payload, ensure_ascii=False), qos=1)


class DeviceState:
    """模拟设备的可变状态：场景故障标志、上报间隔与固件版本，可被下行命令修改。"""

    def __init__(self, scenario: str, interval: float, firmware_version: str = "1.2.0"):
        self.lock = threading.Lock()
        self.mqtt_timeout = scenario == "mqtt_timeout"
        self.wifi_weak = scenario == "wifi_weak"
        self.sensor_error = scenario == "sensor_error"
        self.interval = interval
        self.firmware_version = firmware_version
        self.uptime = 86400

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "mqtt_timeout": self.mqtt_timeout,
                "wifi_weak": self.wifi_weak,
                "sensor_error": self.sensor_error,
                "interval": self.interval,
                "firmware_version": self.firmware_version,
                "uptime": self.uptime,
            }


def handle_command(state: DeviceState, payload: dict) -> dict:
    """处理一条下行命令，返回 cmd_ack 载荷。纯函数便于测试。"""
    command_id = payload.get("command_id", "")
    action = payload.get("action", "")
    parameters = payload.get("parameters") or {}

    def ack(status: str, detail: str) -> dict:
        return {
            "command_id": command_id,
            "status": status,
            "detail": detail,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    if not command_id or not action:
        return ack("failed", "command_id 与 action 不能为空")

    with state.lock:
        if action == "reconnect_mqtt":
            state.mqtt_timeout = False
            return ack("applied", "MQTT 已重新连接")
        if action == "reconnect_wifi":
            state.wifi_weak = False
            return ack("applied", "WiFi 已重新连接")
        if action == "calibrate_sensor":
            state.sensor_error = False
            return ack("applied", "传感器校准完成")
        if action == "set_reporting_interval":
            seconds = parameters.get("seconds")
            if not isinstance(seconds, int) or isinstance(seconds, bool) or seconds < 1:
                return ack("failed", "seconds 必须为正整数")
            state.interval = float(seconds)
            return ack("applied", f"上报间隔已调整为 {seconds} 秒")
        if action == "restart_device":
            state.mqtt_timeout = False
            state.wifi_weak = False
            state.sensor_error = False
            state.uptime = 0
            return ack("applied", "设备已重启")
        if action == "update_firmware":
            version = parameters.get("version")
            if not isinstance(version, str) or not version.strip():
                return ack("failed", "version 不能为空")
            state.mqtt_timeout = False
            state.wifi_weak = False
            state.sensor_error = False
            state.uptime = 0
            state.firmware_version = version.strip()
            return ack("applied", f"固件已升级到 {version.strip()}")
        return ack("failed", f"不支持的动作: {action}")


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

    state = DeviceState(args.scenario, args.interval)
    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"simulator-{args.device_id}",
    )
    # 遗嘱消息：模拟进程崩溃（而非正常退出）时，Broker 代发离线状态，
    # 诊断服务无需等心跳超时即可感知设备掉线。
    client.will_set(
        f"iot/{args.device_id}/status",
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "online": False,
                "offline_reason": "client_lost",
            },
            ensure_ascii=False,
        ),
        qos=1,
        retain=True,
    )

    def on_command(_client, _userdata, message) -> None:
        try:
            payload = json.loads(message.payload.decode("utf-8"))
            if not isinstance(payload, dict):
                return
            ack = handle_command(state, payload)
            publish(client, f"iot/{args.device_id}/cmd_ack", ack)
            if ack["status"] == "applied":
                publish(
                    client,
                    f"iot/{args.device_id}/logs",
                    {
                        "timestamp": ack["timestamp"],
                        "level": "INFO",
                        "module": "remediation",
                        "message": f"命令 {payload.get('command_id')} 已执行: {ack['detail']}",
                    },
                )
                publish(
                    client,
                    f"iot/{args.device_id}/status",
                    current_status(state),
                )
        except Exception:
            import logging

            logging.getLogger(__name__).exception("Failed to handle command")

    client.on_message = on_command
    client.connect(args.host, args.port, keepalive=60)
    client.subscribe(f"iot/{args.device_id}/cmd", qos=1)
    client.loop_start()
    try:
        while True:
            timestamp = datetime.now(timezone.utc).isoformat()
            snapshot = state.snapshot()
            status = current_status(state)
            publish(client, f"iot/{args.device_id}/status", status)
            publish(client, f"iot/{args.device_id}/telemetry", status)
            publish(
                client,
                f"iot/{args.device_id}/heartbeat",
                {"timestamp": timestamp, "online": True, "uptime": snapshot["uptime"]},
            )
            if snapshot["mqtt_timeout"]:
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
            with state.lock:
                state.uptime += int(snapshot["interval"])
            # 分片睡眠，保证下行命令能及时修改上报间隔
            remaining = snapshot["interval"]
            while remaining > 0:
                step = min(0.5, remaining)
                time.sleep(step)
                remaining -= step
    finally:
        client.loop_stop()
        client.disconnect()


def current_status(state: DeviceState, timestamp: str | None = None) -> dict:
    timestamp = timestamp or datetime.now(timezone.utc).isoformat()
    snapshot = state.snapshot()
    return {
        "timestamp": timestamp,
        "online": True,
        "wifi_status": "connected",
        "rssi": -82 if snapshot["wifi_weak"] else random.randint(-55, -42),
        "mqtt_status": "disconnected" if snapshot["mqtt_timeout"] else "connected",
        "temperature": 78.0 if snapshot["sensor_error"] else round(random.uniform(25, 29), 2),
        "uptime": snapshot["uptime"],
        "device_type": "ESP32",
        "firmware_version": snapshot["firmware_version"],
    }


if __name__ == "__main__":
    main()
