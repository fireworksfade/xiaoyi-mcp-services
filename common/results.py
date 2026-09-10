import uuid
from typing import Any


def success(data: Any) -> dict[str, Any]:
    return {"ok": True, "data": data, "error": None, "trace_id": str(uuid.uuid4())}


def failure(code: str, message: str, *, retryable: bool = False) -> dict[str, Any]:
    return {
        "ok": False,
        "data": None,
        "error": {"code": code, "message": message, "retryable": retryable},
        "trace_id": str(uuid.uuid4()),
    }
