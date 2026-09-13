from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from iot_diagnosis.repository import DiagnosisRepository

DOCUMENT_SOURCES = {"mqtt_docs", "wifi_docs", "sensor_docs", "device_docs"}
DOCUMENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$")


def normalize_text(text: str) -> str:
    paragraphs = []
    for paragraph in re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n")):
        cleaned = re.sub(r"[ \t\f\v]+", " ", paragraph)
        cleaned = re.sub(r"\n+", " ", cleaned).strip()
        if cleaned:
            paragraphs.append(cleaned)
    return "\n\n".join(paragraphs)


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 120) -> list[str]:
    if not 300 <= chunk_size <= 4000:
        raise ValueError("CHUNK_SIZE_INVALID")
    if not 0 <= overlap <= 500 or overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP_INVALID")
    normalized = normalize_text(text)
    if not normalized:
        raise ValueError("DOCUMENT_EMPTY")

    chunks: list[str] = []
    cursor = 0
    while cursor < len(normalized):
        proposed_end = min(len(normalized), cursor + chunk_size)
        end = proposed_end
        if proposed_end < len(normalized):
            boundary = normalized.rfind("\n\n", cursor + chunk_size // 2, proposed_end)
            if boundary > cursor:
                end = boundary
        chunk = normalized[cursor:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        cursor = max(cursor + 1, end - overlap)
    return chunks


def ingest_text(
    repository: DiagnosisRepository,
    *,
    source: str,
    document_id: str,
    title: str,
    content: str,
    device_type: str | None = "ESP32",
    chunk_size: int = 1200,
    overlap: int = 120,
) -> dict[str, Any]:
    if source not in DOCUMENT_SOURCES:
        raise ValueError("INVALID_DOCUMENT_SOURCE")
    if not DOCUMENT_ID_PATTERN.fullmatch(document_id):
        raise ValueError("DOCUMENT_ID_INVALID")
    if not title.strip():
        raise ValueError("DOCUMENT_TITLE_INVALID")
    chunks = chunk_text(content, chunk_size, overlap)
    return repository.replace_knowledge_document(
        source=source,
        document_id=document_id,
        title=title.strip(),
        chunks=chunks,
        device_type=device_type,
    )


def extract_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError("DOCUMENT_FORMAT_UNSUPPORTED")
