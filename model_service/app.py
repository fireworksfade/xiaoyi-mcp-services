from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"
)
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL", "Qwen/Qwen3-Reranker-0.6B"
)
MAX_LENGTH = int(os.getenv("MODEL_MAX_LENGTH", "2048"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))
RERANKER_BATCH_SIZE = int(os.getenv("RERANKER_BATCH_SIZE", "4"))
RERANKER_INSTRUCTION = os.getenv(
    "RERANKER_INSTRUCTION",
    "Rank passages and verified cases by relevance to an IoT fault diagnosis query",
)


class EmbeddingRequest(BaseModel):
    model: str | None = None
    input: str | list[str]
    dimensions: int | None = Field(default=None, ge=32, le=1024)


class RerankRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    documents: list[str] = Field(min_length=1, max_length=100)
    top_n: int = Field(default=5, ge=1, le=100)


models: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    models["embedding"] = SentenceTransformer(
        EMBEDDING_MODEL,
        device=device,
        model_kwargs={"dtype": dtype},
        tokenizer_kwargs={"padding_side": "left"},
    )
    models["embedding"].max_seq_length = MAX_LENGTH
    reranker_tokenizer = AutoTokenizer.from_pretrained(
        RERANKER_MODEL, padding_side="left"
    )
    reranker_model = AutoModelForCausalLM.from_pretrained(
        RERANKER_MODEL, dtype=dtype
    ).to(device).eval()
    prefix = (
        '<|im_start|>system\nJudge whether the Document meets the requirements based on '
        'the Query and the Instruct provided. Note that the answer can only be "yes" or '
        '"no".<|im_end|>\n<|im_start|>user\n'
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    models["reranker"] = reranker_model
    models["reranker_tokenizer"] = reranker_tokenizer
    models["reranker_prefix"] = reranker_tokenizer.encode(
        prefix, add_special_tokens=False
    )
    models["reranker_suffix"] = reranker_tokenizer.encode(
        suffix, add_special_tokens=False
    )
    models["reranker_false_id"] = reranker_tokenizer.convert_tokens_to_ids("no")
    models["reranker_true_id"] = reranker_tokenizer.convert_tokens_to_ids("yes")
    models["device"] = device
    yield
    models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="IoT Diagnosis Qwen Retrieval Models", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    ready = "embedding" in models and "reranker" in models
    return {
        "status": "ready" if ready else "loading",
        "device": models.get("device"),
        "embedding_model": EMBEDDING_MODEL,
        "reranker_model": RERANKER_MODEL,
        "dimensions": 1024,
    }


@app.post("/v1/embeddings")
def embeddings(request: EmbeddingRequest) -> dict[str, Any]:
    texts = [request.input] if isinstance(request.input, str) else request.input
    vectors = models["embedding"].encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    dimensions = request.dimensions or int(vectors.shape[1])
    if dimensions < vectors.shape[1]:
        vectors = vectors[:, :dimensions]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
    return {
        "object": "list",
        "model": request.model or EMBEDDING_MODEL,
        "data": [
            {"object": "embedding", "index": index, "embedding": vector.tolist()}
            for index, vector in enumerate(vectors)
        ],
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


@app.post("/rerank")
def rerank(request: RerankRequest) -> dict[str, Any]:
    tokenizer = models["reranker_tokenizer"]
    prefix_tokens = models["reranker_prefix"]
    suffix_tokens = models["reranker_suffix"]
    formatted = [
        (
            f"<Instruct>: {RERANKER_INSTRUCTION}\n"
            f"<Query>: {request.query}\n<Document>: {document}"
        )
        for document in request.documents
    ]
    tokenized = tokenizer(
        formatted,
        padding=False,
        truncation="longest_first",
        return_attention_mask=False,
        max_length=MAX_LENGTH - len(prefix_tokens) - len(suffix_tokens),
    )["input_ids"]
    scores: list[float] = []
    for offset in range(0, len(tokenized), RERANKER_BATCH_SIZE):
        input_ids = [
            prefix_tokens + item + suffix_tokens
            for item in tokenized[offset : offset + RERANKER_BATCH_SIZE]
        ]
        inputs = tokenizer.pad(
            {"input_ids": input_ids}, padding=True, return_tensors="pt"
        ).to(models["device"])
        with torch.no_grad():
            logits = models["reranker"](
                **inputs, logits_to_keep=1
            ).logits[:, -1, :]
        yes_no_logits = torch.stack(
            [
                logits[:, models["reranker_false_id"]],
                logits[:, models["reranker_true_id"]],
            ],
            dim=1,
        )
        scores.extend(
            torch.nn.functional.softmax(yes_no_logits, dim=1)[:, 1]
            .float()
            .cpu()
            .tolist()
        )
    ranked = sorted(
        (
            {"index": index, "score": float(score)}
            for index, score in enumerate(scores)
        ),
        key=lambda item: item["score"],
        reverse=True,
    )
    return {
        "model": RERANKER_MODEL,
        "results": ranked[: min(request.top_n, len(ranked))],
    }
