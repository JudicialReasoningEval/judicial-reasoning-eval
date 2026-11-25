"""Shared sentence-embedding loader for the evaluation metrics.

M3 (semantic alignment) embeds judge-side and LLM-side reasoning sentences
and compares them with cosine similarity. The default model is
**Qwen/Qwen3-Embedding-8B** (strong multilingual / Italian-legal); override
it with the EMBEDDING_MODEL env var.

Qwen3-Embedding expects an *instruction-formatted* input for asymmetric
retrieval. Our task is symmetric (judge sentence vs LLM sentence are both
"reasoning sentences"), so we apply the **same** instruction to both sides:

    Instruct: {task}\nQuery:{sentence}

This is the format documented on the model card. mpnet and other plain
sentence-transformers ignore the instruction (we only prepend it for Qwen3).
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

DEFAULT_MODEL = os.environ.get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")

# Remote mode: when set (e.g. "http://1.2.3.4:8000/v1"), embeddings are
# fetched from an OpenAI-compatible /v1/embeddings endpoint (a vLLM instance
# serving the embedding model on vast.ai) instead of loading the model on the
# local GPU. Auth via EMBEDDING_API_KEY (the vast bearer).
EMBEDDING_BASE_URL = os.environ.get("EMBEDDING_BASE_URL", "")
_REMOTE_BATCH = int(os.environ.get("EMBEDDING_REMOTE_BATCH", "128"))
# Hard per-input character cap for the remote path. Degenerate model outputs
# contain single "sentences" of 11.5k+ chars (minimax, qwen3.6-35b) that can
# exceed the embedding server's context window and 400 the whole batch;
# truncating such a sentence's tail is semantically harmless for M3.
_REMOTE_MAX_CHARS = int(os.environ.get("EMBEDDING_REMOTE_MAX_CHARS", "6000"))

# Symmetric similarity instruction applied to both sides for Qwen3.
DEFAULT_INSTRUCT = os.environ.get(
    "EMBEDDING_INSTRUCT",
    "Valuta la similarità semantica tra due frasi di ragionamento giuridico "
    "di una sentenza della Corte di Cassazione.",
)


def _is_qwen3_embedding(name: str) -> bool:
    return "qwen3-embedding" in name.lower()


@lru_cache(maxsize=4)
def load_model(name: str = DEFAULT_MODEL):
    """Return a cached SentenceTransformer instance.

    Qwen3-Embedding loads in bf16 with left-padding / last-token pooling
    handled by sentence-transformers; we just request an automatic dtype so
    the 8B model fits comfortably on a single 32 GB GPU.
    """
    from sentence_transformers import SentenceTransformer

    if _is_qwen3_embedding(name):
        return SentenceTransformer(
            name,
            model_kwargs={"torch_dtype": "auto"},
            tokenizer_kwargs={"padding_side": "left"},
        )
    return SentenceTransformer(name)


def _format_for_model(texts: list[str], model_name: str, instruct: str | None) -> list[str]:
    """Prepend the Qwen3 instruction prefix; pass other models through.

    ``instruct=""`` deliberately disables formatting.  Production M3 leaves
    ``instruct`` at ``None`` and is therefore unchanged; the empty-string mode
    exists for diagnostic ablations of the common instruction prefix.
    """
    if not _is_qwen3_embedding(model_name):
        return texts
    if instruct == "":
        return texts
    task = instruct if instruct is not None else DEFAULT_INSTRUCT
    return [f"Instruct: {task}\nQuery:{t}" for t in texts]


@lru_cache(maxsize=1)
def _remote_client():
    from openai import OpenAI

    return OpenAI(
        base_url=EMBEDDING_BASE_URL,
        api_key=os.environ.get("EMBEDDING_API_KEY", "EMPTY"),
        timeout=300,
    )


def _embed_remote(formatted: list[str], model_name: str) -> np.ndarray:
    """OpenAI-compatible /v1/embeddings against a vLLM server; L2-normalised
    client-side (vLLM's normalisation depends on the pooling config — don't
    rely on it)."""
    client = _remote_client()
    formatted = [t[:_REMOTE_MAX_CHARS] for t in formatted]
    rows: list[list[float]] = []
    for start in range(0, len(formatted), _REMOTE_BATCH):
        chunk = formatted[start:start + _REMOTE_BATCH]
        resp = client.embeddings.create(model=model_name, input=chunk)
        # API may return out of order in principle; sort by index.
        for item in sorted(resp.data, key=lambda d: d.index):
            rows.append(item.embedding)
    arr = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def embed(
    texts: list[str],
    model_name: str = DEFAULT_MODEL,
    instruct: str | None = None,
    batch_size: int = int(os.environ.get("EMBEDDING_BATCH", "32")),
) -> np.ndarray:
    """Embed a batch of texts; returns (n, dim) float32 ndarray (L2-normalised).

    Empty list returns shape (0, 1). Whitespace-only strings are encoded as
    given (callers in the metric code filter empties before calling).
    """
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    formatted = _format_for_model(texts, model_name, instruct)
    if EMBEDDING_BASE_URL:
        return _embed_remote(formatted, model_name)
    model = load_model(model_name)
    embeds = model.encode(
        formatted,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=batch_size,
    )
    return embeds.astype(np.float32)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between rows of `a` and rows of `b`.

    Both must be L2-normalised (as returned by `embed`). Returns shape
    (len(a), len(b)).
    """
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    return a @ b.T
