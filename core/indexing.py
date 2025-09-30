# core/indexing.py
# Builds FAISS + metadata from repository/meeting_notes.json

import os, json, time, pickle, hashlib
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_DIR     = PROJECT_ROOT / "repository"
JSON_PATH    = REPO_DIR / "meeting_notes.json"
INDEX_FILE   = REPO_DIR / "faiss_index.bin"
META_FILE    = REPO_DIR / "metadata.pkl"
STATE_FILE   = REPO_DIR / "index_state.json"

BATCH = 64  # rows per embedding call

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def build_from_json(json_path: Path = JSON_PATH) -> Dict[str, Any]:
    """
    Read repository JSON -> embed -> write FAISS index + metadata + state.
    Returns a small dict with stats/paths.
    """
    # env
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing in .env")

    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    client = OpenAI(api_key=api_key)

    # data
    if not json_path.exists():
        raise FileNotFoundError(f"{json_path} not found; ingest notes first.")
    data: List[dict] = json.loads(json_path.read_text(encoding="utf-8"))
    if not data:
        raise RuntimeError("Repository JSON is empty.")

    # ---- sanitize repository texts strictly ----
    def _clean_text(x) -> str:
        if x is None:
            return ""
        if not isinstance(x, str):
            try:
                x = str(x)
            except Exception:
                return ""
        x = x.replace("\x00", " ").strip()  # remove NULs just in case
        # truncate to keep within embedding token limits (~8k tokens ≈ ~16k chars, we stay much lower)
        MAX_CHARS = 8000
        if len(x) > MAX_CHARS:
            x = x[:MAX_CHARS]
        return x

    raw_texts = [row.get("embedded_text", "") for row in data]
    texts = [_clean_text(t) for t in raw_texts]
    metas = [row.get("metadata", {}) for row in data]

    # drop rows that are still empty after cleaning
    keep = [i for i, t in enumerate(texts) if t]
    if len(keep) != len(texts):
        # filter both lists to match lengths
        texts = [texts[i] for i in keep]
        metas = [metas[i] for i in keep]

    # embed in batches
    vectors = []
    skipped_batches = 0
    for i in range(0, len(texts), BATCH):
        chunk = texts[i : i + BATCH]

        # final per-batch filter (defensive): ensure non-empty strings only
        clean_chunk = [t for t in chunk if isinstance(t, str) and t.strip()]
        if not clean_chunk:
            skipped_batches += 1
            continue

        try:
            resp = client.embeddings.create(model=model, input=clean_chunk)
        except Exception as e:
            # surface the exact bad items for quick debugging
            bad_preview = clean_chunk[:3]
            raise RuntimeError(
                f"Embedding call failed on a batch of size {len(clean_chunk)}. "
                f"First items (preview) = {bad_preview!r}. Original error: {e}"
            ) from e

        for d in resp.data:
            vectors.append(d.embedding)

    emb = np.asarray(vectors, dtype="float32")
    # normalize for cosine via inner product
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    emb = emb / norms

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    # write artifacts
    REPO_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))
    META_FILE.write_bytes(pickle.dumps(metas))
    STATE_FILE.write_text(
        json.dumps(
            {
                "json_sha256": _sha256_file(json_path),
                "embedding_model": model,
                "num_vectors": len(texts),
                "dimension": int(emb.shape[1]),
                "timestamp": int(time.time()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "num_vectors": len(texts),
        "index_file": str(INDEX_FILE),
        "meta_file": str(META_FILE),
        "state_file": str(STATE_FILE),
    }
