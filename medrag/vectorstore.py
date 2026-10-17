"""Vector store wrapper.

ChromaDB is the primary backend. Chroma ships a compiled Rust core with no
s390x wheels, so a tiny pure-numpy store with the same interface is included
as a fallback (VECTOR_BACKEND=simple). Both support equality filters on
metadata, e.g. where={"role_nurse": True}.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from medrag.config import CHROMA_DIR, SIMPLE_STORE_DIR


@dataclass
class Hit:
    id: str
    text: str
    metadata: dict
    score: float  # cosine similarity, higher is better


class ChromaStore:
    backend = "chroma"

    def __init__(self, collection: str, path: Path = CHROMA_DIR):
        import chromadb

        self.client = chromadb.PersistentClient(path=str(path))
        self.name = collection
        self.col = self.client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})

    def reset(self) -> None:
        self.client.delete_collection(self.name)
        self.col = self.client.get_or_create_collection(self.name, metadata={"hnsw:space": "cosine"})

    def add(self, ids, embeddings, texts, metadatas) -> None:
        self.col.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    def count(self) -> int:
        return self.col.count()

    def query(self, embedding, k: int, where: dict | None = None) -> list[Hit]:
        res = self.col.query(query_embeddings=[embedding], n_results=k, where=where or None)
        return [
            Hit(id=i, text=t, metadata=m, score=1.0 - d)
            for i, t, m, d in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

    def all_chunks(self) -> list[Hit]:
        res = self.col.get(include=["documents", "metadatas"])
        return [Hit(i, t, m, 0.0) for i, t, m in zip(res["ids"], res["documents"], res["metadatas"])]


class SimpleStore:
    """Brute-force cosine search over a numpy matrix, persisted to disk."""

    backend = "simple"

    def __init__(self, collection: str, path: Path = SIMPLE_STORE_DIR):
        self.dir = path / collection
        self.ids, self.texts, self.metas = [], [], []
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        if (self.dir / "chunks.json").exists():
            data = json.loads((self.dir / "chunks.json").read_text(encoding="utf-8"))
            self.ids, self.texts, self.metas = data["ids"], data["texts"], data["metadatas"]
            self.vectors = np.load(self.dir / "vectors.npy")

    def reset(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)
        self.ids, self.texts, self.metas = [], [], []
        self.vectors = np.zeros((0, 0), dtype=np.float32)

    def add(self, ids, embeddings, texts, metadatas) -> None:
        vecs = np.asarray(embeddings, dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        self.vectors = vecs if self.vectors.size == 0 else np.vstack([self.vectors, vecs])
        self.ids += list(ids)
        self.texts += list(texts)
        self.metas += list(metadatas)
        self.dir.mkdir(parents=True, exist_ok=True)
        np.save(self.dir / "vectors.npy", self.vectors)
        payload = {"ids": self.ids, "texts": self.texts, "metadatas": self.metas}
        (self.dir / "chunks.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def count(self) -> int:
        return len(self.ids)

    def query(self, embedding, k: int, where: dict | None = None) -> list[Hit]:
        if not self.ids:
            return []
        q = np.asarray(embedding, dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-12
        scores = self.vectors @ q
        allowed = [i for i, m in enumerate(self.metas) if all(m.get(key) == val for key, val in (where or {}).items())]
        best = sorted(allowed, key=lambda i: scores[i], reverse=True)[:k]
        return [Hit(self.ids[i], self.texts[i], self.metas[i], float(scores[i])) for i in best]

    def all_chunks(self) -> list[Hit]:
        return [Hit(i, t, m, 0.0) for i, t, m in zip(self.ids, self.texts, self.metas)]


def get_store(embedder_name: str, backend: str = "auto"):
    """One collection per embedding model, so switching providers never
    mixes vectors of different dimensions."""
    collection = "hospital_" + "".join(c if c.isalnum() else "_" for c in embedder_name)
    if backend in ("auto", "chroma"):
        try:
            return ChromaStore(collection)
        except ImportError:
            if backend == "chroma":
                raise
            print("[vectorstore] chromadb not installed, using pure-numpy SimpleStore")
    return SimpleStore(collection)
