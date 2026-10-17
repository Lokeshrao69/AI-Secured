"""Ingestion: load documents -> chunk -> embed -> store with access tags.

Run:  python -m medrag.ingest
"""

from __future__ import annotations

import json
import time

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from medrag.config import CATEGORY_ROLES, MANIFEST_PATH, RAW_DIR, ROLES, Settings
from medrag.providers import get_embedder
from medrag.vectorstore import get_store


def load_documents() -> list[Document]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    docs = []
    for entry in manifest["documents"]:
        text = (RAW_DIR / entry["file"]).read_text(encoding="utf-8")
        docs.append(Document(page_content=text, metadata=entry))
    return docs


def access_tags(category: str) -> dict:
    """Flatten allowed roles into boolean metadata (Chroma can filter on these)."""
    allowed = CATEGORY_ROLES[category]
    tags = {f"role_{role}": role in allowed for role in ROLES}
    tags["allowed_roles"] = ",".join(allowed)
    return tags


def chunk_documents(docs: list[Document], settings: Settings) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    chunks = []
    for doc in docs:
        meta = doc.metadata
        for n, piece in enumerate(splitter.split_text(doc.page_content), start=1):
            chunk_id = f"{meta['doc_id']}#c{n}"
            chunks.append(
                Document(
                    page_content=piece,
                    metadata={
                        "chunk_id": chunk_id,
                        "doc_id": meta["doc_id"],
                        "title": meta["title"],
                        "category": meta["category"],
                        "source": meta["file"],
                        **access_tags(meta["category"]),
                    },
                )
            )
    return chunks


def ingest(settings: Settings | None = None) -> dict:
    settings = settings or Settings()
    start = time.time()
    embedder = get_embedder(settings)
    store = get_store(embedder.name, settings.vector_backend)
    store.reset()

    docs = load_documents()
    chunks = chunk_documents(docs, settings)
    # Prefix the title so each chunk carries its document context into the embedding.
    texts_to_embed = [f"{c.metadata['title']}\n{c.page_content}" for c in chunks]
    vectors = embedder.embed_documents(texts_to_embed)
    store.add(
        ids=[c.metadata["chunk_id"] for c in chunks],
        embeddings=vectors,
        texts=[c.page_content for c in chunks],
        metadatas=[c.metadata for c in chunks],
    )
    return {
        "documents": len(docs),
        "chunks": len(chunks),
        "embedder": embedder.name,
        "backend": store.backend,
        "seconds": round(time.time() - start, 1),
    }


def main() -> None:
    stats = ingest()
    print("Ingestion complete:")
    for key, value in stats.items():
        print(f"  {key:<10} {value}")


if __name__ == "__main__":
    main()
