"""Retrieval-augmented answering with inline citations."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from medrag.config import NO_ANSWER, Settings
from medrag.providers import get_embedder, get_llm
from medrag.vectorstore import Hit, get_store

SYSTEM_PROMPT = f"""You are a hospital knowledge assistant for clinical and administrative staff.
Rules:
1. Answer ONLY from the CONTEXT chunks. Do not use outside knowledge.
2. After every sentence that uses a fact, cite the chunk id in square brackets, e.g. [DS-004#c2].
3. If the context does not contain the answer, reply exactly: "{NO_ANSWER}"
4. Personal details appear as placeholders such as <PERSON>; keep them as placeholders.
5. Be concise and factual."""

CITATION_RE = re.compile(r"\[([A-Z]+-\d+#c\d+)\]")


@dataclass
class RAGResult:
    question: str
    answer: str
    citations: list[Hit] = field(default_factory=list)  # chunks the answer cites
    retrieved: list[Hit] = field(default_factory=list)  # chunks given to the LLM
    grounded: bool = False  # True if the answer cites at least one retrieved chunk
    refused: bool = False  # True if the system said it has no information


def build_user_prompt(question: str, hits: list[Hit]) -> str:
    blocks = [f"[{h.id}] ({h.metadata['title']})\n{h.text}" for h in hits]
    return "CONTEXT:\n\n" + "\n\n---\n\n".join(blocks) + f"\n\nQUESTION: {question}"


class RAGPipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.embedder = get_embedder(self.settings)
        self.llm = get_llm(self.settings)
        self.store = get_store(self.embedder.name, self.settings.vector_backend)
        if self.store.count() == 0:
            raise RuntimeError("Vector store is empty. Run: python -m medrag.ingest")

    def retrieve(self, question: str, where: dict | None = None, k: int | None = None) -> list[Hit]:
        vector = self.embedder.embed_query(question)
        hits = self.store.query(vector, k or self.settings.top_k, where)
        return [h for h in hits if h.score >= self.embedder.min_score]

    def ask(self, question: str) -> RAGResult:
        hits = self.retrieve(question)
        return self.answer_from(question, hits)

    def answer_from(self, question: str, hits: list[Hit]) -> RAGResult:
        if not hits:
            return RAGResult(question, NO_ANSWER, refused=True)
        contexts = [(h.id, h.text) for h in hits]
        answer = self.llm.generate(SYSTEM_PROMPT, build_user_prompt(question, hits), contexts)
        by_id = {h.id: h for h in hits}
        # Keep only citations that point at chunks we actually supplied.
        cited_ids = list(dict.fromkeys(cid for cid in CITATION_RE.findall(answer) if cid in by_id))
        refused = NO_ANSWER.lower() in answer.lower()
        return RAGResult(
            question=question,
            answer=answer,
            citations=[by_id[c] for c in cited_ids],
            retrieved=hits,
            grounded=bool(cited_ids),
            refused=refused,
        )
