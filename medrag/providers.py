"""LLM and embedding providers behind one small interface.

Swap providers with env vars (LLM_PROVIDER / EMBED_PROVIDER):
  watsonx  - IBM watsonx.ai over plain REST (no SDK, works on s390x)
  ollama   - a local Ollama server
  local    - fully offline: hashing embeddings + extractive answerer
"""

from __future__ import annotations

import math
import re
import time
import zlib
from abc import ABC, abstractmethod

import numpy as np
import requests
from langchain_core.embeddings import Embeddings

from medrag.config import NO_ANSWER, Settings

# --- Interfaces -----------------------------------------------------------


class EmbeddingProvider(Embeddings, ABC):
    """LangChain-compatible embeddings with a name and a relevance floor."""

    name: str = "base"
    # Minimum cosine similarity for a chunk to count as relevant.
    min_score: float = 0.0


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, system: str, user: str, contexts: list[tuple[str, str]]) -> str:
        """Return the answer text.

        `contexts` holds (chunk_id, text) pairs already rendered inside `user`.
        Real LLMs ignore it; the offline extractive model uses it directly.
        """


# --- Text helpers (used by the offline providers) -------------------------

_STOPWORDS = set(
    """a an the and or of to in on for with by at from is are was were be been being
    this that these those it its as what which who whom whose how when where why
    do does did done can could should would will shall may might must has have had
    i you he she we they me my our your their his her them us not no yes any all
    about into over under than then there here so such per via if also only
    after before given get got patient patients please tell give list show""".split()
)


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [_stem(w) for w in words if w not in _STOPWORDS and len(w) > 1]


def _stem(word: str) -> str:
    for suffix in ("ations", "ation", "ings", "ing", "ies", "es", "ed", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


# --- Local offline providers ----------------------------------------------


class LocalHashEmbeddings(EmbeddingProvider):
    """Deterministic feature-hashing embeddings (unigrams + bigrams).

    No model download, pure numpy, identical results on x86 and s390x.
    Good enough for keyword-heavy hospital text; use watsonx for semantics.
    """

    name = "local-hash"
    min_score = 0.10

    def __init__(self, dim: int = 2048):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        tokens = tokenize(text)
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        vec = np.zeros(self.dim, dtype=np.float32)
        for feat in features:
            h = zlib.crc32(feat.encode("utf-8"))
            vec[h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        vec = np.sign(vec) * np.log1p(np.abs(vec))
        norm = np.linalg.norm(vec)
        return (vec / norm if norm else vec).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class ExtractiveLLM(LLMProvider):
    """Offline stand-in for an LLM: quotes the best-matching sentences
    from the retrieved chunks, each with its citation. It can never
    invent facts, which makes it a safe fallback for demos."""

    name = "local-extractive"

    def __init__(self, max_sentences: int = 4):
        self.max_sentences = max_sentences

    ABBREVIATIONS = ("Rs.", "Dr.", "Mr.", "Ms.", "No.", "Tab.", "Inj.", "e.g.", "i.e.", "vs.")
    LIST_ITEM = re.compile(r"^\s*(\d+\.|-|•)\s+")
    FIELD_LINE = re.compile(r"^[A-Z][A-Za-z /()&.-]{1,40}:\s*\S")

    def _sentences(self, prose: str) -> list[str]:
        """Split prose into sentences without breaking on 'Rs.', 'Dr.' etc."""
        out = []
        for piece in re.split(r"(?<=[.!?])\s+(?=[A-Z])", prose):
            if out and out[-1].endswith(self.ABBREVIATIONS):
                out[-1] += " " + piece
            else:
                out.append(piece)
        return out

    def _units(self, text: str, skip_first_block: bool) -> list[str]:
        """Answerable units: a header with its list items, a 'Field: value'
        line, or a sentence of prose (wrapped lines joined back together)."""
        units = []
        blocks = re.split(r"\n\s*\n", text)
        if skip_first_block:  # the document letterhead, e.g. hospital name + title
            blocks = blocks[1:]
        for block in blocks:
            lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
            if not lines:
                continue
            if any(self.LIST_ITEM.match(ln) for ln in lines):
                head = lines[0] if not self.LIST_ITEM.match(lines[0]) else ""
                items = [self.LIST_ITEM.sub("", ln) for ln in lines if self.LIST_ITEM.match(ln)]
                units.append((head + " " if head else "") + "; ".join(items))
            elif all(self.FIELD_LINE.match(ln) for ln in lines):
                units.extend(lines)
            else:
                units.extend(self._sentences(" ".join(lines)))
        return units

    def generate(self, system: str, user: str, contexts: list[tuple[str, str]]) -> str:
        question = user.rsplit("QUESTION:", 1)[-1]
        q_terms = set(tokenize(question))
        min_overlap = 2 if len(q_terms) >= 3 else 1
        candidates = []
        for rank, (chunk_id, text) in enumerate(contexts):
            for sentence in self._units(text, skip_first_block=chunk_id.endswith("#c1")):
                candidates.append((rank, chunk_id, sentence, set(tokenize(sentence))))
        # Rare question terms (e.g. "ticagrelor") matter more than common ones ("discharge").
        df = {t: sum(t in terms for *_, terms in candidates) for t in q_terms}
        idf = {t: math.log(1 + len(candidates) / (1 + df[t])) for t in q_terms}
        scored = []
        for rank, chunk_id, sentence, s_terms in candidates:
            matched = q_terms & s_terms
            if len(matched) < min_overlap or len(s_terms) < 3:
                continue
            score = sum(idf[t] for t in matched) / len(s_terms) ** 0.3 - 0.1 * rank
            scored.append((score, sentence, chunk_id))
        if not scored:
            return NO_ANSWER
        scored.sort(key=lambda x: x[0], reverse=True)
        seen, lines = set(), []
        for _, sentence, chunk_id in scored:
            if sentence in seen:
                continue
            seen.add(sentence)
            lines.append(f"- {sentence} [{chunk_id}]")
            if len(lines) == self.max_sentences:
                break
        return "From the retrieved records:\n" + "\n".join(lines)


# --- IBM watsonx.ai (REST) -------------------------------------------------


class WatsonxClient:
    """Minimal watsonx.ai REST client: IAM token + POST helper."""

    IAM_URL = "https://iam.cloud.ibm.com/identity/token"

    def __init__(self, settings: Settings):
        if not settings.has_watsonx:
            raise RuntimeError("WATSONX_APIKEY and WATSONX_PROJECT_ID must be set in .env")
        self.s = settings
        self._token, self._expires = "", 0.0

    def _get_token(self) -> str:
        if time.time() < self._expires - 60:
            return self._token
        resp = requests.post(
            self.IAM_URL,
            data={"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": self.s.watsonx_apikey},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        resp.raise_for_status()
        body = resp.json()
        self._token, self._expires = body["access_token"], time.time() + body.get("expires_in", 3600)
        return self._token

    def post(self, path: str, payload: dict) -> dict:
        url = f"{self.s.watsonx_url.rstrip('/')}{path}"
        payload = {"project_id": self.s.watsonx_project_id, **payload}
        resp = requests.post(url, json=payload, headers={"Authorization": f"Bearer {self._get_token()}"}, timeout=120)
        if resp.status_code >= 400:
            raise RuntimeError(f"watsonx error {resp.status_code}: {resp.text[:500]}")
        return resp.json()


class WatsonxEmbeddings(EmbeddingProvider):
    min_score = 0.30

    def __init__(self, settings: Settings, client: WatsonxClient | None = None):
        self.client = client or WatsonxClient(settings)
        self.model = settings.watsonx_embed_model
        self.name = "watsonx-" + self.model.split("/")[-1]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for i in range(0, len(texts), 20):
            body = self.client.post(
                "/ml/v1/text/embeddings?version=2024-05-01",
                {"model_id": self.model, "inputs": texts[i : i + 20], "parameters": {"truncate_input_tokens": 512}},
            )
            vectors.extend(r["embedding"] for r in body["results"])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class WatsonxLLM(LLMProvider):
    def __init__(self, settings: Settings, client: WatsonxClient | None = None):
        self.client = client or WatsonxClient(settings)
        self.model = settings.watsonx_llm_model
        self.name = "watsonx-" + self.model.split("/")[-1]

    def generate(self, system: str, user: str, contexts: list[tuple[str, str]]) -> str:
        body = self.client.post(
            "/ml/v1/text/chat?version=2024-05-01",
            {
                "model_id": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "max_tokens": 600,
                "temperature": 0,
            },
        )
        return body["choices"][0]["message"]["content"].strip()


# --- Ollama (local server) ------------------------------------------------


class OllamaEmbeddings(EmbeddingProvider):
    min_score = 0.30

    def __init__(self, settings: Settings):
        self.url, self.model = settings.ollama_url.rstrip("/"), settings.ollama_embed_model
        self.name = "ollama-" + self.model.replace(":", "-")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        resp = requests.post(f"{self.url}/api/embed", json={"model": self.model, "input": texts}, timeout=300)
        resp.raise_for_status()
        return resp.json()["embeddings"]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class OllamaLLM(LLMProvider):
    def __init__(self, settings: Settings):
        self.url, self.model = settings.ollama_url.rstrip("/"), settings.ollama_llm_model
        self.name = "ollama-" + self.model

    def generate(self, system: str, user: str, contexts: list[tuple[str, str]]) -> str:
        resp = requests.post(
            f"{self.url}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "options": {"temperature": 0},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            },
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()


# --- Factories ------------------------------------------------------------


def _resolve(choice: str, settings: Settings) -> str:
    choice = choice.lower()
    if choice == "auto":
        return "watsonx" if settings.has_watsonx else "local"
    return choice


def get_embedder(settings: Settings | None = None) -> EmbeddingProvider:
    settings = settings or Settings()
    kind = _resolve(settings.embed_provider, settings)
    if kind == "watsonx":
        return WatsonxEmbeddings(settings)
    if kind == "ollama":
        return OllamaEmbeddings(settings)
    return LocalHashEmbeddings()


def get_llm(settings: Settings | None = None) -> LLMProvider:
    settings = settings or Settings()
    kind = _resolve(settings.llm_provider, settings)
    if kind == "watsonx":
        return WatsonxLLM(settings)
    if kind == "ollama":
        return OllamaLLM(settings)
    return ExtractiveLLM()
