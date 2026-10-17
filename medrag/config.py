"""Central configuration: paths, role access rules and runtime settings."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MANIFEST_PATH = DATA_DIR / "manifest.json"
PII_TRUTH_PATH = DATA_DIR / "pii_ground_truth.json"
STORAGE_DIR = ROOT / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma"
SIMPLE_STORE_DIR = STORAGE_DIR / "simple_store"
AUDIT_DB_PATH = STORAGE_DIR / "audit.db"

# --- Access control -------------------------------------------------------
# Every document belongs to exactly one category. A role may read a category
# only if it is listed here. This is the single source of truth for access.
ROLES = ["doctor", "nurse", "billing_clerk", "admin"]

CATEGORY_ROLES = {
    "case_note": ["doctor"],                         # full clinical case notes
    "discharge_summary": ["doctor", "nurse"],        # care instructions, meds
    "billing": ["billing_clerk"],                    # invoices, payments
    "policy": ["doctor", "nurse", "billing_clerk", "admin"],
}

ROLE_CATEGORIES = {
    role: [cat for cat, roles in CATEGORY_ROLES.items() if role in roles]
    for role in ROLES
}

# The exact sentence returned when nothing relevant is available to the user.
NO_ANSWER = "I don't have information on that in the documents available to your role."


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass
class Settings:
    # Providers: "auto" picks watsonx when credentials exist, otherwise local.
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "auto"))
    embed_provider: str = field(default_factory=lambda: _env("EMBED_PROVIDER", "auto"))
    # Vector store: "chroma", "simple" (pure numpy) or "auto".
    vector_backend: str = field(default_factory=lambda: _env("VECTOR_BACKEND", "auto"))

    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "5")))
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "700")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "100")))

    watsonx_apikey: str = field(default_factory=lambda: _env("WATSONX_APIKEY"))
    watsonx_project_id: str = field(default_factory=lambda: _env("WATSONX_PROJECT_ID"))
    watsonx_url: str = field(default_factory=lambda: _env("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"))
    watsonx_llm_model: str = field(default_factory=lambda: _env("WATSONX_LLM_MODEL", "ibm/granite-3-3-8b-instruct"))
    watsonx_embed_model: str = field(default_factory=lambda: _env("WATSONX_EMBED_MODEL", "ibm/slate-125m-english-rtrvr-v2"))

    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_URL", "http://localhost:11434"))
    ollama_llm_model: str = field(default_factory=lambda: _env("OLLAMA_LLM_MODEL", "granite3.3:8b"))
    ollama_embed_model: str = field(default_factory=lambda: _env("OLLAMA_EMBED_MODEL", "granite-embedding:278m"))

    @property
    def has_watsonx(self) -> bool:
        return bool(self.watsonx_apikey and self.watsonx_project_id)
