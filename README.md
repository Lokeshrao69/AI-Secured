# AI-Secured: Privacy-First Hospital RAG Assistant

IBM Z Datathon 2026, theme *"AI Secured: Innovation Without Exposure"*.

Hospital staff ask plain-English questions over confidential documents (case notes,
discharge summaries, billing records, policies). Answers come only from retrieved
chunks, with inline citations like `[DS-004#c2]`.

> Work in progress. Phase 1 (synthetic data + RAG with citations + CLI) is done.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # add watsonx credentials (optional)
python setup_check.py                                 # reports missing packages + fallbacks

python scripts/generate_data.py                       # 40 synthetic documents -> data/raw/
python -m medrag.ingest                               # chunk, embed, store
python scripts/ask.py "What are the ICU visiting hours?"
```

Without watsonx credentials, the app runs fully offline (hashing embeddings plus an
extractive answerer that only quotes retrieved text). Set `LLM_PROVIDER` /
`EMBED_PROVIDER` to `watsonx`, `ollama` or `local` to switch.

## Project layout

| Path | Purpose |
|---|---|
| `medrag/config.py` | Paths, settings, role -> category access map |
| `medrag/providers.py` | LLM + embedding interface (watsonx REST, Ollama, offline) |
| `medrag/vectorstore.py` | ChromaDB wrapper + pure-numpy fallback store |
| `medrag/ingest.py` | Load -> chunk -> embed -> store with access tags |
| `medrag/rag.py` | Retrieval and grounded answering with citations |
| `scripts/generate_data.py` | Synthetic corpus, manifest, PII ground truth |
| `scripts/ask.py` | Command-line assistant |
| `setup_check.py`, `setup_linuxone.sh` | Dependency check / LinuxONE setup |

All data is synthetic. No real patients are represented.
