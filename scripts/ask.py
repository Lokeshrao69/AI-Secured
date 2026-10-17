"""Ask the hospital assistant a question from the command line.

  python scripts/ask.py "What is the visiting time for the ICU?"
  python scripts/ask.py            (interactive mode)
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from medrag.rag import RAGPipeline  # noqa: E402


def show(result) -> None:
    print("\nANSWER\n" + "-" * 70)
    print(result.answer)
    if result.citations:
        print("\nSOURCES")
        for hit in result.citations:
            m = hit.metadata
            print(f"  [{hit.id}] {m['title']}  ({m['source']}, score {hit.score:.2f})")
    print(f"\ngrounded={result.grounded}  refused={result.refused}  retrieved={[h.id for h in result.retrieved]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="*", help="question to ask (omit for interactive mode)")
    parser.add_argument("--show-chunks", action="store_true", help="print the retrieved chunk text")
    args = parser.parse_args()

    rag = RAGPipeline()
    print(f"[providers] llm={rag.llm.name}  embeddings={rag.embedder.name}  store={rag.store.backend}")

    def run(question: str) -> None:
        result = rag.ask(question)
        if args.show_chunks:
            for h in result.retrieved:
                print(f"\n--- {h.id} (score {h.score:.2f}) ---\n" + textwrap.indent(h.text, "  "))
        show(result)

    if args.question:
        run(" ".join(args.question))
        return
    while True:
        try:
            question = input("\nQuestion (blank to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        run(question)


if __name__ == "__main__":
    main()
