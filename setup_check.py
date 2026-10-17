"""Check every dependency, report what failed and which fallback applies.

  python setup_check.py
"""

import importlib
import platform
import sys

# (import name, pip name, required?, fallback when missing)
CHECKS = [
    ("numpy", "numpy", True, None),
    ("requests", "requests", True, None),
    ("dotenv", "python-dotenv", True, None),
    ("langchain_core", "langchain-core", True, None),
    ("langchain_text_splitters", "langchain-text-splitters", True, None),
    ("chromadb", "chromadb", False, "set VECTOR_BACKEND=simple (pure-numpy store)"),
    ("presidio_analyzer", "presidio-analyzer", False, "set REDACTOR=regex (built-in regex redactor)"),
    ("presidio_anonymizer", "presidio-anonymizer", False, "set REDACTOR=regex (built-in regex redactor)"),
    ("spacy", "spacy", False, "set REDACTOR=regex (built-in regex redactor)"),
    ("streamlit", "streamlit", False, "install pyarrow via conda-forge, or use the CLI (scripts/ask.py)"),
]


def main() -> int:
    print(f"Python {sys.version.split()[0]} on {platform.system()} {platform.machine()}")
    if platform.machine() == "s390x":
        print("IBM Z / LinuxONE detected (s390x, big-endian).")
    print()

    failed_required = False
    for module, pip_name, required, fallback in CHECKS:
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "")
            print(f"  OK    {pip_name:<26} {version}")
        except Exception as exc:  # ImportError, or a broken native extension
            tag = "FAIL " if required else "MISS "
            print(f"  {tag} {pip_name:<26} {type(exc).__name__}: {str(exc)[:60]}")
            if fallback:
                print(f"        -> fallback: {fallback}")
            failed_required |= required

    # spaCy needs a language model for Presidio's name detection.
    try:
        import spacy

        spacy.load("en_core_web_sm")
        print("  OK    spaCy model en_core_web_sm")
    except Exception:
        print("  MISS  spaCy model en_core_web_sm  -> python -m spacy download en_core_web_sm")

    print()
    if failed_required:
        print("Core dependencies are missing: pip install -r requirements.txt")
        return 1
    print("Core dependencies OK. Missing optional packages fall back as shown above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
