#!/usr/bin/env bash
# Setup for IBM LinuxONE (s390x) or any Linux/macOS box.
# Installs packages one by one so a single failing wheel does not stop the rest,
# then runs setup_check.py to report what is available and which fallbacks apply.
set -u

PYTHON=${PYTHON:-python3}
cd "$(dirname "$0")"

echo "== Creating virtual environment (.venv)"
$PYTHON -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --quiet --upgrade pip setuptools wheel

echo "== Installing packages individually"
FAILED=()
while read -r line; do
  pkg="${line%%#*}"; pkg="$(echo "$pkg" | xargs)"
  [ -z "$pkg" ] && continue
  if pip install --quiet "$pkg"; then
    echo "  installed $pkg"
  else
    echo "  FAILED    $pkg"
    FAILED+=("$pkg")
  fi
done < requirements.txt

if python -c "import spacy" 2>/dev/null; then
  python -m spacy download en_core_web_sm --quiet || echo "  FAILED    spaCy model en_core_web_sm"
fi

[ -f .env ] || cp .env.example .env

echo
echo "== Dependency report"
python setup_check.py

if [ ${#FAILED[@]} -gt 0 ]; then
  echo
  echo "Packages that failed to install: ${FAILED[*]}"
  echo "The app still runs using the fallbacks listed above. On s390x, many of these"
  echo "are available prebuilt from conda-forge or IBM's Python AI Toolkit for IBM Z."
fi
