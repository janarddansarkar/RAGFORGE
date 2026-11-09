#!/usr/bin/env bash
# One-time setup for the local (Ollama) version: installs the Python packages and downloads
# the Ollama models the settings use. Safe to run again: models you already have are skipped.
#
#   ./setup.sh      then      ./run.sh
set -e
cd "$(dirname "$0")"

# 1. The tools we need
if ! command -v uv >/dev/null; then
    echo "uv is not installed. Install it from https://docs.astral.sh/uv/ and run this again."
    exit 1
fi
if ! command -v ollama >/dev/null; then
    echo "Ollama is not installed. Install it from https://ollama.com and run this again."
    exit 1
fi
if ! ollama list >/dev/null 2>&1; then
    echo "Ollama is installed but not running. Start it (the Ollama app, or: ollama serve) and run this again."
    exit 1
fi

# 2. The Python packages
uv sync

# 3. The Ollama models (the list comes from config.toml and best_config.json)
have=$(ollama list | awk 'NR > 1 {print $1}')
for model in $(uv run python src/needed_models.py); do
    if echo "$have" | grep -qx -e "$model" -e "$model:latest"; then
        echo "already have $model"
    else
        echo "downloading $model ..."
        ollama pull "$model"
    fi
done

echo
echo "Done. Start the app with: ./run.sh"
echo "(The first question also downloads the reranker from Hugging Face, about 1 GB.)"
