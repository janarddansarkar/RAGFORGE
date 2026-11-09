#!/usr/bin/env bash
# One-time setup for the local (Ollama) version: installs the Python packages and downloads
# the Ollama models the settings use. If Ollama is missing (or not running) it offers to fix that.
# Safe to run again: models you already have are skipped.
#
#   ./setup.sh      then      ./run.sh
set -e
cd "$(dirname "$0")"

# Ask a yes/no question; the answer is "no" unless the user types y (or when nobody can answer).
ask() {
    [ -t 0 ] || return 1
    read -r -p "$1 [y/N] " answer
    case "$answer" in
        y | Y | yes | YES) return 0 ;;
        *) return 1 ;;
    esac
}

install_ollama() {
    case "$(uname -s)" in
        Darwin)
            if ! command -v brew >/dev/null; then
                echo "Homebrew is not installed. Download Ollama from https://ollama.com/download and run this again."
                exit 1
            fi
            echo "Running: brew install ollama"
            brew install ollama
            ;;
        Linux)
            echo "Running Ollama's installer: curl -fsSL https://ollama.com/install.sh | sh (it asks for sudo)"
            curl -fsSL https://ollama.com/install.sh | sh
            ;;
        *)
            echo "Download Ollama from https://ollama.com/download and run this again."
            exit 1
            ;;
    esac
}

# Start the Ollama server in the background (it keeps running after this script ends).
start_ollama() {
    echo "Starting Ollama ..."
    nohup ollama serve >/dev/null 2>&1 &
    for _ in $(seq 1 30); do
        ollama list >/dev/null 2>&1 && return 0
        sleep 1
    done
    echo "Ollama did not start. Start it yourself (the Ollama app, or: ollama serve) and run this again."
    exit 1
}

# 1. The tools we need
if ! command -v uv >/dev/null; then
    echo "uv is not installed. Install it from https://docs.astral.sh/uv/ and run this again."
    exit 1
fi
if ! command -v ollama >/dev/null; then
    ask "Ollama is not installed. Install it now?" || {
        echo "Install Ollama from https://ollama.com/download and run this again."
        exit 1
    }
    install_ollama
fi
if ! ollama list >/dev/null 2>&1; then
    ask "Ollama is not running. Start it now?" || {
        echo "Start Ollama (the Ollama app, or: ollama serve) and run this again."
        exit 1
    }
    start_ollama
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
