#!/usr/bin/env bash
# One-time setup: installs the Python packages. No model is downloaded: every model is hosted.
# In the app, pick a provider and paste your API key; for the command line, set OPENAI_API_KEY
# (or ANTHROPIC_API_KEY / GEMINI_API_KEY) in your environment.
#
#   ./setup.sh      then      ./run.sh
set -e
cd "$(dirname "$0")"

if ! command -v uv >/dev/null; then
    echo "uv is not installed. Install it from https://docs.astral.sh/uv/ and run this again."
    exit 1
fi

uv sync

echo
echo "Done. Start the app with: ./run.sh"
