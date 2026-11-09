#!/usr/bin/env bash
# Start the RAGForge web app (run ./setup.sh once first). Open http://localhost:8501
cd "$(dirname "$0")"
exec uv run streamlit run src/app.py "$@"
