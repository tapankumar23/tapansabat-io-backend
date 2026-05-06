#!/bin/bash
cd "$(dirname "$0")"
.venv/bin/uvicorn chatbot_backend:app --host 0.0.0.0 --port 8001
