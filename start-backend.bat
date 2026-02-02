@echo off
setlocal
cd /d %~dp0
python -m uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
