@echo off
setlocal
cd /d %~dp0
python -m backend.caption_review
endlocal
