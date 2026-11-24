@echo off
setlocal
cd /d "%~dp0.."
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -X utf8 -m UI.web_app --open-browser %*
) else (
    python -X utf8 -m UI.web_app --open-browser %*
)
if errorlevel 1 pause
