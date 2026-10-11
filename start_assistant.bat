@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please run setup_assistant.bat first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -X utf8 dynamic_app.py
pause
