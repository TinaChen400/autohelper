@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo First run setup_audio_lab.bat.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -X utf8 recorder.py
pause

