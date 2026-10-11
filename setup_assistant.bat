@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" python -m venv .venv
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo Setup completed. Run start_assistant.bat.
pause
exit /b 0
:failed
echo Setup failed. Check the error above.
pause
exit /b 1
