@echo off
REM Runs the SkinType AI backend API on http://127.0.0.1:8000
REM Not meant to be double-clicked directly -- use start-servers.bat instead
REM (this one is called by it, and stays open so you can see server logs).
REM
REM Uses .venv-new (the rebuilt Python 3.11 environment). The old .venv is
REM broken (Python 3.12 with mismatched compiled packages) -- do not use it.

set "PYEXE=%~dp0.venv-new\Scripts\python.exe"
if not exist "%PYEXE%" (
  echo ERROR: %PYEXE% not found.
  echo Create the environment first:  py -3.11 -m venv .venv-new  ^&^&  .venv-new\Scripts\python.exe -m pip install -r requirements.txt
  echo.
  pause
  exit /b 1
)

cd /d "%~dp0backend"
"%PYEXE%" -m uvicorn app.main:app --port 8000

echo.
echo Backend stopped. Press any key to close this window.
pause >nul
