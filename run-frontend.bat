@echo off
REM Serves the SkinType AI website on http://127.0.0.1:5500
REM Not meant to be double-clicked directly -- use start-servers.bat instead
REM (this one is called by it, and stays open so you can see server logs).
REM
REM Uses .venv-new (the rebuilt Python 3.11 environment).

set "PYEXE=%~dp0.venv-new\Scripts\python.exe"
if not exist "%PYEXE%" (
  echo ERROR: %PYEXE% not found. See run-backend.bat for setup instructions.
  echo.
  pause
  exit /b 1
)

cd /d "%~dp0frontend"
"%PYEXE%" -m http.server 5500

echo.
echo Frontend stopped. Press any key to close this window.
pause >nul
