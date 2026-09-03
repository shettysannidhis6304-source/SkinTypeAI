@echo off
REM Double-click this file to start SkinType AI.
REM Opens the backend API and frontend website each in their own window,
REM then opens the site in your default browser.

echo Starting SkinType AI backend (http://127.0.0.1:8000)...
start "SkinType AI - Backend (port 8000)" "%~dp0run-backend.bat"

echo Starting SkinType AI frontend (http://127.0.0.1:5500)...
start "SkinType AI - Frontend (port 5500)" "%~dp0run-frontend.bat"

echo Waiting for both servers to come up...
timeout /t 3 /nobreak >nul

echo Opening the site in your browser...
start "" "http://127.0.0.1:5500/index.html"

echo.
echo Both servers are now running in their own windows.
echo   - To stop them: close those two windows, or run stop-servers.bat
echo   - This window can be closed safely; it's not running anything itself.
echo.
pause
