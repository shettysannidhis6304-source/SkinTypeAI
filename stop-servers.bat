@echo off
REM Double-click this file to force-stop SkinType AI's backend and frontend
REM servers if you'd rather not close their windows manually.

echo Stopping SkinType AI servers (ports 8000 and 5500)...

for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%p >nul 2>&1
)
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":5500" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%p >nul 2>&1
)

echo Done. Both servers (if they were running) have been stopped.
pause
