@echo off
title OccuSafe — Auth Fix
color 0A
echo.
echo ================================================
echo   OccuSafe Auth Diagnostics ^& Restart
echo ================================================
echo.
cd /d "%~dp0"

echo [Step 1] Testing if backend responds...
curl -s --max-time 3 http://localhost:8000/health > nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo    Backend IS running on port 8000
    echo.
    echo [Step 2] Testing auth endpoints...
    venv\Scripts\python test_auth.py
    echo.
    echo If auth works above, the issue is the Vite proxy.
    echo Try: npm run dev  (in the frontend folder)
) else (
    echo    Backend is NOT responding on port 8000
    echo    Running import check to find crash cause...
    echo.
    venv\Scripts\python check_startup.py
    echo.
    echo [Step 3] Attempting restart...
    echo    Starting uvicorn on 0.0.0.0:8000
    echo    (Press Ctrl+C to stop)
    echo.
    venv\Scripts\python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000 --log-level info
)
pause
