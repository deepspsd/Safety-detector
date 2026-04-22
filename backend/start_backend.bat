@echo off
title OccuSafe Backend
color 0A
echo ============================================
echo   OccuSafe Backend Startup
echo ============================================
echo.

cd /d "%~dp0"

echo [1/3] Running database migration...
venv\Scripts\python migrate_thumbnail.py
echo.

echo [2/3] Checking imports...
venv\Scripts\python -c "from routers import auth, users, alerts, detection, video, faces, cctv; from services import face_service, yolo_service; print('[OK] All imports OK')"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Import check failed - see error above
    pause
    exit /b 1
)
echo.

echo [3/3] Starting backend on port 8000...
echo       Access at: http://localhost:8000
echo       Health:    http://localhost:8000/health
echo.
venv\Scripts\python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
pause
