@echo off
echo =============================================
echo   SafeGuard AI - Starting Backend Server
echo =============================================
cd /d "%~dp0"

if not exist venv\Scripts\activate (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate

echo Installing/checking dependencies...
pip install -q fastapi uvicorn[standard] sqlalchemy python-jose[cryptography] passlib[bcrypt] python-multipart opencv-python Pillow numpy aiofiles pydantic-settings

echo.
echo Starting FastAPI server on http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo.
uvicorn main:app --reload --host 0.0.0.0 --port 8000
pause
