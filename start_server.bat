@echo off
echo ========================================
echo   Deepfake Detection - Starting Server
echo ========================================
echo.

echo Step 1: Installing dependencies...
python -m pip install fastapi uvicorn[standard] python-multipart pillow numpy --quiet

echo.
echo Step 2: Starting FastAPI server...
echo Server will be available at: http://localhost:8000
echo API Documentation at: http://localhost:8000/docs
echo.
echo Press CTRL+C to stop the server
echo.

uvicorn app:app --reload
