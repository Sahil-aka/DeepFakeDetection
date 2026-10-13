@echo off
echo ========================================
echo   Deepfake Detection v2.0 - PyTorch
echo   Image ^& Video Analysis Backend
echo ========================================
echo.

echo Step 1: Installing dependencies...
python -m pip install fastapi uvicorn[standard] python-multipart pillow numpy torch torchvision opencv-python-headless --quiet

echo.
echo Step 2: Starting FastAPI server...
echo Server will be available at: http://localhost:8000
echo API Documentation at: http://localhost:8000/docs
echo.
echo Supported: Image (.jpg, .png) and Video (.mp4, .avi, .mov, .mkv)
echo.
echo Press CTRL+C to stop the server
echo.

uvicorn app:app --reload