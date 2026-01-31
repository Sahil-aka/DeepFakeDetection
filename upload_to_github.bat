@echo off
color 0A
echo ==============================================
echo       FIXING GITHUB UPLOAD ISSUE
echo ==============================================
echo.

REM Kill hanging processes
taskkill /F /IM git.exe 2>nul

echo [1/4] Configuring Remote...
git remote remove origin 2>nul
git remote add origin https://github.com/Sahil-aka/DeepFakeDetection.git

echo.
echo [2/4] Ensuring branch is 'main'...
git branch -M main

echo.
echo [3/4] Adding and Committing files...
git add .
git commit -m "Final showcase upload"

echo.
echo [4/4] PUSHING TO GITHUB...
echo ----------------------------------------------------
echo IMPORTANT: If a login window appears:
echo 1. Browser Login (Recommended if option available)
echo 2. OR Token Login (Username + Personal Access Token)
echo ----------------------------------------------------
echo.
echo Pushing now... (Please wait)
git push -u origin main

echo.
if %errorlevel% neq 0 (
    color 0C
    echo [ERROR] Upload failed!
    echo.
    echo COMMON FIXES:
    echo 1. Check your internet
    echo 2. Make sure repository 'DeepFakeDetection' exists on GitHub
    echo 3. Use a Personal Access Token for password
    echo.
) else (
    color 0A
    echo [SUCCESS] Project uploaded successfully!
    echo Verifying: https://github.com/Sahil-aka/DeepFakeDetection
)
echo.
pause
