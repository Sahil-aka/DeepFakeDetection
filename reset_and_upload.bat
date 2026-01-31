@echo off
color 0A
echo ==============================================
echo      FRESH START UPLOAD (The "nuke" option)
echo ==============================================
echo.
echo This will:
echo 1. Delete your local git history (resets everything)
echo 2. Re-create the repository from scratch locally
echo 3. Ignore the large model files
echo 4. Force upload everything to GitHub
echo.
echo NOTE: Your code files are SAFE. Only the "git history" is reset.
echo.
pause

echo.
echo [1/6] Cleaning up old git configuration...
taskkill /F /IM git.exe 2>nul
rmdir /s /q .git

echo.
echo [2/6] Initializing new repository...
git init
git branch -M main

echo.
echo [3/6] Configuring ignored files...
echo # Large files >> .gitignore
echo *.h5 >> .gitignore
echo *.keras >> .gitignore
echo __pycache__/ >> .gitignore
echo .idea/ >> .gitignore
echo .venv/ >> .gitignore

echo.
echo [4/6] Adding files (This excludes the large models)...
git add .

echo.
echo [5/6] Committing...
git commit -m "Initial upload of DeepFake Detection Project"

echo.
echo [6/6] Connecting and Pushing...
git remote add origin https://github.com/Sahil-aka/DeepFakeDetection.git
echo.
echo Pushing... (If asked, log in!)
git push -u origin main --force

echo.
if %errorlevel% neq 0 (
    color 0C
    echo [ERROR] Upload failed!
    echo.
    echo CHECK THESE:
    echo 1. Did you create the repository 'DeepFakeDetection' on GitHub?
    echo 2. Is it empty? (It should be!)
    echo.
) else (
    color 0B
    echo [SUCCESS] WE ARE DONE! 🚀
    echo.
    echo Your large model files are safe on your computer,
    echo but were NOT uploaded to GitHub (too big).
    echo everything else is online!
)
echo.
pause
