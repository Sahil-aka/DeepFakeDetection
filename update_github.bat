@echo off
echo ==============================================
echo        UPDATING GITHUB REPOSITORY
echo ==============================================
echo.

echo [1/3] Adding all changes...
git add .

echo.
echo [2/3] Committing...
set /p commit_msg="Enter commit message (e.g., 'Updated README'): "
if "%commit_msg%"=="" set commit_msg="Update project files"
git commit -m "%commit_msg%"

echo.
echo [3/3] Pushing to GitHub...
git push

echo.
if %errorlevel% neq 0 (
    color 0C
    echo [ERROR] Update failed. 
) else (
    color 0A
    echo [SUCCESS] GitHub repository updated!
)
echo.
pause
