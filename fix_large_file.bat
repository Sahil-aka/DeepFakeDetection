@echo off
color 0A
echo ==============================================
echo    FIXING LARGE FILE & REMOVING LFS
echo ==============================================
echo.

echo [1/5] Uninstalling Git LFS...
git lfs uninstall
del .gitattributes 2>nul

echo.
echo [2/5] Removing large model file from Git tracking...
REM Remove from index but keep local file
git rm --cached deepfake_cnn_gpu.h5
git rm --cached *.h5 2>nul

echo.
echo [3/5] Cleaning Git History (This might take a moment)...
REM Remove the file from all previous commits to ensure push succeeds
set FILTER_BRANCH_SQUELCH_WARNING=1
git filter-branch --force --index-filter "git rm --cached --ignore-unmatch deepfake_cnn_gpu.h5" --prune-empty --tag-name-filter cat -- --all

echo.
echo [4/5] Updating .gitignore...
echo *.h5 >> .gitignore
echo *.keras >> .gitignore

echo.
echo [5/5] Pushing to GitHub...
git push -u origin main --force

echo.
if %errorlevel% neq 0 (
    color 0C
    echo [ERROR] Push failed.
    echo Please check if you created the repository on GitHub!
) else (
    color 0A
    echo [SUCCESS] Fixed and Uploaded!
    echo The large model file was kept locally but removed from GitHub upload.
)
echo.
pause
