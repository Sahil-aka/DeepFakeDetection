@echo off
color 0A
echo ==============================================
echo        CLEANING UP PROJECT FILES
echo ==============================================
echo.
echo Removing temporary instruction files and scripts...

REM Delete documentation files
del CREATE_REPO_INSTRUCTIONS.md 2>nul
del GITHUB_UPLOAD_GUIDE.md 2>nul
del QUICK_START.txt 2>nul
del READY_TO_UPLOAD.md 2>nul
del TROUBLESHOOTING.md 2>nul

REM Delete helper scripts
del fix_large_file.bat 2>nul
del reset_and_upload.bat 2>nul
del upload_to_github.bat 2>nul

echo.
echo [1/2] Committing cleanup...
git add .
git commit -m "Cleanup: Remove temporary instruction files"

echo.
echo [2/2] Pushing clean version...
git push

echo.
echo ==============================================
echo      CLEANUP COMPLETE! 
echo ==============================================
echo.
echo The repository is now clean and professional.
echo Only your project files remain.
echo.
echo (This script will self-destruct now)
(goto) 2>nul & del "%~f0"
