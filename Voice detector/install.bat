@echo off
echo ========================================
echo Speech Emotion Recognition - Setup
echo ========================================
echo.

echo Step 1: Installing dependencies...
echo This may take several minutes...
pip install -r requirements.txt

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Failed to install dependencies!
    echo Please check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Dependencies installed successfully!
echo ========================================
echo.
echo Next steps:
echo 1. Run 'train.bat' to train the model
echo 2. Then run 'run.bat' to start the web app
echo.
pause
