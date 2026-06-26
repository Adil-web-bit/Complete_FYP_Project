@echo off
echo ========================================
echo Training Emotion Detection Model
echo ========================================
echo.
echo This will take 30-60 minutes...
echo Please be patient and do not close this window.
echo.

python train_model.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Training failed!
    echo Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Training completed successfully!
echo ========================================
echo.
echo Next step: Run 'run.bat' to start the web application
echo.
pause
