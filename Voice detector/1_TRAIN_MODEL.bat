@echo off
echo ========================================
echo   Training Emotion Detection Model
echo ========================================
echo.

cd "C:\Users\AT\OneDrive - University of Engineering and Technology Taxila\Desktop\Voice detector"
C:\VoiceDetector\venv\Scripts\python.exe train_model.py

echo.
echo ========================================
echo   Training Complete!
echo ========================================
pause
