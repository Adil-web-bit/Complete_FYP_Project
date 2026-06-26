@echo off
echo ========================================
echo   Starting Web Application
echo ========================================
echo.
echo Web app will open at: http://127.0.0.1:5000
echo Press Ctrl+C to stop the server
echo.

cd "C:\Users\AT\OneDrive - University of Engineering and Technology Taxila\Desktop\Voice detector"
C:\VoiceDetector\venv\Scripts\python.exe app.py

pause
