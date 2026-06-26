@echo off
setlocal

cd /d "%~dp0"

if exist "C:\fyp_venv311\Scripts\python.exe" (
    set "PYTHON_EXE=C:\fyp_venv311\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

"%PYTHON_EXE%" -m streamlit run Home.py

endlocal
