@echo off
title Smart Attendance AI Platform Launcher
echo ====================================================================
echo Starting Smart Attendance AI Platform (FastAPI + Streamlit)
echo ====================================================================
echo.

echo [1/2] Launching FastAPI Backend Server on http://127.0.0.1:8000...
start "FastAPI Backend" cmd /k ".venv\Scripts\uvicorn.exe backend.main:app --host 127.0.0.1 --port 8000 --reload"

echo [2/2] Waiting for backend to initialize...
timeout /t 3 /nobreak >nul

echo Launching Streamlit Dashboard on http://localhost:8501...
start "Streamlit Dashboard" cmd /k ".venv\Scripts\streamlit.exe run frontend/app.py"

echo.
echo ====================================================================
echo Both services are now running!
echo - API Docs:   http://127.0.0.1:8000/api/v1/docs
echo - Dashboard:  http://localhost:8501
echo ====================================================================
