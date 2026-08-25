@echo off
echo ========================================================
echo        Matera Vision - OMR Studio (Dark Mode)
echo ========================================================
echo Starting server. Please wait...

:: Start the FastAPI server
start /B python -m uvicorn src.matera.api.server:app --host 0.0.0.0 --port 8000 > nul 2>&1

:: Wait 2 seconds
timeout /t 2 /nobreak > nul

:: Open browser
echo Opening workspace in your browser...
start http://localhost:8000

echo.
echo System is running at http://localhost:8000
echo To shut down, close this window.
pause
