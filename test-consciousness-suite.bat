@echo off
REM 🧪 CONSCIOUSNESS SUITE COMPREHENSIVE TEST (Windows)
REM ===================================================

echo 🧪 CONSCIOUSNESS SUITE COMPREHENSIVE TEST SUITE
echo ================================================
echo Testing all components of your Consciousness Suite deployment...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ ERROR: Python is not available. Please install Python 3.8+.
    pause
    exit /b 1
)
echo ✅ Python is available

REM Run the comprehensive test
python test-consciousness-suite.py

REM Keep window open to see results
echo.
echo Press any key to close this window...
pause >nul
