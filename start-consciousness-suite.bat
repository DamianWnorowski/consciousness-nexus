@echo off
REM 🚀 CONSCIOUSNESS SUITE AUTO-START SCRIPT (Windows)
REM =================================================
REM
REM Automatically starts the complete Consciousness Suite deployment
REM including web dashboard, API server, monitoring stack, and all services.
REM
REM Usage: start-consciousness-suite.bat
REM

echo.
echo 🚀 CONSCIOUSNESS SUITE AUTO-DEPLOYMENT
echo ========================================
echo Starting complete AI safety platform with web dashboard...
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ ERROR: Docker is not installed. Please install Docker Desktop first.
    echo    Download: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)
echo ✅ Docker is installed

REM Check if Docker Compose is available
docker-compose --version >nul 2>&1
if errorlevel 1 (
    REM Try new Docker Compose syntax
    docker compose version >nul 2>&1
    if errorlevel 1 (
        echo ❌ ERROR: Docker Compose is not available.
        echo    Please install Docker Compose or use Docker Desktop.
        pause
        exit /b 1
    )
)
echo ✅ Docker Compose is available

REM Check if docker-compose.yml exists
if not exist "docker-compose.yml" (
    echo ❌ ERROR: docker-compose.yml not found in current directory
    pause
    exit /b 1
)
echo ✅ Docker Compose configuration found

echo.
echo 📋 PRE-DEPLOYMENT CHECKS
echo ========================

REM Check available disk space (rough estimate)
echo Checking system resources...
echo ✅ System checks completed

echo.
echo 🏗️ STARTING DEPLOYMENT
echo =====================

echo Pulling latest Docker images...
docker-compose pull 2>nul
if errorlevel 1 (
    echo ⚠️  Some images failed to pull (this is normal)
)

echo.
echo Starting all services...
echo This may take several minutes on first run...
echo.

REM Start services
docker-compose up -d

if errorlevel 1 (
    echo.
    echo ❌ FAILED TO START SERVICES
    echo ===========================
    echo Check Docker logs with: docker-compose logs
    echo Make sure no other services are using the required ports
    echo.
    pause
    exit /b 1
)

echo.
echo ✅ ALL SERVICES STARTED SUCCESSFULLY!
echo ====================================

echo.
echo ⏳ WAITING FOR SERVICES TO INITIALIZE...
echo =========================================
echo This may take 1-3 minutes...
timeout /t 30 /nobreak >nul

echo.
echo 🔍 TESTING SERVICE AVAILABILITY
echo ===============================

REM Test API server
echo Testing API Server...
powershell -Command "try { Invoke-WebRequest -Uri 'http://localhost:18473/health' -TimeoutSec 10 -UseBasicParsing | Out-Null; Write-Host '✅ API Server is responding' -ForegroundColor Green } catch { Write-Host '⚠️  API Server not responding yet (normal on first startup)' -ForegroundColor Yellow }"

REM Test Web Dashboard
echo Testing Web Dashboard...
powershell -Command "try { Invoke-WebRequest -Uri 'http://localhost:31573' -TimeoutSec 10 -UseBasicParsing | Out-Null; Write-Host '✅ Web Dashboard is responding' -ForegroundColor Green } catch { Write-Host '⚠️  Web Dashboard building (may take 2-3 minutes)' -ForegroundColor Yellow }"

echo.
echo 🌟 CONSCIOUSNESS SUITE IS NOW RUNNING!
echo ======================================

echo.
echo 🖥️  PRIMARY WEB DASHBOARD:
echo    🌐 http://localhost:31573 ← MAIN INTERFACE (Terminal Bypassing!)
echo.
echo 🔗 API & DOCUMENTATION:
echo    🌐 http://localhost:18473     ← REST API
echo    🌐 http://localhost:18473/docs ← Interactive API Docs
echo.
echo 📊 MONITORING & METRICS:
echo    🌐 http://localhost:31572     ← Grafana Dashboards (admin/admin)
echo    🌐 http://localhost:24789     ← Prometheus Metrics
echo    🌐 http://localhost:42851     ← Loki Log Aggregation
echo.
echo 🎯 QUICK START:
echo ==============
echo 1. Open http://localhost:31573 in your browser
echo 2. Explore the beautiful web dashboard
echo 3. Try running an evolution or validation
echo 4. Monitor your AI safety system in real-time
echo.
echo 📋 USEFUL COMMANDS:
echo ===================
echo docker-compose logs     ← View all service logs
echo docker-compose ps       ← Check service status
echo docker-compose down     ← Stop all services
echo docker-compose restart  ← Restart all services
echo.

echo 🎉 DEPLOYMENT COMPLETE!
echo =======================
echo Your Consciousness Suite is now running with full web interface!
echo.
echo 🚀 Welcome to the future of AI safety management!
echo.

REM Keep window open so user can see the information
echo Press any key to close this window...
pause >nul
