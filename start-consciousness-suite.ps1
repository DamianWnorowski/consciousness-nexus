# 🚀 CONSCIOUSNESS SUITE AUTO-START SCRIPT (PowerShell)
# ===================================================
#
# Automatically starts the complete Consciousness Suite deployment
# including web dashboard, API server, monitoring stack, and all services.
#
# Usage: .\start-consciousness-suite.ps1
#

param(
    [switch]$Logs,
    [switch]$Status,
    [switch]$Help
)

# Colors for output
$Red = "Red"
$Green = "Green"
$Yellow = "Yellow"
$Blue = "Blue"
$Magenta = "Magenta"
$Cyan = "Cyan"
$White = "White"

function Write-Info {
    param([string]$Message)
    Write-Host "ℹ️  $Message" -ForegroundColor $Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor $Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠️  $Message" -ForegroundColor $Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "❌ $Message" -ForegroundColor $Red
}

function Write-Header {
    param([string]$Message)
    Write-Host "🚀 $Message" -ForegroundColor $Magenta
    Write-Host ("=" * 50) -ForegroundColor $Magenta
}

function Write-Service {
    param([string]$Message)
    Write-Host "🔧 $Message" -ForegroundColor $Cyan
}

# Help function
if ($Help) {
    Write-Host "Consciousness Suite Auto-Start Script"
    Write-Host ""
    Write-Host "Usage: .\start-consciousness-suite.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Logs     Show service logs after startup"
    Write-Host "  -Status   Show current service status"
    Write-Host "  -Help     Show this help message"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\start-consciousness-suite.ps1        # Start all services"
    Write-Host "  .\start-consciousness-suite.ps1 -Logs  # Start and show logs"
    exit
}

# Main execution
Write-Host ""
Write-Header "CONSCIOUSNESS SUITE AUTO-DEPLOYMENT"
Write-Host "Starting complete AI safety platform with web dashboard..." -ForegroundColor $White
Write-Host ""

# Check prerequisites
Write-Header "CHECKING PREREQUISITES"

# Check Docker
try {
    $dockerVersion = docker --version 2>$null
    Write-Success "Docker is installed"
} catch {
    Write-Error "Docker is not installed. Please install Docker Desktop first."
    Write-Host "Download: https://www.docker.com/products/docker-desktop" -ForegroundColor $White
    Read-Host "Press Enter to exit"
    exit 1
}

# Check Docker Compose
try {
    $composeVersion = docker-compose --version 2>$null
    Write-Success "Docker Compose is available"
} catch {
    try {
        $composeVersion = docker compose version 2>$null
        Write-Success "Docker Compose V2 is available"
    } catch {
        Write-Error "Docker Compose is not available. Please install Docker Compose."
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Check docker-compose.yml
if (!(Test-Path "docker-compose.yml")) {
    Write-Error "docker-compose.yml not found in current directory"
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Success "Docker Compose configuration found"

Write-Host ""

# Show status if requested
if ($Status) {
    Write-Header "SERVICE STATUS"
    docker-compose ps

    Write-Host ""
    Write-Info "Service health checks:"

    Write-Host "API Server: " -NoNewline
    try {
        Invoke-WebRequest -Uri "http://localhost:18473/health" -TimeoutSec 5 -UseBasicParsing | Out-Null
        Write-Success "OK"
    } catch {
        Write-Error "DOWN"
    }

    Write-Host "Web Dashboard: " -NoNewline
    try {
        Invoke-WebRequest -Uri "http://localhost:31573" -TimeoutSec 5 -UseBasicParsing | Out-Null
        Write-Success "OK"
    } catch {
        Write-Error "DOWN"
    }

    exit
}

# Pre-deployment checks
Write-Header "PRE-DEPLOYMENT CHECKS"
Write-Info "Checking system resources..."

# Get disk space (rough estimate)
$diskSpace = Get-WmiObject -Class Win32_LogicalDisk -Filter "DeviceID='C:'" | Select-Object -ExpandProperty FreeSpace
$diskSpaceGB = [math]::Round($diskSpace / 1GB, 1)

if ($diskSpaceGB -lt 5) {
    Write-Warning "Low disk space: ${diskSpaceGB}GB available"
    Write-Info "Docker containers may fail to start"
} else {
    Write-Success "Sufficient disk space available: ${diskSpaceGB}GB"
}

Write-Success "System checks completed"
Write-Host ""

# Start deployment
Write-Header "STARTING CONSCIOUSNESS SUITE DEPLOYMENT"

Write-Info "Pulling latest Docker images..."
docker-compose pull 2>$null

Write-Info "Starting all services..."
Write-Info "This may take several minutes on first run..."

# Start services
docker-compose up -d

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Error "FAILED TO START SERVICES"
    Write-Info "Check Docker logs with: docker-compose logs"
    Write-Info "Make sure no other services are using the required ports"
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Success "ALL SERVICES STARTED SUCCESSFULLY!"

# Wait for services
Write-Header "WAITING FOR SERVICES TO BE READY"
Write-Info "Waiting 30 seconds for initial startup..."
Start-Sleep -Seconds 30

# Test services
Write-Header "TESTING SERVICE AVAILABILITY"

Write-Service "Testing API Server (port 18473)..."
try {
    Invoke-WebRequest -Uri "http://localhost:18473/health" -TimeoutSec 10 -UseBasicParsing | Out-Null
    Write-Success "API Server is responding"
} catch {
    Write-Warning "API Server not responding yet (this is normal on first startup)"
}

Write-Service "Testing Web Dashboard (port 31573)..."
try {
    Invoke-WebRequest -Uri "http://localhost:31573" -TimeoutSec 10 -UseBasicParsing | Out-Null
    Write-Success "Web Dashboard is responding"
} catch {
    Write-Warning "Web Dashboard not responding yet (may take 2-3 minutes to build)"
}

# Display service information
Write-Host ""
Write-Header "CONSCIOUSNESS SUITE IS NOW RUNNING!"

Write-Host ""
Write-Host "🌟 ACCESS YOUR SERVICES:" -ForegroundColor $White
Write-Host ""

Write-Host "🖥️  PRIMARY WEB DASHBOARD:" -ForegroundColor $Cyan
Write-Host "   🌐 http://localhost:31573 ← MAIN INTERFACE (Terminal Bypassing!)" -ForegroundColor $Green
Write-Host ""

Write-Host "🔗 API & DOCUMENTATION:" -ForegroundColor $Cyan
Write-Host "   🌐 http://localhost:18473     ← REST API" -ForegroundColor $Green
Write-Host "   🌐 http://localhost:18473/docs ← Interactive API Docs" -ForegroundColor $Green
Write-Host ""

Write-Host "📊 MONITORING & METRICS:" -ForegroundColor $Cyan
Write-Host "   🌐 http://localhost:31572     ← Grafana Dashboards (admin/admin)" -ForegroundColor $Green
Write-Host "   🌐 http://localhost:24789     ← Prometheus Metrics" -ForegroundColor $Green
Write-Host "   🌐 http://localhost:42851     ← Loki Log Aggregation" -ForegroundColor $Green
Write-Host ""

Write-Host "🎯 QUICK START:" -ForegroundColor $Magenta
Write-Host "   1. Open http://localhost:31573 in your browser" -ForegroundColor $White
Write-Host "   2. Explore the dashboard - no terminal commands needed!" -ForegroundColor $White
Write-Host "   3. Try running an evolution or validation" -ForegroundColor $White
Write-Host ""

Write-Host "📋 USEFUL COMMANDS:" -ForegroundColor $Blue
Write-Host "   docker-compose logs     ← View all service logs" -ForegroundColor $White
Write-Host "   docker-compose ps       ← Check service status" -ForegroundColor $White
Write-Host "   docker-compose down     ← Stop all services" -ForegroundColor $White
Write-Host "   docker-compose restart  ← Restart all services" -ForegroundColor $White

Write-Host ""
Write-Success "DEPLOYMENT COMPLETE! Your Consciousness Suite is now running."
Write-Host ""
Write-Host "🚀 Welcome to the future of AI safety management!" -ForegroundColor $Magenta

# Show logs if requested
if ($Logs) {
    Write-Host ""
    Write-Info "Showing service logs (Ctrl+C to exit)..."
    docker-compose logs -f
} else {
    Write-Host ""
    Read-Host "Press Enter to close this window"
}
