# 🧪 CONSCIOUSNESS SUITE COMPREHENSIVE TEST (PowerShell)
# =========================================================

Write-Host "🧪 CONSCIOUSNESS SUITE COMPREHENSIVE TEST SUITE" -ForegroundColor Cyan
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host "Testing all components of your Consciousness Suite deployment..." -ForegroundColor White
Write-Host ""

# Check Python availability
try {
    $pythonVersion = python --version 2>$null
    Write-Host "✅ Python is available" -ForegroundColor Green
} catch {
    Write-Host "❌ ERROR: Python is not available. Please install Python 3.8+." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Run the comprehensive test
try {
    python test-consciousness-suite.py
} catch {
    Write-Host "❌ Test execution failed: $($_.Exception.Message)" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Keep window open to see results
Write-Host ""
Read-Host "Press Enter to close this window"
