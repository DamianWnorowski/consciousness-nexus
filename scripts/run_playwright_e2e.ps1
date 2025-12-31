param(
    [Parameter(Mandatory = $false)]
    [ValidateSet("smoke", "full", "visual")]
    [string]$Profile = "full",

    [Parameter(Mandatory = $false)]
    [switch]$Headless = $true,

    [Parameter(Mandatory = $false)]
    [string]$Browser = "chromium",

    [Parameter(Mandatory = $false)]
    [switch]$Debug
)

# Consciousness Nexus E2E Testing Runner
# =====================================

Write-Host "🔬 Consciousness Nexus - E2E Testing Suite" -ForegroundColor Cyan
Write-Host "=" * 50 -ForegroundColor Cyan

# Set paths
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$PlaywrightDir = Join-Path $ProjectRoot "playwright-e2e-testing"
$TestResultsDir = Join-Path $ProjectRoot "test-results"

# Ensure we're in the right directory
Push-Location $PlaywrightDir

try {
    # Check if Playwright is installed
    if (!(Test-Path "node_modules/.bin/playwright")) {
        Write-Host "📦 Installing Playwright dependencies..." -ForegroundColor Yellow
        & npm install
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install Playwright dependencies"
        }
    }

    # Install browsers if needed
    Write-Host "🌐 Installing Playwright browsers..." -ForegroundColor Yellow
    & npx playwright install
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Browser installation failed, but continuing..."
    }

    # Create test results directory
    if (!(Test-Path $TestResultsDir)) {
        New-Item -ItemType Directory -Path $TestResultsDir -Force | Out-Null
    }

    # Determine test command based on profile
    $TestCommand = "npx playwright test"

    switch ($Profile) {
        "smoke" {
            Write-Host "🚀 Running SMOKE tests (@critical only, Chromium only)" -ForegroundColor Green
            $TestCommand += " --grep @critical --project=chromium"
        }
        "visual" {
            Write-Host "👁️ Running VISUAL tests only" -ForegroundColor Green
            $TestCommand += " tests/visual/"
        }
        "full" {
            Write-Host "🎭 Running FULL test suite (all browsers, all tests)" -ForegroundColor Green
            # Full test suite - no additional parameters needed
        }
    }

    # Add common parameters
    if ($Headless) {
        $TestCommand += " --headed=false"
    } else {
        $TestCommand += " --headed=true"
    }

    if ($Debug) {
        $TestCommand += " --debug"
        Write-Host "🐛 Debug mode enabled" -ForegroundColor Yellow
    }

    # Add output directory
    $TestCommand += " --output=$TestResultsDir"

    Write-Host "⚡ Executing: $TestCommand" -ForegroundColor Magenta
    Write-Host ""

    # Run the tests
    $StartTime = Get-Date
    Invoke-Expression $TestCommand
    $ExitCode = $LASTEXITCODE
    $EndTime = Get-Date
    $Duration = $EndTime - $StartTime

    Write-Host ""
    Write-Host "📊 Test Results Summary" -ForegroundColor Cyan
    Write-Host "=" * 30 -ForegroundColor Cyan

    if (Test-Path "test-results.json") {
        $TestResults = Get-Content "test-results.json" | ConvertFrom-Json

        $Passed = ($TestResults.suites | ForEach-Object { $_.specs } | Where-Object { $_.ok }).Count
        $Failed = ($TestResults.suites | ForEach-Object { $_.specs } | Where-Object { -not $_.ok }).Count
        $Total = $Passed + $Failed

        Write-Host "Total Tests: $Total" -ForegroundColor White
        Write-Host "Passed: $Passed" -ForegroundColor Green
        Write-Host "Failed: $Failed" -ForegroundColor $(if ($Failed -gt 0) { "Red" } else { "White" })
        Write-Host "Duration: $($Duration.TotalSeconds.ToString("F2"))s" -ForegroundColor White
        Write-Host "Success Rate: $((($Passed / $Total) * 100).ToString("F1"))%" -ForegroundColor $(if ($Passed -eq $Total) { "Green" } else { "Yellow" })
    }

    # Generate HTML report
    Write-Host ""
    Write-Host "📄 Generating HTML report..." -ForegroundColor Yellow
    & npx playwright show-report --save
    $ReportUrl = "file://$PlaywrightDir/playwright-report/index.html"
    Write-Host "Report available at: $ReportUrl" -ForegroundColor Cyan

    # Exit with test result code
    if ($ExitCode -eq 0) {
        Write-Host ""
        Write-Host "✅ ALL TESTS PASSED" -ForegroundColor Green
        Write-Host "🎉 Consciousness Nexus E2E validation successful!" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "❌ SOME TESTS FAILED" -ForegroundColor Red
        Write-Host "🔍 Check the HTML report for details" -ForegroundColor Yellow
    }

    exit $ExitCode

} catch {
    Write-Host ""
    Write-Host "💥 E2E Testing Failed" -ForegroundColor Red
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
} finally {
    Pop-Location
}
