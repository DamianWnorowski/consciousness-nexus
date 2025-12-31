param(
    [Parameter(Mandatory = $false)]
    [switch]$IncludeDev,

    [Parameter(Mandatory = $false)]
    [string]$OutputDir = "MASTER_INDEX/sbom",

    [Parameter(Mandatory = $false)]
    [switch]$Quiet
)

# Consciousness Nexus - Software Bill of Materials Generator
# ==========================================================

if (!$Quiet) {
    Write-Host "📦 Consciousness Nexus - SBOM Generator" -ForegroundColor Cyan
    Write-Host "=" * 45 -ForegroundColor Cyan
}

# Set paths
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$OutputPath = Join-Path $ProjectRoot $OutputDir

# Create output directory if it doesn't exist
if (!(Test-Path $OutputPath)) {
    New-Item -ItemType Directory -Path $OutputPath -Force | Out-Null
    if (!$Quiet) {
        Write-Host "📁 Created output directory: $OutputPath" -ForegroundColor Yellow
    }
}

$Timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ"
$PythonPackagesFile = Join-Path $OutputPath "python_packages.json"

# Function to get Python packages
function Get-PythonPackages {
    param(
        [bool]$IncludeDevDeps = $false
    )

    if (!$Quiet) {
        Write-Host "🐍 Analyzing Python dependencies..." -ForegroundColor Yellow
    }

    $Packages = @()

    # Try to get packages from pip
    try {
        $PipList = & python -m pip list --format json 2>$null | ConvertFrom-Json

        foreach ($Package in $PipList) {
            $PackageInfo = @{
                name = $Package.name
                version = $Package.version
                source = "pip"
                type = "runtime"
                license = "Unknown"  # Would need additional tooling for license info
                description = ""
                homepage = ""
                dependencies = @()  # Would need additional tooling for dep tree
                vulnerabilities = @()  # Would need security scanning
            }

            # Classify as dev or runtime (basic heuristic)
            if ($Package.name -match "(pytest|black|mypy|flake8|playwright|types-)") {
                $PackageInfo.type = "development"
            }

            # Only include dev deps if requested
            if ($IncludeDevDeps -or $PackageInfo.type -eq "runtime") {
                $Packages += $PackageInfo
            }
        }
    } catch {
        if (!$Quiet) {
            Write-Warning "Could not enumerate pip packages: $($_.Exception.Message)"
        }
    }

    # Add known project dependencies
    $KnownDeps = @(
        @{
            name = "consciousness_nexus"
            version = "1.0.0"
            source = "local"
            type = "application"
            license = "MIT"
            description = "Consciousness Computing Suite Core"
            homepage = "https://github.com/consciousness-nexus/core"
            dependencies = @("asyncio", "typing", "dataclasses", "json")
            vulnerabilities = @()
        }
    )

    $Packages += $KnownDeps

    return $Packages
}

# Function to analyze Rust dependencies (placeholder for future)
function Get-RustDependencies {
    if (!$Quiet) {
        Write-Host "🦀 Analyzing Rust dependencies..." -ForegroundColor Yellow
    }

    # Placeholder for Rust Cargo.toml analysis
    # Would use cargo metadata --format-version 1
    $RustDeps = @(
        @{
            name = "consciousness-nexus-rust"
            version = "0.1.0"
            source = "local"
            type = "application"
            license = "MIT"
            description = "Rust components for consciousness computing"
            homepage = "https://github.com/consciousness-nexus/rust"
            dependencies = @("tokio", "serde", "anyhow")
            vulnerabilities = @()
        }
    )

    return $RustDeps
}

# Function to generate SBOM document
function New-SBOMDocument {
    param(
        [array]$PythonPackages,
        [array]$RustPackages,
        [string]$Timestamp
    )

    $SBOM = @{
        sbom = @{
            version = "1.0"
            format = "Consciousness-Nexus-SBOM"
            specVersion = "1.4"
            creationInfo = @{
                created = $Timestamp
                creators = @("Tool: Consciousness Nexus SBOM Generator v1.0")
                tool = @{
                    name = "Consciousness Nexus SBOM Generator"
                    version = "1.0.0"
                    vendor = "Consciousness Nexus Team"
                }
            }
        }
        components = @()
    }

    # Add Python components
    foreach ($Package in $PythonPackages) {
        $Component = @{
            type = "library"
            name = $Package.name
            version = $Package.version
            purl = "pkg:pypi/$($Package.name)@$($Package.version)"
            properties = @(
                @{
                    name = "source"
                    value = $Package.source
                },
                @{
                    name = "type"
                    value = $Package.type
                }
            )
        }

        if ($Package.license -and $Package.license -ne "Unknown") {
            $Component.properties += @{
                name = "license"
                value = $Package.license
            }
        }

        if ($Package.description) {
            $Component.properties += @{
                name = "description"
                value = $Package.description
            }
        }

        if ($Package.homepage) {
            $Component.properties += @{
                name = "homepage"
                value = $Package.homepage
            }
        }

        $SBOM.components += $Component
    }

    # Add Rust components (when implemented)
    foreach ($Package in $RustPackages) {
        $Component = @{
            type = "library"
            name = $Package.name
            version = $Package.version
            purl = "pkg:cargo/$($Package.name)@$($Package.version)"
            properties = @(
                @{
                    name = "source"
                    value = $Package.source
                },
                @{
                    name = "type"
                    value = $Package.type
                },
                @{
                    name = "language"
                    value = "rust"
                }
            )
        }

        if ($Package.license) {
            $Component.properties += @{
                name = "license"
                value = $Package.license
            }
        }

        $SBOM.components += $Component
    }

    return $SBOM
}

# Main execution
try {
    # Get dependencies
    $PythonPackages = Get-PythonPackages -IncludeDevDeps $IncludeDev
    $RustPackages = Get-RustDependencies

    if (!$Quiet) {
        Write-Host "📊 Found $($PythonPackages.Count) Python packages" -ForegroundColor Green
        Write-Host "📊 Found $($RustPackages.Count) Rust packages" -ForegroundColor Green
    }

    # Generate SBOM
    $SBOM = New-SBOMDocument -PythonPackages $PythonPackages -RustPackages $RustPackages -Timestamp $Timestamp

    # Save Python packages specifically (as per original spec)
    $PythonSBOM = @{
        metadata = @{
            timestamp = $Timestamp
            generator = "Consciousness Nexus SBOM Generator"
            includeDev = $IncludeDev.ToString()
        }
        packages = $PythonPackages
    }

    # Save to files
    $SBOM | ConvertTo-Json -Depth 10 | Out-File -FilePath (Join-Path $OutputPath "sbom.json") -Encoding UTF8
    $PythonSBOM | ConvertTo-Json -Depth 10 | Out-File -FilePath $PythonPackagesFile -Encoding UTF8

    # Save Rust metadata (placeholder)
    $RustSBOM = @{
        metadata = @{
            timestamp = $Timestamp
            generator = "Consciousness Nexus SBOM Generator"
            note = "Rust SBOM generation not fully implemented - placeholder"
        }
        packages = $RustPackages
    }

    $RustSBOM | ConvertTo-Json -Depth 10 | Out-File -FilePath (Join-Path $OutputPath "rust_packages.json") -Encoding UTF8

    if (!$Quiet) {
        Write-Host ""
        Write-Host "✅ SBOM Generation Complete" -ForegroundColor Green
        Write-Host "📄 Files saved to: $OutputPath" -ForegroundColor Cyan
        Write-Host "   ├── sbom.json (Complete SBOM)" -ForegroundColor White
        Write-Host "   ├── python_packages.json (Python dependencies)" -ForegroundColor White
        Write-Host "   └── rust_packages.json (Rust dependencies)" -ForegroundColor White
        Write-Host ""
        Write-Host "📦 Total components: $($SBOM.components.Count)" -ForegroundColor Green
        Write-Host "🐍 Python packages: $($PythonPackages.Count)" -ForegroundColor Green
        Write-Host "🦀 Rust packages: $($RustPackages.Count)" -ForegroundColor Green
    }
}
catch {
    if (!$Quiet) {
        Write-Host ""
        Write-Host "❌ SBOM Generation Failed" -ForegroundColor Red
        Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    }
    exit 1
}
