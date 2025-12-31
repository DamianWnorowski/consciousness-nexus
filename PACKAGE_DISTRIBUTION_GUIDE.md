# 📦 CONSCIOUSNESS SUITE - PACKAGE DISTRIBUTION GUIDE

## 🚀 Publishing to Package Registries

Your Consciousness Suite is now ready for universal distribution across all major package ecosystems.

---

## 📦 1. PYTHON PACKAGE (PYPI)

### Prerequisites
```bash
pip install build twine
```

### Build Package
```bash
# Build source and wheel distributions
python -m build

# Check contents
tar -tf dist/*.tar.gz
```

### Test Locally
```bash
# Install in development mode
pip install -e .

# Test import
python -c "from consciousness_suite import AutoRecursiveChainAI; print('✅ Import successful')"
```

### Publish to PyPI
```bash
# Upload to PyPI (requires API token)
twine upload dist/*

# Or upload to TestPyPI first
twine upload --repository testpypi dist/*
```

### PyPI Configuration
Create `~/.pypirc`:
```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-your-api-token-here

[testpypi]
username = __token__
password = testpypi-your-api-token-here
```

---

## 📦 2. JAVASCRIPT SDK (NPM)

### Prerequisites
```bash
cd consciousness-sdk-js
npm install
```

### Build SDK
```bash
# Install dependencies
npm ci

# Build for production
npm run build

# Verify build output
ls -la dist/
```

### Test Locally
```bash
# Test in Node.js
node -e "const sdk = require('./dist/index.js'); console.log('✅ SDK loaded');"

# Test in browser (optional)
npm install -g http-server
http-server . -p 3000
```

### Publish to NPM
```bash
# Login to npm
npm login

# Or use token
npm config set //registry.npmjs.org/:_authToken your-npm-token

# Publish
npm publish

# Check publication
npm view consciousness-suite-sdk
```

### NPM Configuration
Update `package.json` if needed:
```json
{
  "name": "consciousness-suite-sdk",
  "version": "2.0.0",
  "description": "JavaScript SDK for Consciousness Computing Suite",
  "main": "dist/index.js",
  "module": "dist/index.mjs",
  "types": "dist/index.d.ts",
  "files": ["dist", "README.md", "LICENSE"],
  "scripts": {
    "build": "rollup -c",
    "prepublishOnly": "npm run build && npm run test"
  }
}
```

---

## 📦 3. RUST SDK (CRATES.IO)

### Prerequisites
```bash
cd consciousness-sdk-rust
cargo --version  # Should be 1.70+
```

### Build SDK
```bash
# Check compilation
cargo check

# Build release version
cargo build --release

# Run tests
cargo test

# Generate documentation
cargo doc --open
```

### Test Locally
```bash
# Create test project
cd ..
cargo new test_project
cd test_project

# Add dependency
echo 'consciousness-suite-sdk = { path = "../consciousness-sdk-rust" }' >> Cargo.toml

# Test usage
cargo run
```

### Publish to Crates.io
```bash
# Login (one-time)
cargo login your-api-token

# Dry run
cargo publish --dry-run

# Publish
cargo publish

# Check publication
cargo search consciousness-suite-sdk
```

### Crates.io Configuration
Update `Cargo.toml`:
```toml
[package]
name = "consciousness-suite-sdk"
version = "2.0.0"
edition = "2021"
description = "Rust SDK for Consciousness Computing Suite"
license = "MIT"
repository = "https://github.com/DAMIANWNOROWSKI/consciousness-suite"
keywords = ["ai", "consciousness", "safety", "evolution"]
categories = ["api-bindings"]

[dependencies]
reqwest = { version = "0.11", features = ["json", "stream"] }
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
thiserror = "1.0"
uuid = { version = "1.0", features = ["v4"] }
```

---

## 📦 4. GO SDK (GO MODULES)

### Prerequisites
```bash
cd consciousness-sdk-go
go version  # Should be 1.21+
```

### Setup Module
```bash
# Initialize module (if not done)
go mod init github.com/DAMIANWNOROWSKI/consciousness-suite/go-sdk

# Download dependencies
go mod tidy

# Test compilation
go build .
```

### Test Locally
```bash
# Create test program
cat > test.go << 'EOF'
package main

import (
    "fmt"
    "log"

    consciousness "github.com/DAMIANWNOROWSKI/consciousness-suite/go-sdk"
)

func main() {
    client := consciousness.NewClient("http://localhost:18473", "test-key")

    health, err := client.Health()
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Health: %+v\n", health)
}
EOF

# Run test
go run test.go
```

### Publish Go Module
```bash
# Go modules are published by:
# 1. Pushing code to GitHub with proper tags
# 2. Ensuring go.mod has correct module path
# 3. Users can import directly from GitHub

# Tag the release
git tag v2.0.0
git push origin v2.0.0

# Users can now import:
# go get github.com/DAMIANWNOROWSKI/consciousness-suite/go-sdk@v2.0.0
```

---

## 🔄 CONTINUOUS INTEGRATION (GITHUB ACTIONS)

### Automatic Publishing Setup

The CI/CD workflows will automatically publish when you create GitHub releases:

```yaml
# .github/workflows/release.yml
name: Release
on:
  release:
    types: [published]

jobs:
  pypi:
    # Publishes Python package to PyPI

  npm:
    # Publishes JavaScript SDK to NPM

  crates:
    # Publishes Rust SDK to Crates.io

  go:
    # Tags Go module release
```

### Required Secrets

Add these to your GitHub repository secrets:

```
PYPI_API_TOKEN      # Python Package Index API token
NPM_TOKEN          # NPM registry authentication token
CRATES_IO_TOKEN    # Rust Crates.io API token
```

### Manual Publishing

If you prefer manual control:

```bash
# Python
python -m build
twine upload dist/*

# JavaScript
cd consciousness-sdk-js && npm publish

# Rust
cd consciousness-sdk-rust && cargo publish

# Go (just tag and push)
git tag v2.0.0 && git push origin v2.0.0
```

---

## 📊 VERIFICATION

### Check All Publications

```bash
# Python
pip search consciousness-suite
pip install consciousness-suite

# JavaScript
npm view consciousness-suite-sdk
npm install consciousness-suite-sdk

# Rust
cargo search consciousness-suite-sdk
cargo add consciousness-suite-sdk

# Go
go list -m github.com/DAMIANWNOROWSKI/consciousness-suite/go-sdk
go get github.com/DAMIANWNOROWSKI/consciousness-suite/go-sdk
```

---

## 🌍 USAGE EXAMPLES

### After Publication

```python
# Python (anywhere)
pip install consciousness-suite
from consciousness_suite import AutoRecursiveChainAI

orchestrator = AutoRecursiveChainAI()
await orchestrator.run_orchestration()
```

```javascript
// JavaScript (anywhere)
npm install consciousness-suite-sdk
import { ConsciousnessClient } from 'consciousness-suite-sdk';

const client = new ConsciousnessClient({
  baseURL: 'http://localhost:18473'
});
```

```rust
// Rust (anywhere)
// Cargo.toml: consciousness-suite-sdk = "2.0"
use consciousness_suite_sdk::prelude::*;

let client = create_local_client(18473)?;
let result = client.run_evolution(EvolutionOperation::Recursive, "app", None, None).await?;
```

```go
// Go (anywhere)
go get github.com/DAMIANWNOROWSKI/consciousness-suite/go-sdk

client := consciousness.NewClient("http://localhost:18473", "key")
result, err := client.RunEvolution(consciousness.EvolutionRequest{...})
```

---

## 🎯 NEXT STEPS

1. **Push to GitHub**: `git push origin main`
2. **Create Release**: Go to GitHub → Releases → Create v2.0.0
3. **Automatic Publishing**: CI/CD will publish to all registries
4. **Verify**: Check all registries show your packages
5. **Update Docs**: Point users to the published packages

**Your Consciousness Suite will then be available universally across all programming ecosystems!** 🚀

---

*See individual SDK READMEs for detailed usage instructions.*
