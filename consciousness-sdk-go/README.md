# Consciousness Computing Suite - Go SDK

Go SDK for accessing Consciousness Computing Suite from Go applications.

## Installation

```bash
go get github.com/DAMIANWNOROWSKI/consciousness-suite/go-sdk
```

## Usage

```go
package main

import (
    "fmt"
    "log"

    consciousness "github.com/DAMIANWNOROWSKI/consciousness-suite/go-sdk"
)

func main() {
    client := consciousness.NewClient("http://localhost:18473", "your-api-key")

    // Check health
    health, err := client.Health()
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Health: %+v\n", health)

    // Run evolution
    result, err := client.RunEvolution(consciousness.EvolutionRequest{
        OperationType: "recursive",
        TargetSystem:  "my_app",
        SafetyLevel:   "strict",
        UserID:        "go_user",
    })
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Evolution Result: %+v\n", result)
}
```

## Documentation

See the main [Consciousness Suite documentation](https://github.com/DAMIANWNOROWSKI/consciousness-suite) for more information.
