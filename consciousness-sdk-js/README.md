# Consciousness Computing Suite - JavaScript/TypeScript SDK

[![npm version](https://img.shields.io/npm/v/consciousness-suite-sdk.svg)](https://www.npmjs.com/package/consciousness-suite-sdk)
[![License](https://img.shields.io/npm/l/consciousness-suite-sdk.svg)](https://github.com/DAMIANWNOROWSKI/consciousness-suite/blob/main/LICENSE)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue.svg)](https://www.typescriptlang.org/)
[![Node.js](https://img.shields.io/badge/Node.js-14+-green.svg)](https://nodejs.org/)

JavaScript/TypeScript SDK for accessing Consciousness Computing Suite from JavaScript applications. Makes enterprise-grade AI safety and evolution tools available in Node.js, React, Vue, Angular, Electron, Deno, and browser environments.

## Installation

```bash
# npm
npm install consciousness-suite-sdk

# yarn
yarn add consciousness-suite-sdk

# pnpm
pnpm add consciousness-suite-sdk
```

## Quick Start

```typescript
import { ConsciousnessSDK, ConsciousnessClient } from 'consciousness-suite-sdk';

// Create a client for local development
const client = ConsciousnessSDK.createLocalClient(8000, 'your-api-key');

// Or create a production client
const prodClient = ConsciousnessSDK.createProductionClient(
  'your-api-key',
  'https://api.consciousness-suite.com'
);
```

## Usage Examples

### Health Check

```typescript
const health = await client.getHealth();
console.log('System status:', health.data);
```

### Running Evolution Operations

```typescript
// Run verified evolution
const result = await client.runEvolution({
  operationType: 'verified',
  targetSystem: 'my_application',
  safetyLevel: 'strict',
  userId: 'developer_1'
});

if (result.success) {
  console.log('Evolution ID:', result.data.evolutionId);
  console.log('Fitness Score:', result.data.metrics.fitnessScore);
}
```

### Streaming Evolution (Long-running operations)

```typescript
const stream = await client.runEvolutionStream({
  operationType: 'recursive',
  targetSystem: 'my_application',
  parameters: { maxIterations: 50 }
});

for await (const update of stream) {
  console.log(`Progress: ${update.data.progress * 100}%`);
  console.log(`Stage: ${update.data.stage}`);
}
```

### Running Validation

```typescript
const validation = await client.runValidation({
  files: ['src/main.ts', 'src/utils.ts'],
  validationScope: 'comprehensive'
});

console.log('Valid:', validation.data.isValid);
console.log('Issues:', validation.data.issues);
```

### Running Analysis

```typescript
const analysis = await client.runAnalysis({
  data: { systemMetrics: {...} },
  analysisType: 'fitness'
});

console.log('Analysis results:', analysis.data);
```

### Authentication

```typescript
// Login
const loginResult = await client.login('username', 'password');
if (loginResult.success) {
  console.log('Session ID:', loginResult.sessionId);
}

// Logout
await client.logout();
```

### WebSocket Events

```typescript
// Listen for real-time updates
client.on('connected', () => console.log('WebSocket connected'));
client.on('message', (msg) => console.log('Received:', msg));
client.on('error', (err) => console.error('Error:', err));
client.on('disconnected', () => console.log('WebSocket disconnected'));

// Cleanup
client.close();
```

## API Reference

### ConsciousnessClient

Main client class for interacting with the Consciousness API.

#### Constructor Options

```typescript
interface ConsciousnessConfig {
  baseURL: string;           // API server URL
  apiKey?: string;           // Optional API key
  timeout?: number;          // Request timeout (default: 30000ms)
  enableWebSocket?: boolean; // Enable WebSocket (default: true)
  autoReconnect?: boolean;   // Auto-reconnect WebSocket (default: true)
}
```

#### Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `login(username, password)` | Authenticate user | `APIResponse<LoginData>` |
| `logout()` | End session | `APIResponse` |
| `runEvolution(request)` | Run evolution operation | `APIResponse<EvolutionResult>` |
| `runEvolutionStream(request)` | Stream evolution progress | `AsyncIterable<APIResponse>` |
| `runValidation(request)` | Validate files | `APIResponse<ValidationResult>` |
| `runAnalysis(request)` | Run analysis | `APIResponse<AnalysisData>` |
| `getHealth()` | Get health status | `APIResponse<HealthData>` |
| `getSystemStatus()` | Get system status | `APIResponse<SystemStatus>` |
| `customRequest(method, endpoint, data)` | Custom API request | `APIResponse` |
| `close()` | Close connections | `void` |

### Types

```typescript
// Evolution Request
interface EvolutionRequest {
  operationType: 'verified' | 'recursive';
  targetSystem: string;
  parameters?: Record<string, any>;
  safetyLevel?: 'minimal' | 'standard' | 'strict' | 'paranoid';
  userId?: string;
  sessionId?: string;
}

// Evolution Result
interface EvolutionResult {
  evolutionId: string;
  status: 'completed' | 'failed' | 'running';
  results: Record<string, any>;
  metrics: {
    fitnessScore: number;
    executionTime: number;
    safetyChecks: number;
    warnings: string[];
  };
}

// Validation Request
interface ValidationRequest {
  files: string[];
  validationScope?: 'basic' | 'full' | 'comprehensive';
  userId?: string;
}

// Validation Result
interface ValidationResult {
  isValid: boolean;
  totalChecks: number;
  passedChecks: number;
  issues: ValidationIssue[];
  fitnessScore: number;
  warnings: string[];
}
```

## Environment Support

- **Node.js**: 14.0.0+
- **React/Vue/Angular**: With bundler (Vite, webpack, etc.)
- **Electron**: Main and renderer processes
- **Deno**: With npm compatibility
- **Browser**: With CORS proxy for cross-origin requests

## Configuration

### Environment Variables

```bash
# API Configuration
CONSCIOUSNESS_API_URL=http://localhost:8000
CONSCIOUSNESS_API_KEY=your-api-key

# WebSocket Configuration
CONSCIOUSNESS_WS_ENABLED=true
CONSCIOUSNESS_WS_AUTO_RECONNECT=true
```

## Error Handling

```typescript
try {
  const result = await client.runEvolution({
    operationType: 'verified',
    targetSystem: 'my_app'
  });

  if (!result.success) {
    console.error('Evolution failed:', result.error);
  }
} catch (error) {
  console.error('Request failed:', error.message);
}
```

## Testing

```bash
# Run tests
npm test

# Run tests with coverage
npm run test:coverage
```

## Building

```bash
# Build the library
npm run build

# Watch mode for development
npm run dev

# Generate documentation
npm run docs
```

## License

MIT License - see [LICENSE](../LICENSE) for details.

## Related

- [Main Consciousness Suite](https://github.com/DAMIANWNOROWSKI/consciousness-suite)
- [Go SDK](../consciousness-sdk-go/README.md)
- [Rust SDK](../consciousness-sdk-rust/README.md)
- [API Documentation](../docs/api/README.md)
