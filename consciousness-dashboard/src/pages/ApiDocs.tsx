import { ExternalLink, Code, Zap, Shield, Brain, Activity } from 'lucide-react'

export function ApiDocs() {
  const apiEndpoints = [
    {
      method: 'GET',
      path: '/health',
      description: 'Check system health status',
      icon: Activity,
      color: 'text-green-600',
    },
    {
      method: 'GET',
      path: '/status',
      description: 'Get comprehensive system status',
      icon: Activity,
      color: 'text-blue-600',
    },
    {
      method: 'POST',
      path: '/evolution/run',
      description: 'Run evolution operations',
      icon: Brain,
      color: 'text-purple-600',
    },
    {
      method: 'POST',
      path: '/validation/run',
      description: 'Run code validation',
      icon: Shield,
      color: 'text-emerald-600',
    },
    {
      method: 'POST',
      path: '/analysis/run',
      description: 'Run system analysis',
      icon: Zap,
      color: 'text-orange-600',
    },
    {
      method: 'POST',
      path: '/auth/login',
      description: 'Authenticate user',
      icon: Shield,
      color: 'text-red-600',
    },
  ]

  const exampleRequests = [
    {
      title: 'Health Check',
      method: 'GET',
      endpoint: '/health',
      code: `curl -H "X-API-Key: your-key" \\
  http://localhost:18473/health`,
    },
    {
      title: 'Run Evolution',
      method: 'POST',
      endpoint: '/evolution/run',
      code: `curl -X POST http://localhost:18473/evolution/run \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: your-key" \\
  -d '{
    "operation_type": "recursive",
    "target_system": "my_app",
    "safety_level": "strict",
    "parameters": {"max_iterations": 100}
  }'`,
    },
    {
      title: 'Code Validation',
      method: 'POST',
      endpoint: '/validation/run',
      code: `curl -X POST http://localhost:18473/validation/run \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: your-key" \\
  -d '{
    "files": ["src/main.py", "src/api.py"],
    "validation_scope": "full"
  }'`,
    },
  ]

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">API Documentation</h1>
          <p className="mt-2 text-gray-600 dark:text-gray-300">
            Complete REST API reference for the Consciousness Computing Suite
          </p>
        </div>
        <a
          href="/api/docs"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
        >
          <ExternalLink className="h-5 w-5 mr-2" />
          Interactive API Docs
        </a>
      </div>

      {/* Authentication */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
        <div className="p-6 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-lg font-medium text-gray-900 dark:text-white">
            Authentication
          </h2>
        </div>
        <div className="p-6">
          <div className="bg-yellow-50 dark:bg-yellow-900 border border-yellow-200 dark:border-yellow-700 rounded-md p-4">
            <div className="flex">
              <Shield className="h-5 w-5 text-yellow-400" />
              <div className="ml-3">
                <h3 className="text-sm font-medium text-yellow-800 dark:text-yellow-200">
                  API Key Required
                </h3>
                <p className="mt-2 text-sm text-yellow-700 dark:text-yellow-300">
                  All API requests require an <code className="font-mono">X-API-Key</code> header.
                  Get your API key from the Consciousness Suite configuration.
                </p>
              </div>
            </div>
          </div>

          <div className="mt-4">
            <h3 className="text-sm font-medium text-gray-900 dark:text-white mb-2">
              Example Header
            </h3>
            <code className="block bg-gray-100 dark:bg-gray-700 p-2 rounded text-sm">
              X-API-Key: consciousness-api-key-2024
            </code>
          </div>
        </div>
      </div>

      {/* API Endpoints */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
        <div className="p-6 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-lg font-medium text-gray-900 dark:text-white">
            API Endpoints
          </h2>
          <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
            All available REST API endpoints
          </p>
        </div>

        <div className="divide-y divide-gray-200 dark:divide-gray-700">
          {apiEndpoints.map((endpoint, index) => (
            <div key={index} className="p-6">
              <div className="flex items-center space-x-3">
                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                  endpoint.method === 'GET' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' :
                  endpoint.method === 'POST' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200' :
                  'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
                }`}>
                  {endpoint.method}
                </span>
                <code className="text-sm font-mono text-gray-900 dark:text-white">
                  {endpoint.path}
                </code>
                <endpoint.icon className={`h-5 w-5 ${endpoint.color}`} />
              </div>
              <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
                {endpoint.description}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Example Requests */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
        <div className="p-6 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-lg font-medium text-gray-900 dark:text-white">
            Example Requests
          </h2>
          <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
            Common API usage examples with curl
          </p>
        </div>

        <div className="p-6 space-y-6">
          {exampleRequests.map((example, index) => (
            <div key={index} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
              <div className="flex items-center space-x-3 mb-3">
                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                  example.method === 'GET' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' :
                  'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
                }`}>
                  {example.method}
                </span>
                <h3 className="text-sm font-medium text-gray-900 dark:text-white">
                  {example.title}
                </h3>
                <code className="text-xs font-mono text-gray-500 dark:text-gray-400">
                  {example.endpoint}
                </code>
              </div>
              <pre className="bg-gray-100 dark:bg-gray-700 p-3 rounded text-xs overflow-x-auto">
                <code>{example.code}</code>
              </pre>
            </div>
          ))}
        </div>
      </div>

      {/* Response Formats */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
        <div className="p-6 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-lg font-medium text-gray-900 dark:text-white">
            Response Format
          </h2>
          <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
            Standard JSON response structure
          </p>
        </div>

        <div className="p-6">
          <div className="space-y-4">
            <div>
              <h3 className="text-sm font-medium text-gray-900 dark:text-white mb-2">
                Success Response
              </h3>
              <pre className="bg-green-50 dark:bg-green-900 p-3 rounded text-xs overflow-x-auto">
                <code>{`{
  "success": true,
  "data": { ... },
  "execution_time": 1.23
}`}</code>
              </pre>
            </div>

            <div>
              <h3 className="text-sm font-medium text-gray-900 dark:text-white mb-2">
                Error Response
              </h3>
              <pre className="bg-red-50 dark:bg-red-900 p-3 rounded text-xs overflow-x-auto">
                <code>{`{
  "success": false,
  "error": "Error message",
  "execution_time": 0.01
}`}</code>
              </pre>
            </div>
          </div>
        </div>
      </div>

      {/* SDK Information */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
        <div className="p-6 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-lg font-medium text-gray-900 dark:text-white">
            SDK Libraries
          </h2>
          <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
            Official client libraries for different languages
          </p>
        </div>

        <div className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
              <Code className="h-8 w-8 text-blue-600 mx-auto mb-2" />
              <h3 className="text-sm font-medium text-gray-900 dark:text-white">JavaScript</h3>
              <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                npm install consciousness-suite-sdk
              </p>
            </div>

            <div className="text-center p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
              <Code className="h-8 w-8 text-orange-600 mx-auto mb-2" />
              <h3 className="text-sm font-medium text-gray-900 dark:text-white">Rust</h3>
              <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                cargo add consciousness-suite-sdk
              </p>
            </div>

            <div className="text-center p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
              <Code className="h-8 w-8 text-blue-400 mx-auto mb-2" />
              <h3 className="text-sm font-medium text-gray-900 dark:text-white">Go</h3>
              <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                go get github.com/DAMIANWNOROWSKI/consciousness-suite/go-sdk
              </p>
            </div>
          </div>

          <div className="mt-4 text-center">
            <a
              href="https://github.com/DAMIANWNOROWSKI/consciousness-suite"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-blue-600 hover:text-blue-500 dark:text-blue-400"
            >
              View all SDK documentation →
            </a>
          </div>
        </div>
      </div>
    </div>
  )
}
