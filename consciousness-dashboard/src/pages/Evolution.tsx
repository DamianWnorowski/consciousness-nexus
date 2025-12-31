import { useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Brain, Play, Loader2, CheckCircle, XCircle } from 'lucide-react'
import { apiClient } from '../lib/api'

export function Evolution() {
  const [formData, setFormData] = useState({
    operationType: 'recursive' as 'recursive' | 'verified',
    targetSystem: '',
    parameters: '{}',
    safetyLevel: 'standard' as 'minimal' | 'standard' | 'strict' | 'paranoid',
    userId: 'web_user',
  })

  const [results, setResults] = useState<any>(null)
  const [isRunning, setIsRunning] = useState(false)

  const queryClient = useQueryClient()

  const evolutionMutation = useMutation({
    mutationFn: (data: typeof formData) => {
      const params = data.parameters ? JSON.parse(data.parameters) : {}
      return apiClient.runEvolution({
        operation_type: data.operationType,
        target_system: data.targetSystem,
        parameters: params,
        safety_level: data.safetyLevel,
        user_id: data.userId,
      })
    },
    onSuccess: (data) => {
      setResults(data)
      queryClient.invalidateQueries({ queryKey: ['health'] })
      queryClient.invalidateQueries({ queryKey: ['system-status'] })
    },
    onError: (error: any) => {
      setResults({ success: false, error: error.message })
    },
  })

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsRunning(true)
    setResults(null)

    try {
      await evolutionMutation.mutateAsync(formData)
    } finally {
      setIsRunning(false)
    }
  }

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }))
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Evolution</h1>
        <p className="mt-2 text-gray-600 dark:text-gray-300">
          Run intelligent evolution operations with enterprise safety
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Evolution Form */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
          <div className="p-6 border-b border-gray-200 dark:border-gray-700">
            <h2 className="text-lg font-medium text-gray-900 dark:text-white">
              Start Evolution
            </h2>
            <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
              Configure and launch evolution operations
            </p>
          </div>

          <form onSubmit={handleSubmit} className="p-6 space-y-6">
            {/* Operation Type */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Operation Type
              </label>
              <select
                value={formData.operationType}
                onChange={(e) => handleInputChange('operationType', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
              >
                <option value="recursive">Recursive Evolution</option>
                <option value="verified">Verified Evolution</option>
              </select>
            </div>

            {/* Target System */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Target System
              </label>
              <input
                type="text"
                value={formData.targetSystem}
                onChange={(e) => handleInputChange('targetSystem', e.target.value)}
                placeholder="e.g., web_app, api_service, mobile_app"
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
                required
              />
            </div>

            {/* Safety Level */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Safety Level
              </label>
              <select
                value={formData.safetyLevel}
                onChange={(e) => handleInputChange('safetyLevel', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
              >
                <option value="minimal">Minimal</option>
                <option value="standard">Standard</option>
                <option value="strict">Strict</option>
                <option value="paranoid">Paranoid</option>
              </select>
            </div>

            {/* Parameters */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Parameters (JSON)
              </label>
              <textarea
                value={formData.parameters}
                onChange={(e) => handleInputChange('parameters', e.target.value)}
                placeholder='{"max_iterations": 100, "fitness_threshold": 0.95}'
                rows={4}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white font-mono text-sm"
              />
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={isRunning || evolutionMutation.isLoading}
              className="w-full flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isRunning || evolutionMutation.isLoading ? (
                <>
                  <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                  Running Evolution...
                </>
              ) : (
                <>
                  <Play className="h-5 w-5 mr-2" />
                  Start Evolution
                </>
              )}
            </button>
          </form>
        </div>

        {/* Results Panel */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
          <div className="p-6 border-b border-gray-200 dark:border-gray-700">
            <h2 className="text-lg font-medium text-gray-900 dark:text-white">
              Results
            </h2>
            <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
              Evolution operation results and metrics
            </p>
          </div>

          <div className="p-6">
            {results ? (
              <div className="space-y-4">
                {/* Status */}
                <div className="flex items-center space-x-2">
                  {results.success ? (
                    <CheckCircle className="h-5 w-5 text-green-500" />
                  ) : (
                    <XCircle className="h-5 w-5 text-red-500" />
                  )}
                  <span className={`text-sm font-medium ${
                    results.success ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {results.success ? 'Evolution Completed' : 'Evolution Failed'}
                  </span>
                </div>

                {/* Results Data */}
                {results.data && (
                  <div className="bg-gray-50 dark:bg-gray-700 rounded-md p-4">
                    <h3 className="text-sm font-medium text-gray-900 dark:text-white mb-2">
                      Evolution Results
                    </h3>
                    <pre className="text-xs text-gray-600 dark:text-gray-300 whitespace-pre-wrap">
                      {JSON.stringify(results.data, null, 2)}
                    </pre>
                  </div>
                )}

                {/* Error */}
                {results.error && (
                  <div className="bg-red-50 dark:bg-red-900 rounded-md p-4">
                    <h3 className="text-sm font-medium text-red-800 dark:text-red-200 mb-2">
                      Error
                    </h3>
                    <p className="text-sm text-red-700 dark:text-red-300">
                      {results.error}
                    </p>
                  </div>
                )}

                {/* Execution Time */}
                {results.execution_time && (
                  <div className="text-sm text-gray-600 dark:text-gray-400">
                    Execution time: {results.execution_time.toFixed(2)}s
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-12">
                <Brain className="mx-auto h-12 w-12 text-gray-400" />
                <h3 className="mt-2 text-sm font-medium text-gray-900 dark:text-white">
                  No Results Yet
                </h3>
                <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                  Run an evolution operation to see results here
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Recent Evolutions */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
        <div className="p-6 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-lg font-medium text-gray-900 dark:text-white">
            Recent Evolutions
          </h2>
        </div>
        <div className="p-6">
          <div className="space-y-4">
            {[
              {
                id: 'ev-001',
                type: 'recursive',
                target: 'web_app',
                status: 'completed',
                time: '2 minutes ago',
                fitness: 0.97,
              },
              {
                id: 'ev-002',
                type: 'verified',
                target: 'api_service',
                status: 'running',
                time: '5 minutes ago',
                fitness: null,
              },
              {
                id: 'ev-003',
                type: 'recursive',
                target: 'mobile_app',
                status: 'failed',
                time: '10 minutes ago',
                fitness: 0.45,
              },
            ].map((evolution) => (
              <div key={evolution.id} className="flex items-center justify-between p-4 border border-gray-200 dark:border-gray-700 rounded-md">
                <div className="flex items-center space-x-3">
                  <Brain className="h-5 w-5 text-blue-500" />
                  <div>
                    <p className="text-sm font-medium text-gray-900 dark:text-white">
                      {evolution.type} evolution on {evolution.target}
                    </p>
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      {evolution.time}
                    </p>
                  </div>
                </div>
                <div className="flex items-center space-x-3">
                  {evolution.fitness && (
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                      Fitness: {(evolution.fitness * 100).toFixed(1)}%
                    </span>
                  )}
                  <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                    evolution.status === 'completed'
                      ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                      : evolution.status === 'running'
                      ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
                      : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                  }`}>
                    {evolution.status}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
