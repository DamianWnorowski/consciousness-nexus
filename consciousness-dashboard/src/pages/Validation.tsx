import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Shield, Upload, CheckCircle, XCircle, AlertTriangle, Loader2 } from 'lucide-react'
import { apiClient } from '../lib/api'

export function Validation() {
  const [files, setFiles] = useState<string>('')
  const [validationScope, setValidationScope] = useState<'basic' | 'full' | 'comprehensive'>('full')
  const [results, setResults] = useState<any>(null)
  const [isValidating, setIsValidating] = useState(false)

  const validationMutation = useMutation({
    mutationFn: (data: { files: string[]; validation_scope: string; user_id: string }) =>
      apiClient.runValidation(data),
    onSuccess: (data) => {
      setResults(data)
    },
    onError: (error: any) => {
      setResults({ success: false, error: error.message })
    },
  })

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsValidating(true)
    setResults(null)

    const fileList = files.split('\n').map(f => f.trim()).filter(f => f)

    try {
      await validationMutation.mutateAsync({
        files: fileList,
        validation_scope: validationScope,
        user_id: 'web_user',
      })
    } finally {
      setIsValidating(false)
    }
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'text-red-600 bg-red-100'
      case 'high': return 'text-orange-600 bg-orange-100'
      case 'medium': return 'text-yellow-600 bg-yellow-100'
      case 'low': return 'text-blue-600 bg-blue-100'
      default: return 'text-gray-600 bg-gray-100'
    }
  }

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'security': return Shield
      case 'performance': return Loader2
      default: return AlertTriangle
    }
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Validation</h1>
        <p className="mt-2 text-gray-600 dark:text-gray-300">
          Validate code and systems with enterprise-grade security checks
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Validation Form */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
          <div className="p-6 border-b border-gray-200 dark:border-gray-700">
            <h2 className="text-lg font-medium text-gray-900 dark:text-white">
              Code Validation
            </h2>
            <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
              Run comprehensive security and quality checks
            </p>
          </div>

          <form onSubmit={handleSubmit} className="p-6 space-y-6">
            {/* Files Input */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Files to Validate
              </label>
              <textarea
                value={files}
                onChange={(e) => setFiles(e.target.value)}
                placeholder={`src/main.py
src/utils.py
src/api.py
tests/test_api.py`}
                rows={6}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white font-mono text-sm"
                required
              />
              <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                One file path per line
              </p>
            </div>

            {/* Validation Scope */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Validation Scope
              </label>
              <select
                value={validationScope}
                onChange={(e) => setValidationScope(e.target.value as any)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
              >
                <option value="basic">Basic - Syntax and imports</option>
                <option value="full">Full - Security, performance, best practices</option>
                <option value="comprehensive">Comprehensive - Everything + AI analysis</option>
              </select>
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={isValidating || validationMutation.isLoading}
              className="w-full flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isValidating || validationMutation.isLoading ? (
                <>
                  <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                  Running Validation...
                </>
              ) : (
                <>
                  <Shield className="h-5 w-5 mr-2" />
                  Start Validation
                </>
              )}
            </button>
          </form>
        </div>

        {/* Results Panel */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
          <div className="p-6 border-b border-gray-200 dark:border-gray-700">
            <h2 className="text-lg font-medium text-gray-900 dark:text-white">
              Validation Results
            </h2>
            <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
              Security, performance, and quality analysis
            </p>
          </div>

          <div className="p-6">
            {results ? (
              <div className="space-y-6">
                {/* Overall Status */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    {results.isValid ? (
                      <CheckCircle className="h-6 w-6 text-green-500" />
                    ) : (
                      <XCircle className="h-6 w-6 text-red-500" />
                    )}
                    <span className={`text-lg font-medium ${
                      results.isValid ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {results.isValid ? 'Validation Passed' : 'Validation Failed'}
                    </span>
                  </div>
                  <div className="text-right">
                    <div className="text-2xl font-bold text-gray-900 dark:text-white">
                      {((results.passedChecks / results.totalChecks) * 100).toFixed(1)}%
                    </div>
                    <div className="text-sm text-gray-500 dark:text-gray-400">
                      {results.passedChecks}/{results.totalChecks} checks
                    </div>
                  </div>
                </div>

                {/* Fitness Score */}
                <div className="bg-blue-50 dark:bg-blue-900 rounded-md p-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-blue-800 dark:text-blue-200">
                      Code Fitness Score
                    </span>
                    <span className="text-lg font-bold text-blue-800 dark:text-blue-200">
                      {(results.fitnessScore * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                {/* Issues List */}
                {results.issues && results.issues.length > 0 && (
                  <div>
                    <h3 className="text-sm font-medium text-gray-900 dark:text-white mb-3">
                      Issues Found ({results.issues.length})
                    </h3>
                    <div className="space-y-2 max-h-96 overflow-y-auto">
                      {results.issues.slice(0, 10).map((issue: any, index: number) => {
                        const IconComponent = getCategoryIcon(issue.category)
                        return (
                          <div key={index} className="flex items-start space-x-3 p-3 bg-gray-50 dark:bg-gray-700 rounded-md">
                            <IconComponent className="h-5 w-5 text-gray-400 mt-0.5" />
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center space-x-2">
                                <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${getSeverityColor(issue.severity)}`}>
                                  {issue.severity}
                                </span>
                                <span className="text-xs text-gray-500 dark:text-gray-400">
                                  {issue.category}
                                </span>
                              </div>
                              <p className="text-sm text-gray-900 dark:text-white mt-1">
                                {issue.title}
                              </p>
                              <p className="text-xs text-gray-600 dark:text-gray-300 mt-1">
                                {issue.description}
                              </p>
                              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                                File: {issue.file}
                              </p>
                            </div>
                          </div>
                        )
                      })}
                      {results.issues.length > 10 && (
                        <p className="text-xs text-gray-500 dark:text-gray-400 text-center">
                          ... and {results.issues.length - 10} more issues
                        </p>
                      )}
                    </div>
                  </div>
                )}

                {/* Warnings */}
                {results.warnings && results.warnings.length > 0 && (
                  <div>
                    <h3 className="text-sm font-medium text-gray-900 dark:text-white mb-3">
                      Warnings ({results.warnings.length})
                    </h3>
                    <div className="space-y-1">
                      {results.warnings.map((warning: string, index: number) => (
                        <div key={index} className="flex items-center space-x-2 text-sm text-yellow-600 dark:text-yellow-400">
                          <AlertTriangle className="h-4 w-4" />
                          <span>{warning}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-12">
                <Shield className="mx-auto h-12 w-12 text-gray-400" />
                <h3 className="mt-2 text-sm font-medium text-gray-900 dark:text-white">
                  No Validation Results
                </h3>
                <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                  Run validation on your code to see security and quality analysis
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
