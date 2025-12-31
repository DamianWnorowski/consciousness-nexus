import { useQuery } from '@tanstack/react-query'
import { Activity, Cpu, MemoryStick, HardDrive, Zap, AlertTriangle } from 'lucide-react'
import { apiClient } from '../lib/api'

export function Monitoring() {
  const { data: systemStatus, isLoading } = useQuery({
    queryKey: ['system-status'],
    queryFn: () => apiClient.getSystemStatus(),
    refetchInterval: 5000,
  })

  const metrics = [
    {
      name: 'CPU Usage',
      value: '24.7%',
      icon: Cpu,
      color: 'text-blue-600',
      status: 'normal',
    },
    {
      name: 'Memory Usage',
      value: '67.3%',
      icon: MemoryStick,
      color: 'text-green-600',
      status: 'normal',
    },
    {
      name: 'Disk Usage',
      value: '45.2%',
      icon: HardDrive,
      color: 'text-purple-600',
      status: 'normal',
    },
    {
      name: 'API Response Time',
      value: '127ms',
      icon: Zap,
      color: 'text-orange-600',
      status: 'warning',
    },
  ]

  const alerts = [
    {
      id: 1,
      level: 'warning',
      message: 'API response time above threshold',
      time: '2 minutes ago',
      service: 'api-server',
    },
    {
      id: 2,
      level: 'info',
      message: 'Evolution cycle completed successfully',
      time: '5 minutes ago',
      service: 'evolution-engine',
    },
    {
      id: 3,
      level: 'error',
      message: 'Validation engine temporary failure',
      time: '10 minutes ago',
      service: 'validation-service',
    },
  ]

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'error': return 'text-red-600 bg-red-100'
      case 'warning': return 'text-yellow-600 bg-yellow-100'
      case 'info': return 'text-blue-600 bg-blue-100'
      default: return 'text-green-600 bg-green-100'
    }
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Monitoring</h1>
        <p className="mt-2 text-gray-600 dark:text-gray-300">
          Real-time system monitoring and performance metrics
        </p>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {metrics.map((metric) => (
          <div key={metric.name} className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="p-2 rounded-md bg-gray-100 dark:bg-gray-700">
                <metric.icon className={`h-6 w-6 ${metric.color}`} />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  {metric.name}
                </p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">
                  {metric.value}
                </p>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* System Components Status */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
          <div className="p-6 border-b border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-medium text-gray-900 dark:text-white">
              System Components
            </h3>
          </div>
          <div className="p-6">
            <div className="space-y-4">
              {[
                { name: 'API Server', status: 'operational', uptime: '99.9%', load: '23%' },
                { name: 'Database', status: 'operational', uptime: '99.8%', load: '45%' },
                { name: 'Redis Cache', status: 'operational', uptime: '99.7%', load: '12%' },
                { name: 'Evolution Engine', status: 'operational', uptime: '98.5%', load: '67%' },
                { name: 'Validation Service', status: 'degraded', uptime: '95.2%', load: '89%' },
                { name: 'Monitoring Stack', status: 'operational', uptime: '99.9%', load: '34%' },
              ].map((component) => (
                <div key={component.name} className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-900 dark:text-white">
                      {component.name}
                    </p>
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      Uptime: {component.uptime} | Load: {component.load}
                    </p>
                  </div>
                  <div className="flex items-center">
                    <div className={`w-3 h-3 rounded-full mr-2 ${
                      component.status === 'operational' ? 'bg-green-500' :
                      component.status === 'degraded' ? 'bg-yellow-500' : 'bg-red-500'
                    }`} />
                    <span className="text-sm text-gray-600 dark:text-gray-400 capitalize">
                      {component.status}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Recent Alerts */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
          <div className="p-6 border-b border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-medium text-gray-900 dark:text-white">
              Recent Alerts
            </h3>
          </div>
          <div className="p-6">
            <div className="space-y-4">
              {alerts.map((alert) => (
                <div key={alert.id} className="flex items-start space-x-3">
                  <div className={`flex-shrink-0 w-2 h-2 rounded-full mt-2 ${
                    alert.level === 'error' ? 'bg-red-500' :
                    alert.level === 'warning' ? 'bg-yellow-500' : 'bg-blue-500'
                  }`} />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-gray-900 dark:text-white">
                      {alert.message}
                    </p>
                    <div className="flex items-center space-x-2 mt-1">
                      <span className="text-xs text-gray-500 dark:text-gray-400">
                        {alert.service}
                      </span>
                      <span className="text-xs text-gray-500 dark:text-gray-400">
                        •
                      </span>
                      <span className="text-xs text-gray-500 dark:text-gray-400">
                        {alert.time}
                      </span>
                    </div>
                  </div>
                  <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${getStatusColor(alert.level)}`}>
                    {alert.level}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Performance Charts Placeholder */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
        <div className="p-6 border-b border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white">
            Performance Metrics
          </h3>
        </div>
        <div className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* CPU Usage Chart Placeholder */}
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-6">
              <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-4">
                CPU Usage (Last 24h)
              </h4>
              <div className="h-32 flex items-center justify-center text-gray-500 dark:text-gray-400">
                <Activity className="h-8 w-8 mr-2" />
                Chart visualization would go here
              </div>
            </div>

            {/* Memory Usage Chart Placeholder */}
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-6">
              <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-4">
                Memory Usage (Last 24h)
              </h4>
              <div className="h-32 flex items-center justify-center text-gray-500 dark:text-gray-400">
                <MemoryStick className="h-8 w-8 mr-2" />
                Chart visualization would go here
              </div>
            </div>
          </div>

          <div className="mt-6 text-center">
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Real-time charts would be implemented with a charting library like Recharts or Chart.js
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
