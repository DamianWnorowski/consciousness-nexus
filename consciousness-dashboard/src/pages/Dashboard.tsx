import { useQuery } from '@tanstack/react-query'
import {
  Brain,
  Shield,
  Activity,
  AlertTriangle,
  CheckCircle,
  Clock,
  Zap,
  Users,
  Database,
  Cpu,
  MemoryStick
} from 'lucide-react'
import { apiClient } from '../lib/api'

export function Dashboard() {
  // Fetch system data
  const { data: health, isLoading: healthLoading } = useQuery({
    queryKey: ['health'],
    queryFn: () => apiClient.getHealth(),
    refetchInterval: 5000,
  })

  const { data: systemStatus, isLoading: statusLoading } = useQuery({
    queryKey: ['system-status'],
    queryFn: () => apiClient.getSystemStatus(),
    refetchInterval: 10000,
  })

  const stats = [
    {
      name: 'Active Sessions',
      value: health?.active_sessions || 0,
      icon: Users,
      color: 'text-blue-600',
      bgColor: 'bg-blue-100',
    },
    {
      name: 'System Uptime',
      value: health ? `${Math.floor(health.uptime / 3600)}h ${Math.floor((health.uptime % 3600) / 60)}m` : '0m',
      icon: Clock,
      color: 'text-green-600',
      bgColor: 'bg-green-100',
    },
    {
      name: 'Evolution Cycles',
      value: '1,247',
      icon: Brain,
      color: 'text-purple-600',
      bgColor: 'bg-purple-100',
    },
    {
      name: 'Safety Score',
      value: '98.7%',
      icon: Shield,
      color: 'text-emerald-600',
      bgColor: 'bg-emerald-100',
    },
  ]

  const recentActivities = [
    {
      id: 1,
      type: 'evolution',
      message: 'Recursive evolution completed for web_app',
      time: '2 minutes ago',
      status: 'success',
    },
    {
      id: 2,
      type: 'validation',
      message: 'Code validation passed for api.py',
      time: '5 minutes ago',
      status: 'success',
    },
    {
      id: 3,
      type: 'alert',
      message: 'Resource usage exceeded threshold',
      time: '10 minutes ago',
      status: 'warning',
    },
    {
      id: 4,
      type: 'evolution',
      message: 'Verified evolution started for mobile_app',
      time: '15 minutes ago',
      status: 'running',
    },
  ]

  const systemComponents = [
    { name: 'API Server', status: 'operational', icon: Zap },
    { name: 'Database', status: 'operational', icon: Database },
    { name: 'Redis Cache', status: 'operational', icon: MemoryStick },
    { name: 'Monitoring', status: 'degraded', icon: Activity },
    { name: 'Validation Engine', status: 'operational', icon: Shield },
    { name: 'Evolution Engine', status: 'operational', icon: Brain },
  ]

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Dashboard</h1>
        <p className="mt-2 text-gray-600 dark:text-gray-300">
          Overview of your Consciousness Computing Suite
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat) => (
          <div key={stat.name} className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className={`p-2 rounded-md ${stat.bgColor}`}>
                <stat.icon className={`h-6 w-6 ${stat.color}`} />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  {stat.name}
                </p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">
                  {stat.value}
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
              {systemComponents.map((component) => (
                <div key={component.name} className="flex items-center justify-between">
                  <div className="flex items-center">
                    <component.icon className="h-5 w-5 text-gray-400 mr-3" />
                    <span className="text-sm font-medium text-gray-900 dark:text-white">
                      {component.name}
                    </span>
                  </div>
                  <div className="flex items-center">
                    {component.status === 'operational' && (
                      <CheckCircle className="h-5 w-5 text-green-500" />
                    )}
                    {component.status === 'degraded' && (
                      <AlertTriangle className="h-5 w-5 text-yellow-500" />
                    )}
                    <span className={`ml-2 text-sm ${
                      component.status === 'operational'
                        ? 'text-green-600'
                        : component.status === 'degraded'
                        ? 'text-yellow-600'
                        : 'text-red-600'
                    }`}>
                      {component.status}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Recent Activity */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
          <div className="p-6 border-b border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-medium text-gray-900 dark:text-white">
              Recent Activity
            </h3>
          </div>
          <div className="p-6">
            <div className="space-y-4">
              {recentActivities.map((activity) => (
                <div key={activity.id} className="flex items-start space-x-3">
                  <div className={`flex-shrink-0 w-2 h-2 rounded-full mt-2 ${
                    activity.status === 'success' ? 'bg-green-500' :
                    activity.status === 'warning' ? 'bg-yellow-500' :
                    activity.status === 'running' ? 'bg-blue-500' : 'bg-red-500'
                  }`} />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-gray-900 dark:text-white">
                      {activity.message}
                    </p>
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      {activity.time}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
        <div className="p-6 border-b border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white">
            Quick Actions
          </h3>
        </div>
        <div className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <button className="flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500">
              <Brain className="h-5 w-5 mr-2" />
              Start Evolution
            </button>
            <button className="flex items-center justify-center px-4 py-2 border border-gray-300 dark:border-gray-600 text-sm font-medium rounded-md text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700">
              <Shield className="h-5 w-5 mr-2" />
              Run Validation
            </button>
            <button className="flex items-center justify-center px-4 py-2 border border-gray-300 dark:border-gray-600 text-sm font-medium rounded-md text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700">
              <Activity className="h-5 w-5 mr-2" />
              View Monitoring
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
