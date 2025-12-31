import { ReactNode } from 'react'
import { Link, useLocation } from 'react-router-dom'
import {
  Brain,
  Activity,
  Shield,
  Settings,
  FileText,
  BarChart3,
  Zap,
  CheckCircle,
  AlertTriangle,
  XCircle
} from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { apiClient } from '../lib/api'

interface LayoutProps {
  children: ReactNode
}

const navigation = [
  { name: 'Dashboard', href: '/', icon: BarChart3 },
  { name: 'Evolution', href: '/evolution', icon: Brain },
  { name: 'Validation', href: '/validation', icon: Shield },
  { name: 'Monitoring', href: '/monitoring', icon: Activity },
  { name: 'Settings', href: '/settings', icon: Settings },
  { name: 'API Docs', href: '/docs', icon: FileText },
]

export function Layout({ children }: LayoutProps) {
  const location = useLocation()

  // Fetch system health
  const { data: health, isLoading: healthLoading } = useQuery({
    queryKey: ['health'],
    queryFn: () => apiClient.getHealth(),
    refetchInterval: 5000, // Refresh every 5 seconds
  })

  const getStatusIcon = () => {
    if (healthLoading) return <Activity className="h-4 w-4 animate-spin text-yellow-500" />
    if (health?.status === 'healthy') return <CheckCircle className="h-4 w-4 text-green-500" />
    if (health?.status === 'warning') return <AlertTriangle className="h-4 w-4 text-yellow-500" />
    return <XCircle className="h-4 w-4 text-red-500" />
  }

  const getStatusText = () => {
    if (healthLoading) return 'Loading...'
    return health?.status === 'healthy' ? 'System Healthy' : 'System Issues'
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <header className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <Brain className="h-8 w-8 text-blue-600" />
              <span className="ml-2 text-xl font-bold text-gray-900 dark:text-white">
                Consciousness Suite
              </span>
            </div>

            <div className="flex items-center space-x-4">
              {getStatusIcon()}
              <span className="text-sm text-gray-600 dark:text-gray-300">
                {getStatusText()}
              </span>
              <div className="flex items-center space-x-1">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-xs text-gray-500 dark:text-gray-400">Live</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="flex">
        {/* Sidebar */}
        <nav className="w-64 bg-white dark:bg-gray-800 shadow-sm min-h-screen border-r border-gray-200 dark:border-gray-700">
          <div className="p-6">
            <div className="space-y-2">
              {navigation.map((item) => {
                const isActive = location.pathname === item.href
                return (
                  <Link
                    key={item.name}
                    to={item.href}
                    className={`flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                      isActive
                        ? 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-200'
                        : 'text-gray-600 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-700'
                    }`}
                  >
                    <item.icon className="mr-3 h-5 w-5" />
                    {item.name}
                  </Link>
                )
              })}
            </div>
          </div>

          {/* System Status Panel */}
          <div className="absolute bottom-0 left-0 right-0 w-64 p-4 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
            <div className="text-xs text-gray-500 dark:text-gray-400 mb-2">System Status</div>
            <div className="space-y-1">
              <div className="flex justify-between">
                <span className="text-xs">API</span>
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              </div>
              <div className="flex justify-between">
                <span className="text-xs">Database</span>
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              </div>
              <div className="flex justify-between">
                <span className="text-xs">Monitoring</span>
                <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
              </div>
            </div>
          </div>
        </nav>

        {/* Main Content */}
        <main className="flex-1 p-8">
          <div className="max-w-7xl mx-auto">
            {children}
          </div>
        </main>
      </div>
    </div>
  )
}
