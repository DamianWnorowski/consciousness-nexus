import { useState } from 'react'
import { Settings as SettingsIcon, Save, RefreshCw } from 'lucide-react'

export function Settings() {
  const [settings, setSettings] = useState({
    apiEndpoint: 'http://localhost:18473',
    safetyLevel: 'standard',
    autoRefresh: true,
    refreshInterval: 30,
    notifications: true,
    theme: 'system',
    language: 'en',
  })

  const handleSave = () => {
    // In a real app, this would save to backend
    alert('Settings saved successfully!')
  }

  const handleReset = () => {
    setSettings({
      apiEndpoint: 'http://localhost:18473',
      safetyLevel: 'standard',
      autoRefresh: true,
      refreshInterval: 30,
      notifications: true,
      theme: 'system',
      language: 'en',
    })
  }

  const updateSetting = (key: string, value: any) => {
    setSettings(prev => ({ ...prev, [key]: value }))
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Settings</h1>
        <p className="mt-2 text-gray-600 dark:text-gray-300">
          Configure your Consciousness Suite preferences and system settings
        </p>
      </div>

      {/* Settings Sections */}
      <div className="space-y-6">
        {/* API Configuration */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
          <div className="p-6 border-b border-gray-200 dark:border-gray-700">
            <h2 className="text-lg font-medium text-gray-900 dark:text-white">
              API Configuration
            </h2>
            <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
              Configure how the dashboard connects to your Consciousness Suite
            </p>
          </div>

          <div className="p-6 space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                API Endpoint
              </label>
              <input
                type="url"
                value={settings.apiEndpoint}
                onChange={(e) => updateSetting('apiEndpoint', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Default Safety Level
              </label>
              <select
                value={settings.safetyLevel}
                onChange={(e) => updateSetting('safetyLevel', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
              >
                <option value="minimal">Minimal</option>
                <option value="standard">Standard</option>
                <option value="strict">Strict</option>
                <option value="paranoid">Paranoid</option>
              </select>
            </div>
          </div>
        </div>

        {/* Dashboard Settings */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
          <div className="p-6 border-b border-gray-200 dark:border-gray-700">
            <h2 className="text-lg font-medium text-gray-900 dark:text-white">
              Dashboard Settings
            </h2>
            <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
              Customize your dashboard experience
            </p>
          </div>

          <div className="p-6 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Auto Refresh
                </label>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  Automatically refresh dashboard data
                </p>
              </div>
              <input
                type="checkbox"
                checked={settings.autoRefresh}
                onChange={(e) => updateSetting('autoRefresh', e.target.checked)}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Refresh Interval (seconds)
              </label>
              <input
                type="number"
                min="5"
                max="300"
                value={settings.refreshInterval}
                onChange={(e) => updateSetting('refreshInterval', parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
              />
            </div>

            <div className="flex items-center justify-between">
              <div>
                <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Notifications
                </label>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  Show desktop notifications for important events
                </p>
              </div>
              <input
                type="checkbox"
                checked={settings.notifications}
                onChange={(e) => updateSetting('notifications', e.target.checked)}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
            </div>
          </div>
        </div>

        {/* Appearance */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
          <div className="p-6 border-b border-gray-200 dark:border-gray-700">
            <h2 className="text-lg font-medium text-gray-900 dark:text-white">
              Appearance
            </h2>
            <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
              Customize the look and feel of your dashboard
            </p>
          </div>

          <div className="p-6 space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Theme
              </label>
              <select
                value={settings.theme}
                onChange={(e) => updateSetting('theme', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
              >
                <option value="light">Light</option>
                <option value="dark">Dark</option>
                <option value="system">System</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Language
              </label>
              <select
                value={settings.language}
                onChange={(e) => updateSetting('language', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
              >
                <option value="en">English</option>
                <option value="es">Español</option>
                <option value="fr">Français</option>
                <option value="de">Deutsch</option>
                <option value="zh">中文</option>
              </select>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex space-x-4">
          <button
            onClick={handleSave}
            className="flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            <Save className="h-5 w-5 mr-2" />
            Save Settings
          </button>

          <button
            onClick={handleReset}
            className="flex items-center px-4 py-2 border border-gray-300 dark:border-gray-600 text-sm font-medium rounded-md text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700"
          >
            <RefreshCw className="h-5 w-5 mr-2" />
            Reset to Defaults
          </button>
        </div>
      </div>

      {/* System Information */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
        <div className="p-6 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-lg font-medium text-gray-900 dark:text-white">
            System Information
          </h2>
        </div>
        <div className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="text-sm font-medium text-gray-900 dark:text-white mb-2">
                Dashboard Version
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">v2.0.0</p>
            </div>
            <div>
              <h3 className="text-sm font-medium text-gray-900 dark:text-white mb-2">
                API Version
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">v2.0.0</p>
            </div>
            <div>
              <h3 className="text-sm font-medium text-gray-900 dark:text-white mb-2">
                Node.js Version
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">18.x.x</p>
            </div>
            <div>
              <h3 className="text-sm font-medium text-gray-900 dark:text-white mb-2">
                Browser Support
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">Chrome, Firefox, Safari, Edge</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
