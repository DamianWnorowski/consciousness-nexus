import { Routes, Route } from 'react-router-dom'
import { Layout } from './components/Layout'
import { Dashboard } from './pages/Dashboard'
import { Evolution } from './pages/Evolution'
import { Validation } from './pages/Validation'
import { Monitoring } from './pages/Monitoring'
import { Settings } from './pages/Settings'
import { ApiDocs } from './pages/ApiDocs'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/evolution" element={<Evolution />} />
        <Route path="/validation" element={<Validation />} />
        <Route path="/monitoring" element={<Monitoring />} />
        <Route path="/settings" element={<Settings />} />
        <Route path="/docs" element={<ApiDocs />} />
      </Routes>
    </Layout>
  )
}

export default App
