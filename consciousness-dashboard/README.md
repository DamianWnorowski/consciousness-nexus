# 🖥️ Consciousness Suite Web Dashboard

Complete web interface for the Consciousness Computing Suite - **terminal bypassing** made easy!

## ✨ Features

- **Complete Terminal Bypass**: Run all operations through an intuitive web interface
- **Real-time Monitoring**: Live system health, metrics, and performance data
- **Evolution Operations**: Start and monitor AI evolution processes
- **Code Validation**: Security and quality analysis with visual results
- **Interactive API Docs**: Built-in documentation and testing
- **Settings Management**: Configure safety levels and preferences
- **Responsive Design**: Works on desktop, tablet, and mobile

## 🚀 Quick Start

### Prerequisites
```bash
# Node.js 18+ and npm
node --version  # Should be 18.x.x or higher
npm --version   # Should be 9.x.x or higher
```

### Development
```bash
# Install dependencies
cd consciousness-dashboard
npm install

# Start development server
npm run dev

# Open http://localhost:3000
```

### Production Build
```bash
# Build for production
npm run build

# Preview production build
npm run preview
```

## 🐳 Docker Deployment

The dashboard is automatically included in the main Docker Compose setup:

```bash
# From the root directory
docker-compose up -d

# Dashboard available at: http://localhost:31573
```

### Standalone Docker
```bash
# Build dashboard image
docker build -t consciousness-dashboard ./consciousness-dashboard

# Run dashboard
docker run -p 3000:3000 consciousness-dashboard
```

## 🔗 API Connection

The dashboard connects to your Consciousness Suite API server:

- **Default API URL**: `http://localhost:18473`
- **Configuration**: Update in Settings → API Configuration
- **Authentication**: Uses API keys configured in the backend

## 📱 Interface Overview

### Dashboard
- System health overview
- Active sessions and uptime
- Recent activity feed
- Quick action buttons

### Evolution
- Start recursive or verified evolution
- Configure safety levels and parameters
- Real-time progress monitoring
- Results visualization

### Validation
- File upload or path input
- Multiple validation scopes
- Detailed issue reporting
- Security and performance metrics

### Monitoring
- System component status
- Performance metrics
- Alert management
- Resource usage charts

### Settings
- API endpoint configuration
- Safety level preferences
- UI customization
- Notification settings

### API Docs
- Interactive OpenAPI documentation
- Request/response examples
- Authentication guide
- SDK information

## 🛠️ Development

### Project Structure
```
consciousness-dashboard/
├── src/
│   ├── components/     # Reusable UI components
│   ├── pages/         # Main application pages
│   ├── lib/           # Utilities and API client
│   └── ...
├── public/            # Static assets
├── package.json       # Dependencies and scripts
├── vite.config.ts     # Build configuration
├── tailwind.config.js # Styling configuration
└── README.md          # This file
```

### Tech Stack
- **Frontend**: React 18 with TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS
- **State Management**: TanStack Query (React Query)
- **Icons**: Lucide React
- **HTTP Client**: Axios

### Available Scripts
```bash
npm run dev          # Start development server
npm run build        # Build for production
npm run preview      # Preview production build
npm run lint         # Run ESLint
npm run type-check   # Run TypeScript type checking
```

## 🔧 Configuration

### Environment Variables
```bash
# API Configuration
VITE_API_BASE_URL=http://localhost:18473

# Development
VITE_DEV_TOOLS=true
```

### Build Configuration
The dashboard is configured to proxy API requests to the backend server. Update `vite.config.ts` if needed:

```typescript
export default defineConfig({
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:18473',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  }
})
```

## 🚀 Deployment Options

### Nginx + Static Files
```bash
# Build the app
npm run build

# Serve with nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        root /path/to/consciousness-dashboard/dist;
        try_files $uri $uri/ /index.html;
    }

    # Proxy API requests
    location /api {
        proxy_pass http://localhost:18473;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Cloud Deployment
- **Vercel**: Connect GitHub repo, automatic deployments
- **Netlify**: Drag & drop dist folder or connect repo
- **AWS S3 + CloudFront**: Static hosting with CDN
- **Docker**: Use provided Docker setup

## 🔐 Security

- **API Key Authentication**: All requests require valid API keys
- **HTTPS Only**: Production deployments should use HTTPS
- **CORS Configuration**: Properly configured for cross-origin requests
- **Input Validation**: All user inputs are validated on both frontend and backend

## 📊 Performance

- **Lazy Loading**: Components loaded on demand
- **Code Splitting**: Automatic chunk splitting for optimal loading
- **Caching**: React Query for efficient API data caching
- **Compression**: Gzip compression enabled in production builds

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

**API Connection Failed**
- Ensure the Consciousness Suite API server is running on port 18473
- Check API key configuration
- Verify CORS settings

**Build Errors**
- Clear node_modules: `rm -rf node_modules && npm install`
- Check Node.js version compatibility
- Verify all dependencies are installed

**Performance Issues**
- Enable production build optimizations
- Check network tab for large assets
- Optimize images and assets

## 📞 Support

- **Documentation**: https://docs.consciousness-suite.com
- **Issues**: https://github.com/DAMIANWNOROWSKI/consciousness-suite/issues
- **Discussions**: https://github.com/DAMIANWNOROWSKI/consciousness-suite/discussions

---

**Transform your terminal-based workflow into a beautiful, intuitive web experience!** 🎨✨
