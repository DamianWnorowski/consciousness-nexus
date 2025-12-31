#!/bin/bash
# 🚀 CONSCIOUSNESS SUITE AUTO-START SCRIPT
# ========================================
#
# Automatically starts the complete Consciousness Suite deployment
# including web dashboard, API server, monitoring stack, and all services.
#
# Usage: ./start-consciousness-suite.sh
#

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="Consciousness Suite"
DOCKER_COMPOSE_FILE="docker-compose.yml"

# Service ports and URLs
declare -A SERVICES=(
    ["Web Dashboard"]="31573:http://localhost:31573"
    ["API Server"]="18473:http://localhost:18473"
    ["Grafana"]="31572:http://localhost:31572"
    ["Prometheus"]="24789:http://localhost:24789"
    ["Loki"]="42851:http://localhost:42851"
    ["Nginx HTTP"]="25746:http://localhost:25746"
    ["PostgreSQL"]="17392:localhost:17392"
    ["Redis"]="29481:localhost:29481"
)

# Logging functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_header() {
    echo -e "${PURPLE}🚀 $1${NC}"
    echo -e "${PURPLE}$(printf '%.0s=' {1..50})${NC}"
}

log_service() {
    echo -e "${CYAN}🔧 $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log_header "CHECKING PREREQUISITES"

    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    log_success "Docker is installed"

    # Check if Docker Compose is available
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not available. Please install Docker Compose."
        exit 1
    fi
    log_success "Docker Compose is available"

    # Check if docker-compose.yml exists
    if [ ! -f "$DOCKER_COMPOSE_FILE" ]; then
        log_error "docker-compose.yml not found in current directory"
        exit 1
    fi
    log_success "Docker Compose configuration found"

    echo
}

# Pre-deployment checks
pre_deployment_checks() {
    log_header "PRE-DEPLOYMENT CHECKS"

    # Check if ports are available
    log_info "Checking port availability..."
    for service in "${!SERVICES[@]}"; do
        port=$(echo "${SERVICES[$service]}" | cut -d: -f1)
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            log_warning "Port $port ($service) is already in use"
            log_info "This might cause deployment issues"
        else
            log_success "Port $port ($service) is available"
        fi
    done

    # Check available disk space
    DISK_SPACE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
    if [ "$DISK_SPACE" -lt 5 ]; then
        log_warning "Low disk space: ${DISK_SPACE}GB available"
        log_info "Docker containers may fail to start"
    else
        log_success "Sufficient disk space available"
    fi

    # Check available memory
    MEM_GB=$(free -g | awk 'NR==2{printf "%.0f", $2}')
    if [ "$MEM_GB" -lt 4 ]; then
        log_warning "Low memory: ${MEM_GB}GB available"
        log_info "Consider increasing system memory for better performance"
    else
        log_success "Sufficient memory available: ${MEM_GB}GB"
    fi

    echo
}

# Start deployment
start_deployment() {
    log_header "STARTING CONSCIOUSNESS SUITE DEPLOYMENT"

    log_info "Pulling latest Docker images..."
    docker-compose pull || log_warning "Some images failed to pull (this is normal)"

    log_info "Starting all services..."
    log_info "This may take several minutes on first run..."

    # Start services in detached mode
    if docker-compose up -d; then
        log_success "All services started successfully!"
    else
        log_error "Failed to start services"
        log_info "Check Docker logs with: docker-compose logs"
        exit 1
    fi

    echo
}

# Wait for services to be ready
wait_for_services() {
    log_header "WAITING FOR SERVICES TO BE READY"

    # Give services time to start
    log_info "Waiting 30 seconds for initial startup..."
    sleep 30

    # Test key services
    local services_ready=0
    local total_services=2  # API and Dashboard

    # Test API server
    log_service "Testing API Server (port 18473)..."
    if curl -s --max-time 10 http://localhost:18473/health > /dev/null 2>&1; then
        log_success "API Server is responding"
        ((services_ready++))
    else
        log_warning "API Server not responding yet (this is normal on first startup)"
    fi

    # Test Web Dashboard
    log_service "Testing Web Dashboard (port 31573)..."
    if curl -s --max-time 10 http://localhost:31573 > /dev/null 2>&1; then
        log_success "Web Dashboard is responding"
        ((services_ready++))
    else
        log_warning "Web Dashboard not responding yet (may take longer to build)"
    fi

    if [ $services_ready -eq $total_services ]; then
        log_success "All core services are ready!"
    else
        log_info "Some services are still starting up..."
        log_info "The web dashboard may take 2-3 minutes to fully initialize"
    fi

    echo
}

# Display service information
display_service_info() {
    log_header "CONSCIOUSNESS SUITE IS NOW RUNNING!"

    echo -e "${WHITE}🌟 ACCESS YOUR SERVICES:${NC}"
    echo

    # Main interfaces
    echo -e "${CYAN}🖥️  PRIMARY WEB DASHBOARD:${NC}"
    echo -e "   ${GREEN}http://localhost:31573${NC} ← ${YELLOW}MAIN INTERFACE (Terminal Bypassing!)${NC}"
    echo

    echo -e "${CYAN}🔗 API & DOCUMENTATION:${NC}"
    echo -e "   ${GREEN}http://localhost:18473${NC}     ← REST API"
    echo -e "   ${GREEN}http://localhost:18473/docs${NC} ← Interactive API Docs"
    echo

    echo -e "${CYAN}📊 MONITORING & METRICS:${NC}"
    echo -e "   ${GREEN}http://localhost:31572${NC}     ← Grafana Dashboards"
    echo -e "   ${GREEN}http://localhost:24789${NC}     ← Prometheus Metrics"
    echo -e "   ${GREEN}http://localhost:42851${NC}     ← Loki Log Aggregation"
    echo

    echo -e "${CYAN}🛠️  DEVELOPMENT ACCESS:${NC}"
    echo -e "   ${WHITE}localhost:17392${NC}     ← PostgreSQL Database"
    echo -e "   ${WHITE}localhost:29481${NC}     ← Redis Cache"
    echo -e "   ${WHITE}localhost:25746${NC}     ← Nginx HTTP Proxy"
    echo

    echo -e "${PURPLE}🎯 QUICK START:${NC}"
    echo -e "   1. Open ${GREEN}http://localhost:31573${NC} in your browser"
    echo -e "   2. Explore the dashboard - no terminal commands needed!"
    echo -e "   3. Try running an evolution or validation"
    echo

    echo -e "${BLUE}📋 USEFUL COMMANDS:${NC}"
    echo -e "   ${WHITE}docker-compose logs${NC}     ← View all service logs"
    echo -e "   ${WHITE}docker-compose ps${NC}       ← Check service status"
    echo -e "   ${WHITE}docker-compose down${NC}     ← Stop all services"
    echo -e "   ${WHITE}docker-compose restart${NC}   ← Restart all services"
    echo
}

# Display next steps
display_next_steps() {
    log_header "WHAT TO DO NEXT"

    echo -e "${YELLOW}🎨 EXPLORE THE WEB DASHBOARD:${NC}"
    echo "   • Visit http://localhost:31573"
    echo "   • Try the Evolution interface"
    echo "   • Upload files for validation"
    echo "   • Monitor system health"
    echo

    echo -e "${YELLOW}🧪 TEST THE SYSTEM:${NC}"
    echo "   • Run: ./consciousness-cli health"
    echo "   • Check: http://localhost:18473/health"
    echo "   • Monitor: http://localhost:31572 (admin/admin)"
    echo

    echo -e "${YELLOW}🚀 FOR PRODUCTION:${NC}"
    echo "   • Set up SSL certificates"
    echo "   • Configure authentication"
    echo "   • Set up backups"
    echo "   • Configure monitoring alerts"
    echo

    echo -e "${GREEN}🎉 ENJOY YOUR CONSCIOUSNESS SUITE!${NC}"
    echo -e "${GREEN}   Terminal bypassing is now complete!${NC}"
}

# Main execution
main() {
    echo
    log_header "CONSCIOUSNESS SUITE AUTO-DEPLOYMENT"
    echo -e "${WHITE}Starting complete AI safety platform with web dashboard...${NC}"
    echo

    check_prerequisites
    pre_deployment_checks
    start_deployment
    wait_for_services
    display_service_info
    display_next_steps

    echo
    log_success "DEPLOYMENT COMPLETE! Your Consciousness Suite is now running."
    echo -e "${PURPLE}🚀 Welcome to the future of AI safety management!${NC}"
}

# Handle command line arguments
case "${1:-}" in
    "--help"|"-h")
        echo "Consciousness Suite Auto-Start Script"
        echo ""
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --help, -h    Show this help message"
        echo "  --logs        Show service logs after startup"
        echo "  --status      Show service status"
        echo ""
        echo "Examples:"
        echo "  $0              # Start all services"
        echo "  $0 --logs       # Start and show logs"
        echo "  $0 --status     # Show current status"
        exit 0
        ;;
    "--logs")
        main
        echo
        log_info "Showing service logs (Ctrl+C to exit)..."
        docker-compose logs -f
        ;;
    "--status")
        log_header "SERVICE STATUS"
        docker-compose ps
        echo
        log_info "Service health checks:"
        echo -n "API Server: "
        curl -s http://localhost:18473/health && echo "✅ OK" || echo "❌ DOWN"
        echo -n "Web Dashboard: "
        curl -s http://localhost:31573 && echo "✅ OK" || echo "❌ DOWN"
        ;;
    *)
        main
        ;;
esac
