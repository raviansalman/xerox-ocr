#!/bin/bash

# Ultimate Search Processor - Combined Build and Start Script

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    print_success "Docker is running"
}

# Function to check if we're in the right directory
check_directory() {
    if [ ! -f "docker-compose.ultimate.yml" ]; then
        print_error "Please run this script from the ultimate directory"
        print_error "Expected files: docker-compose.ultimate.yml, Dockerfile.ultimate"
        exit 1
    fi
    print_success "Running from correct directory"
}

# Function to build Docker image
build_image() {
    print_status "Building Ultimate Search Processor Docker image..."
    
    # Build from parent directory since Dockerfile expects that context
    print_status "Building from parent directory..."
    cd .. && docker build -f ultimate/Dockerfile.ultimate -t ultimate-search-processor:latest . && cd ultimate
    
    if [ $? -eq 0 ]; then
        print_success "Docker image built successfully!"
    else
        print_error "Docker build failed!"
        exit 1
    fi
}

# Function to start services
start_services() {
    print_status "Starting Ultimate Search Processor services..."
    
    # Start services in detached mode
    docker compose -f docker-compose.ultimate.yml up --build -d
    
    if [ $? -eq 0 ]; then
        print_success "Services started successfully!"
    else
        print_error "Failed to start services!"
        exit 1
    fi
}

# Function to wait for services to be ready
wait_for_services() {
    print_status "Waiting for services to initialize..."
    print_warning "This may take 2-10 minutes on first run (includes model downloads)"
    
    # Wait for initial startup
    sleep 30
    
    # Check if API is responding
    local max_attempts=20
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        print_status "Health check attempt $attempt/$max_attempts..."
        
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            print_success "API is responding!"
            break
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            print_warning "API not responding after $max_attempts attempts"
            print_warning "Services may still be starting up. Check logs for details."
            break
        fi
        
        sleep 15
        ((attempt++))
    done
}

# Function to check system health
check_health() {
    print_status "Checking system health..."
    
    # Check API health
    if curl -s http://localhost:8000/health | jq '.' 2>/dev/null; then
        print_success "System health check passed"
    else
        print_warning "System health check failed or jq not available"
        print_warning "You can manually check: curl http://localhost:8000/health"
    fi
}

# Function to show container status
show_status() {
    print_status "Container status:"
    docker compose -f docker-compose.ultimate.yml ps
}

# Function to show access information
show_access_info() {
    echo ""
    print_success "Ultimate Search Processor is running!"
    echo ""
    echo -e "${GREEN}Access Information:${NC}"
    echo "  Main Interface:    http://localhost:8000"
    echo "  API Documentation: http://localhost:8000/docs"
    echo "  Health Check:      http://localhost:8000/health"
    echo ""
    echo -e "${BLUE}Useful Commands:${NC}"
    echo "  View logs:         docker logs vector-storage-processing-service-python-ultimate-search-1"
    echo "  View worker logs:  docker logs vector-storage-processing-service-python-ultimate-celery-worker-1"
    echo "  View all logs:     docker compose -f docker-compose.ultimate.yml logs -f"
    echo "  Stop services:     docker compose -f docker-compose.ultimate.yml down"
    echo "  Restart services:  ./start.sh"
    echo ""
}

# Function to show help
show_help() {
    echo "Ultimate Search Processor - Build and Start Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --build-only    Only build the Docker image, don't start services"
    echo "  --start-only    Only start services (assumes image is already built)"
    echo "  --no-wait       Don't wait for services to be ready"
    echo "  --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Build and start with full health checks"
    echo "  $0 --build-only       # Only build the Docker image"
    echo "  $0 --start-only       # Only start services"
    echo "  $0 --no-wait          # Build and start without waiting"
}

# Parse command line arguments
BUILD_ONLY=false
START_ONLY=false
NO_WAIT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        --start-only)
            START_ONLY=true
            shift
            ;;
        --no-wait)
            NO_WAIT=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
main() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Ultimate Search Processor Setup${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    
    # Pre-flight checks
    check_docker
    check_directory
    
    # Build image if not start-only
    if [ "$START_ONLY" = false ]; then
        build_image
    fi
    
    # Start services if not build-only
    if [ "$BUILD_ONLY" = false ]; then
        start_services
        show_status
        
        # Wait for services if not no-wait
        if [ "$NO_WAIT" = false ]; then
            wait_for_services
            check_health
        fi
        
        show_access_info
    else
        echo ""
        print_success "Build completed successfully!"
        echo ""
        echo -e "${BLUE}Next steps:${NC}"
        echo "  Run container:     docker run -p 8000:8000 ultimate-search-processor:latest"
        echo "  Use docker compose: docker compose -f docker-compose.ultimate.yml up"
        echo "  Start services:    ./start.sh --start-only"
    fi
}

# Run main function
main "$@"