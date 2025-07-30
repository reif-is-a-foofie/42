#!/bin/bash

# 42 Setup Script
# Makes it easy to work with hidden configuration files

echo "🚀 42 Setup Script"
echo "=================="

# Function to show usage
show_usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  install     - Install Python dependencies"
    echo "  start       - Start Docker services"
    echo "  stop        - Stop Docker services"
    echo "  create      - Initialize 42 system"
    echo "  test        - Run all tests"
    echo "  clean       - Clean up temporary files"
    echo "  help        - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 install"
    echo "  $0 start"
    echo "  $0 create"
}

# Install dependencies
install() {
    echo "📦 Installing Python dependencies..."
    pip install -r .config/requirements.txt
    echo "✅ Dependencies installed!"
}

# Start services
start() {
    echo "🐳 Starting Docker services..."
    docker-compose -f .config/docker-compose.yml up -d
    echo "✅ Services started!"
}

# Stop services
stop() {
    echo "🛑 Stopping Docker services..."
    docker-compose -f .config/docker-compose.yml down
    echo "✅ Services stopped!"
}

# Create system
create() {
    echo "🔧 Initializing 42 system..."
    python3 -m 42 create
    echo "✅ System initialized!"
}

# Run tests
test() {
    echo "🧪 Running tests..."
    python3 -m pytest tests/ -v
    echo "✅ Tests completed!"
}

# Clean up
clean() {
    echo "🧹 Cleaning up temporary files..."
    find . -name "*.pyc" -delete
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
    echo "✅ Cleanup completed!"
}

# Main script logic
case "${1:-help}" in
    install)
        install
        ;;
    start)
        start
        ;;
    stop)
        stop
        ;;
    create)
        create
        ;;
    test)
        test
        ;;
    clean)
        clean
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        echo "❌ Unknown command: $1"
        show_usage
        exit 1
        ;;
esac 