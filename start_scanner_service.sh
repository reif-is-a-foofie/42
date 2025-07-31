#!/bin/bash

# 42 Autonomous Scanner Production Service
# Run the scanner as a background service with proper logging and monitoring

echo "🚀 Starting 42 Autonomous Scanner Production Service"
echo "=================================================="

# Configuration
SCANNER_LOG_FILE="scanner.log"
SCANNER_PID_FILE="scanner.pid"
SCANNER_SCRIPT="deploy_scanner.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if scanner is running
check_scanner_status() {
    if [ -f "$SCANNER_PID_FILE" ]; then
        PID=$(cat "$SCANNER_PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Scanner is running (PID: $PID)${NC}"
            return 0
        else
            echo -e "${YELLOW}⚠️  Scanner PID file exists but process not running${NC}"
            rm -f "$SCANNER_PID_FILE"
            return 1
        fi
    else
        echo -e "${RED}❌ Scanner is not running${NC}"
        return 1
    fi
}

# Function to start scanner
start_scanner() {
    echo "🔧 Starting autonomous scanner..."
    
    # Check if already running
    if check_scanner_status > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  Scanner is already running${NC}"
        return 1
    fi
    
    # Start scanner in background
    nohup python3 "$SCANNER_SCRIPT" > "$SCANNER_LOG_FILE" 2>&1 &
    PID=$!
    
    # Save PID
    echo $PID > "$SCANNER_PID_FILE"
    
    echo -e "${GREEN}✅ Scanner started (PID: $PID)${NC}"
    echo "📝 Logs: $SCANNER_LOG_FILE"
    echo "🔄 Check status with: $0 status"
    
    # Wait a moment and check if it started successfully
    sleep 3
    if check_scanner_status > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Scanner is running successfully${NC}"
    else
        echo -e "${RED}❌ Scanner failed to start. Check logs: $SCANNER_LOG_FILE${NC}"
        return 1
    fi
}

# Function to stop scanner
stop_scanner() {
    echo "🛑 Stopping autonomous scanner..."
    
    if [ -f "$SCANNER_PID_FILE" ]; then
        PID=$(cat "$SCANNER_PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            kill $PID
            echo -e "${GREEN}✅ Scanner stopped (PID: $PID)${NC}"
        else
            echo -e "${YELLOW}⚠️  Scanner process not found${NC}"
        fi
        rm -f "$SCANNER_PID_FILE"
    else
        echo -e "${YELLOW}⚠️  No PID file found${NC}"
    fi
}

# Function to restart scanner
restart_scanner() {
    echo "🔄 Restarting autonomous scanner..."
    stop_scanner
    sleep 2
    start_scanner
}

# Function to show logs
show_logs() {
    if [ -f "$SCANNER_LOG_FILE" ]; then
        echo "📝 Scanner logs (last 50 lines):"
        echo "=================================="
        tail -n 50 "$SCANNER_LOG_FILE"
    else
        echo -e "${YELLOW}⚠️  No log file found${NC}"
    fi
}

# Function to show status
show_status() {
    echo "📊 Scanner Status"
    echo "================"
    
    if check_scanner_status; then
        PID=$(cat "$SCANNER_PID_FILE")
        echo "🟢 Status: Running"
        echo "🆔 PID: $PID"
        echo "📝 Logs: $SCANNER_LOG_FILE"
        
        # Show recent activity
        if [ -f "$SCANNER_LOG_FILE" ]; then
            echo ""
            echo "📈 Recent Activity:"
            tail -n 5 "$SCANNER_LOG_FILE" | grep -E "(Crawling|Discovered|Added)" || echo "No recent activity"
        fi
    else
        echo "🔴 Status: Stopped"
    fi
}

# Function to add custom URLs
add_urls() {
    if [ $# -eq 0 ]; then
        echo -e "${RED}❌ Please provide URLs to add${NC}"
        echo "Usage: $0 add-urls <url1> <url2> ..."
        return 1
    fi
    
    echo "🌱 Adding custom URLs to scanner..."
    python3 "$SCANNER_SCRIPT" --custom-urls "$@" --add-only
    
    if check_scanner_status > /dev/null 2>&1; then
        echo -e "${GREEN}✅ URLs added to running scanner${NC}"
    else
        echo -e "${YELLOW}⚠️  Scanner not running. URLs will be used when scanner starts.${NC}"
    fi
}

# Function to show help
show_help() {
    echo "42 Autonomous Scanner Service"
    echo "============================"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  start       - Start the scanner service"
    echo "  stop        - Stop the scanner service"
    echo "  restart     - Restart the scanner service"
    echo "  status      - Show scanner status"
    echo "  logs        - Show recent logs"
    echo "  add-urls    - Add custom seed URLs"
    echo "  help        - Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 start"
    echo "  $0 status"
    echo "  $0 add-urls https://example.com https://blog.example.com"
    echo "  $0 logs"
}

# Main script logic
case "${1:-help}" in
    start)
        start_scanner
        ;;
    stop)
        stop_scanner
        ;;
    restart)
        restart_scanner
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    add-urls)
        shift
        add_urls "$@"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}❌ Unknown command: $1${NC}"
        show_help
        exit 1
        ;;
esac 