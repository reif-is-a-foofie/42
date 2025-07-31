#!/bin/bash

# Global 42 Installation Script
# Makes the 42 command available globally on your system

echo "🚀 Installing 42 Global Command"
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the current directory (should be the 42 project root)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GLOBAL_SCRIPT="$PROJECT_DIR/42_global.py"

echo "📁 Project directory: $PROJECT_DIR"
echo "📄 Global script: $GLOBAL_SCRIPT"

# Check if the global script exists
if [ ! -f "$GLOBAL_SCRIPT" ]; then
    echo -e "${RED}❌ Global script not found: $GLOBAL_SCRIPT${NC}"
    exit 1
fi

# Make the global script executable
chmod +x "$GLOBAL_SCRIPT"
echo -e "${GREEN}✅ Made global script executable${NC}"

# Determine the installation method
echo ""
echo "🔧 Installation Options:"
echo "1. Create symlink in /usr/local/bin (requires sudo)"
echo "2. Create symlink in ~/bin (user directory)"
echo "3. Add to PATH manually"

read -p "Choose installation method (1-3): " choice

case $choice in
    1)
        # Method 1: System-wide installation
        echo -e "${BLUE}Installing system-wide...${NC}"
        sudo ln -sf "$GLOBAL_SCRIPT" /usr/local/bin/42
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Installed to /usr/local/bin/42${NC}"
            echo "You can now run: 42 [command]"
        else
            echo -e "${RED}❌ Installation failed${NC}"
            exit 1
        fi
        ;;
    2)
        # Method 2: User directory installation
        echo -e "${BLUE}Installing to user directory...${NC}"
        mkdir -p ~/bin
        ln -sf "$GLOBAL_SCRIPT" ~/bin/42
        
        # Add to PATH if not already there
        if [[ ":$PATH:" != *":$HOME/bin:"* ]]; then
            echo 'export PATH="$HOME/bin:$PATH"' >> ~/.zshrc
            echo -e "${YELLOW}⚠️  Added ~/bin to PATH in ~/.zshrc${NC}"
            echo -e "${YELLOW}⚠️  Please restart your terminal or run: source ~/.zshrc${NC}"
        fi
        
        echo -e "${GREEN}✅ Installed to ~/bin/42${NC}"
        echo "You can now run: 42 [command]"
        ;;
    3)
        # Method 3: Manual PATH addition
        echo -e "${BLUE}Manual installation...${NC}"
        echo ""
        echo "To make 42 globally available, add this to your shell profile (~/.zshrc):"
        echo ""
        echo -e "${YELLOW}export PATH=\"$PROJECT_DIR:\$PATH\"${NC}"
        echo ""
        echo "Then restart your terminal or run:"
        echo -e "${YELLOW}source ~/.zshrc${NC}"
        echo ""
        echo "Or you can run 42 directly with:"
        echo -e "${YELLOW}python3 $GLOBAL_SCRIPT [command]${NC}"
        ;;
    *)
        echo -e "${RED}❌ Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
echo "🎉 Installation complete!"
echo ""
echo "📋 Usage Examples:"
echo "  42 help                    # Show all commands"
echo "  42 status                  # Check system status"
echo "  42 scanner start           # Start autonomous scanner"
echo "  42 scanner status          # Check scanner status"
echo "  42 search 'your query'     # Search knowledge base"
echo "  42 scanner add-urls https://example.com  # Add custom URLs"
echo ""
echo "🔍 Test the installation:"
echo "  42 --help" 