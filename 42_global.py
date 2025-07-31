#!/usr/bin/env python3
"""
Global 42 Command

This script makes the 42 system globally accessible from anywhere on the machine.
Usage: 42 [command] [args]
"""

import sys
import os
import subprocess
from pathlib import Path

# Find the 42 project directory
def find_42_project():
    """Find the 42 project directory from anywhere on the system."""
    # Common locations to search
    search_paths = [
        Path.home() / "42" / "42",  # ~/42/42
        Path.home() / "42",          # ~/42
        Path.cwd() / "42",           # ./42 (if run from project root)
        Path("/Users/reif/42/42"),   # Your specific path
    ]
    
    for path in search_paths:
        if path.exists() and (path / "42" / "__init__.py").exists():
            return path
    
    # If not found in common locations, try to find it
    for root, dirs, files in os.walk(Path.home()):
        if "42" in dirs:
            potential_path = Path(root) / "42"
            if (potential_path / "42" / "__init__.py").exists():
                return potential_path
    
    raise FileNotFoundError("Could not find 42 project directory")

def main():
    """Main entry point for global 42 command."""
    try:
        # Find the 42 project
        project_path = find_42_project()
        
        # Change to project directory
        os.chdir(project_path)
        
        # Get command line arguments
        args = sys.argv[1:] if len(sys.argv) > 1 else ["help"]
        
        # Special handling for scanner commands
        if args[0] == "scanner":
            # Use the scanner service script
            scanner_script = project_path / "start_scanner_service.sh"
            if scanner_script.exists():
                cmd = [str(scanner_script)] + args[1:]
            else:
                # Fallback to Python module
                cmd = [sys.executable, "-m", "42"] + args
        else:
            # Use the 42 Python module
            cmd = [sys.executable, "-m", "42"] + args
        
        # Execute the command
        result = subprocess.run(cmd, cwd=project_path)
        sys.exit(result.returncode)
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please ensure the 42 project is installed in one of these locations:")
        print("  - ~/42/42")
        print("  - ~/42")
        print("  - /Users/reif/42/42")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running 42 command: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 