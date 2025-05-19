#!/usr/bin/env python3
"""
Universal run script for GenAI Document Ingestion API
Works on both Windows and Linux platforms
"""
import os
import sys
import subprocess
import argparse
import platform
from pathlib import Path

def setup_environment():
    """Setup the virtual environment if it doesn't exist"""
    venv_dir = ".venv"
    if not Path(venv_dir).exists():
        print("Setting up the environment for the first time...")
        
        try:
            subprocess.run([sys.executable, "setup.py"], check=True)
            print("Environment setup complete.")
        except subprocess.CalledProcessError:
            print("Setup failed, please check the error message above.")
            return False
    return True

def activate_and_run(args):
    """Activate virtual environment and run the application"""
    print("Starting the application...")
    
    # Determine platform-specific commands
    is_windows = platform.system() == "Windows"
    
    # Prepare the command
    if args.direct:
        # Direct run without using start.py
        cmd = [sys.executable, "-m", "app.main"]
    else:
        # First activate the virtual environment
        if is_windows:
            python_path = os.path.join(".venv", "Scripts", "python.exe")
        else:
            python_path = os.path.join(".venv", "bin", "python")
        
        cmd = [python_path, "start.py"]
        
        # Add additional arguments
        if args.reload:
            cmd.append("--reload")
    
    try:
        # Run the command
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        print("Application failed to start, please check the error message above.")
        return False

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="GenAI Document Ingestion API Runner")
    parser.add_argument("--direct", action="store_true", 
                      help="Run the app.main module directly (no start.py)")
    parser.add_argument("--reload", action="store_true", 
                      help="Enable auto-reload for development")
    return parser.parse_args()

def main():
    """Main entry point"""
    print("GenAI Document Ingestion API Runner")
    print("----------------------------------")
    
    args = parse_arguments()
    
    # Setup if not direct run
    if not args.direct and not setup_environment():
        return 1
    
    # Run the application
    if not activate_and_run(args):
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 