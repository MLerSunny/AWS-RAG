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
import logging
import shutil
from pathlib import Path
from typing import Optional, List

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_python_version() -> bool:
    """Check if Python version meets requirements."""
    required_major = 3
    required_minor = 8
    
    current_major = sys.version_info.major
    current_minor = sys.version_info.minor
    
    if current_major < required_major or (current_major == required_major and current_minor < required_minor):
        logger.error(f"Python {required_major}.{required_minor}+ is required. You have {current_major}.{current_minor}")
        return False
    return True

def setup_environment() -> bool:
    """Setup the virtual environment if it doesn't exist"""
    venv_dir = ".venv"
    if not Path(venv_dir).exists():
        logger.info("Setting up the environment for the first time...")
        
        try:
            subprocess.run([sys.executable, "setup.py"], check=True, capture_output=True, text=True)
            logger.info("Environment setup complete.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Setup failed: {e.stderr}")
            return False
    return True

def cleanup_temp_files() -> None:
    """Clean up temporary files."""
    temp_dir = "temp"
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)
            logger.info("Cleaned up temporary files")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary files: {e}")

def get_python_path() -> Optional[str]:
    """Get the path to the Python executable in the virtual environment."""
    is_windows = platform.system() == "Windows"
    if is_windows:
        python_path = os.path.join(".venv", "Scripts", "python.exe")
    else:
        python_path = os.path.join(".venv", "bin", "python")
    
    if not os.path.exists(python_path):
        logger.error(f"Python executable not found at {python_path}")
        return None
    return python_path

def activate_and_run(args: argparse.Namespace) -> bool:
    """Activate virtual environment and run the application"""
    logger.info("Starting the application...")
    
    if not check_python_version():
        return False
    
    # Prepare the command
    if args.direct:
        # Direct run without using start.py
        cmd: List[str] = [sys.executable, "-m", "app.main"]
    else:
        python_path = get_python_path()
        if not python_path:
            return False
        
        cmd = [python_path, "start.py"]
        
        # Add additional arguments
        if args.reload:
            cmd.append("--reload")
    
    try:
        # Clean up before running
        cleanup_temp_files()
        
        # Run the command
        process = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        # Log output
        if process.stdout:
            logger.info(process.stdout)
        if process.stderr:
            logger.error(process.stderr)
            
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Application failed to start: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="GenAI Document Ingestion API Runner")
    parser.add_argument("--direct", action="store_true", 
                      help="Run the app.main module directly (no start.py)")
    parser.add_argument("--reload", action="store_true", 
                      help="Enable auto-reload for development")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug logging")
    return parser.parse_args()

def main() -> int:
    """Main entry point"""
    logger.info("GenAI Document Ingestion API Runner")
    logger.info("----------------------------------")
    
    args = parse_arguments()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Setup if not direct run
    if not args.direct and not setup_environment():
        return 1
    
    # Run the application
    if not activate_and_run(args):
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 