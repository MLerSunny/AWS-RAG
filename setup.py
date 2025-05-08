"""
GenAI Document Ingestion API Setup Script

This script helps set up the environment for the application.
"""
import os
import subprocess
import sys
import platform

def check_python_version():
    """Check if the Python version is compatible."""
    required_major = 3
    required_minor = 8
    
    current_major = sys.version_info.major
    current_minor = sys.version_info.minor
    
    if current_major < required_major or (current_major == required_major and current_minor < required_minor):
        print(f"Error: Python {required_major}.{required_minor}+ is required. You have {current_major}.{current_minor}")
        return False
    return True

def create_virtual_env():
    """Create a virtual environment if it doesn't exist."""
    venv_dir = ".venv"
    
    if not os.path.exists(venv_dir):
        print(f"Creating virtual environment in {venv_dir}...")
        try:
            subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
            print("Virtual environment created successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error creating virtual environment: {e}")
            return False
    else:
        print(f"Virtual environment already exists in {venv_dir}")
    
    return True

def install_dependencies():
    """Install dependencies from requirements.txt."""
    print("Installing dependencies...")
    
    # Determine virtual environment activation script
    if platform.system() == "Windows":
        activate_script = os.path.join(".venv", "Scripts", "activate")
        pip_path = os.path.join(".venv", "Scripts", "pip")
    else:
        activate_script = os.path.join(".venv", "bin", "activate")
        pip_path = os.path.join(".venv", "bin", "pip")
    
    try:
        if platform.system() == "Windows":
            # On Windows, run pip directly from the virtual environment
            subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
        else:
            # On Unix-like systems, source the activation script and then run pip
            subprocess.run(
                f"source {activate_script} && pip install -r requirements.txt",
                shell=True,
                check=True
            )
        print("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False

def setup_environment():
    """Set up the environment by creating .env file if it doesn't exist."""
    if not os.path.exists(".env"):
        print("Creating default .env file...")
        with open(".env", "w") as f:
            f.write("""# AWS Configuration
AWS_REGION=us-east-1
# AWS_ACCESS_KEY_ID=
# AWS_SECRET_ACCESS_KEY=

# OpenSearch Configuration
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
OPENSEARCH_INDEX=documents

# Bedrock Configuration
BEDROCK_MODEL_ID=anthropic.claude-v2

# Application Settings
ENVIRONMENT=development
DEBUG=True
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
""")
        print(".env file created")
    else:
        print(".env file already exists")

def create_directories():
    """Create necessary directories."""
    directories = ["logs", "uploads", "temp"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")

def main():
    """Main setup process."""
    print("Starting GenAI Document Ingestion API setup...")
    
    if not check_python_version():
        sys.exit(1)
    
    if not create_virtual_env():
        sys.exit(1)
    
    if not install_dependencies():
        sys.exit(1)
    
    setup_environment()
    create_directories()
    
    print("\nSetup completed successfully!")
    
    # Print instructions for next steps
    if platform.system() == "Windows":
        activate_cmd = r".venv\Scripts\activate"
        run_cmd = "python start.py --reload"
    else:
        activate_cmd = "source .venv/bin/activate"
        run_cmd = "python start.py --reload"
    
    print("\nTo run the application:")
    print(f"1. Activate the virtual environment: {activate_cmd}")
    print(f"2. Start the application: {run_cmd}")
    print("\nOr use the following command if you're using PowerShell:")
    print(r".\.venv\Scripts\python.exe start.py --reload")

if __name__ == "__main__":
    main() 