#!/bin/bash

echo "GenAI Document Ingestion API Runner"

# Make script executable if not already
chmod +x "$0"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Setting up the environment for the first time..."
    python3 setup.py
    if [ $? -ne 0 ]; then
        echo "Setup failed, please check the error message above."
        read -p "Press Enter to continue..."
        exit 1
    fi
fi

# Activate virtual environment and run the application
echo "Starting the application..."
source .venv/bin/activate
python start.py --reload

if [ $? -ne 0 ]; then
    echo "Application failed to start, please check the error message above."
    read -p "Press Enter to continue..."
fi 