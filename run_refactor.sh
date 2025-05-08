#!/bin/bash
# Cursor Agent Refactoring Automation Runner
# This shell script runs the refactor_until_green.py script with common options

echo "Cursor Agent Refactoring Automation"
echo "-----------------------------------"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Default arguments
PROJECT_ROOT="."
MAX_ITERATIONS=10
TEST_COMMAND="pytest"
LINT_COMMAND="flake8"
TARGET_DIRS="genai-doc-ingestion"
INSTRUCTIONS="Refactor the codebase to fix any failing tests and lint errors"

# Run the refactoring script
echo "Starting refactoring process..."
python3 refactor_until_green.py \
    --project-root "$PROJECT_ROOT" \
    --max-iterations $MAX_ITERATIONS \
    --test-command "$TEST_COMMAND" \
    --lint-command "$LINT_COMMAND" \
    --target-dirs $TARGET_DIRS \
    --initial-instructions "$INSTRUCTIONS"

# Check exit code
exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo "Refactoring completed successfully. Codebase is green!"
elif [ $exit_code -eq 1 ]; then
    echo "Refactoring completed but codebase is not green. See report for details."
elif [ $exit_code -eq 2 ]; then
    echo "Refactoring process was interrupted by user."
else
    echo "Error occurred during refactoring. Check logs for details."
fi

echo "See refactor_report.md for detailed results." 