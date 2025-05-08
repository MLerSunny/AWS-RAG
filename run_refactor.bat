@echo off
REM Cursor Agent Refactoring Automation Runner
REM This batch file runs the refactor_until_green.py script with common options

echo Cursor Agent Refactoring Automation
echo -----------------------------------

REM Check if Python is available
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is not installed or not in PATH
    exit /b 1
)

REM Parse command line arguments
set PROJECT_ROOT=.
set MAX_ITERATIONS=10
set TEST_COMMAND=pytest
set LINT_COMMAND=flake8
set TARGET_DIRS=genai-doc-ingestion
set INSTRUCTIONS=Refactor the codebase to fix any failing tests and lint errors

REM Run the refactoring script
echo Starting refactoring process...
python refactor_until_green.py --project-root "%PROJECT_ROOT%" --max-iterations %MAX_ITERATIONS% --test-command "%TEST_COMMAND%" --lint-command "%LINT_COMMAND%" --target-dirs %TARGET_DIRS% --initial-instructions "%INSTRUCTIONS%"

REM Check exit code
if %ERRORLEVEL% EQU 0 (
    echo Refactoring completed successfully. Codebase is green!
) else if %ERRORLEVEL% EQU 1 (
    echo Refactoring completed but codebase is not green. See report for details.
) else if %ERRORLEVEL% EQU 2 (
    echo Refactoring process was interrupted by user.
) else (
    echo Error occurred during refactoring. Check logs for details.
)

echo See refactor_report.md for detailed results. 