@echo off
echo GenAI Document Ingestion API Runner

REM Check if virtual environment exists
if not exist ".venv" (
    echo Setting up the environment for the first time...
    python setup.py
    if %ERRORLEVEL% neq 0 (
        echo Setup failed, please check the error message above.
        pause
        exit /b 1
    )
)

REM Activate virtual environment and run the application
echo Starting the application...
call .venv\Scripts\activate.bat
python start.py --reload

if %ERRORLEVEL% neq 0 (
    echo Application failed to start, please check the error message above.
    pause
)

pause 