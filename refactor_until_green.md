# Cursor Agent Refactoring Automation

This utility automates the process of using Cursor's Agent to refactor code until all tests pass and linting checks are clean (a "green" codebase).

## Overview

The script works by:

1. Checking if the codebase is already "green" (passing tests and linting)
2. If not, it runs Cursor's Agent with refactoring instructions
3. After each refactoring attempt, it checks if the codebase is green
4. If not, it generates new refactoring instructions based on the failures
5. This process continues until the codebase is green or the maximum number of iterations is reached

## Prerequisites

- Python 3.7+
- Cursor IDE with Agent functionality 
- Required Python packages: `argparse`, `pathlib`
- Test framework (default: pytest)
- Linter (default: flake8)

## Installation

1. Place the `refactor_until_green.py` script in your project's root directory
2. Make the script executable: `chmod +x refactor_until_green.py` (Unix-based systems)

## Usage

Basic usage:

```bash
python refactor_until_green.py
```

With custom options:

```bash
python refactor_until_green.py --project-root /path/to/project --max-iterations 5 --test-command "python -m pytest" --lint-command "flake8 genai-doc-ingestion" --target-dirs genai-doc-ingestion --initial-instructions "Fix test failures and improve code quality"
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--project-root` | Path to project root directory | Current directory (`.`) |
| `--max-iterations` | Maximum number of refactoring iterations | 10 |
| `--test-command` | Command to run tests | `pytest` |
| `--lint-command` | Command to run linter | `flake8` |
| `--target-dirs` | Target directories to refactor (space separated) | `genai-doc-ingestion` |
| `--initial-instructions` | Initial refactoring instructions | "Refactor the codebase to fix any failing tests and lint errors" |

## Output

The script produces:

1. A log file (`refactor_logs.log`) with detailed information about each step
2. A backup of the codebase before and after each refactoring attempt in the `refactor_backups` directory
3. A final report (`refactor_report.md`) summarizing the refactoring process and results

## Customization

### Adjusting the Success Criteria

By default, "green" is defined as all tests passing and no linting errors. You can modify the `is_green()` method in the script to change this definition.

### Custom Refactoring Logic

The `generate_refactor_instructions()` method can be customized to create more sophisticated refactoring instructions based on specific patterns in test failures or linting errors.

## Notes on Cursor Agent Integration

This script contains a placeholder for the Cursor Agent command:

```python
cursor_command = f'cursor agent "{refactor_instructions}"'
```

You may need to adjust this command based on how Cursor's CLI is configured on your system.

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success (codebase is green) |
| 1 | Script completed but codebase is not green |
| 2 | User interrupted the process |
| 3 | An error occurred during execution | 