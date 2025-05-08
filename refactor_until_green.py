#!/usr/bin/env python3
"""
Cursor Agent Refactoring Script
-------------------------------
This script automates Cursor's Agent to refactor code until all tests pass.
It runs in a loop, executing refactoring commands and then checking test results.
"""

import os
import sys
import subprocess
import json
import time
import logging
import argparse
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("refactor_logs.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class CursorRefactorAgent:
    def __init__(
        self,
        project_root: str,
        max_iterations: int = 10,
        test_command: str = "pytest",
        lint_command: str = "flake8",
        target_directories: Optional[List[str]] = None,
    ):
        self.project_root = Path(project_root).resolve()
        self.max_iterations = max_iterations
        self.test_command = test_command
        self.lint_command = lint_command
        self.target_directories = target_directories or ["genai-doc-ingestion"]
        self.iteration = 0
        self.changes_made = []
        self.start_time = datetime.now()
        
        # Create backup directory
        self.backup_dir = self.project_root / "refactor_backups" / self.start_time.strftime("%Y%m%d_%H%M%S")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
    def backup_code(self) -> None:
        """Create a backup of the current codebase state"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"backup_{self.iteration}_{timestamp}"
        
        logger.info(f"Creating backup at {backup_path}")
        
        for target_dir in self.target_directories:
            target_path = self.project_root / target_dir
            if target_path.exists():
                backup_target = backup_path / target_dir
                backup_target.mkdir(parents=True, exist_ok=True)
                
                # Use subprocess to copy files (platform-independent)
                if os.name == 'nt':  # Windows
                    subprocess.run(
                        f'xcopy "{target_path}" "{backup_target}" /E /I /H /Y',
                        shell=True, check=False
                    )
                else:  # Unix-like
                    subprocess.run(
                        f'cp -r "{target_path}/." "{backup_target}"',
                        shell=True, check=False
                    )
        
        logger.info(f"Backup completed for iteration {self.iteration}")
        
    def is_green(self) -> Tuple[bool, Dict]:
        """
        Check if the codebase is 'green':
        1. All tests pass
        2. No linting errors
        
        Returns a tuple of (is_green, results_dict)
        """
        results = {
            "tests_pass": False,
            "lint_pass": False,
            "test_output": "",
            "lint_output": ""
        }
        
        # Run tests
        logger.info(f"Running tests: {self.test_command}")
        test_process = subprocess.run(
            self.test_command,
            shell=True,
            cwd=self.project_root,
            capture_output=True,
            text=True
        )
        results["test_output"] = test_process.stdout + test_process.stderr
        results["tests_pass"] = test_process.returncode == 0
        
        # Run linter
        logger.info(f"Running linter: {self.lint_command}")
        lint_process = subprocess.run(
            self.lint_command,
            shell=True, 
            cwd=self.project_root,
            capture_output=True,
            text=True
        )
        results["lint_output"] = lint_process.stdout + lint_process.stderr
        results["lint_pass"] = lint_process.returncode == 0
        
        is_green = results["tests_pass"] and results["lint_pass"]
        logger.info(f"Green check: Tests pass: {results['tests_pass']}, Lint pass: {results['lint_pass']}")
        
        return is_green, results
    
    def run_cursor_agent(self, refactor_instructions: str) -> Dict:
        """
        Run Cursor's agent with specific refactoring instructions.
        This requires having Cursor's CLI installed and configured.
        
        Returns dict with operation status and any output from Cursor.
        """
        logger.info(f"Running Cursor Agent with refactoring instructions")
        
        # The actual command will depend on how Cursor's CLI is configured
        # This is a placeholder that should be replaced with actual Cursor CLI commands
        cursor_command = f'cursor agent "{refactor_instructions}"'
        
        try:
            cursor_process = subprocess.run(
                cursor_command,
                shell=True,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            result = {
                "success": cursor_process.returncode == 0,
                "output": cursor_process.stdout,
                "error": cursor_process.stderr
            }
            
            if result["success"]:
                self.changes_made.append({
                    "iteration": self.iteration,
                    "instructions": refactor_instructions,
                    "timestamp": datetime.now().isoformat()
                })
                
            return result
        except subprocess.TimeoutExpired:
            logger.error("Cursor Agent operation timed out")
            return {
                "success": False,
                "output": "",
                "error": "Operation timed out after 5 minutes"
            }
        except Exception as e:
            logger.error(f"Error running Cursor Agent: {str(e)}")
            return {
                "success": False,
                "output": "",
                "error": str(e)
            }
    
    def generate_refactor_instructions(self, green_results: Dict) -> str:
        """
        Generate refactoring instructions based on test and lint results.
        This is where you'd implement logic to parse test failures and lint errors
        to create targeted refactoring instructions.
        """
        if green_results["tests_pass"] and green_results["lint_pass"]:
            return "Perform general code cleanup and optimization"
        
        instructions = []
        
        # Parse lint errors - this is a simple pattern and may need adjustment
        if not green_results["lint_pass"]:
            lint_output = green_results["lint_output"]
            # Basic parsing of lint output to identify issues
            # This is a simplified example - real parsing would be more complex
            for line in lint_output.split("\n"):
                if ".py:" in line and ":" in line:
                    instructions.append(f"Fix lint error: {line.strip()}")
        
        # Parse test failures
        if not green_results["tests_pass"]:
            test_output = green_results["test_output"]
            # Look for FAILED lines in pytest output
            for line in test_output.split("\n"):
                if "FAILED" in line and ".py::" in line:
                    instructions.append(f"Fix test failure: {line.strip()}")
        
        # If we couldn't extract specific instructions, use a general one
        if not instructions:
            return "Address test failures and lint errors in the codebase"
        
        # Limit to 3 instructions to focus effort
        return "\n".join(instructions[:3])
    
    def run_until_green(self, initial_instructions: Optional[str] = None) -> Dict:
        """
        Main loop that runs the refactoring process until green or max iterations reached.
        """
        logger.info(f"Starting refactoring process. Max iterations: {self.max_iterations}")
        
        # Initial check
        is_green, green_results = self.is_green()
        if is_green:
            logger.info("Codebase is already green! No refactoring needed.")
            return {
                "success": True,
                "iterations": 0,
                "green": True,
                "message": "Codebase is already green"
            }
            
        # Take initial backup
        self.backup_code()
        
        # Main refactoring loop
        while not is_green and self.iteration < self.max_iterations:
            self.iteration += 1
            logger.info(f"Starting iteration {self.iteration} of {self.max_iterations}")
            
            # Generate refactoring instructions
            if self.iteration == 1 and initial_instructions:
                refactor_instructions = initial_instructions
            else:
                refactor_instructions = self.generate_refactor_instructions(green_results)
                
            logger.info(f"Refactoring instructions: {refactor_instructions}")
            
            # Run Cursor Agent
            cursor_result = self.run_cursor_agent(refactor_instructions)
            
            if not cursor_result["success"]:
                logger.error(f"Cursor Agent failed: {cursor_result['error']}")
                continue
                
            # Check if we're green now
            is_green, green_results = self.is_green()
            
            # Backup after changes
            self.backup_code()
            
            if is_green:
                logger.info(f"Success! Codebase is green after {self.iteration} iterations.")
                break
                
            logger.info(f"Codebase still not green after iteration {self.iteration}.")
            
        # Generate final report
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        return {
            "success": True,
            "green": is_green,
            "iterations": self.iteration,
            "duration_seconds": duration,
            "changes": self.changes_made,
            "message": f"Completed {self.iteration} iterations. Codebase is {'green' if is_green else 'not green'}."
        }
            
    def generate_report(self, result: Dict) -> str:
        """Generate a detailed report of the refactoring process"""
        report = [
            "# Cursor Agent Refactoring Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Project: {self.project_root}",
            "",
            f"## Summary",
            f"- Status: {'GREEN' if result['green'] else 'NOT GREEN'}",
            f"- Iterations performed: {result['iterations']}/{self.max_iterations}",
            f"- Duration: {result['duration_seconds']:.2f} seconds",
            "",
            "## Changes Made"
        ]
        
        for change in self.changes_made:
            report.append(f"### Iteration {change['iteration']} - {change['timestamp']}")
            report.append(f"Instructions:")
            report.append(f"```")
            report.append(change["instructions"])
            report.append(f"```")
            report.append("")
            
        report.append("## Final Status")
        report.append(f"Message: {result['message']}")
        
        return "\n".join(report)
        
def main():
    parser = argparse.ArgumentParser(description="Run Cursor Agent refactoring until codebase is green")
    parser.add_argument("--project-root", default=".", help="Path to project root directory")
    parser.add_argument("--max-iterations", type=int, default=10, help="Maximum number of refactoring iterations")
    parser.add_argument("--test-command", default="pytest", help="Command to run tests")
    parser.add_argument("--lint-command", default="flake8", help="Command to run linter")
    parser.add_argument("--target-dirs", nargs="+", default=["genai-doc-ingestion"], help="Target directories to refactor")
    parser.add_argument("--initial-instructions", help="Initial refactoring instructions", 
                        default="Refactor the codebase to fix any failing tests and lint errors")
    
    args = parser.parse_args()
    
    agent = CursorRefactorAgent(
        project_root=args.project_root,
        max_iterations=args.max_iterations,
        test_command=args.test_command,
        lint_command=args.lint_command,
        target_directories=args.target_dirs
    )
    
    try:
        result = agent.run_until_green(args.initial_instructions)
        
        # Generate and save report
        report = agent.generate_report(result)
        report_path = Path("refactor_report.md")
        report_path.write_text(report)
        
        print(f"Refactoring complete. Report saved to {report_path}")
        
        # Return success code based on whether we reached green state
        sys.exit(0 if result["green"] else 1)
        
    except KeyboardInterrupt:
        print("\nRefactoring process interrupted by user")
        sys.exit(2)
    except Exception as e:
        logger.exception("Error in refactoring process")
        print(f"Error: {str(e)}")
        sys.exit(3)

if __name__ == "__main__":
    main() 