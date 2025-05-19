#!/usr/bin/env python
"""
Test imports after code cleanup to ensure no modules are missing.
"""
import sys
import os
import traceback

# Add project root to path
sys.path.insert(0, os.path.abspath("."))

def test_import(module_path, expected_error=False):
    """Test importing a module and report success or failure."""
    try:
        print(f"Attempting to import {module_path}...")
        module = __import__(module_path, fromlist=["*"])
        if expected_error:
            print(f"UNEXPECTED SUCCESS: {module_path} imported successfully but error was expected")
            return False
        print(f"SUCCESS: {module_path} imported successfully")
        return True
    except ImportError as e:
        if expected_error:
            print(f"EXPECTED ERROR: {module_path} import failed as expected")
            return True
        print(f"ERROR: Failed to import {module_path}")
        print(f"Exception: {str(e)}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"UNEXPECTED ERROR: Error importing {module_path}")
        print(f"Exception: {str(e)}")
        traceback.print_exc()
        return False

def run_tests():
    """Run import tests for various modules."""
    results = []
    
    # Test imports that should succeed
    results.append(test_import("app.services.experiment.ab_testing"))
    results.append(test_import("app.services.finetune.bedrock_finetune"))
    
    # Test imports that should fail (removed modules)
    results.append(test_import("app.services.evaluation.ab_testing", expected_error=True))
    results.append(test_import("app.services.fine_tuning.bedrock_tuning", expected_error=True))
    
    # Summary
    total = len(results)
    passed = sum(results)
    print(f"\nTEST SUMMARY: {passed}/{total} tests passed")
    if passed == total:
        print("All import tests passed successfully!")
        return 0
    else:
        print("Some import tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(run_tests()) 