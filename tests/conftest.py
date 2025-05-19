"""
Configuration for the test suite
"""
import os
import sys
import pytest
from pathlib import Path

# Add project root to path for proper imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Fixture for temporary test files
@pytest.fixture
def temp_test_file(tmp_path):
    """Creates a temporary file for testing"""
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("Test content")
    return file_path

# Fixture for application test client
@pytest.fixture
def test_client():
    """Creates a test client for the application"""
    from app.main import app
    from fastapi.testclient import TestClient
    
    client = TestClient(app)
    return client

# Fixture for environment variables
@pytest.fixture
def env_setup():
    """Sets up test environment variables"""
    old_env = dict(os.environ)
    os.environ["AWS_REGION"] = "us-east-1"
    os.environ["TESTING"] = "True"
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(old_env) 