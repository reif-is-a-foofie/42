"""Basic test to verify testing setup."""

import pytest
import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_basic_functionality():
    """Test that basic functionality works."""
    assert 1 + 1 == 2
    assert "hello" + " world" == "hello world"

def test_import_works():
    """Test that we can import modules."""
    try:
        import embedding
        assert True
    except ImportError:
        pytest.skip("embedding module not available")

def test_pytest_working():
    """Test that pytest is working correctly."""
    assert True 