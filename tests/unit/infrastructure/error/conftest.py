import pytest
from src.infrastructure.error.error_handler import ErrorHandler

@pytest.fixture
def error_handler():
    """ErrorHandler测试夹具"""
    return ErrorHandler(logger=None)
