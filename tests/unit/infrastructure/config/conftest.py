import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_loader():
    """模拟配置加载器"""
    mock = MagicMock()
    mock.load.return_value = {"key": "value"}  # 默认返回值
    return mock

@pytest.fixture
def mock_validator():
    """模拟配置验证器"""
    mock = MagicMock()
    mock.validate.return_value = True  # 默认验证通过
    return mock


@pytest.fixture
def mock_log_manager():
    """日志管理器mock"""
    with patch('src.infrastructure.config.config_manager.LogManager') as mock:
        mock_logger = MagicMock()
        mock.get_logger.return_value = mock_logger
        yield mock

@pytest.fixture
def config_manager(mock_loader, mock_validator, mock_log_manager):
    """配置管理器fixture"""
    from src.infrastructure.config.config_manager import ConfigManager
    return ConfigManager(
        loader_service=mock_loader,
        validator=mock_validator
    )