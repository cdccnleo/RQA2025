# tests/data/conftest.py
import os
import re
import shutil
from pathlib import Path
import pytest
from unittest.mock import patch
from src.data.data_manager import DataManager


@pytest.fixture(scope="session", autouse=True)
def setup_temp_dir():
    """使用项目目录下的临时目录"""
    temp_dir = Path("pytest_temp_dir")

    # 确保目录存在
    temp_dir.mkdir(exist_ok=True)

    # 设置临时目录环境变量
    os.environ["PYTEST_TEMP_DIR"] = str(temp_dir.absolute())

    yield

    # 测试完成后清理
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def mock_config(tmp_path):
    config = {
        "General": {"max_concurrent_workers": "4"},
        "Stock": {"save_path": str(tmp_path / "stock"), "max_retries": "3", "cache_days": "30","frequency": "daily"},
        "News": {"save_path": str(tmp_path / "news"), "max_retries": "5", "cache_days": "7"},
        "Financial": {
            "save_path": str(tmp_path / "financial"),
            "statement_types": "income,balance,cashflow",
            "max_retries": "3",
            "cache_days": "30"
        },
        "Index": {"save_path": str(tmp_path / "index"), "max_retries": "3", "cache_days": "30"},
        "Metadata": {"save_path": str(tmp_path / "metadata")}
    }
    return config

@pytest.fixture
def data_manager(mock_config):
    with patch("src.data.data_manager.ConfigParser") as mock_config_parser:
        mock_config_parser.return_value.read_dict.return_value = None
        manager = DataManager()
        manager.config = mock_config_parser.return_value
        
        # 更完善的mock行为
        def config_get(section, key, fallback=None):
            try:
                return mock_config[section][key]
            except KeyError:
                return fallback
                
        def config_getint(section, key, fallback=None):
            try:
                return int(mock_config[section][key])
            except (KeyError, ValueError):
                return fallback if fallback is None else int(fallback)
                
        manager.config.get.side_effect = config_get
        manager.config.getint.side_effect = config_getint
        manager.config.has_section.side_effect = lambda section: section in mock_config
        manager.config.has_option.side_effect = lambda section, key: section in mock_config and key in mock_config[section]
        
        return manager


@pytest.fixture
def mock_akshare(mocker):
    """模拟akshare的财务数据接口"""
    return mocker.patch("akshare.stock_financial_analysis_indicator")


@pytest.fixture
def default_stock_config():
    return {
        "save_path": "data/test/stock",  # 测试专用路径
        "max_retries": "3",
        "frequency": "daily"
    }


def assert_loader_error(error_msg, patterns):
    for pattern in patterns:
        if re.search(pattern, error_msg):
            return
    pytest.fail(f"Error message '{error_msg}' doesn't match any patterns: {patterns}")


@pytest.fixture
def minimal_config():
    """提供最小化配置"""
    return {
        "Stock": {
            "save_path": "data/stock",
            "max_retries": "3",  # 确保所有数值都是字符串
            "cache_days": "30"
        },
        "News": {
            "save_path": "data/news",
            "max_retries": "5",
            "cache_days": "7"
        },
        "Financial": {
            "save_path": "data/fundamental",
            "max_retries": "3",
            "cache_days": "30"
        },
        "Index": {
            "save_path": "data/index",
            "max_retries": "3",
            "cache_days": "30"
        }
    }