import asyncio
import pandas as pd
from unittest.mock import Mock

# Mock数据管理器模块以绕过复杂的导入问题
mock_data_manager = Mock()
mock_data_manager.DataManager = Mock()
mock_data_manager.DataLoaderError = Exception

# 配置DataManager实例方法
mock_instance = Mock()
mock_instance.validate_all_configs.return_value = True
mock_instance.health_check.return_value = {"status": "healthy"}
mock_instance.store_data.return_value = True
mock_instance.has_data.return_value = True
mock_instance.get_metadata.return_value = {"data_type": "test", "symbol": "X"}
mock_instance.retrieve_data.return_value = pd.DataFrame({"col": [1, 2, 3]})
mock_instance.get_stats.return_value = {"total_items": 1}
mock_instance.validate_data.return_value = {"valid": True}
mock_instance.shutdown.return_value = None

mock_data_manager.DataManager.return_value = mock_instance

# Mock整个模块
import sys
sys.modules["src.data.data_manager"] = mock_data_manager


import copy

from src.data.integration.enhanced_data_integration_modules.config import IntegrationConfig


def test_integration_config_applies_defaults_when_none():
    config = IntegrationConfig()

    assert config.parallel_loading["max_workers"] == 12
    assert config.parallel_loading["thread_pool_strategy"] == "adaptive"

    assert config.cache_strategy["approach"] == "adaptive"
    assert config.cache_strategy["enable_cache_warming"] is True

    assert config.quality_monitor["enable_anomaly_detection"] is True
    assert config.quality_monitor["quality_threshold"] == 0.95

    assert config.data_manager["cache_enabled"] is True
    assert config.data_manager["enable_performance_monitoring"] is True

    assert config.performance_optimization["max_connection_pool_size"] == 50
    assert config.performance_optimization["compression_level"] == 6


def test_integration_config_respects_overrides_and_isolation():
    custom_parallel = {"max_workers": 4, "thread_pool_strategy": "fixed"}
    custom_cache = {"approach": "manual", "max_items": 10}

    config = IntegrationConfig(parallel_loading=custom_parallel, cache_strategy=custom_cache)

    assert config.parallel_loading is custom_parallel
    assert config.parallel_loading["max_workers"] == 4
    assert config.cache_strategy["approach"] == "manual"
    assert config.cache_strategy["max_items"] == 10

    assert config.quality_monitor["enable_alerting"] is True

    config.parallel_loading["max_workers"] = 1
    default_config = IntegrationConfig()
    assert default_config.parallel_loading["max_workers"] == 12


def test_integration_config_creates_independent_defaults():
    config_a = IntegrationConfig()
    config_b = IntegrationConfig()

    config_a.performance_optimization["compression_level"] = 9
    assert config_b.performance_optimization["compression_level"] == 6

    snapshot = copy.deepcopy(config_a.cache_strategy)
    config_a.cache_strategy["max_size"] = 100
    assert config_b.cache_strategy["max_size"] == 200 * 1024 * 1024
    assert snapshot["max_size"] == 200 * 1024 * 1024

