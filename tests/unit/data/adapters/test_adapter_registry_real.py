# -*- coding: utf-8 -*-
"""
适配器注册表真实实现测试
测试 AdapterRegistry 的核心功能
"""

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

# Mock src.data.adapters
if "src.data.adapters" not in sys.modules:
    mock_module = Mock()
    mock_module.BaseAdapter = Mock()
    mock_module.DataAdapter = Mock()
    mock_module.AdapterFactory = Mock()
    sys.modules["src.data.adapters"] = mock_module


import pytest
from src.data.adapters import (
    AdapterRegistry,
    AdapterInfo,
    AdapterStatus
)
from src.data.adapters import (
    GenericAdapter,
    MarketAdapter,
    AdapterConfig,
    AdapterType
)


@pytest.fixture
def registry():
    """创建适配器注册表实例"""
    return AdapterRegistry()


@pytest.fixture
def adapter_info():
    """创建适配器信息"""
    return AdapterInfo(
        name="test_stock_adapter",
        adapter_type=AdapterType.STOCK.value,
        status=AdapterStatus.AVAILABLE,
        version="1.0.0",
        description="测试股票适配器"
    )


@pytest.fixture
def adapter_config():
    """创建适配器配置"""
    return AdapterConfig(
        name="test_stock_adapter",
        adapter_type=AdapterType.STOCK.value
    )


def test_registry_initialization(registry):
    """测试注册表初始化"""
    assert len(registry.list_adapters()) == 0
    assert len(registry.get_active_adapters()) == 0


def test_register_adapter(registry, adapter_info):
    """测试注册适配器"""
    result = registry.register_adapter(
        "test_stock_adapter",
        GenericAdapter,
        adapter_info
    )
    
    assert result is True
    assert "test_stock_adapter" in registry.list_adapters()
    assert registry.get_adapter_class("test_stock_adapter") == GenericAdapter
    assert registry.get_adapter_info("test_stock_adapter") == adapter_info


def test_register_duplicate_adapter(registry, adapter_info):
    """测试重复注册适配器（应覆盖）"""
    registry.register_adapter("test_adapter", GenericAdapter, adapter_info)
    
    # 再次注册同名适配器
    new_info = AdapterInfo(
        name="test_adapter",
        adapter_type=AdapterType.CRYPTO.value,
        status=AdapterStatus.AVAILABLE
    )
    result = registry.register_adapter("test_adapter", MarketAdapter, new_info)
    
    assert result is True
    assert registry.get_adapter_class("test_adapter") == MarketAdapter


def test_unregister_adapter(registry, adapter_info):
    """测试注销适配器"""
    registry.register_adapter("test_adapter", GenericAdapter, adapter_info)
    registry.activate_adapter("test_adapter", AdapterConfig(name="test_adapter"))
    
    result = registry.unregister_adapter("test_adapter")
    assert result is True
    assert "test_adapter" not in registry.list_adapters()
    assert "test_adapter" not in registry.get_active_adapters()


def test_unregister_nonexistent_adapter(registry):
    """测试注销不存在的适配器"""
    result = registry.unregister_adapter("nonexistent")
    assert result is False


def test_list_adapters_by_type(registry, adapter_info):
    """测试按类型列出适配器"""
    # 注册不同类型的适配器
    stock_info = AdapterInfo(
        name="stock_adapter",
        adapter_type=AdapterType.STOCK.value,
        status=AdapterStatus.AVAILABLE
    )
    crypto_info = AdapterInfo(
        name="crypto_adapter",
        adapter_type=AdapterType.CRYPTO.value,
        status=AdapterStatus.AVAILABLE
    )
    
    registry.register_adapter("stock_adapter", GenericAdapter, stock_info)
    registry.register_adapter("crypto_adapter", GenericAdapter, crypto_info)
    
    stock_adapters = registry.list_adapters_by_type(AdapterType.STOCK.value)
    assert "stock_adapter" in stock_adapters
    assert "crypto_adapter" not in stock_adapters


def test_create_adapter(registry, adapter_info, adapter_config):
    """测试创建适配器实例"""
    registry.register_adapter("test_adapter", GenericAdapter, adapter_info)
    
    adapter = registry.create_adapter("test_adapter", adapter_config)
    assert adapter is not None
    assert isinstance(adapter, GenericAdapter)
    assert "test_adapter" in registry.get_active_adapters()


def test_create_nonexistent_adapter(registry, adapter_config):
    """测试创建不存在的适配器"""
    adapter = registry.create_adapter("nonexistent", adapter_config)
    assert adapter is None


def test_activate_adapter(registry, adapter_info, adapter_config):
    """测试激活适配器"""
    registry.register_adapter("test_adapter", GenericAdapter, adapter_info)
    
    result = registry.activate_adapter("test_adapter", adapter_config)
    assert result is True
    assert "test_adapter" in registry.get_active_adapters()
    assert registry.get_adapter_status("test_adapter") == AdapterStatus.AVAILABLE


def test_activate_already_active_adapter(registry, adapter_info, adapter_config):
    """测试激活已激活的适配器"""
    registry.register_adapter("test_adapter", GenericAdapter, adapter_info)
    registry.activate_adapter("test_adapter", adapter_config)
    
    # 再次激活
    result = registry.activate_adapter("test_adapter", adapter_config)
    assert result is True


def test_deactivate_adapter(registry, adapter_info, adapter_config):
    """测试停用适配器"""
    registry.register_adapter("test_adapter", GenericAdapter, adapter_info)
    registry.activate_adapter("test_adapter", adapter_config)
    
    result = registry.deactivate_adapter("test_adapter")
    assert result is True
    assert "test_adapter" not in registry.get_active_adapters()


def test_deactivate_nonexistent_adapter(registry):
    """测试停用不存在的适配器"""
    result = registry.deactivate_adapter("nonexistent")
    assert result is False


def test_get_adapter_status(registry, adapter_info, adapter_config):
    """测试获取适配器状态"""
    registry.register_adapter("test_adapter", GenericAdapter, adapter_info)
    
    # 未激活状态
    status = registry.get_adapter_status("test_adapter")
    assert status == AdapterStatus.UNAVAILABLE
    
    # 激活后状态
    registry.activate_adapter("test_adapter", adapter_config)
    status = registry.get_adapter_status("test_adapter")
    assert status == AdapterStatus.AVAILABLE
    
    # 不存在的适配器
    status = registry.get_adapter_status("nonexistent")
    assert status == AdapterStatus.ERROR


def test_get_registry_info(registry, adapter_info, adapter_config):
    """测试获取注册表信息"""
    registry.register_adapter("stock_adapter", GenericAdapter, adapter_info)
    registry.register_adapter("crypto_adapter", GenericAdapter, adapter_info)
    registry.activate_adapter("stock_adapter", adapter_config)
    
    info = registry.get_registry_info()
    assert info['total_adapters'] == 2
    assert info['active_adapters'] == 1
    assert AdapterType.STOCK.value in info['adapter_types']
    assert 'stock_adapter' in info['adapters']
    assert info['adapters']['stock_adapter']['status'] == AdapterStatus.AVAILABLE.value


