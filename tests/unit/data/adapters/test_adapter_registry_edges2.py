"""
边界测试：adapter_registry.py
测试边界情况和异常场景
"""
import asyncio
import pandas as pd
from unittest.mock import Mock, patch

# 正确导入AdapterStatus枚举
from src.data.adapters.adapter_registry import AdapterStatus, AdapterInfo, AdapterRegistry

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
from unittest.mock import Mock, MagicMock, patch

from src.data.adapters import (
    AdapterStatus,
    AdapterInfo,
    AdapterRegistry,
    get_adapter_registry,
)
from src.data.adapters import BaseAdapter, AdapterConfig


class MockAdapter(BaseAdapter):
    """Mock适配器用于测试"""
    
    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        self._connected = False
    
    def connect(self) -> bool:
        self._connected = True
        return True
    
    def disconnect(self) -> bool:
        self._connected = False
        return True
    
    def is_connected(self) -> bool:
        return self._connected
    
    def fetch_data(self, **kwargs):
        return {"data": "test"}
    
    def get_adapter_info(self):
        return {"name": "mock", "type": "test"}
    
    def validate_config(self, config: AdapterConfig) -> bool:
        return True


def test_adapter_status_enum():
    """测试 AdapterStatus 枚举"""
    assert AdapterStatus.AVAILABLE.value == "available"
    assert AdapterStatus.UNAVAILABLE.value == "unavailable"
    assert AdapterStatus.ERROR.value == "error"
    assert AdapterStatus.MAINTENANCE.value == "maintenance"


def test_adapter_info_init_default():
    """测试 AdapterInfo（初始化，默认值）"""
    info = AdapterInfo(
        name="test_adapter",
        adapter_type="test",
        status=AdapterStatus.AVAILABLE
    )
    
    assert info.name == "test_adapter"
    assert info.adapter_type == "test"
    assert info.status == AdapterStatus.AVAILABLE
    assert info.version == "1.0.0"
    assert info.description == ""
    assert info.capabilities == []
    assert info.config_schema == {}


def test_adapter_info_init_custom():
    """测试 AdapterInfo（初始化，自定义值）"""
    info = AdapterInfo(
        name="custom_adapter",
        adapter_type="custom",
        status=AdapterStatus.UNAVAILABLE,
        version="2.0.0",
        description="Custom adapter",
        capabilities=["cap1", "cap2"],
        config_schema={"key": "value"}
    )
    
    assert info.name == "custom_adapter"
    assert info.version == "2.0.0"
    assert info.description == "Custom adapter"
    assert info.capabilities == ["cap1", "cap2"]
    assert info.config_schema == {"key": "value"}


def test_adapter_info_post_init_none():
    """测试 AdapterInfo（__post_init__，None 值）"""
    info = AdapterInfo(
        name="test",
        adapter_type="test",
        status=AdapterStatus.AVAILABLE,
        capabilities=None,
        config_schema=None
    )
    
    assert info.capabilities == []
    assert info.config_schema == {}


def test_adapter_registry_init():
    """测试 AdapterRegistry（初始化）"""
    registry = AdapterRegistry()
    
    assert registry._adapters == {}
    assert registry._adapter_infos == {}
    assert registry._active_adapters == {}


def test_adapter_registry_register_adapter_success():
    """测试 AdapterRegistry（注册适配器，成功）"""
    registry = AdapterRegistry()
    config = AdapterConfig(name="test")
    info = AdapterInfo(
        name="test_adapter",
        adapter_type="test",
        status=AdapterStatus.AVAILABLE
    )
    
    result = registry.register_adapter("test_adapter", MockAdapter, info)
    
    assert result is True
    assert "test_adapter" in registry._adapters
    assert "test_adapter" in registry._adapter_infos
    assert registry._adapters["test_adapter"] == MockAdapter
    assert registry._adapter_infos["test_adapter"] == info


def test_adapter_registry_register_adapter_duplicate():
    """测试 AdapterRegistry（注册适配器，重复注册）"""
    registry = AdapterRegistry()
    info1 = AdapterInfo("test", "type1", AdapterStatus.AVAILABLE)
    info2 = AdapterInfo("test", "type2", AdapterStatus.UNAVAILABLE)
    
    registry.register_adapter("test", MockAdapter, info1)
    result = registry.register_adapter("test", MockAdapter, info2)
    
    assert result is True
    assert registry._adapter_infos["test"] == info2  # 被覆盖


def test_adapter_registry_register_adapter_exception():
    """测试 AdapterRegistry（注册适配器，异常）"""
    registry = AdapterRegistry()
    info = AdapterInfo("test", "type", AdapterStatus.AVAILABLE)
    
    # 使用无效的适配器类导致异常
    class InvalidAdapter:
        pass
    
    # 注册时不会立即验证，所以应该成功
    result = registry.register_adapter("test", InvalidAdapter, info)
    
    # 注册本身应该成功（异常会在创建时发生）
    assert result is True


def test_adapter_registry_unregister_adapter_existing():
    """测试 AdapterRegistry（注销适配器，存在）"""
    registry = AdapterRegistry()
    info = AdapterInfo("test", "type", AdapterStatus.AVAILABLE)
    registry.register_adapter("test", MockAdapter, info)
    
    result = registry.unregister_adapter("test")
    
    assert result is True
    assert "test" not in registry._adapters
    assert "test" not in registry._adapter_infos


def test_adapter_registry_unregister_adapter_nonexistent():
    """测试 AdapterRegistry（注销适配器，不存在）"""
    registry = AdapterRegistry()
    
    result = registry.unregister_adapter("nonexistent")
    
    assert result is False


def test_adapter_registry_unregister_adapter_active():
    """测试 AdapterRegistry（注销适配器，活跃状态）"""
    registry = AdapterRegistry()
    config = AdapterConfig(name="test")
    info = AdapterInfo("test", "type", AdapterStatus.AVAILABLE)
    registry.register_adapter("test", MockAdapter, info)
    registry.create_adapter("test", config)
    
    result = registry.unregister_adapter("test")
    
    assert result is True
    assert "test" not in registry._active_adapters


def test_adapter_registry_get_adapter_class_existing():
    """测试 AdapterRegistry（获取适配器类，存在）"""
    registry = AdapterRegistry()
    info = AdapterInfo("test", "type", AdapterStatus.AVAILABLE)
    registry.register_adapter("test", MockAdapter, info)
    
    adapter_class = registry.get_adapter_class("test")
    
    assert adapter_class == MockAdapter


def test_adapter_registry_get_adapter_class_nonexistent():
    """测试 AdapterRegistry（获取适配器类，不存在）"""
    registry = AdapterRegistry()
    
    adapter_class = registry.get_adapter_class("nonexistent")
    
    assert adapter_class is None


def test_adapter_registry_get_adapter_info_existing():
    """测试 AdapterRegistry（获取适配器信息，存在）"""
    registry = AdapterRegistry()
    info = AdapterInfo("test", "type", AdapterStatus.AVAILABLE)
    registry.register_adapter("test", MockAdapter, info)
    
    retrieved_info = registry.get_adapter_info("test")
    
    assert retrieved_info == info


def test_adapter_registry_get_adapter_info_nonexistent():
    """测试 AdapterRegistry（获取适配器信息，不存在）"""
    registry = AdapterRegistry()
    
    retrieved_info = registry.get_adapter_info("nonexistent")
    
    assert retrieved_info is None


def test_adapter_registry_list_adapters_empty():
    """测试 AdapterRegistry（列出适配器，空）"""
    registry = AdapterRegistry()
    
    adapters = registry.list_adapters()
    
    assert adapters == []


def test_adapter_registry_list_adapters_with_data():
    """测试 AdapterRegistry（列出适配器，有数据）"""
    registry = AdapterRegistry()
    info1 = AdapterInfo("adapter1", "type1", AdapterStatus.AVAILABLE)
    info2 = AdapterInfo("adapter2", "type2", AdapterStatus.UNAVAILABLE)
    registry.register_adapter("adapter1", MockAdapter, info1)
    registry.register_adapter("adapter2", MockAdapter, info2)
    
    adapters = registry.list_adapters()
    
    assert len(adapters) == 2
    assert "adapter1" in adapters
    assert "adapter2" in adapters


def test_adapter_registry_list_adapters_by_type():
    """测试 AdapterRegistry（按类型列出适配器）"""
    registry = AdapterRegistry()
    info1 = AdapterInfo("adapter1", "type1", AdapterStatus.AVAILABLE)
    info2 = AdapterInfo("adapter2", "type1", AdapterStatus.AVAILABLE)
    info3 = AdapterInfo("adapter3", "type2", AdapterStatus.AVAILABLE)
    registry.register_adapter("adapter1", MockAdapter, info1)
    registry.register_adapter("adapter2", MockAdapter, info2)
    registry.register_adapter("adapter3", MockAdapter, info3)
    
    adapters = registry.list_adapters_by_type("type1")
    
    assert len(adapters) == 2
    assert "adapter1" in adapters
    assert "adapter2" in adapters
    assert "adapter3" not in adapters


def test_adapter_registry_list_adapters_by_type_nonexistent():
    """测试 AdapterRegistry（按类型列出适配器，类型不存在）"""
    registry = AdapterRegistry()
    info = AdapterInfo("adapter1", "type1", AdapterStatus.AVAILABLE)
    registry.register_adapter("adapter1", MockAdapter, info)
    
    adapters = registry.list_adapters_by_type("nonexistent_type")
    
    assert adapters == []


def test_adapter_registry_create_adapter_success():
    """测试 AdapterRegistry（创建适配器，成功）"""
    registry = AdapterRegistry()
    config = AdapterConfig(name="test")
    info = AdapterInfo("test", "type", AdapterStatus.AVAILABLE)
    registry.register_adapter("test", MockAdapter, info)
    
    adapter = registry.create_adapter("test", config)
    
    assert adapter is not None
    assert isinstance(adapter, MockAdapter)
    assert "test" in registry._active_adapters


def test_adapter_registry_create_adapter_nonexistent():
    """测试 AdapterRegistry（创建适配器，不存在）"""
    registry = AdapterRegistry()
    config = AdapterConfig(name="test")
    
    adapter = registry.create_adapter("nonexistent", config)
    
    assert adapter is None


def test_adapter_registry_create_adapter_exception():
    """测试 AdapterRegistry（创建适配器，异常）"""
    registry = AdapterRegistry()
    config = AdapterConfig(name="test")
    info = AdapterInfo("test", "type", AdapterStatus.AVAILABLE)
    
    # 创建一个会抛出异常的适配器类
    class BadAdapter(BaseAdapter):
        def __init__(self, config):
            raise RuntimeError("Initialization failed")
        
        def connect(self): return True
        def disconnect(self): return True
        def is_connected(self): return False
        def fetch_data(self, **kwargs): return {}
    
    registry.register_adapter("test", BadAdapter, info)
    
    adapter = registry.create_adapter("test", config)
    
    assert adapter is None


def test_adapter_registry_activate_adapter_success():
    """测试 AdapterRegistry（激活适配器，成功）"""
    registry = AdapterRegistry()
    config = AdapterConfig(name="test")
    info = AdapterInfo("test", "type", AdapterStatus.AVAILABLE)
    registry.register_adapter("test", MockAdapter, info)
    
    result = registry.activate_adapter("test", config)
    
    assert result is True
    assert "test" in registry._active_adapters


def test_adapter_registry_activate_adapter_already_active():
    """测试 AdapterRegistry（激活适配器，已激活）"""
    registry = AdapterRegistry()
    config = AdapterConfig(name="test")
    info = AdapterInfo("test", "type", AdapterStatus.AVAILABLE)
    registry.register_adapter("test", MockAdapter, info)
    registry.create_adapter("test", config)
    
    result = registry.activate_adapter("test", config)
    
    assert result is True


def test_adapter_registry_activate_adapter_connect_fails():
    """测试 AdapterRegistry（激活适配器，连接失败）"""
    registry = AdapterRegistry()
    config = AdapterConfig(name="test")
    info = AdapterInfo("test", "type", AdapterStatus.AVAILABLE)
    
    class NonConnectingAdapter(MockAdapter):
        def connect(self):
            return False
    
    registry.register_adapter("test", NonConnectingAdapter, info)
    
    result = registry.activate_adapter("test", config)
    
    assert result is False


def test_adapter_registry_deactivate_adapter_success():
    """测试 AdapterRegistry（停用适配器，成功）"""
    registry = AdapterRegistry()
    config = AdapterConfig(name="test")
    info = AdapterInfo("test", "type", AdapterStatus.AVAILABLE)
    registry.register_adapter("test", MockAdapter, info)
    registry.activate_adapter("test", config)
    
    result = registry.deactivate_adapter("test")
    
    assert result is True
    assert "test" not in registry._active_adapters


def test_adapter_registry_deactivate_adapter_not_active():
    """测试 AdapterRegistry（停用适配器，未激活）"""
    registry = AdapterRegistry()
    
    result = registry.deactivate_adapter("test")
    
    assert result is False


def test_adapter_registry_deactivate_adapter_disconnect_fails():
    """测试 AdapterRegistry（停用适配器，断开失败）"""
    registry = AdapterRegistry()
    config = AdapterConfig(name="test")
    info = AdapterInfo("test", "type", AdapterStatus.AVAILABLE)
    
    class NonDisconnectingAdapter(MockAdapter):
        def disconnect(self):
            return False
    
    registry.register_adapter("test", NonDisconnectingAdapter, info)
    registry.activate_adapter("test", config)
    
    result = registry.deactivate_adapter("test")
    
    assert result is False


def test_adapter_registry_get_active_adapters_empty():
    """测试 AdapterRegistry（获取活跃适配器，空）"""
    registry = AdapterRegistry()
    
    active = registry.get_active_adapters()
    
    assert active == []


def test_adapter_registry_get_active_adapters_with_data():
    """测试 AdapterRegistry（获取活跃适配器，有数据）"""
    registry = AdapterRegistry()
    config = AdapterConfig(name="test")
    info = AdapterInfo("test", "type", AdapterStatus.AVAILABLE)
    registry.register_adapter("test", MockAdapter, info)
    registry.activate_adapter("test", config)
    
    active = registry.get_active_adapters()
    
    assert "test" in active
    assert len(active) == 1


def test_adapter_registry_get_adapter_status_active():
    """测试 AdapterRegistry（获取适配器状态，活跃）"""
    registry = AdapterRegistry()
    config = AdapterConfig(name="test")
    info = AdapterInfo("test", "type", AdapterStatus.AVAILABLE)
    registry.register_adapter("test", MockAdapter, info)
    registry.activate_adapter("test", config)
    
    status = registry.get_adapter_status("test")
    
    assert status == AdapterStatus.AVAILABLE


def test_adapter_registry_get_adapter_status_unavailable():
    """测试 AdapterRegistry（获取适配器状态，未激活）"""
    registry = AdapterRegistry()
    info = AdapterInfo("test", "type", AdapterStatus.AVAILABLE)
    registry.register_adapter("test", MockAdapter, info)
    
    status = registry.get_adapter_status("test")
    
    assert status == AdapterStatus.UNAVAILABLE


def test_adapter_registry_get_adapter_status_error():
    """测试 AdapterRegistry（获取适配器状态，错误）"""
    registry = AdapterRegistry()
    
    status = registry.get_adapter_status("nonexistent")
    
    assert status == AdapterStatus.ERROR


def test_adapter_registry_get_registry_info_empty():
    """测试 AdapterRegistry（获取注册表信息，空）"""
    registry = AdapterRegistry()
    
    info = registry.get_registry_info()
    
    assert info["total_adapters"] == 0
    assert info["active_adapters"] == 0
    assert info["adapter_types"] == []
    assert info["adapters"] == {}


def test_adapter_registry_get_registry_info_with_data():
    """测试 AdapterRegistry（获取注册表信息，有数据）"""
    registry = AdapterRegistry()
    config = AdapterConfig(name="test")
    info = AdapterInfo("test", "type1", AdapterStatus.AVAILABLE)
    registry.register_adapter("test", MockAdapter, info)
    registry.activate_adapter("test", config)
    
    registry_info = registry.get_registry_info()
    
    assert registry_info["total_adapters"] == 1
    assert registry_info["active_adapters"] == 1
    assert "type1" in registry_info["adapter_types"]
    assert "test" in registry_info["adapters"]
    assert registry_info["adapters"]["test"]["type"] == "type1"


def test_get_adapter_registry():
    """测试 get_adapter_registry（获取全局注册实例）"""
    registry = get_adapter_registry()
    
    assert isinstance(registry, AdapterRegistry)


def test_get_adapter_registry_singleton():
    """测试 get_adapter_registry（单例模式）"""
    registry1 = get_adapter_registry()
    registry2 = get_adapter_registry()
    
    assert registry1 is registry2


def test_adapter_registry_register_adapter_exception():
    """测试 AdapterRegistry（注册适配器，异常处理，覆盖 71-73 行）"""
    registry = AdapterRegistry()
    # 模拟注册时抛出异常
    # 通过 patch logger.info 来触发异常，因为 logger.info 在注册成功时被调用
    import src.data.adapters.adapter_registry as ar_module
    original_logger = ar_module.logger
    def mock_info(*args, **kwargs):
        raise Exception("Register error")
    with patch.object(ar_module, 'logger', Mock(info=mock_info, warning=Mock(), error=Mock())):
        info = AdapterInfo(
            name="test_adapter",
            adapter_type="test",
            status=AdapterStatus.AVAILABLE
        )
        result = registry.register_adapter("test_adapter", MockAdapter, info)
        assert result is False


def test_adapter_registry_unregister_adapter_exception():
    """测试 AdapterRegistry（注销适配器，异常处理，覆盖 91-93 行）"""
    registry = AdapterRegistry()
    # 先注册并激活一个适配器
    info = AdapterInfo(
        name="test_adapter",
        adapter_type="test",
        status=AdapterStatus.AVAILABLE
    )
    registry.register_adapter("test_adapter", MockAdapter, info)
    config = AdapterConfig()
    registry.activate_adapter("test_adapter", config)
    # 模拟注销时抛出异常
    # 通过 patch deactivate_adapter 来触发异常
    with patch.object(registry, 'deactivate_adapter', side_effect=Exception("Unregister error")):
        result = registry.unregister_adapter("test_adapter")
        assert result is False


def test_adapter_registry_activate_adapter_exception():
    """测试 AdapterRegistry（激活适配器，异常处理，覆盖 146-148 行）"""
    registry = AdapterRegistry()
    # 先注册一个适配器
    info = AdapterInfo(
        name="test_adapter",
        adapter_type="test",
        status=AdapterStatus.AVAILABLE
    )
    registry.register_adapter("test_adapter", MockAdapter, info)
    # 模拟激活时抛出异常
    with patch.object(registry, 'create_adapter', side_effect=Exception("Activate error")):
        config = AdapterConfig()
        result = registry.activate_adapter("test_adapter", config)
        assert result is False


def test_adapter_registry_deactivate_adapter_exception():
    """测试 AdapterRegistry（停用适配器，异常处理，覆盖 166-168 行）"""
    registry = AdapterRegistry()
    # 先注册并激活一个适配器
    info = AdapterInfo(
        name="test_adapter",
        adapter_type="test",
        status=AdapterStatus.AVAILABLE
    )
    registry.register_adapter("test_adapter", MockAdapter, info)
    config = AdapterConfig()
    registry.activate_adapter("test_adapter", config)
    # 模拟停用时抛出异常
    with patch.object(registry._active_adapters["test_adapter"], 'disconnect', side_effect=Exception("Deactivate error")):
        result = registry.deactivate_adapter("test_adapter")
        assert result is False


def test_adapter_registry_get_adapter_status_not_connected():
    """测试 AdapterRegistry（获取适配器状态，未连接，覆盖 181 行）"""
    registry = AdapterRegistry()
    # 先注册并激活一个适配器
    info = AdapterInfo(
        name="test_adapter",
        adapter_type="test",
        status=AdapterStatus.AVAILABLE
    )
    registry.register_adapter("test_adapter", MockAdapter, info)
    config = AdapterConfig()
    adapter = registry.create_adapter("test_adapter", config)
    registry._active_adapters["test_adapter"] = adapter
    # 断开连接
    adapter.disconnect()
    # 获取状态应该返回 ERROR
    status = registry.get_adapter_status("test_adapter")
    assert status == AdapterStatus.ERROR


