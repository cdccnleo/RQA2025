#!/usr/bin/env python3
"""
测试iconfigstorage模块

测试覆盖：
- IConfigStorage抽象接口
- BaseConfigStorage基类的具体方法
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import sys
import os
from unittest.mock import Mock

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../'))

try:
    from src.infrastructure.config.storage.types.iconfigstorage import IConfigStorage, BaseConfigStorage
    from src.infrastructure.config.storage.types.configscope import ConfigScope
    MODULE_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as e:
    MODULE_AVAILABLE = False
    IMPORT_ERROR = e


# 创建具体实现类用于测试抽象接口
class MockConfigStorage(IConfigStorage):
    """Mock配置存储实现，用于测试抽象接口"""
    
    def __init__(self):
        self._data = {}
    
    def get(self, key: str, scope: ConfigScope = ConfigScope.APPLICATION) -> any:
        return self._data.get(f"{scope.value}:{key}")
    
    def set(self, key: str, value: any, scope: ConfigScope = ConfigScope.APPLICATION) -> bool:
        self._data[f"{scope.value}:{key}"] = value
        return True
    
    def delete(self, key: str, scope: ConfigScope = ConfigScope.APPLICATION) -> bool:
        key_to_delete = f"{scope.value}:{key}"
        if key_to_delete in self._data:
            del self._data[key_to_delete]
            return True
        return False
    
    def exists(self, key: str, scope: ConfigScope = ConfigScope.APPLICATION) -> bool:
        return f"{scope.value}:{key}" in self._data
    
    def list_keys(self, scope: any = None) -> list:
        if scope:
            prefix = f"{scope.value}:"
            return [key[len(prefix):] for key in self._data.keys() if key.startswith(prefix)]
        return list(self._data.keys())
    
    def save(self) -> bool:
        return True
    
    def load(self) -> bool:
        return True


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR if IMPORT_ERROR else 'Unknown error'}")
class TestIConfigStorage:
    """测试IConfigStorage抽象接口"""

    def setup_method(self):
        """测试前准备"""
        self.storage = MockConfigStorage()

    def test_interface_methods_exist(self):
        """测试接口方法存在"""
        assert hasattr(self.storage, 'get')
        assert hasattr(self.storage, 'set')
        assert hasattr(self.storage, 'delete')
        assert hasattr(self.storage, 'exists')
        assert hasattr(self.storage, 'list_keys')
        assert hasattr(self.storage, 'save')
        assert hasattr(self.storage, 'load')

    def test_get_and_set_basic(self):
        """测试基本的get和set操作"""
        # 设置值
        result = self.storage.set("test_key", "test_value")
        assert result is True
        
        # 获取值
        value = self.storage.get("test_key")
        assert value == "test_value"

    def test_exists_method(self):
        """测试exists方法"""
        # 不存在的键
        assert self.storage.exists("nonexistent") is False
        
        # 存在的键
        self.storage.set("existing_key", "value")
        assert self.storage.exists("existing_key") is True

    def test_delete_method(self):
        """测试delete方法"""
        # 删除不存在的键
        result = self.storage.delete("nonexistent")
        assert result is False
        
        # 删除存在的键
        self.storage.set("to_delete", "value")
        assert self.storage.exists("to_delete") is True
        result = self.storage.delete("to_delete")
        assert result is True
        assert self.storage.exists("to_delete") is False

    def test_list_keys_method(self):
        """测试list_keys方法"""
        # 空存储
        keys = self.storage.list_keys()
        assert isinstance(keys, list)
        
        # 有数据的存储
        self.storage.set("key1", "value1")
        self.storage.set("key2", "value2")
        keys = self.storage.list_keys()
        assert len(keys) >= 2

    def test_save_and_load_methods(self):
        """测试save和load方法"""
        assert self.storage.save() is True
        assert self.storage.load() is True


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR if IMPORT_ERROR else 'Unknown error'}")
class TestBaseConfigStorage:
    """测试BaseConfigStorage基类"""

    def setup_method(self):
        """测试前准备"""
        self.base_storage = BaseConfigStorage()

    def test_initialization(self):
        """测试初始化"""
        assert hasattr(self.base_storage, '_data')
        assert hasattr(self.base_storage, '_lock')
        assert isinstance(self.base_storage._data, dict)

    def test_list_keys_with_scope(self):
        """测试带scope的list_keys方法"""
        # 测试空数据
        keys = self.base_storage.list_keys(ConfigScope.APPLICATION)
        assert isinstance(keys, list)
        assert len(keys) == 0
        
        # 测试有数据的情况（通过直接操作内部数据结构）
        self.base_storage._data[ConfigScope.APPLICATION] = {"key1": "value1", "key2": "value2"}
        keys = self.base_storage.list_keys(ConfigScope.APPLICATION)
        assert len(keys) == 2
        assert "key1" in keys
        assert "key2" in keys

    def test_list_keys_without_scope(self):
        """测试不带scope的list_keys方法"""
        # 设置多个scope的数据
        self.base_storage._data[ConfigScope.APPLICATION] = {"app_key1": "value1"}
        self.base_storage._data[ConfigScope.USER] = {"user_key1": "value2"}
        
        keys = self.base_storage.list_keys()
        assert len(keys) == 2
        assert "app_key1" in keys
        assert "user_key1" in keys

    def test_exists_method(self):
        """测试exists方法"""
        # 测试不存在的scope和key
        assert self.base_storage.exists("nonexistent_key") is False
        
        # 测试存在的scope但不存在的key
        self.base_storage._data[ConfigScope.APPLICATION] = {}
        assert self.base_storage.exists("still_nonexistent") is False
        
        # 测试存在的key
        self.base_storage._data[ConfigScope.APPLICATION]["existing_key"] = "value"
        assert self.base_storage.exists("existing_key", ConfigScope.APPLICATION) is True
        assert self.base_storage.exists("existing_key", ConfigScope.USER) is False

    def test_get_item_method(self):
        """测试_get_item内部方法"""
        # 测试不存在的item
        item = self.base_storage._get_item("nonexistent")
        assert item is None
        
        # 测试存在的item
        self.base_storage._data[ConfigScope.APPLICATION] = {"test_key": "test_value"}
        item = self.base_storage._get_item("test_key", ConfigScope.APPLICATION)
        assert item == "test_value"
        
        # 测试不存在的scope
        item = self.base_storage._get_item("test_key", ConfigScope.USER)
        assert item is None
