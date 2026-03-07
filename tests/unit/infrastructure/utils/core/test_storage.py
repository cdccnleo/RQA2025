#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施工具层存储适配器测试

测试目标：提升utils/core/storage.py的真实覆盖率
实际导入和使用src.infrastructure.utils.core.storage模块
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Any


class TestStorageAdapter:
    """测试存储适配器基类"""
    
    def test_init_with_base_path(self):
        """测试使用基础路径初始化"""
        from src.infrastructure.utils.core.storage import StorageAdapter
        
        class ConcreteStorageAdapter(StorageAdapter):
            def save(self, key: str, data: Any, **kwargs) -> bool:
                return True
            
            def load(self, key: str, **kwargs):
                return None
            
            def delete(self, key: str, **kwargs) -> bool:
                return True
            
            def exists(self, key: str, **kwargs) -> bool:
                return False
            
            def list_keys(self, prefix: str = "", **kwargs):
                return []
        
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            adapter = ConcreteStorageAdapter(base_path="/tmp/storage")
            # Windows路径分隔符问题，使用Path对象比较
            assert adapter.base_path == Path("/tmp/storage")
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    
    def test_init_without_base_path(self):
        """测试不使用基础路径初始化"""
        from src.infrastructure.utils.core.storage import StorageAdapter
        
        class ConcreteStorageAdapter(StorageAdapter):
            def save(self, key: str, data: Any, **kwargs) -> bool:
                return True
            
            def load(self, key: str, **kwargs):
                return None
            
            def delete(self, key: str, **kwargs) -> bool:
                return True
            
            def exists(self, key: str, **kwargs) -> bool:
                return False
            
            def list_keys(self, prefix: str = "", **kwargs):
                return []
        
        adapter = ConcreteStorageAdapter()
        assert adapter.base_path == Path(".")
    
    def test_init_with_permission_error(self):
        """测试权限错误处理"""
        from src.infrastructure.utils.core.storage import StorageAdapter
        
        class ConcreteStorageAdapter(StorageAdapter):
            def save(self, key: str, data: Any, **kwargs) -> bool:
                return True
            
            def load(self, key: str, **kwargs):
                return None
            
            def delete(self, key: str, **kwargs) -> bool:
                return True
            
            def exists(self, key: str, **kwargs) -> bool:
                return False
            
            def list_keys(self, prefix: str = "", **kwargs):
                return []
        
        with patch('pathlib.Path.mkdir', side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError, match="创建存储目录失败"):
                ConcreteStorageAdapter(base_path="/tmp/storage")
    
    def test_init_with_os_error(self):
        """测试OS错误处理"""
        from src.infrastructure.utils.core.storage import StorageAdapter
        
        class ConcreteStorageAdapter(StorageAdapter):
            def save(self, key: str, data: Any, **kwargs) -> bool:
                return True
            
            def load(self, key: str, **kwargs):
                return None
            
            def delete(self, key: str, **kwargs) -> bool:
                return True
            
            def exists(self, key: str, **kwargs) -> bool:
                return False
            
            def list_keys(self, prefix: str = "", **kwargs):
                return []
        
        with patch('pathlib.Path.mkdir', side_effect=OSError("Disk full")):
            with pytest.raises(OSError, match="创建存储目录失败"):
                ConcreteStorageAdapter(base_path="/tmp/storage")
    
    def test_init_with_generic_exception(self):
        """测试通用异常处理"""
        from src.infrastructure.utils.core.storage import StorageAdapter
        
        class ConcreteStorageAdapter(StorageAdapter):
            def save(self, key: str, data: Any, **kwargs) -> bool:
                return True
            
            def load(self, key: str, **kwargs):
                return None
            
            def delete(self, key: str, **kwargs) -> bool:
                return True
            
            def exists(self, key: str, **kwargs) -> bool:
                return False
            
            def list_keys(self, prefix: str = "", **kwargs):
                return []
        
        with patch('pathlib.Path.mkdir', side_effect=Exception("Unexpected error")):
            with pytest.raises(RuntimeError, match="存储初始化失败"):
                ConcreteStorageAdapter(base_path="/tmp/storage")
    
    def test_get_stats(self):
        """测试获取统计信息"""
        from src.infrastructure.utils.core.storage import StorageAdapter
        
        class ConcreteStorageAdapter(StorageAdapter):
            def save(self, key: str, data: Any, **kwargs) -> bool:
                return True
            
            def load(self, key: str, **kwargs):
                return None
            
            def delete(self, key: str, **kwargs) -> bool:
                return True
            
            def exists(self, key: str, **kwargs) -> bool:
                return False
            
            def list_keys(self, prefix: str = "", **kwargs):
                return []
        
        adapter = ConcreteStorageAdapter(base_path="/tmp/storage")
        stats = adapter.get_stats()
        
        assert stats["adapter_type"] == "ConcreteStorageAdapter"
        # Windows路径分隔符问题，使用Path对象比较
        assert Path(stats["base_path"]) == Path("/tmp/storage")
    
    def test_abstract_methods(self):
        """测试抽象方法不能直接实例化"""
        from src.infrastructure.utils.core.storage import StorageAdapter
        
        with pytest.raises(TypeError):
            StorageAdapter()


class TestConcreteStorageAdapter:
    """测试具体存储适配器实现"""
    
    def test_save_load_delete_exists(self):
        """测试保存、加载、删除、存在检查"""
        from src.infrastructure.utils.core.storage import StorageAdapter
        
        class MemoryStorageAdapter(StorageAdapter):
            def __init__(self):
                super().__init__()
                self._storage = {}
            
            def save(self, key: str, data: Any, **kwargs) -> bool:
                self._storage[key] = data
                return True
            
            def load(self, key: str, **kwargs):
                return self._storage.get(key)
            
            def delete(self, key: str, **kwargs) -> bool:
                if key in self._storage:
                    del self._storage[key]
                    return True
                return False
            
            def exists(self, key: str, **kwargs) -> bool:
                return key in self._storage
            
            def list_keys(self, prefix: str = "", **kwargs):
                return [k for k in self._storage.keys() if k.startswith(prefix)]
        
        adapter = MemoryStorageAdapter()
        
        # 测试保存
        assert adapter.save("key1", "value1") is True
        assert adapter.exists("key1") is True
        
        # 测试加载
        assert adapter.load("key1") == "value1"
        assert adapter.load("nonexistent") is None
        
        # 测试删除
        assert adapter.delete("key1") is True
        assert adapter.exists("key1") is False
        assert adapter.delete("nonexistent") is False
    
    def test_list_keys(self):
        """测试列出键"""
        from src.infrastructure.utils.core.storage import StorageAdapter
        
        class MemoryStorageAdapter(StorageAdapter):
            def __init__(self):
                super().__init__()
                self._storage = {}
            
            def save(self, key: str, data: Any, **kwargs) -> bool:
                self._storage[key] = data
                return True
            
            def load(self, key: str, **kwargs):
                return self._storage.get(key)
            
            def delete(self, key: str, **kwargs) -> bool:
                if key in self._storage:
                    del self._storage[key]
                    return True
                return False
            
            def exists(self, key: str, **kwargs) -> bool:
                return key in self._storage
            
            def list_keys(self, prefix: str = "", **kwargs):
                return [k for k in self._storage.keys() if k.startswith(prefix)]
        
        adapter = MemoryStorageAdapter()
        
        adapter.save("user:1", "data1")
        adapter.save("user:2", "data2")
        adapter.save("order:1", "data3")
        
        all_keys = adapter.list_keys()
        assert len(all_keys) == 3
        
        user_keys = adapter.list_keys(prefix="user:")
        assert len(user_keys) == 2
        assert "user:1" in user_keys
        assert "user:2" in user_keys

