"""
日志存储基础实现单元测试

测试日志存储的基础接口和实现。
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from src.infrastructure.logging.storage.base import (
    ILogStorage,
    BaseStorage,
)


class ConcreteStorage(BaseStorage):
    """具体的存储实现，用于测试"""

    def __init__(self, config=None):
        super().__init__(config)
        self._data = []

    def _store(self, record):
        self._data.append(record)
        return True

    def _retrieve(self, query=None, limit=None):
        data = self._data
        if limit:
            data = data[:limit]
        return data

    def _delete(self, query):
        # 简单的删除逻辑
        self._data = []
        return len(self._data)

    def _count(self, query=None):
        return len(self._data)

    def _clear(self):
        self._data.clear()

    def _get_status(self):
        return {
            'record_count': len(self._data),
            'implementation': 'concrete'
        }
from src.infrastructure.logging.core.exceptions import LogStorageError


class TestILogStorage:
    """测试日志存储接口"""

    def test_interface_is_abstract(self):
        """测试接口是抽象的"""
        with pytest.raises(TypeError):
            ILogStorage()

    def test_interface_has_required_methods(self):
        """测试接口有必需的方法"""
        required_methods = ['store', 'retrieve', 'delete', 'count', 'clear', 'get_status']
        for method in required_methods:
            assert hasattr(ILogStorage, method)


class TestBaseStorage:
    """测试基础存储实现"""

    @pytest.fixture
    def base_storage(self):
        """创建基础存储实例"""
        return ConcreteStorage()

    @pytest.fixture
    def configured_storage(self):
        """创建配置化的存储实例"""
        config = {
            'name': 'TestStorage',
            'enabled': True,
            'max_records': 1000,
            'compression': True
        }
        return ConcreteStorage(config)

    def test_init_default(self, base_storage):
        """测试默认初始化"""
        assert base_storage.name == 'ConcreteStorage'  # 使用具体类的名称
        assert base_storage.enabled is True
        assert base_storage.max_records == 0
        assert base_storage.compression is False

    def test_init_with_config(self, configured_storage):
        """测试带配置初始化"""
        assert configured_storage.name == 'TestStorage'
        assert configured_storage.enabled is True
        assert configured_storage.max_records == 1000
        assert configured_storage.compression is True

    def test_store_enabled(self, base_storage):
        """测试存储启用时的工作"""
        record = {'message': 'test', 'level': 'INFO'}

        with patch.object(base_storage, '_store', return_value=True) as mock_store:
            result = base_storage.store(record)
            assert result is True
            mock_store.assert_called_once_with(record)

    def test_store_disabled(self, base_storage):
        """测试存储禁用时的行为"""
        base_storage.enabled = False
        record = {'message': 'test', 'level': 'INFO'}

        result = base_storage.store(record)
        assert result is False

    def test_store_exception_handling(self, base_storage):
        """测试存储异常处理"""
        record = {'message': 'test', 'level': 'INFO'}

        with patch.object(base_storage, '_store', side_effect=Exception('Storage error')):
            with pytest.raises(LogStorageError):
                base_storage.store(record)

    def test_retrieve_enabled(self, base_storage):
        """测试检索启用时的工作"""
        query = {'level': 'ERROR'}
        expected_records = [{'message': 'error1'}, {'message': 'error2'}]

        with patch.object(base_storage, '_retrieve', return_value=expected_records) as mock_retrieve:
            result = base_storage.retrieve(query, limit=10)
            assert result == expected_records
            mock_retrieve.assert_called_once_with(query, 10)

    def test_retrieve_disabled(self, base_storage):
        """测试检索禁用时的行为"""
        base_storage.enabled = False

        result = base_storage.retrieve()
        assert result == []

    def test_retrieve_exception_handling(self, base_storage):
        """测试检索异常处理"""
        with patch.object(base_storage, '_retrieve', side_effect=Exception('Retrieve error')):
            with pytest.raises(LogStorageError):
                base_storage.retrieve()

    def test_delete_enabled(self, base_storage):
        """测试删除启用时的工作"""
        query = {'level': 'DEBUG'}

        with patch.object(base_storage, '_delete', return_value=5) as mock_delete:
            result = base_storage.delete(query)
            assert result == 5
            mock_delete.assert_called_once_with(query)

    def test_delete_disabled(self, base_storage):
        """测试删除禁用时的行为"""
        base_storage.enabled = False

        result = base_storage.delete({'level': 'DEBUG'})
        assert result == 0

    def test_delete_exception_handling(self, base_storage):
        """测试删除异常处理"""
        with patch.object(base_storage, '_delete', side_effect=Exception('Delete error')):
            with pytest.raises(LogStorageError):
                base_storage.delete({'level': 'DEBUG'})

    def test_count_enabled(self, base_storage):
        """测试计数启用时的工作"""
        query = {'level': 'INFO'}

        with patch.object(base_storage, '_count', return_value=42) as mock_count:
            result = base_storage.count(query)
            assert result == 42
            mock_count.assert_called_once_with(query)

    def test_count_disabled(self, base_storage):
        """测试计数禁用时的行为"""
        base_storage.enabled = False

        result = base_storage.count()
        assert result == 0

    def test_count_exception_handling(self, base_storage):
        """测试计数异常处理"""
        with patch.object(base_storage, '_count', side_effect=Exception('Count error')):
            with pytest.raises(LogStorageError):
                base_storage.count()

    def test_clear_enabled(self, base_storage):
        """测试清空启用时的工作"""
        with patch.object(base_storage, '_clear') as mock_clear:
            base_storage.clear()
            mock_clear.assert_called_once()

    def test_clear_disabled(self, base_storage):
        """测试清空禁用时的行为"""
        base_storage.enabled = False

        # 不应该抛出异常，只是静默忽略
        base_storage.clear()

    def test_clear_exception_handling(self, base_storage):
        """测试清空异常处理"""
        with patch.object(base_storage, '_clear', side_effect=Exception('Clear error')):
            with pytest.raises(LogStorageError):
                base_storage.clear()

    def test_get_status(self, base_storage):
        """测试获取状态"""
        status = base_storage.get_status()

        assert isinstance(status, dict)
        assert 'name' in status
        assert 'enabled' in status
        assert 'record_count' in status
        assert 'max_records' in status
        assert 'compression' in status

    def test_get_status_with_config(self, configured_storage):
        """测试获取带配置的状态"""
        status = configured_storage.get_status()

        assert status['name'] == 'TestStorage'
        assert status['enabled'] is True
        assert status['max_records'] == 1000
        assert status['compression'] is True

    def test_abstract_methods_not_implemented(self):
        """测试抽象方法没有实现"""
        # 创建BaseStorage实例（会失败，因为是抽象类）
        with pytest.raises(TypeError):
            BaseStorage()

    def test_concrete_methods_implemented(self, base_storage):
        """测试具体方法已实现"""
        record = {'message': 'test'}

        # 这些方法应该正常工作，不抛出NotImplementedError
        result = base_storage._store(record)
        assert result is True

        data = base_storage._retrieve(None, None)
        assert isinstance(data, list)

        count = base_storage._count(None)
        assert isinstance(count, int)

        base_storage._clear()

        status = base_storage._get_status()
        assert isinstance(status, dict)

    def test_max_records_limit(self, configured_storage):
        """测试最大记录数限制"""
        configured_storage._records = []  # 假设有_records属性

        # 这个测试需要具体的实现类来验证
        # 这里只是一个占位符

    def test_compression_enabled(self, configured_storage):
        """测试压缩启用"""
        # 验证配置中启用了压缩
        assert configured_storage.compression is True

    def test_config_preservation(self, configured_storage):
        """测试配置保存"""
        assert configured_storage.config['name'] == 'TestStorage'
        assert configured_storage.config['enabled'] is True
