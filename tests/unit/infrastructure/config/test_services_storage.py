#!/usr/bin/env python3
"""
测试配置存储服务

测试覆盖：
- ConfigStorageService类的基本功能
- 存储和缓存操作
- 统计信息收集
- 存储后端管理
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
from unittest.mock import MagicMock, patch, Mock

from src.infrastructure.config.services.config_storage_service import ConfigStorageService


class TestConfigStorageService:
    """测试配置存储服务"""

    def setup_method(self):
        """测试前准备"""
        self.service = ConfigStorageService()

    def test_initialization(self):
        """测试初始化"""
        assert self.service is not None
        assert hasattr(self.service, '_storage_backend')
        assert hasattr(self.service, '_cache')
        assert hasattr(self.service, '_cache_enabled')
        assert hasattr(self.service, '_cache_size')

    def test_initialization_with_backend(self):
        """测试带存储后端的初始化"""
        mock_backend = MagicMock()
        service = ConfigStorageService(storage_backend=mock_backend)
        assert service._storage_backend == mock_backend

    def test_initialization_cache_disabled(self):
        """测试禁用缓存的初始化"""
        service = ConfigStorageService(cache_enabled=False)
        assert service._cache_enabled is False

    def test_initialization_custom_cache_size(self):
        """测试自定义缓存大小的初始化"""
        service = ConfigStorageService(cache_size=500)
        assert service._cache_size == 500

    @patch('src.infrastructure.config.services.config_storage_service.time')
    def test_get_from_cache(self, mock_time):
        """测试从缓存获取数据"""
        mock_time.time.return_value = 1000.0

        # 先设置缓存
        self.service._cache['test_key'] = 'cached_value'
        self.service._cache_timestamps['test_key'] = 1000.0

        result = self.service.get('test_key', 'default')
        assert result == 'cached_value'

    @patch('src.infrastructure.config.services.config_storage_service.time')
    def test_get_cache_expired(self, mock_time):
        """测试缓存过期的情况"""
        mock_time.time.return_value = 2000.0  # 模拟过期

        # 设置过期的缓存
        self.service._cache['test_key'] = 'expired_value'
        self.service._cache_timestamps['test_key'] = 1000.0  # 1小时前

        # 模拟后端返回新值
        mock_backend = MagicMock()
        mock_backend.get.return_value = 'fresh_value'
        self.service._storage_backend = mock_backend

        result = self.service.get('test_key', 'default')
        assert result == 'fresh_value'
        mock_backend.get.assert_called_once_with('test_key', 'default')

    def test_get_without_backend(self):
        """测试没有存储后端的情况"""
        self.service._storage_backend = None

        result = self.service.get('test_key', 'default')
        assert result == 'default'

    def test_set_with_cache(self):
        """测试带缓存的设置操作"""
        mock_backend = MagicMock()
        mock_backend.set.return_value = True
        self.service._storage_backend = mock_backend

        result = self.service.set('test_key', 'test_value')
        assert result is True

        # 检查缓存是否更新
        assert self.service._cache['test_key'] == 'test_value'
        assert 'test_key' in self.service._cache_timestamps

        mock_backend.set.assert_called_once_with('test_key', 'test_value')

    def test_set_without_cache(self):
        """测试禁用缓存的设置操作"""
        service = ConfigStorageService(cache_enabled=False)
        mock_backend = MagicMock()
        mock_backend.set.return_value = True
        service._storage_backend = mock_backend

        result = service.set('test_key', 'test_value')
        assert result is True

        # 检查缓存是否为空
        assert len(service._cache) == 0

    def test_delete_operation(self):
        """测试删除操作"""
        # 设置缓存
        self.service._cache['test_key'] = 'test_value'

        mock_backend = MagicMock()
        mock_backend.delete.return_value = True
        self.service._storage_backend = mock_backend

        result = self.service.delete('test_key')
        assert result is True

        # 检查缓存是否清理
        assert 'test_key' not in self.service._cache

        mock_backend.delete.assert_called_once_with('test_key')

    def test_exists_operation(self):
        """测试存在检查操作"""
        # 设置缓存
        self.service._cache['test_key'] = 'test_value'

        result = self.service.exists('test_key')
        assert result is True

    def test_exists_operation_not_in_cache(self):
        """测试缓存中不存在的键"""
        mock_backend = MagicMock()
        mock_backend.exists.return_value = True
        self.service._storage_backend = mock_backend

        result = self.service.exists('test_key')
        assert result is True
        mock_backend.exists.assert_called_once_with('test_key')

    def test_keys_operation(self):
        """测试键列表操作"""
        mock_backend = MagicMock()
        mock_backend.keys.return_value = ['key1', 'key2']
        self.service._storage_backend = mock_backend

        result = self.service.keys('pattern')
        assert result == ['key1', 'key2']
        mock_backend.keys.assert_called_once_with('pattern')

    def test_clear_operation(self):
        """测试清空操作"""
        # 设置一些缓存
        self.service._cache['key1'] = 'value1'
        self.service._cache['key2'] = 'value2'

        mock_backend = MagicMock()
        mock_backend.clear.return_value = True
        self.service._storage_backend = mock_backend

        result = self.service.clear()
        assert result is True

        # 检查缓存是否清空
        assert len(self.service._cache) == 0

        mock_backend.clear.assert_called_once()

    def test_load_operation(self):
        """测试加载操作"""
        mock_backend = MagicMock()
        mock_backend.load.return_value = {'loaded': 'config'}
        self.service._storage_backend = mock_backend

        result = self.service.load('source')
        assert result == {'loaded': 'config'}
        mock_backend.load.assert_called_once_with('source')

    def test_save_operation(self):
        """测试保存操作"""
        mock_backend = MagicMock()
        mock_backend.save.return_value = True
        self.service._storage_backend = mock_backend

        result = self.service.save({'config': 'data'}, 'target')
        assert result is True
        mock_backend.save.assert_called_once_with({'config': 'data'}, 'target')

    def test_reload_operation(self):
        """测试重新加载操作"""
        mock_backend = MagicMock()
        mock_backend.reload.return_value = True
        self.service._storage_backend = mock_backend

        result = self.service.reload()
        assert result is True
        mock_backend.reload.assert_called_once()

    def test_set_storage_backend(self):
        """测试设置存储后端"""
        mock_backend = MagicMock()
        self.service.set_storage_backend(mock_backend)
        assert self.service._storage_backend == mock_backend

    def test_get_storage_stats(self):
        """测试获取存储统计"""
        # 设置匹配source的stats
        self.service._stats = {
            'loads': 100,
            'saves': 10,
            'cache_hits': 80,
            'cache_misses': 20,
            'errors': 5
        }

        stats = self.service.get_storage_stats()

        assert 'loads' in stats
        assert stats['loads'] == 100
        assert 'saves' in stats
        assert stats['saves'] == 10
        assert 'cache_hits' in stats
        assert stats['cache_hits'] == 80
        assert 'cache_misses' in stats
        assert stats['cache_misses'] == 20
        assert 'errors' in stats
        assert stats['errors'] == 5
        assert 'total_operations' in stats
        assert 'cache_size' in stats

    def test_get_service_info(self):
        """测试获取服务信息"""
        info = self.service.get_service_info()
        assert isinstance(info, dict)
        assert 'service_name' in info
        assert info['service_name'] == 'config_storage_service'

    def test_thread_safety(self):
        """测试线程安全性"""
        # 不override service方法，而是mock backend
        mock_backend = Mock()
        mock_backend.set.return_value = True
        mock_backend.get.return_value = 'value'  # 固定返回，调整test逻辑
        self.service._storage_backend = mock_backend

        import threading

        results = []
        errors = []

        def worker(worker_id):
            try:
                key = f'thread_{worker_id}_key'
                value = f'value_{worker_id}'  # 保持一致

                # 每个线程执行操作 - 现在调用真实方法，会用mock_backend
                self.service.set(key, value)
                retrieved = self.service.get(key)

                if retrieved == value:
                    results.append(True)
                else:
                    results.append(False)

            except Exception as e:
                errors.append(str(e))
                results.append(False)

        # 创建多个线程
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)

        # 启动和等待线程
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        # 验证结果
        assert len(results) == 5
        assert all(results), f'线程安全测试失败，错误: {errors}'

    def test_cache_eviction(self):
        """测试缓存淘汰"""
        # 填充缓存到最大容量 - 使用set触发evict
        for i in range(self.service._cache_size + 10):
            self.service.set(f'key_{i}', f'value_{i}')

        # 验证缓存大小被控制
        assert len(self.service._cache) <= self.service._cache_size

    def test_cache_access_tracking(self):
        """测试缓存访问跟踪"""
        # 设置mock存储后端
        mock_backend = MagicMock()
        mock_backend.set.return_value = True
        mock_backend.get.return_value = 'test_value'
        self.service._storage_backend = mock_backend

        # 设置缓存项 - 使用set
        self.service.set('test_key', 'test_value')

        # 访问缓存项 - 触发_update_cache_access
        result = self.service.get('test_key')
        assert result == 'test_value'  # backend返回'test_value'

        # 检查访问时间是否被记录
        assert 'test_key' in self.service._cache_access_times

    def test_performance_stats_update(self):
        """测试性能统计更新"""
        # 设置mock存储后端
        mock_backend = MagicMock()
        mock_backend.set.return_value = True
        mock_backend.get.return_value = 'test_value'
        self.service._storage_backend = mock_backend
        
        # 清除缓存以确保get操作会产生缓存未命中
        if hasattr(self.service, '_cache'):
            self.service._cache.clear()
        
        initial_saves = self.service._stats['saves']
        initial_misses = self.service._stats['cache_misses']

        # 执行一些操作
        self.service.set('test_key', 'test_value')
        # 获取一个不存在的键以确保产生缓存未命中
        self.service.get('nonexistent_key')

        # 检查相关统计字段是否增加
        assert self.service._stats['saves'] > initial_saves  # set 操作会增加 saves
        assert self.service._stats['cache_misses'] > initial_misses  # get 操作会增加 cache_misses
