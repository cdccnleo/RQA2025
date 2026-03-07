#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试 - 日志存储组件

测试存储后端的数据存储、检索、删除等功能。
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from src.infrastructure.logging.storage.base import ILogStorage, BaseStorage, MemoryStorage


class TestLogStorageInterface:
    """日志存储接口测试"""

    def test_interface_definition(self):
        """测试接口定义"""
        # 确保接口类存在且是抽象类
        assert hasattr(ILogStorage, 'store')
        assert hasattr(ILogStorage, 'retrieve')
        assert hasattr(ILogStorage, 'delete')
        assert hasattr(ILogStorage, 'count')
        assert hasattr(ILogStorage, 'clear')
        assert hasattr(ILogStorage, 'get_status')

    def test_interface_is_abstract(self):
        """测试接口是抽象的"""
        # 尝试实例化接口应该失败
        with pytest.raises(TypeError):
            ILogStorage()


class TestBaseStorage:
    """基础存储接口测试"""

    def test_interface_abstract_methods(self):
        """测试接口定义了抽象方法"""
        # 验证抽象方法存在
        assert hasattr(BaseStorage, 'store')
        assert hasattr(BaseStorage, 'retrieve')
        assert hasattr(BaseStorage, 'delete')
        assert hasattr(BaseStorage, 'count')
        assert hasattr(BaseStorage, 'clear')
        assert hasattr(BaseStorage, 'get_status')

    def test_interface_cannot_instantiate(self):
        """测试抽象类不能直接实例化"""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseStorage()


class TestMemoryStorage:
    """简单内存存储测试"""

    def setup_method(self):
        """测试前准备"""
        self.config = {'max_records': 100}
        self.storage = MemoryStorage(self.config)

    def test_initialization(self):
        """测试初始化"""
        assert len(self.storage._storage) == 0
        assert self.storage._max_records == 100

    def test_store_single_record(self):
        """测试存储单条记录"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'level': 'INFO',
            'message': '测试消息',
            'logger': 'test.logger'
        }

        result = self.storage.store(record)
        assert result is True
        assert len(self.storage._storage) == 1
        assert self.storage._storage[0] == record

    def test_store_multiple_records(self):
        """测试存储多条记录"""
        records = []
        for i in range(5):
            record = {
                'id': i,
                'message': f'消息{i}',
                'timestamp': datetime.now().isoformat()
            }
            records.append(record)
            self.storage.store(record)

        assert len(self.storage._storage) == 5
        for i, record in enumerate(self.storage._storage):
            assert record['id'] == i
            assert record['message'] == f'消息{i}'

    def test_retrieve_all_records(self):
        """测试检索所有记录"""
        # 存储一些记录
        for i in range(3):
            self.storage.store({'id': i, 'data': f'value{i}'})

        results = self.storage.retrieve()
        assert len(results) == 3
        assert results[0]['id'] == 0
        assert results[2]['id'] == 2

    def test_retrieve_with_query(self):
        """测试带查询的检索"""
        # 存储不同类型的记录
        self.storage.store({'type': 'error', 'message': '错误1'})
        self.storage.store({'type': 'info', 'message': '信息1'})
        self.storage.store({'type': 'error', 'message': '错误2'})
        self.storage.store({'type': 'warning', 'message': '警告1'})

        # 查询错误类型的记录
        results = self.storage.retrieve({'type': 'error'})
        assert len(results) == 2
        for result in results:
            assert result['type'] == 'error'

    def test_retrieve_with_limit(self):
        """测试带限制的检索"""
        # 存储10条记录
        for i in range(10):
            self.storage.store({'id': i})

        # 限制返回5条
        results = self.storage.retrieve(limit=5)
        assert len(results) == 5

        # 应该返回最新的5条记录
        assert results[0]['id'] == 5
        assert results[4]['id'] == 9

    def test_retrieve_with_query_and_limit(self):
        """测试带查询和限制的检索"""
        # 存储多种记录
        for i in range(10):
            record_type = 'even' if i % 2 == 0 else 'odd'
            self.storage.store({'id': i, 'type': record_type})

        # 查询even类型，限制2条
        results = self.storage.retrieve({'type': 'even'}, limit=2)
        assert len(results) == 2
        for result in results:
            assert result['type'] == 'even'
            assert result['id'] % 2 == 0

    def test_count_all_records(self):
        """测试统计所有记录数量"""
        assert self.storage.count() == 0

        # 存储记录
        for i in range(5):
            self.storage.store({'id': i})

        assert self.storage.count() == 5

    def test_count_with_query(self):
        """测试带查询的统计"""
        # 存储不同级别的记录
        levels = ['INFO', 'ERROR', 'WARNING', 'DEBUG', 'ERROR', 'INFO']
        for level in levels:
            self.storage.store({'level': level})

        assert self.storage.count({'level': 'ERROR'}) == 2
        assert self.storage.count({'level': 'INFO'}) == 2
        assert self.storage.count({'level': 'WARNING'}) == 1
        assert self.storage.count({'level': 'DEBUG'}) == 1

    def test_delete_records(self):
        """测试删除记录"""
        # 存储记录
        self.storage.store({'type': 'keep', 'id': 1})
        self.storage.store({'type': 'delete', 'id': 2})
        self.storage.store({'type': 'keep', 'id': 3})
        self.storage.store({'type': 'delete', 'id': 4})

        assert self.storage.count() == 4

        # 删除delete类型的记录
        deleted_count = self.storage.delete({'type': 'delete'})
        assert deleted_count == 2
        assert self.storage.count() == 2

        # 验证剩余的记录
        remaining = self.storage.retrieve()
        types = [r['type'] for r in remaining]
        assert 'delete' not in types
        assert types.count('keep') == 2

    def test_clear_storage(self):
        """测试清空存储"""
        # 存储一些记录
        for i in range(5):
            self.storage.store({'id': i})

        assert self.storage.count() == 5

        # 清空存储
        self.storage.clear()
        assert self.storage.count() == 0
        assert len(self.storage._storage) == 0

    def test_capacity_management(self):
        """测试容量管理"""
        # 创建小容量存储
        small_storage = MemoryStorage({'max_records': 3})

        # 存储超过容量的记录
        for i in range(5):
            small_storage.store({'id': i})

        # 应该只保留最新的3条记录
        assert small_storage.count() == 3
        records = small_storage.retrieve()
        assert records[0]['id'] == 2  # 最旧的被移除
        assert records[1]['id'] == 3
        assert records[2]['id'] == 4  # 最新的保留

    def test_get_status(self):
        """测试获取状态"""
        status = self.storage.get_status()

        assert isinstance(status, dict)
        # BaseStorage添加的字段
        assert 'name' in status
        assert 'enabled' in status
        assert 'type' in status
        assert 'max_records' in status
        assert 'compression' in status

        assert status['name'] == 'MemoryStorage'
        assert status['type'] == 'MemoryStorage'

    def test_get_status_functionality(self):
        """测试获取状态功能"""
        # 存储一些记录
        for i in range(10):
            self.storage.store({'id': i})

        status = self.storage.get_status()

        # 验证状态包含必要字段
        assert isinstance(status, dict)
        assert 'name' in status
        assert 'enabled' in status
        assert 'type' in status

    def test_thread_safety_simulation(self):
        """测试线程安全模拟"""
        import threading
        import concurrent.futures

        errors = []

        def worker(worker_id):
            try:
                # 每个线程存储一些记录
                for i in range(10):
                    self.storage.store({
                        'worker': worker_id,
                        'sequence': i,
                        'thread_id': threading.current_thread().ident
                    })
            except Exception as e:
                errors.append(f"Worker {worker_id}: {e}")

        # 启动多个线程
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker, i) for i in range(5)]
            concurrent.futures.wait(futures)

        # 验证没有错误
        assert len(errors) == 0

        # 验证所有记录都被存储
        total_records = self.storage.count()
        assert total_records == 50  # 5 workers * 10 records each

        # 验证记录完整性
        records = self.storage.retrieve()
        worker_counts = {}
        for record in records:
            worker_id = record['worker']
            worker_counts[worker_id] = worker_counts.get(worker_id, 0) + 1

        # 每个worker应该有10条记录
        for worker_id in range(5):
            assert worker_id in worker_counts
            assert worker_counts[worker_id] == 10

    def test_data_integrity(self):
        """测试数据完整性"""
        # 存储复杂数据结构
        complex_record = {
            'timestamp': datetime.now(),
            'level': 'ERROR',
            'message': '复杂错误信息',
            'context': {
                'user': {'id': 12345, 'name': '测试用户'},
                'request': {'method': 'POST', 'url': '/api/test'},
                'error': {'code': 'VALIDATION_ERROR', 'details': ['字段缺失', '类型错误']}
            },
            'tags': ['api', 'validation', 'error'],
            'metadata': {
                'service': 'test_service',
                'version': '1.2.3',
                'environment': 'testing'
            }
        }

        # 存储记录
        self.storage.store(complex_record)

        # 检索记录
        results = self.storage.retrieve()
        assert len(results) == 1

        retrieved = results[0]

        # 验证复杂数据结构完整性
        assert retrieved['level'] == 'ERROR'
        assert retrieved['message'] == '复杂错误信息'
        assert retrieved['context']['user']['id'] == 12345
        assert retrieved['context']['error']['code'] == 'VALIDATION_ERROR'
        assert 'api' in retrieved['tags']
        assert retrieved['metadata']['service'] == 'test_service'

    def test_performance_basic(self):
        """测试基本性能"""
        import time

        # 存储100条记录（受max_records限制）
        start_time = time.time()
        for i in range(100):
            self.storage.store({'id': i, 'data': f'value_{i}'})
        store_time = time.time() - start_time

        # 检索记录
        start_time = time.time()
        results = self.storage.retrieve()
        retrieve_time = time.time() - start_time

        # 验证性能在合理范围内
        assert len(results) == 100
        assert store_time < 1.0  # 存储100条记录应该小于1秒
        assert retrieve_time < 0.1  # 检索应该很快


if __name__ == "__main__":
    pytest.main([__file__])
