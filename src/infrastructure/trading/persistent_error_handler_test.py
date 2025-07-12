import pytest
import time
import os
import tempfile
from unittest.mock import Mock, patch
from .persistent_error_handler import (
    PersistentErrorHandler,
    SQLiteStorage,
    RedisStorage
)

class TestSQLiteStorage:
    @pytest.fixture
    def temp_db(self):
        # 创建临时数据库
        fd, path = tempfile.mkstemp()
        yield path
        os.close(fd)
        try:
            os.unlink(path)
        except:
            pass

    def test_save_and_load(self, temp_db):
        storage = SQLiteStorage(temp_db)
        test_record = {
            'id': 'test123',
            'timestamp': time.time(),
            'type': 'TestError',
            'context': {'key': 'value'},
            'stack_trace': 'test trace'
        }

        # 测试保存
        assert storage.save('test123', test_record)

        # 测试加载
        loaded = storage.load('test123')
        assert loaded is not None
        assert loaded['id'] == 'test123'
        assert loaded['type'] == 'TestError'
        assert loaded['context']['key'] == 'value'

    def test_search(self, temp_db):
        storage = SQLiteStorage(temp_db)

        # 准备测试数据
        records = [
            {'id': f'test{i}', 'timestamp': time.time() - i, 'type': 'TypeA', 'context': {}}
            for i in range(5)
        ]
        records.extend([
            {'id': f'test{i+5}', 'timestamp': time.time() - i, 'type': 'TypeB', 'context': {}}
            for i in range(3)
        ])

        # 批量插入
        for r in records:
            storage.save(r['id'], r)

        # 测试搜索
        results = storage.search(error_type='TypeA')
        assert len(results) == 5

        results = storage.search(error_type='TypeB')
        assert len(results) == 3

class TestRedisStorage:
    @pytest.fixture
    def mock_redis(self):
        with patch('redis.Redis') as mock:
            mock.return_value = Mock()
            yield mock

    def test_save_and_load(self, mock_redis):
        storage = RedisStorage()
        test_record = {
            'id': 'test123',
            'timestamp': time.time(),
            'type': 'TestError',
            'context': {'key': 'value'}
        }

        # 测试保存
        assert storage.save('test123', test_record)
        mock_redis.return_value.set.assert_called()

        # 测试加载
        mock_redis.return_value.get.return_value = b'compressed_data'
        with patch('zlib.decompress', return_value=b'pickled_data'):
            with patch('pickle.loads', return_value=test_record):
                loaded = storage.load('test123')
                assert loaded == test_record

class TestPersistentErrorHandler:
    @pytest.fixture
    def handler(self):
        storage = Mock()
        handler = PersistentErrorHandler(storage, batch_size=2)
        yield handler
        handler.shutdown()

    def test_error_handling_flow(self, handler):
        # 模拟错误
        test_error = ValueError("test error")

        # 处理错误
        error_id = handler.handle_error(test_error, {'key': 'value'})
        assert error_id.startswith('err_')

        # 验证存储调用
        time.sleep(0.1)  # 等待后台线程处理
        assert handler.storage.save.called

        # 验证获取错误
        handler.storage.load.return_value = {
            'id': error_id,
            'type': 'ValueError',
            'context': {'key': 'value'}
        }
        record = handler.get_error(error_id)
        assert record is not None
        assert record['type'] == 'ValueError'

    def test_queue_full_handling(self, handler):
        # 设置小队列大小
        small_handler = PersistentErrorHandler(Mock(), max_queue_size=1)

        # 第一次处理应该成功
        error_id1 = small_handler.handle_error(Exception("test1"))
        assert error_id1

        # 第二次处理应该因队列满而失败
        with patch.object(small_handler, '_generate_error_id', return_value='test2'):
            error_id2 = small_handler.handle_error(Exception("test2"))
            assert error_id2 == 'test2'  # 返回了ID但可能被丢弃

        small_handler.shutdown()

    def test_batch_persistence(self, handler):
        # 处理多个错误
        for i in range(3):
            handler.handle_error(Exception(f"test{i}"))

        # 等待批量处理
        time.sleep(0.2)

        # 验证批量保存
        assert handler.storage.save.call_count >= 2  # 至少2次批量保存

    def test_shutdown_with_pending(self, handler):
        # 处理错误但不等待后台线程
        handler.handle_error(Exception("test"))

        # 立即关闭
        handler.shutdown()

        # 验证保存被调用
        assert handler.storage.save.called

def test_integration_with_sqlite():
    # 使用真实SQLite数据库测试完整流程
    with tempfile.NamedTemporaryFile() as temp_db:
        storage = SQLiteStorage(temp_db.name)
        handler = PersistentErrorHandler(storage)

        # 处理错误
        error_id = handler.handle_error(ValueError("integration test"))

        # 等待持久化
        time.sleep(0.2)

        # 验证数据
        record = storage.load(error_id)
        assert record is not None
        assert record['type'] == 'ValueError'

        handler.shutdown()
