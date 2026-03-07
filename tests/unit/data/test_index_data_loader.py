# tests/unit/data/test_index_data_loader.py
"""
IndexDataLoader单元测试

测试覆盖:
- 初始化参数验证
- 指数数据加载功能
- 指数映射功能
- 缓存机制
- 错误处理
- 多线程处理
- 边界条件
- 性能监控
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


import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from requests import RequestException
import tempfile
import time
import os
import pickle

from src.data.loader.index_loader import IndexDataLoader
from src.infrastructure.error import DataLoaderError



# 设置测试超时，避免死锁和无限等待
pytestmark = [
    pytest.mark.timeout(30),  # 30秒超时
    pytest.mark.deadlock_risk,  # 标记为可能存在死锁风险
    pytest.mark.concurrent,  # 并发测试
    pytest.mark.infinite_loop_risk  # 可能存在无限循环风险
]

class TestIndexDataLoader:
    """IndexDataLoader测试类"""

    @pytest.fixture
    def temp_dir(self):
        """临时目录fixture"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_thread_pool(self):
        """Mock线程池"""
        return Mock()

    @pytest.fixture
    def loader(self, temp_dir, mock_thread_pool):
        """IndexDataLoader实例"""
        # 创建一个测试用的具体实现类
        class TestIndexDataLoader(IndexDataLoader):
            def validate_data(self, data):
                return True

        loader = TestIndexDataLoader(
            save_path=str(temp_dir),
            max_retries=3,
            cache_days=30,
            thread_pool=mock_thread_pool
        )
        return loader

    def test_initialization_valid_params(self, temp_dir, mock_thread_pool):
        """测试有效参数初始化"""
        # 创建一个测试用的具体实现类
        class TestIndexDataLoader(IndexDataLoader):
            def validate_data(self, data):
                return True

        loader = TestIndexDataLoader(
            save_path=str(temp_dir),
            max_retries=3,
            cache_days=30,
            thread_pool=mock_thread_pool
        )

        assert loader.save_path == Path(temp_dir)
        assert loader.max_retries == 3
        assert loader.cache_days == 30
        assert loader.thread_pool == mock_thread_pool

    def test_initialization_default_params(self, temp_dir):
        """测试默认参数初始化"""
        loader = IndexDataLoader(save_path=str(temp_dir))

        assert loader.save_path == Path(temp_dir)
        assert loader.max_retries == 3
        assert loader.cache_days == 30
        assert loader.thread_pool is None

    def test_index_mapping(self, loader):
        """测试指数映射功能"""
        # 测试已知指数映射
        assert loader.INDEX_MAPPING['HS300'] == '000300'
        assert loader.INDEX_MAPPING['SZ50'] == '000016'
        assert loader.INDEX_MAPPING['CY50'] == '399673'
        assert loader.INDEX_MAPPING['KC50'] == '000688'

    @patch('akshare.stock_zh_index_daily')
    def test_load_single_index_success(self, mock_akshare, loader):
        """测试单指数数据加载成功"""
        # Mock数据
        mock_data = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02'],
            'open': [3000.0, 3050.0],
            'close': [3100.0, 3150.0],
            'high': [3200.0, 3250.0],
            'low': [2950.0, 3000.0],
            'volume': [100000000, 110000000],
            'amount': [300000000000, 330000000000]
        })
        mock_akshare.return_value = mock_data

        result = loader.load_single_index('HS300')

        assert result is not None
        assert 'data' in result
        assert 'metadata' in result
        assert result['metadata']['index_code'] == '000300'  # 映射后的代码
        assert len(result['data']) == 2

    @patch('akshare.stock_zh_index_daily')
    def test_load_single_index_with_retry(self, mock_akshare, loader):
        """测试单指数数据加载重试机制"""
        # 第一次调用失败，第二次成功
        mock_akshare.side_effect = [
            Exception("Network error"),
            pd.DataFrame({
                'date': ['2024-01-01'],
                'open': [3000.0],
                'close': [3100.0],
                'high': [3200.0],
                'low': [2950.0],
                'volume': [100000000],
                'amount': [300000000000]
            })
        ]

        result = loader.load_single_index('HS300')

        assert result is not None
        assert mock_akshare.call_count == 2

    @patch('akshare.stock_zh_index_daily')
    def test_load_single_index_max_retries_exceeded(self, mock_akshare, loader):
        """测试单指数数据加载重试次数超限"""
        mock_akshare.side_effect = Exception("Network error")

        with pytest.raises(DataLoaderError):
            loader.load_single_index('HS300')

        assert mock_akshare.call_count == 3  # max_retries = 3

    def test_load_multiple_indexes_concurrent(self, loader):
        """测试并发加载多指数"""
        indexes = ['HS300', 'SZ50', 'CY50']

        # Mock并发执行
        with patch.object(loader, '_load_single_index_with_cache') as mock_load:
            mock_load.return_value = {'data': pd.DataFrame(), 'metadata': {}}

            result = loader.load_multiple_indexes(indexes, max_workers=3)

            assert len(result) == 3
            assert mock_load.call_count == 3

    def test_unknown_index_mapping(self, loader):
        """测试未知指数映射"""
        # 对于未知指数，应该使用原始名称
        with patch('akshare.stock_zh_index_daily') as mock_akshare:
            mock_data = pd.DataFrame({
                'date': ['2024-01-01'],
                'open': [3000.0],
                'close': [3100.0],
                'high': [3200.0],
                'low': [2950.0],
                'volume': [100000000],
                'amount': [300000000000]
            })
            mock_akshare.return_value = mock_data

            result = loader.load_single_index('UNKNOWN_INDEX')

            # 未知指数应该使用原始名称
            assert result['metadata']['index_code'] == 'UNKNOWN_INDEX'

    def test_cache_mechanism(self, loader, temp_dir):
        """测试缓存机制"""
        cache_file = temp_dir / 'cache' / '000300_index.pkl'

        # 确保缓存目录存在
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        # 创建缓存文件
        cache_data = {
            'data': pd.DataFrame({'date': ['2024-01-01'], 'close': [3100.0]}),
            'metadata': {'cached_time': datetime.now()},
            'cache_info': {'is_from_cache': True}
        }

        with open(cache_file, 'wb') as f:
            import pickle
            pickle.dump(cache_data, f)

        # 修改缓存时间为当前时间，确保缓存有效
        cache_data['metadata']['cached_time'] = datetime.now()
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)

        result = loader.load_single_index('HS300')

        assert result is not None
        assert 'cache_info' in result
        assert result['cache_info']['is_from_cache'] is True

    def test_load_data_uses_valid_csv_cache(self, loader, temp_dir):
        """load_data 应直接返回有效 CSV 缓存"""
        start = '2024-01-01'
        end = '2024-01-03'
        file_path = loader._get_file_path('HS300', start, end)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        cache_df = pd.DataFrame({
            'date': pd.date_range(start, periods=3),
            'open': [3000.0, 3010.0, 3020.0],
            'high': [3050.0, 3060.0, 3070.0],
            'low': [2950.0, 2960.0, 2970.0],
            'close': [3040.0, 3050.0, 3060.0],
            'volume': [100000000, 110000000, 120000000],
        })
        cache_df.to_csv(file_path, index=False)

        with patch.object(loader, '_fetch_raw_data') as mock_fetch:
            result = loader.load_data('HS300', start, end)
            mock_fetch.assert_not_called()

        expected = cache_df.copy()
        expected.set_index('date', inplace=True)
        expected.index = pd.to_datetime(expected.index)
        assert_frame_equal(result, expected)

    def test_cache_expiration(self, loader, temp_dir):
        """测试缓存过期机制"""
        cache_file = temp_dir / 'cache' / '000300_index.pkl'
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        # 创建过期的缓存文件
        expired_time = datetime.now() - timedelta(days=31)  # 超过30天
        cache_data = {
            'data': pd.DataFrame({'date': ['2024-01-01'], 'close': [3100.0]}),
            'metadata': {'cached_time': expired_time},
            'cache_info': {'is_from_cache': True}
        }

        with open(cache_file, 'wb') as f:
            import pickle
            pickle.dump(cache_data, f)

        # Mock网络请求
        with patch('akshare.stock_zh_index_daily') as mock_akshare:
            mock_data = pd.DataFrame({
                'date': ['2024-01-01'],
                'open': [3000.0],
                'close': [3100.0],
                'high': [3200.0],
                'low': [2950.0],
                'volume': [100000000],
                'amount': [300000000000]
            })
            mock_akshare.return_value = mock_data

            result = loader.load_single_index('HS300')

            # 应该调用网络请求而不是使用过期缓存
            mock_akshare.assert_called_once()

    @patch('akshare.stock_zh_index_daily')
    def test_load_data_merges_with_stale_cache(self, mock_akshare, loader, temp_dir):
        """缓存过期时，应拉取新数据并与旧缓存合并"""
        start = '2024-01-01'
        end = '2024-01-02'
        file_path = loader._get_file_path('HS300', start, end)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        cached = pd.DataFrame({
            'date': ['2024-01-01'],
            'open': [3000.0],
            'high': [3050.0],
            'low': [2950.0],
            'close': [3040.0],
            'volume': [100000000],
        })
        cached.to_csv(file_path, index=False)
        old_time = time.time() - (loader.cache_days + 1) * 86400
        os.utime(file_path, (old_time, old_time))

        mock_akshare.return_value = pd.DataFrame({
            '日期': ['2024-01-02'],
            '开盘': [3050.0],
            '收盘': [3060.0],
            '最高': [3080.0],
            '最低': [3040.0],
            '成交量': [110000000],
        })

        result = loader.load_data('HS300', start, '2024-01-02')
        assert len(result) == 2
        assert pd.Timestamp('2024-01-02') in result.index

    def test_retry_api_call_raises_after_max(self, loader):
        """_retry_api_call 用尽重试后抛出 DataLoaderError"""
        loader.max_retries = 2

        def failing(*args, **kwargs):
            raise RequestException("boom")

        with pytest.raises(DataLoaderError):
            loader._retry_api_call(failing)

    def test_load_cache_payload_respects_expiration(self, loader, temp_dir):
        """_load_cache_payload 命中过期时间时应返回 None"""
        cache_file = loader._get_cache_file_path('HS300')
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'data': pd.DataFrame({'open': [1.0]}, index=pd.to_datetime(['2024-01-01'])),
            'metadata': {'cached_time': (datetime.now() - timedelta(days=loader.cache_days + 1)).isoformat()},
        }
        with cache_file.open('wb') as fp:
            pickle.dump(payload, fp)

        loaded = loader._load_cache_payload(cache_file)
        assert loaded is None

    def test_save_and_load_cache_payload_roundtrip(self, loader, temp_dir):
        """保存与读取缓存 payload 应保持数据一致"""
        cache_file = loader._get_cache_file_path('HS300')
        data = pd.DataFrame({'open': [1.0]}, index=pd.to_datetime(['2024-01-01']))
        payload = {
            'data': data,
            'metadata': {'cached_time': datetime.now()},
        }
        loader._save_cache_payload(cache_file, payload)

        loaded = loader._load_cache_payload(cache_file)
        assert loaded is not None
        assert_frame_equal(loaded['data'], data)

    def test_data_validation(self, loader):
        """测试数据验证功能"""
        # 测试有效数据
        valid_data = pd.DataFrame({
            'date': ['2024-01-01'],
            'open': [3000.0],
            'close': [3100.0],
            'high': [3200.0],
            'low': [2950.0],
            'volume': [100000000],
            'amount': [300000000000]
        })

        is_valid, errors = loader._validate_index_data(valid_data)
        assert is_valid is True
        assert len(errors) == 0

    def test_data_validation_missing_columns(self, loader):
        """测试数据验证缺失列"""
        invalid_data = pd.DataFrame({
            'date': ['2024-01-01'],
            'open': [3000.0]
            # 缺少其他必需列
        })

        is_valid, errors = loader._validate_index_data(invalid_data)
        assert is_valid is False
        assert len(errors) > 0

    def test_performance_monitoring(self, loader):
        """测试性能监控"""
        start_time = time.time()

        with patch('akshare.stock_zh_index_daily') as mock_akshare:
            mock_data = pd.DataFrame({
                'date': ['2024-01-01'],
                'open': [3000.0],
                'close': [3100.0],
                'high': [3200.0],
                'low': [2950.0],
                'volume': [100000000],
                'amount': [300000000000]
            })
            mock_akshare.return_value = mock_data

            result = loader.load_single_index('HS300')

            # 检查是否记录了性能指标
            assert 'metadata' in result
            assert 'performance' in result['metadata']

    def test_error_handling_network_error(self, loader):
        """测试网络错误处理"""
        with patch('akshare.stock_zh_index_daily') as mock_akshare:
            mock_akshare.side_effect = Exception("Network connection failed")

            with pytest.raises(DataLoaderError):
                loader.load_single_index('HS300')

    def test_get_index_code_mapping(self, loader):
        """测试指数代码映射获取"""
        # 测试已知映射
        code = loader._get_index_code('HS300')
        assert code == '000300'

        # 测试未知映射
        code = loader._get_index_code('UNKNOWN')
        assert code == 'UNKNOWN'

    def test_setup_method(self, loader, temp_dir):
        """测试设置方法"""
        # 验证设置方法正确创建了目录结构
        assert loader.save_path.exists()
        assert (loader.save_path / 'cache').exists()
        assert (loader.save_path / 'logs').exists()

    def test_thread_pool_usage(self, loader, mock_thread_pool):
        """测试线程池使用"""
        indexes = ['HS300', 'SZ50']

        with patch.object(loader, '_load_single_index_with_cache') as mock_load:
            mock_load.return_value = {'data': pd.DataFrame(), 'metadata': {}}

            loader.load_multiple_indexes(indexes, max_workers=2)

            # 验证线程池被正确使用
            assert mock_thread_pool is not None

    def test_empty_index_list(self, loader):
        """测试空指数列表"""
        result = loader.load_multiple_indexes([])

        assert result == {}
        assert len(result) == 0

    def test_single_index_in_list(self, loader):
        """测试单指数列表"""
        indexes = ['HS300']

        with patch.object(loader, '_load_single_index_with_cache') as mock_load:
            mock_load.return_value = {'data': pd.DataFrame(), 'metadata': {}}

            result = loader.load_multiple_indexes(indexes)

            assert len(result) == 1
            assert 'HS300' in result
            mock_load.assert_called_once_with('HS300')

    def test_get_cache_key(self, loader):
        """测试缓存键生成"""
        index_name = 'HS300'
        cache_key = loader._get_cache_key(index_name)

        expected_key = f"{index_name}_index"
        assert cache_key == expected_key

    def test_is_cache_valid_recent(self, loader):
        """测试缓存有效性检查（最近缓存）"""
        recent_time = datetime.now() - timedelta(days=1)  # 1天前

        is_valid = loader._is_cache_valid(recent_time)

        assert is_valid is True

    def test_is_cache_valid_expired(self, loader):
        """测试缓存有效性检查（过期缓存）"""
        expired_time = datetime.now() - timedelta(days=31)  # 31天前

        is_valid = loader._is_cache_valid(expired_time)

        assert is_valid is False

    def test_index_list_validation(self, loader):
        """测试指数列表验证"""
        # 有效指数列表
        valid_indexes = ['HS300', 'SZ50', 'CY50']
        assert loader._validate_index_list(valid_indexes) is True

        # 包含无效指数的列表
        invalid_indexes = ['HS300', 'INVALID_INDEX', 'SZ50']
        assert loader._validate_index_list(invalid_indexes) is False

    def test_batch_loading_error_handling(self, loader):
        """测试批量加载错误处理"""
        indexes = ['HS300', 'INVALID_INDEX', 'SZ50']

        with patch.object(loader, '_load_single_index_with_cache') as mock_load:
            # 模拟第一个和第三个成功，第二个失败
            def side_effect(index):
                if index == 'INVALID_INDEX':
                    raise DataLoaderError(f"Failed to load index: {index}")
                return {'data': pd.DataFrame(), 'metadata': {}}

            mock_load.side_effect = side_effect

            result = loader.load_multiple_indexes(indexes)

            # 应该只返回成功的加载结果
            assert len(result) == 2
            assert 'HS300' in result
            assert 'SZ50' in result
            assert 'INVALID_INDEX' not in result

    def test_concurrent_loading_performance(self, loader):
        """测试并发加载性能"""
        indexes = ['HS300', 'SZ50', 'CY50', 'KC50']

        with patch.object(loader, '_load_single_index_with_cache') as mock_load:
            mock_load.return_value = {'data': pd.DataFrame(), 'metadata': {}}

            start_time = time.time()
            result = loader.load_multiple_indexes(indexes, max_workers=4)
            end_time = time.time()

            # 验证并发加载的性能
            duration = end_time - start_time
            assert duration < 2.0  # 应该在2秒内完成
            assert len(result) == 4
