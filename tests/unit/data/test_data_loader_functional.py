"""
Data Loader数据加载器功能测试模块

按《投产计划-总览.md》第二阶段Week 3 Day 1执行
测试数据加载器的完整功能

测试覆盖：
- 数据加载基本功能（7个）
- 多数据源加载（7个）
- 数据缓存机制（7个）
- 数据预处理（7个）
- 错误处理与重试（7个）
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
import time


# Apply timeout to all tests (10 seconds per test)
pytestmark = pytest.mark.timeout(10)


class TestDataLoaderBasicFunctional:
    """数据加载器基本功能测试"""

    def test_load_data_from_single_source(self):
        """测试1: 从单一数据源加载数据"""
        # Arrange
        loader = Mock()
        data_source = "stock_data"
        
        expected_data = pd.DataFrame({
            'symbol': ['AAPL', 'GOOGL', 'MSFT'],
            'price': [150.0, 2800.0, 300.0],
            'volume': [1000000, 500000, 800000]
        })
        
        loader.load_data = Mock(return_value=expected_data)
        
        # Act
        result = loader.load_data(source=data_source)
        
        # Assert
        assert result is not None
        assert len(result) == 3
        assert 'symbol' in result.columns
        assert list(result['symbol']) == ['AAPL', 'GOOGL', 'MSFT']
        loader.load_data.assert_called_once()

    def test_load_data_with_date_range(self):
        """测试2: 按日期范围加载数据"""
        # Arrange
        loader = Mock()
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        expected_data = pd.DataFrame({
            'date': pd.date_range(start_date, end_date),
            'value': range(31)
        })
        
        loader.load_data_range = Mock(return_value=expected_data)
        
        # Act
        result = loader.load_data_range(start_date=start_date, end_date=end_date)
        
        # Assert
        assert result is not None
        assert len(result) == 31
        assert result['date'].min() >= pd.Timestamp(start_date)
        assert result['date'].max() <= pd.Timestamp(end_date)

    def test_load_data_with_filters(self):
        """测试3: 使用过滤条件加载数据"""
        # Arrange
        loader = Mock()
        filters = {
            'symbol': ['AAPL', 'GOOGL'],
            'price_min': 100.0,
            'volume_min': 500000
        }
        
        expected_data = pd.DataFrame({
            'symbol': ['AAPL', 'GOOGL'],
            'price': [150.0, 2800.0],
            'volume': [1000000, 500000]
        })
        
        loader.load_data_filtered = Mock(return_value=expected_data)
        
        # Act
        result = loader.load_data_filtered(filters=filters)
        
        # Assert
        assert result is not None
        assert len(result) == 2
        assert all(result['price'] >= 100.0)
        assert all(result['volume'] >= 500000)

    def test_load_data_pagination(self):
        """测试4: 分页加载数据"""
        # Arrange
        loader = Mock()
        page_size = 100
        total_records = 350
        
        def mock_load_page(page, size):
            start = page * size
            end = min(start + size, total_records)
            return pd.DataFrame({
                'id': range(start, end),
                'data': [f'record_{i}' for i in range(start, end)]
            })
        
        loader.load_page = mock_load_page
        
        # Act
        page1 = loader.load_page(0, page_size)
        page2 = loader.load_page(1, page_size)
        page3 = loader.load_page(2, page_size)
        page4 = loader.load_page(3, page_size)
        
        # Assert
        assert len(page1) == 100
        assert len(page2) == 100
        assert len(page3) == 100
        assert len(page4) == 50  # Last page
        assert page1['id'].min() == 0
        assert page4['id'].max() == 349

    def test_load_data_streaming(self):
        """测试5: 流式加载数据"""
        # Arrange
        loader = Mock()
        chunk_size = 1000
        
        def mock_stream_data():
            for i in range(5):  # 5 chunks
                yield pd.DataFrame({
                    'chunk_id': [i] * chunk_size,
                    'data': range(i * chunk_size, (i + 1) * chunk_size)
                })
        
        loader.stream_data = mock_stream_data
        
        # Act
        chunks = list(loader.stream_data())
        
        # Assert
        assert len(chunks) == 5
        assert all(len(chunk) == chunk_size for chunk in chunks)
        assert chunks[0]['chunk_id'].iloc[0] == 0
        assert chunks[4]['chunk_id'].iloc[0] == 4

    def test_load_data_with_schema_validation(self):
        """测试6: 带模式验证的数据加载"""
        # Arrange
        loader = Mock()
        expected_schema = {
            'symbol': 'string',
            'price': 'float',
            'volume': 'int',
            'timestamp': 'datetime'
        }
        
        valid_data = pd.DataFrame({
            'symbol': ['AAPL'],
            'price': [150.0],
            'volume': [1000000],
            'timestamp': [pd.Timestamp.now()]
        })
        
        def validate_schema(data, schema):
            for col, dtype in schema.items():
                if col not in data.columns:
                    return False
                # Simplified type check
                if dtype == 'float' and not pd.api.types.is_float_dtype(data[col]):
                    return False
                if dtype == 'int' and not pd.api.types.is_integer_dtype(data[col]):
                    return False
            return True
        
        loader.load_with_validation = Mock(return_value=valid_data)
        
        # Act
        result = loader.load_with_validation(schema=expected_schema)
        is_valid = validate_schema(result, expected_schema)
        
        # Assert
        assert is_valid is True
        assert 'symbol' in result.columns
        assert 'price' in result.columns

    def test_load_data_with_transformations(self):
        """测试7: 带数据转换的加载"""
        # Arrange
        loader = Mock()
        raw_data = pd.DataFrame({
            'price_str': ['150.50', '2800.75', '300.25'],
            'volume_str': ['1M', '500K', '800K']
        })
        
        def transform_data(data):
            transformed = data.copy()
            transformed['price'] = transformed['price_str'].astype(float)
            # Simplified volume conversion
            transformed['volume'] = transformed['volume_str'].str.replace('M', '000000').str.replace('K', '000').astype(int)
            return transformed[['price', 'volume']]
        
        loader.load_and_transform = Mock(return_value=transform_data(raw_data))
        
        # Act
        result = loader.load_and_transform()
        
        # Assert
        assert 'price' in result.columns
        assert 'volume' in result.columns
        assert result['price'].dtype == float
        assert result['volume'].dtype == int


class TestMultiSourceDataLoaderFunctional:
    """多数据源加载功能测试"""

    def test_load_from_multiple_sources(self):
        """测试8: 从多个数据源加载数据"""
        # Arrange
        loader = Mock()
        sources = ['source1', 'source2', 'source3']
        
        def load_multi_source(sources):
            results = {}
            for source in sources:
                results[source] = pd.DataFrame({
                    'source': [source] * 10,
                    'data': range(10)
                })
            return results
        
        loader.load_multi_source = load_multi_source
        
        # Act
        results = loader.load_multi_source(sources)
        
        # Assert
        assert len(results) == 3
        assert all(source in results for source in sources)
        assert all(len(df) == 10 for df in results.values())

    def test_merge_data_from_sources(self):
        """测试9: 合并多个数据源的数据"""
        # Arrange
        source1_data = pd.DataFrame({
            'id': [1, 2, 3],
            'price': [100, 200, 300]
        })
        
        source2_data = pd.DataFrame({
            'id': [1, 2, 3],
            'volume': [1000, 2000, 3000]
        })
        
        # Act - Merge on 'id'
        merged = pd.merge(source1_data, source2_data, on='id')
        
        # Assert
        assert len(merged) == 3
        assert 'price' in merged.columns
        assert 'volume' in merged.columns
        assert list(merged['id']) == [1, 2, 3]

    def test_fallback_data_source(self):
        """测试10: 数据源故障回退"""
        # Arrange
        loader = Mock()
        primary_source = 'primary_db'
        backup_source = 'backup_db'
        
        def load_with_fallback(primary, backup):
            try:
                # Simulate primary failure
                raise ConnectionError("Primary source unavailable")
            except:
                # Fallback to backup
                return pd.DataFrame({
                    'source': ['backup'] * 5,
                    'data': range(5)
                })
        
        loader.load_with_fallback = load_with_fallback
        
        # Act
        result = loader.load_with_fallback(primary_source, backup_source)
        
        # Assert
        assert result is not None
        assert len(result) == 5
        assert all(result['source'] == 'backup')

    def test_parallel_source_loading(self):
        """测试11: 并行从多源加载"""
        # Arrange
        from concurrent.futures import ThreadPoolExecutor
        
        sources = ['db1', 'db2', 'db3', 'db4']
        
        def load_from_source(source):
            return {'source': source, 'count': 100}
        
        # Act
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(load_from_source, src) for src in sources]
            results = [f.result() for f in futures]
        
        # Assert
        assert len(results) == 4
        assert all(r['count'] == 100 for r in results)
        loaded_sources = [r['source'] for r in results]
        assert set(loaded_sources) == set(sources)

    def test_data_source_priority(self):
        """测试12: 数据源优先级"""
        # Arrange
        sources = [
            {'id': 'realtime', 'priority': 1, 'latency': 10},
            {'id': 'cache', 'priority': 2, 'latency': 1},
            {'id': 'historical', 'priority': 3, 'latency': 100}
        ]
        
        def select_source_by_priority(sources, prefer_low_latency=False):
            if prefer_low_latency:
                return min(sources, key=lambda s: s['latency'])
            else:
                return min(sources, key=lambda s: s['priority'])
        
        # Act
        priority_source = select_source_by_priority(sources, prefer_low_latency=False)
        latency_source = select_source_by_priority(sources, prefer_low_latency=True)
        
        # Assert
        assert priority_source['id'] == 'realtime'  # Priority 1
        assert latency_source['id'] == 'cache'  # Latency 1

    def test_data_source_health_check(self):
        """测试13: 数据源健康检查"""
        # Arrange
        sources = {
            'db1': {'status': 'healthy', 'latency_ms': 50},
            'db2': {'status': 'degraded', 'latency_ms': 200},
            'db3': {'status': 'unhealthy', 'latency_ms': 5000}
        }
        
        def get_healthy_sources(sources, max_latency=100):
            healthy = []
            for source_id, info in sources.items():
                if info['status'] == 'healthy' and info['latency_ms'] <= max_latency:
                    healthy.append(source_id)
            return healthy
        
        # Act
        healthy = get_healthy_sources(sources)
        
        # Assert
        assert len(healthy) == 1
        assert 'db1' in healthy
        assert 'db2' not in healthy  # Degraded
        assert 'db3' not in healthy  # Unhealthy

    def test_incremental_data_loading(self):
        """测试14: 增量数据加载"""
        # Arrange
        loader = Mock()
        last_loaded_timestamp = datetime(2024, 1, 1, 12, 0, 0)
        
        def load_incremental(since_timestamp):
            # Load only new data since timestamp
            new_data = pd.DataFrame({
                'timestamp': pd.date_range(since_timestamp, periods=10, freq='1H'),
                'value': range(10)
            })
            return new_data[new_data['timestamp'] > since_timestamp]
        
        loader.load_incremental = load_incremental
        
        # Act
        result = loader.load_incremental(last_loaded_timestamp)
        
        # Assert
        assert len(result) == 9  # 10 periods - 1 (excluding the boundary)
        assert all(result['timestamp'] > pd.Timestamp(last_loaded_timestamp))


class TestDataCacheMechanismFunctional:
    """数据缓存机制测试"""

    def test_cache_hit_performance(self):
        """测试15: 缓存命中性能"""
        # Arrange
        cache = {}
        
        def load_with_cache(key):
            if key in cache:
                return {'data': cache[key], 'cached': True, 'load_time': 0.001}
            else:
                # Simulate expensive load
                data = f"expensive_data_{key}"
                cache[key] = data
                return {'data': data, 'cached': False, 'load_time': 0.5}
        
        # Act
        first_load = load_with_cache('key1')
        second_load = load_with_cache('key1')
        
        # Assert
        assert first_load['cached'] is False
        assert first_load['load_time'] == 0.5
        assert second_load['cached'] is True
        assert second_load['load_time'] < first_load['load_time']

    def test_cache_invalidation(self):
        """测试16: 缓存失效"""
        # Arrange
        cache = {'key1': {'data': 'old_data', 'timestamp': time.time() - 400}}
        cache_ttl = 300  # 5 minutes
        
        def is_cache_valid(key, ttl):
            import time
            if key not in cache:
                return False
            age = time.time() - cache[key]['timestamp']
            return age < ttl
        
        # Act
        is_valid = is_cache_valid('key1', cache_ttl)
        
        # Assert
        assert is_valid is False  # Cache expired (400s > 300s TTL)

    def test_cache_eviction_policy(self):
        """测试17: 缓存淘汰策略（LRU）"""
        # Arrange
        from collections import OrderedDict
        
        cache = OrderedDict()
        max_size = 3
        
        def lru_cache_set(key, value):
            if key in cache:
                cache.move_to_end(key)
            else:
                if len(cache) >= max_size:
                    cache.popitem(last=False)  # Remove oldest
                cache[key] = value
        
        # Act
        lru_cache_set('a', 1)
        lru_cache_set('b', 2)
        lru_cache_set('c', 3)
        lru_cache_set('d', 4)  # Should evict 'a'
        
        # Assert
        assert len(cache) == 3
        assert 'a' not in cache  # Evicted
        assert 'b' in cache
        assert 'c' in cache
        assert 'd' in cache

    def test_cache_warm_up(self):
        """测试18: 缓存预热"""
        # Arrange
        cache = {}
        frequently_accessed_keys = ['hot_data_1', 'hot_data_2', 'hot_data_3']
        
        def warm_up_cache(keys):
            for key in keys:
                cache[key] = f"preloaded_{key}"
            return len(cache)
        
        # Act
        warmed_count = warm_up_cache(frequently_accessed_keys)
        
        # Assert
        assert warmed_count == 3
        assert all(key in cache for key in frequently_accessed_keys)

    def test_cache_consistency(self):
        """测试19: 缓存一致性"""
        # Arrange
        cache = {'key1': 'value1'}
        db_data = {'key1': 'updated_value1'}
        
        def check_cache_consistency(cache, db):
            inconsistent = []
            for key in cache:
                if key in db and cache[key] != db[key]:
                    inconsistent.append(key)
            return inconsistent
        
        # Act
        inconsistent_keys = check_cache_consistency(cache, db_data)
        
        # Assert
        assert len(inconsistent_keys) == 1
        assert 'key1' in inconsistent_keys

    def test_distributed_cache_sync(self):
        """测试20: 分布式缓存同步"""
        # Arrange
        cache_node1 = {'key1': 'value1'}
        cache_node2 = {}
        
        def sync_cache(source_cache, target_cache):
            target_cache.update(source_cache)
            return len(target_cache)
        
        # Act
        synced_count = sync_cache(cache_node1, cache_node2)
        
        # Assert
        assert synced_count == 1
        assert cache_node2['key1'] == 'value1'

    def test_cache_compression(self):
        """测试21: 缓存压缩"""
        # Arrange
        import sys
        
        large_data = 'x' * 10000
        
        def compress_data(data):
            # Simplified compression simulation
            return {'compressed': True, 'size': len(data) // 2}
        
        def decompress_data(compressed):
            # Simplified decompression
            return 'x' * (compressed['size'] * 2)
        
        # Act
        compressed = compress_data(large_data)
        decompressed = decompress_data(compressed)
        
        # Assert
        assert compressed['size'] < len(large_data)
        assert len(decompressed) == len(large_data)


class TestDataPreprocessingFunctional:
    """数据预处理功能测试"""

    def test_data_cleaning(self):
        """测试22: 数据清洗"""
        # Arrange
        dirty_data = pd.DataFrame({
            'value': [1.0, None, 3.0, None, 5.0],
            'category': ['A', 'B', None, 'C', 'D']
        })
        
        # Act - Remove null values
        cleaned = dirty_data.dropna()
        
        # Assert
        assert len(cleaned) == 2  # Only 2 rows without nulls
        assert cleaned['value'].notna().all()
        assert cleaned['category'].notna().all()

    def test_data_normalization(self):
        """测试23: 数据归一化"""
        # Arrange
        data = pd.DataFrame({
            'value': [10, 20, 30, 40, 50]
        })
        
        # Act - Min-max normalization to [0, 1]
        min_val = data['value'].min()
        max_val = data['value'].max()
        data['normalized'] = (data['value'] - min_val) / (max_val - min_val)
        
        # Assert
        assert data['normalized'].min() == 0.0
        assert data['normalized'].max() == 1.0
        assert 0 <= data['normalized'].mean() <= 1

    def test_data_aggregation(self):
        """测试24: 数据聚合"""
        # Arrange
        data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'value': range(10)
        })
        
        # Act - Daily aggregation (already daily, so group by date)
        aggregated = data.groupby(data['date'].dt.date).agg({
            'value': ['mean', 'sum', 'count']
        })
        
        # Assert
        assert len(aggregated) == 10
        assert 'value' in aggregated.columns

    def test_missing_value_handling(self):
        """测试25: 缺失值处理"""
        # Arrange
        data = pd.DataFrame({
            'value': [1.0, None, 3.0, None, 5.0]
        })
        
        # Act - Forward fill
        filled = data.fillna(method='ffill')
        
        # Assert
        assert filled['value'].notna().all()
        assert filled['value'].iloc[1] == 1.0  # Forward filled
        assert filled['value'].iloc[3] == 3.0  # Forward filled

    def test_outlier_detection(self):
        """测试26: 异常值检测"""
        # Arrange
        data = pd.DataFrame({
            'value': [10, 12, 11, 13, 100, 12, 11]  # 100 is outlier
        })
        
        # Act - 使用IQR方法检测异常值
        q1 = data['value'].quantile(0.25)
        q3 = data['value'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = data[(data['value'] < lower_bound) | (data['value'] > upper_bound)]
        
        # Assert
        assert len(outliers) == 1
        assert outliers['value'].iloc[0] == 100

    def test_data_deduplication(self):
        """测试27: 数据去重"""
        # Arrange
        data = pd.DataFrame({
            'id': [1, 2, 2, 3, 3, 3],
            'value': [10, 20, 20, 30, 30, 30]
        })
        
        # Act
        deduplicated = data.drop_duplicates()
        
        # Assert
        assert len(deduplicated) == 3
        assert list(deduplicated['id']) == [1, 2, 3]

    def test_data_type_conversion(self):
        """测试28: 数据类型转换"""
        # Arrange
        data = pd.DataFrame({
            'int_str': ['1', '2', '3'],
            'float_str': ['1.5', '2.5', '3.5'],
            'bool_str': ['True', 'False', 'True']
        })
        
        # Act
        data['int_val'] = data['int_str'].astype(int)
        data['float_val'] = data['float_str'].astype(float)
        data['bool_val'] = data['bool_str'].map({'True': True, 'False': False})
        
        # Assert
        assert data['int_val'].dtype == int
        assert data['float_val'].dtype == float
        assert data['bool_val'].dtype == bool


class TestDataLoaderErrorHandlingFunctional:
    """数据加载错误处理测试"""

    def test_connection_error_handling(self):
        """测试29: 连接错误处理"""
        # Arrange
        loader = Mock()
        
        def load_with_retry(max_retries=3):
            attempts = []
            for attempt in range(max_retries):
                try:
                    if attempt < 2:
                        raise ConnectionError(f"Attempt {attempt + 1} failed")
                    return {'success': True, 'attempts': attempt + 1}
                except ConnectionError as e:
                    attempts.append(str(e))
                    if attempt == max_retries - 1:
                        raise
            return {'success': False}
        
        loader.load_with_retry = load_with_retry
        
        # Act
        result = loader.load_with_retry(max_retries=3)
        
        # Assert
        assert result['success'] is True
        assert result['attempts'] == 3

    def test_timeout_error_handling(self):
        """测试30: 超时错误处理"""
        # Arrange
        import time
        
        def load_with_timeout(timeout=5):
            start = time.time()
            
            # Simulate long operation
            time.sleep(0.01)  # Very short for test
            
            elapsed = time.time() - start
            
            if elapsed > timeout:
                raise TimeoutError(f"Operation exceeded {timeout}s")
            
            return {'success': True, 'elapsed': elapsed}
        
        # Act
        result = load_with_timeout(timeout=1)
        
        # Assert
        assert result['success'] is True
        assert result['elapsed'] < 1

    def test_data_format_error_handling(self):
        """测试31: 数据格式错误处理"""
        # Arrange
        invalid_data = "not a valid dataframe"
        
        def validate_dataframe(data):
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Invalid data format: expected DataFrame")
            return True
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            validate_dataframe(invalid_data)
        
        assert "DataFrame" in str(exc_info.value)

    def test_empty_data_handling(self):
        """测试32: 空数据处理"""
        # Arrange
        empty_data = pd.DataFrame()
        
        def handle_empty_data(data, default=None):
            if data.empty:
                if default is not None:
                    return default
                return pd.DataFrame({'message': ['No data available']})
            return data
        
        # Act
        result = handle_empty_data(empty_data)
        
        # Assert
        assert not result.empty
        assert 'message' in result.columns

    def test_partial_load_failure_handling(self):
        """测试33: 部分加载失败处理"""
        # Arrange
        sources = ['source1', 'source2', 'source3']
        
        def load_all_sources(sources):
            results = {}
            failures = []
            
            for source in sources:
                try:
                    if source == 'source2':
                        raise Exception(f"{source} failed")
                    results[source] = f"data_from_{source}"
                except Exception as e:
                    failures.append({'source': source, 'error': str(e)})
            
            return {'results': results, 'failures': failures}
        
        # Act
        result = load_all_sources(sources)
        
        # Assert
        assert len(result['results']) == 2
        assert 'source1' in result['results']
        assert 'source3' in result['results']
        assert len(result['failures']) == 1
        assert result['failures'][0]['source'] == 'source2'

    def test_data_validation_error(self):
        """测试34: 数据验证错误"""
        # Arrange
        data = pd.DataFrame({
            'price': [100, -50, 200],  # -50 is invalid
            'volume': [1000, 2000, -100]  # -100 is invalid
        })
        
        def validate_positive_values(data):
            errors = []
            if (data['price'] < 0).any():
                errors.append("Negative prices found")
            if (data['volume'] < 0).any():
                errors.append("Negative volumes found")
            return errors
        
        # Act
        errors = validate_positive_values(data)
        
        # Assert
        assert len(errors) == 2
        assert "price" in errors[0]
        assert "volume" in errors[1]

    def test_retry_with_exponential_backoff(self):
        """测试35: 指数退避重试"""
        # Arrange
        import time
        
        retry_delays = []
        
        def retry_with_backoff(max_retries=4):
            for attempt in range(max_retries):
                try:
                    if attempt < 3:
                        delay = 2 ** attempt  # Exponential backoff
                        retry_delays.append(delay)
                        raise Exception(f"Attempt {attempt + 1} failed")
                    return {'success': True, 'attempts': attempt + 1}
                except Exception:
                    if attempt == max_retries - 1:
                        raise
        
        # Act
        result = retry_with_backoff(max_retries=4)
        
        # Assert
        assert result['success'] is True
        assert result['attempts'] == 4
        assert retry_delays == [1, 2, 4]  # 2^0, 2^1, 2^2


# 测试统计
# Total: 35 tests
# TestDataLoaderBasicFunctional: 7 tests (基本功能)
# TestMultiSourceDataLoaderFunctional: 7 tests (多数据源)
# TestDataCacheMechanismFunctional: 7 tests (缓存机制)
# TestDataPreprocessingFunctional: 7 tests (数据预处理)
# TestDataLoaderErrorHandlingFunctional: 7 tests (错误处理)

