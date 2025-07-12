import pytest
from src.infrastructure.cache.thread_safe_cache import ThreadSafeTTLCache

class TestThreadSafeTTLCache:
    def test_memory_usage_percentage_with_non_zero_values(self):
        """
        Test memory usage percentage calculation when both max_memory and memory_usage are non-zero
        """
        # Setup
        max_memory = 2048
        memory_usage = 1024
        cache = ThreadSafeTTLCache(max_memory=max_memory)
        
        # Mock the memory_usage value since it's normally private
        cache._memory_usage = memory_usage
        
        # Test
        metrics = cache.get_metrics()
        memory_percentage = metrics['memory_usage_percentage']
        
        # Assert
        assert memory_percentage == 50.0, \
            f"Expected memory usage percentage to be 50.0% but got {memory_percentage}%"

    def test_memory_usage_percentage_with_zero_max_memory(self):
        """
        Test memory usage percentage when max_memory is 0
        Expected: Returns 0 when max_memory is 0 regardless of memory_usage
        """
        # Create cache instance with max_memory=0
        cache = ThreadSafeTTLCache(max_memory=0)
        
        # Set memory_usage to 1024 (though in real usage this would happen through operations)
        # Note: This directly accesses the protected attribute for testing purposes
        cache._memory_usage = 1024
        
        # Call the method to test
        metrics = cache.get_metrics()
        
        # Assert the memory usage percentage is 0 when max_memory is 0
        assert metrics['memory_usage_percentage'] == 0

    def test_calculate_eviction_rate_with_some_operations(self):
        """
        Test eviction rate calculation with some operations
        Input: 2 evictions, 3 hits, 5 misses
        Expected Outcome: Returns 0.25 (2/8)
        """
        # Create cache instance
        cache = ThreadSafeTTLCache()
        
        # Set the operation counts directly for testing
        cache._eviction_count = 2
        cache._hit_count = 3
        cache._miss_count = 5
        
        # Calculate eviction rate
        eviction_rate = cache.eviction_rate()
        
        # Assert the expected outcome
        assert eviction_rate == 0.25

    def test_calculate_eviction_rate_with_zero_operations(self):
        """
        Test eviction rate calculation with no operations
        Input: No evictions or other operations
        Expected Outcome: Returns 0.0
        """
        # Initialize cache with arbitrary max_memory (not relevant for this test)
        cache = ThreadSafeTTLCache(max_memory=100)
        
        # Ensure no operations have been performed
        assert cache._eviction_count == 0
        
        # Calculate eviction rate
        rate = cache.eviction_rate()
        
        # Verify the rate is 0.0 when no operations occurred
        assert rate == 0.0

    def test_calculate_hit_rate_with_some_operations(self):
        """
        Test hit rate calculation with some operations
        Input: 4 hits, 1 miss
        Expected Outcome: Returns 0.8 (4/5)
        """
        # Create cache instance
        cache = ThreadSafeTTLCache()
        
        # Simulate operations (4 hits, 1 miss)
        cache._hit_count = 4
        cache._miss_count = 1
        
        # Calculate hit rate
        hit_rate = cache.hit_rate()
        
        # Assert the expected outcome
        assert hit_rate == 0.8

    def test_calculate_hit_rate_with_zero_operations(self):
        """
        Test hit rate calculation with no operations
        Input: No hits or misses recorded
        Expected Outcome: Returns 0.0
        """
        # Initialize the cache with default max_memory
        cache = ThreadSafeTTLCache()
        
        # Verify initial hit rate is 0.0 when no operations have occurred
        assert cache.hit_rate() == 0.0

    def test_get_metrics_after_multiple_operations(self):
        """
        Test getting metrics after various operations:
        - 3 hits
        - 2 misses
        - 1 compression
        - 1 eviction
        Should return correct counts and calculated rates in the metrics dict
        """
        # Initialize cache
        cache = ThreadSafeTTLCache()
        
        # Simulate operations
        for _ in range(3):
            cache._hit_count += 1  # Simulate 3 hits
        for _ in range(2):
            cache._miss_count += 1  # Simulate 2 misses
        cache._compression_count += 1  # Simulate 1 compression
        cache._eviction_count += 1  # Simulate 1 eviction
        
        # Get metrics
        metrics = cache.get_metrics()
        
        # Assert counts
        assert metrics['hit_count'] == 3
        assert metrics['miss_count'] == 2
        assert metrics['compression_count'] == 1
        assert metrics['eviction_count'] == 1
        
        # Assert calculated rates
        total_operations = 3 + 2  # hits + misses
        assert metrics['hit_rate'] == 3 / total_operations
        assert metrics['miss_rate'] == 2 / total_operations

    def test_get_metrics_with_zero_operations(self):
        """
        Test getting metrics with no operations recorded.
        Verifies that all metrics show 0 values when no operations have been performed.
        """
        # Initialize the cache with default max_memory (0)
        cache = ThreadSafeTTLCache()
        
        # Get the metrics
        metrics = cache.get_metrics()
        
        # Assert all metrics are 0
        assert metrics.hit_count == 0
        assert metrics.miss_count == 0
        assert metrics.compression_count == 0
        assert metrics.eviction_count == 0
        assert metrics.memory_usage == 0
        assert metrics.write_count == 0

    def test_update_memory_usage_sets_correct_value(self):
        """
        Test that update_memory_usage correctly sets the _memory_usage value.
        """
        # Arrange
        cache = ThreadSafeTTLCache()
        bytes_used = 1024
        
        # Act
        cache.update_memory_usage(bytes_used)
        
        # Assert
        assert cache._memory_usage == 1024

    def test_record_delete_operation_increments_eviction_count(self):
        """
        Test that calling record_delete() increments the _eviction_count by 1.
        """
        # Arrange
        cache = ThreadSafeTTLCache(max_memory=100)
        initial_eviction_count = cache._eviction_count
        
        # Act
        cache.record_delete()
        
        # Assert
        assert cache._eviction_count == initial_eviction_count + 1

    def test_record_cache_write_operation(self):
        """
        Test recording a cache write operation
        Verifies that _write_count increments to 1 after calling record_set()
        """
        # Initialize the cache with default max_memory
        cache = ThreadSafeTTLCache()
        
        # Verify initial state
        assert cache._write_count == 0
        
        # Perform the operation to test
        cache.record_set()
        
        # Verify the expected outcome
        assert cache._write_count == 1

    def test_record_eviction_operation_increments_count(self):
        """
        Test that calling record_eviction() increments the _eviction_count by 1.
        """
        # Arrange
        cache = ThreadSafeTTLCache()
        initial_count = cache._eviction_count
        
        # Act
        cache.record_eviction()
        
        # Assert
        assert cache._eviction_count == initial_count + 1

    def test_record_compression_operation(self):
        """
        Test recording a compression operation
        Input: Call record_compression() once
        Expected Outcome: _compression_count increments to 1
        """
        # Initialize cache with arbitrary max_memory (not relevant for this test)
        cache = ThreadSafeTTLCache(max_memory=100)
        
        # Verify initial count is 0
        assert cache._compression_count == 0
        
        # Call the method being tested
        cache.record_compression()
        
        # Verify the count has incremented to 1
        assert cache._compression_count == 1

    def test_record_single_cache_miss(self):
        """
        Test recording a single cache miss
        Verifies that _miss_count increments to 1 after calling record_miss()
        """
        # Initialize the cache
        cache = ThreadSafeTTLCache()
        
        # Verify initial state
        assert cache._miss_count == 0
        
        # Call the method being tested
        cache.record_miss()
        
        # Verify the expected outcome
        assert cache._miss_count == 1

    def test_record_multiple_cache_hits(self):
        """
        Test recording multiple cache hits
        Input: Call record_hit() 5 times
        Expected Outcome: _hit_count increments to 5
        """
        # Initialize the cache
        cache = ThreadSafeTTLCache()
        
        # Verify initial hit count is 0
        assert cache._hit_count == 0
        
        # Call record_hit() 5 times
        for _ in range(5):
            cache.record_hit()
        
        # Verify hit count is now 5
        assert cache._hit_count == 5

    def test_record_single_cache_hit(self):
        """
        Test recording a single cache hit
        Input: Call record_hit() once
        Expected Outcome: _hit_count increments to 1
        """
        # Initialize the cache
        cache = ThreadSafeTTLCache()
        
        # Verify initial hit count is 0
        assert cache._hit_count == 0
        
        # Call the method being tested
        cache.record_hit()
        
        # Verify hit count has incremented to 1
        assert cache._hit_count == 1

    def test_initialization_with_custom_max_memory(self):
        """
        Test initialization with custom max_memory value
        Verifies that max_memory is set correctly and all counters are initialized to 0
        """
        # Arrange
        max_memory = 1024
        
        # Act
        cache = ThreadSafeTTLCache(max_memory=max_memory)
        
        # Assert
        assert cache._max_memory == max_memory, "Max memory should be set to 1024"
        assert cache._hit_count == 0, "Hit count should be initialized to 0"
        assert cache._miss_count == 0, "Miss count should be initialized to 0"
        assert cache._compression_count == 0, "Compression count should be initialized to 0"
        assert cache._eviction_count == 0, "Eviction count should be initialized to 0"
        assert cache._memory_usage == 0, "Memory usage should be initialized to 0"
        assert cache._write_count == 0, "Write count should be initialized to 0"

    def test_initialization_with_default_max_memory(self):
        """
        Test initialization with default max_memory value (0)
        Verifies all counters initialized to 0 and max_memory set to 0
        """
        # Create cache instance with default max_memory
        cache = ThreadSafeTTLCache()
        
        # Assert all counters are initialized to 0
        assert cache._hit_count == 0
        assert cache._miss_count == 0
        assert cache._compression_count == 0
        assert cache._eviction_count == 0
        assert cache._memory_usage == 0
        assert cache._write_count == 0
        
        # Assert max_memory is set to default 0
        assert cache._max_memory == 0

    def test_thread_safety_under_concurrent_access(self):
        """
        测试多线程并发读写缓存的线程安全性
        """
        import threading
        cache = ThreadSafeTTLCache(maxsize=100, ttl=2)
        results = []
        def writer(idx):
            cache.set(f"key{idx}", idx)
        def reader(idx):
            try:
                val = cache.get(f"key{idx}")
                results.append(val)
            except KeyError:
                pass
        threads = []
        for i in range(50):
            t = threading.Thread(target=writer, args=(i,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        threads = []
        for i in range(50):
            t = threading.Thread(target=reader, args=(i,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        assert len(results) == 50

    def test_expiry_and_eviction(self):
        """
        测试缓存项过期和淘汰
        """
        import time
        cache = ThreadSafeTTLCache(maxsize=2, ttl=1)
        cache.set("a", 1)
        cache.set("b", 2)
        time.sleep(1.1)
        with pytest.raises(KeyError):
            _ = cache["a"]
        cache.set("c", 3)
        assert "b" not in cache or "a" not in cache
        assert "c" in cache

    def test_bulk_set_and_bulk_get(self):
        """
        测试批量设置和批量获取
        """
        cache = ThreadSafeTTLCache(maxsize=10)
        items = {f"k{i}": i for i in range(5)}
        cache.bulk_set(items)
        result = cache.bulk_get(["k0", "k1", "k2", "k3", "k4", "not_exist"])
        assert result["k0"] == 0
        assert result["k4"] == 4
        assert "not_exist" not in result

    def test_bulk_delete(self):
        """
        测试批量删除
        """
        cache = ThreadSafeTTLCache(maxsize=10)
        for i in range(5):
            cache.set(f"k{i}", i)
        deleted = cache.bulk_delete(["k0", "k1", "k2"])
        assert deleted == 3
        assert "k0" not in cache
        assert "k3" in cache

    def test_set_with_ttl(self):
        """
        测试自定义过期时间写入
        """
        import time
        cache = ThreadSafeTTLCache(maxsize=10, ttl=10)
        cache.set_with_ttl("foo", "bar", ttl=1)
        assert cache["foo"] == "bar"
        time.sleep(1.1)
        with pytest.raises(KeyError):
            _ = cache["foo"]

    def test_memory_limit_eviction(self):
        """
        测试内存限制下自动淘汰
        """
        from src.infrastructure.error.exceptions import CacheError
        
        cache = ThreadSafeTTLCache(maxsize=100, max_memory=64)
        # 插入大对象，触发淘汰
        for i in range(10):
            try:
                cache.set(f"k{i}", "x" * 32)
            except CacheError:
                # 预期行为：内存不足时抛出CacheError
                break
        
        # 验证缓存状态
        assert len(cache) <= 10
        # 验证内存使用情况
        assert cache._memory_usage <= cache._max_memory

    def test_exception_on_oom(self):
        """
        测试内存溢出异常分支
        """
        from src.infrastructure.error.exceptions import CacheError
        
        cache = ThreadSafeTTLCache(maxsize=2, max_memory=1)
        
        # 尝试插入大对象，应该抛出CacheError
        with pytest.raises(CacheError, match="无法释放足够内存"):
            cache.set("a", "x" * 1024)
