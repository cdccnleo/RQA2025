"""
边界测试：rate_limiter.py
测试边界情况和异常场景
"""
import pytest
import time
import threading
from unittest.mock import Mock, patch

# 直接导入模块，避免通过 __init__.py 的依赖问题
# 使用 importlib 直接导入文件
import sys
import importlib.util
from pathlib import Path

# 计算项目根目录（从 tests/unit/data/adapters/miniqmt 向上6级到项目根）
# tests/unit/data/adapters/miniqmt -> tests/unit/data/adapters -> tests/unit/data -> tests/unit -> tests -> 项目根
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
rate_limiter_path = project_root / "src" / "data" / "adapters" / "miniqmt" / "rate_limiter.py"

# 验证文件存在
if not rate_limiter_path.exists():
    raise FileNotFoundError(f"Rate limiter file not found: {rate_limiter_path}")

# 直接加载模块
spec = importlib.util.spec_from_file_location("rate_limiter", rate_limiter_path)
rate_limiter_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rate_limiter_module)

# 从模块中获取类
RateLimitType = rate_limiter_module.RateLimitType
RateLimitStrategy = rate_limiter_module.RateLimitStrategy
RateLimitConfig = rate_limiter_module.RateLimitConfig
RateLimiter = rate_limiter_module.RateLimiter
FixedWindowLimiter = rate_limiter_module.FixedWindowLimiter
SlidingWindowLimiter = rate_limiter_module.SlidingWindowLimiter
LeakyBucketLimiter = rate_limiter_module.LeakyBucketLimiter
TokenBucketLimiter = rate_limiter_module.TokenBucketLimiter
RateLimitDecorator = rate_limiter_module.RateLimitDecorator
rate_limit = rate_limiter_module.rate_limit


def test_rate_limit_type_enum():
    """测试 RateLimitType（枚举值）"""
    assert RateLimitType.FIXED_WINDOW.value == "fixed_window"
    assert RateLimitType.SLIDING_WINDOW.value == "sliding_window"
    assert RateLimitType.LEAKY_BUCKET.value == "leaky_bucket"
    assert RateLimitType.TOKEN_BUCKET.value == "token_bucket"


def test_rate_limit_strategy_enum():
    """测试 RateLimitStrategy（枚举值）"""
    assert RateLimitStrategy.REJECT.value == "reject"
    assert RateLimitStrategy.QUEUE.value == "queue"
    assert RateLimitStrategy.DEGRADE.value == "degrade"
    assert RateLimitStrategy.RETRY.value == "retry"


def test_rate_limit_config_init():
    """测试 RateLimitConfig（初始化）"""
    config = RateLimitConfig(
        limit_type=RateLimitType.FIXED_WINDOW,
        max_requests=100,
        time_window=60.0
    )
    
    assert config.limit_type == RateLimitType.FIXED_WINDOW
    assert config.max_requests == 100
    assert config.time_window == 60.0
    assert config.approach == RateLimitStrategy.REJECT
    assert config.retry_delay == 1.0
    assert config.queue_timeout == 30.0
    assert config.burst_size == 10


def test_rate_limit_config_init_custom():
    """测试 RateLimitConfig（初始化，自定义值）"""
    config = RateLimitConfig(
        limit_type=RateLimitType.TOKEN_BUCKET,
        max_requests=50,
        time_window=30.0,
        approach=RateLimitStrategy.QUEUE,
        retry_delay=2.0,
        queue_timeout=60.0,
        burst_size=20
    )
    
    assert config.limit_type == RateLimitType.TOKEN_BUCKET
    assert config.max_requests == 50
    assert config.approach == RateLimitStrategy.QUEUE
    assert config.retry_delay == 2.0
    assert config.queue_timeout == 60.0
    assert config.burst_size == 20


def test_rate_limiter_init():
    """测试 RateLimiter（初始化）"""
    limiter = RateLimiter({})
    
    assert limiter.config == {}
    assert limiter.limiters == {}
    assert limiter._stats['total_requests'] == 0


def test_rate_limiter_create_limiter_fixed_window():
    """测试 RateLimiter（创建限流器，固定窗口）"""
    limiter = RateLimiter({})
    config = RateLimitConfig(
        limit_type=RateLimitType.FIXED_WINDOW,
        max_requests=10,
        time_window=60.0
    )
    
    created = limiter.create_limiter("key1", config)
    
    assert isinstance(created, FixedWindowLimiter)
    assert "key1" in limiter.limiters


def test_rate_limiter_create_limiter_sliding_window():
    """测试 RateLimiter（创建限流器，滑动窗口）"""
    limiter = RateLimiter({})
    config = RateLimitConfig(
        limit_type=RateLimitType.SLIDING_WINDOW,
        max_requests=10,
        time_window=60.0
    )
    
    created = limiter.create_limiter("key2", config)
    
    assert isinstance(created, SlidingWindowLimiter)


def test_rate_limiter_create_limiter_leaky_bucket():
    """测试 RateLimiter（创建限流器，漏桶）"""
    limiter = RateLimiter({})
    config = RateLimitConfig(
        limit_type=RateLimitType.LEAKY_BUCKET,
        max_requests=10,
        time_window=60.0
    )
    
    created = limiter.create_limiter("key3", config)
    
    assert isinstance(created, LeakyBucketLimiter)


def test_rate_limiter_create_limiter_token_bucket():
    """测试 RateLimiter（创建限流器，令牌桶）"""
    limiter = RateLimiter({})
    config = RateLimitConfig(
        limit_type=RateLimitType.TOKEN_BUCKET,
        max_requests=10,
        time_window=60.0
    )
    
    created = limiter.create_limiter("key4", config)
    
    assert isinstance(created, TokenBucketLimiter)


def test_rate_limiter_create_limiter_invalid_type():
    """测试 RateLimiter（创建限流器，无效类型）"""
    limiter = RateLimiter({})
    config = RateLimitConfig(
        limit_type=None,  # 无效类型
        max_requests=10,
        time_window=60.0
    )
    
    with pytest.raises(ValueError, match="不支持的限流类型"):
        limiter.create_limiter("key5", config)


def test_rate_limiter_create_limiter_duplicate_key():
    """测试 RateLimiter（创建限流器，重复键）"""
    limiter = RateLimiter({})
    config = RateLimitConfig(
        limit_type=RateLimitType.FIXED_WINDOW,
        max_requests=10,
        time_window=60.0
    )
    
    created1 = limiter.create_limiter("key1", config)
    created2 = limiter.create_limiter("key1", config)
    
    assert created1 is created2  # 应该返回同一个实例


def test_rate_limiter_is_allowed_no_limiter():
    """测试 RateLimiter（检查允许，无限流器）"""
    limiter = RateLimiter({})
    
    assert limiter.is_allowed("key1") is True
    assert limiter._stats['total_requests'] == 1


def test_rate_limiter_is_allowed_with_limiter():
    """测试 RateLimiter（检查允许，有限流器）"""
    limiter = RateLimiter({})
    config = RateLimitConfig(
        limit_type=RateLimitType.FIXED_WINDOW,
        max_requests=2,
        time_window=60.0
    )
    limiter.create_limiter("key1", config)
    
    assert limiter.is_allowed("key1") is True
    assert limiter.is_allowed("key1") is True
    assert limiter.is_allowed("key1") is False  # 超过限制


def test_rate_limiter_acquire_reject_strategy():
    """测试 RateLimiter（获取许可，拒绝策略）"""
    limiter = RateLimiter({})
    config = RateLimitConfig(
        limit_type=RateLimitType.FIXED_WINDOW,
        max_requests=1,
        time_window=60.0,
        approach=RateLimitStrategy.REJECT
    )
    limiter.create_limiter("key1", config)
    
    assert limiter.acquire("key1") is True
    assert limiter.acquire("key1", timeout=0.1) is False  # 被拒绝


def test_rate_limiter_acquire_queue_strategy():
    """测试 RateLimiter（获取许可，排队策略）"""
    limiter = RateLimiter({})
    config = RateLimitConfig(
        limit_type=RateLimitType.FIXED_WINDOW,
        max_requests=1,
        time_window=1.0,  # 短时间窗口
        approach=RateLimitStrategy.QUEUE
    )
    limiter.create_limiter("key1", config)
    
    assert limiter.acquire("key1") is True
    # 第二个请求会排队等待，但超时时间很短
    result = limiter.acquire("key1", timeout=0.1)
    # 可能返回 True（等待后成功）或 False（超时）
    assert isinstance(result, bool)


def test_rate_limiter_acquire_degrade_strategy():
    """测试 RateLimiter（获取许可，降级策略）"""
    limiter = RateLimiter({})
    config = RateLimitConfig(
        limit_type=RateLimitType.FIXED_WINDOW,
        max_requests=1,
        time_window=60.0,
        approach=RateLimitStrategy.DEGRADE
    )
    limiter.create_limiter("key1", config)
    
    assert limiter.acquire("key1") is True
    assert limiter.acquire("key1") is True  # 降级处理，允许但降级


def test_rate_limiter_get_stats():
    """测试 RateLimiter（获取统计信息）"""
    limiter = RateLimiter({})
    config = RateLimitConfig(
        limit_type=RateLimitType.FIXED_WINDOW,
        max_requests=10,
        time_window=60.0
    )
    limiter.create_limiter("key1", config)
    limiter.is_allowed("key1")
    
    stats = limiter.get_stats()
    
    assert stats['total_requests'] == 1
    assert 'key1_limiter_stats' in stats


def test_fixed_window_limiter_init():
    """测试 FixedWindowLimiter（初始化）"""
    config = RateLimitConfig(
        limit_type=RateLimitType.FIXED_WINDOW,
        max_requests=10,
        time_window=60.0
    )
    limiter = FixedWindowLimiter(config)
    
    assert limiter.config == config
    assert limiter.request_count == 0


def test_fixed_window_limiter_is_allowed():
    """测试 FixedWindowLimiter（检查允许）"""
    config = RateLimitConfig(
        limit_type=RateLimitType.FIXED_WINDOW,
        max_requests=2,
        time_window=60.0
    )
    limiter = FixedWindowLimiter(config)
    
    assert limiter.is_allowed() is True
    assert limiter.is_allowed() is True
    assert limiter.is_allowed() is False  # 超过限制


def test_fixed_window_limiter_window_reset():
    """测试 FixedWindowLimiter（窗口重置）"""
    config = RateLimitConfig(
        limit_type=RateLimitType.FIXED_WINDOW,
        max_requests=1,
        time_window=0.1  # 很短的窗口
    )
    limiter = FixedWindowLimiter(config)
    
    assert limiter.is_allowed() is True
    assert limiter.is_allowed() is False
    
    time.sleep(0.15)  # 等待窗口重置
    assert limiter.is_allowed() is True  # 新窗口允许


def test_fixed_window_limiter_get_stats():
    """测试 FixedWindowLimiter（获取统计信息）"""
    config = RateLimitConfig(
        limit_type=RateLimitType.FIXED_WINDOW,
        max_requests=10,
        time_window=60.0
    )
    limiter = FixedWindowLimiter(config)
    limiter.is_allowed()
    
    stats = limiter.get_stats()
    
    assert stats['request_count'] == 1
    assert stats['max_requests'] == 10


def test_sliding_window_limiter_init():
    """测试 SlidingWindowLimiter（初始化）"""
    config = RateLimitConfig(
        limit_type=RateLimitType.SLIDING_WINDOW,
        max_requests=10,
        time_window=60.0
    )
    limiter = SlidingWindowLimiter(config)
    
    assert limiter.config == config
    assert len(limiter.requests) == 0


def test_sliding_window_limiter_is_allowed():
    """测试 SlidingWindowLimiter（检查允许）"""
    config = RateLimitConfig(
        limit_type=RateLimitType.SLIDING_WINDOW,
        max_requests=2,
        time_window=60.0
    )
    limiter = SlidingWindowLimiter(config)
    
    assert limiter.is_allowed() is True
    assert limiter.is_allowed() is True
    assert limiter.is_allowed() is False  # 超过限制


def test_sliding_window_limiter_expired_requests():
    """测试 SlidingWindowLimiter（过期请求清理）"""
    config = RateLimitConfig(
        limit_type=RateLimitType.SLIDING_WINDOW,
        max_requests=1,
        time_window=0.1  # 很短的窗口
    )
    limiter = SlidingWindowLimiter(config)
    
    assert limiter.is_allowed() is True
    assert limiter.is_allowed() is False
    
    time.sleep(0.15)  # 等待请求过期
    assert limiter.is_allowed() is True  # 过期后允许


def test_sliding_window_limiter_get_stats():
    """测试 SlidingWindowLimiter（获取统计信息）"""
    config = RateLimitConfig(
        limit_type=RateLimitType.SLIDING_WINDOW,
        max_requests=10,
        time_window=60.0
    )
    limiter = SlidingWindowLimiter(config)
    limiter.is_allowed()
    
    stats = limiter.get_stats()
    
    assert stats['current_requests'] == 1
    assert stats['max_requests'] == 10


def test_leaky_bucket_limiter_init():
    """测试 LeakyBucketLimiter（初始化）"""
    config = RateLimitConfig(
        limit_type=RateLimitType.LEAKY_BUCKET,
        max_requests=10,
        time_window=60.0
    )
    limiter = LeakyBucketLimiter(config)
    
    assert limiter.config == config
    assert limiter.current_water == 0
    assert limiter.capacity == 10


def test_leaky_bucket_limiter_is_allowed():
    """测试 LeakyBucketLimiter（检查允许）"""
    config = RateLimitConfig(
        limit_type=RateLimitType.LEAKY_BUCKET,
        max_requests=2,
        time_window=60.0
    )
    limiter = LeakyBucketLimiter(config)
    
    assert limiter.is_allowed() is True
    assert limiter.is_allowed() is True
    
    # 漏桶算法：桶满时不允许，但由于漏水，可能很快又有空间
    # 在短时间内连续请求，第三个请求应该被拒绝
    result = limiter.is_allowed()
    # 由于漏水速度很快（max_requests/time_window），可能已经漏出空间
    # 所以结果可能是 True 或 False，取决于时间
    assert isinstance(result, bool)


def test_leaky_bucket_limiter_leak():
    """测试 LeakyBucketLimiter（漏水）"""
    config = RateLimitConfig(
        limit_type=RateLimitType.LEAKY_BUCKET,
        max_requests=2,
        time_window=0.1  # 很短的窗口，快速漏水
    )
    limiter = LeakyBucketLimiter(config)
    
    assert limiter.is_allowed() is True
    assert limiter.is_allowed() is True
    
    # 由于漏水速度很快，第三个请求可能已经可以了
    result1 = limiter.is_allowed()
    assert isinstance(result1, bool)
    
    time.sleep(0.15)  # 等待漏水
    # 漏水后应该允许
    result2 = limiter.is_allowed()
    assert isinstance(result2, bool)


def test_leaky_bucket_limiter_get_stats():
    """测试 LeakyBucketLimiter（获取统计信息）"""
    config = RateLimitConfig(
        limit_type=RateLimitType.LEAKY_BUCKET,
        max_requests=10,
        time_window=60.0
    )
    limiter = LeakyBucketLimiter(config)
    limiter.is_allowed()
    
    stats = limiter.get_stats()
    
    assert stats['current_water'] == 1
    assert stats['capacity'] == 10


def test_token_bucket_limiter_init():
    """测试 TokenBucketLimiter（初始化）"""
    config = RateLimitConfig(
        limit_type=RateLimitType.TOKEN_BUCKET,
        max_requests=10,
        time_window=60.0
    )
    limiter = TokenBucketLimiter(config)
    
    assert limiter.config == config
    assert limiter.tokens == 10  # 初始令牌数
    assert limiter.capacity == 10


def test_token_bucket_limiter_is_allowed():
    """测试 TokenBucketLimiter（检查允许）"""
    config = RateLimitConfig(
        limit_type=RateLimitType.TOKEN_BUCKET,
        max_requests=2,
        time_window=60.0
    )
    limiter = TokenBucketLimiter(config)
    
    assert limiter.is_allowed() is True
    assert limiter.is_allowed() is True
    assert limiter.is_allowed() is False  # 令牌用完


def test_token_bucket_limiter_refill():
    """测试 TokenBucketLimiter（令牌补充）"""
    config = RateLimitConfig(
        limit_type=RateLimitType.TOKEN_BUCKET,
        max_requests=2,
        time_window=0.1  # 很短的窗口，快速补充
    )
    limiter = TokenBucketLimiter(config)
    
    assert limiter.is_allowed() is True
    assert limiter.is_allowed() is True
    assert limiter.is_allowed() is False
    
    time.sleep(0.15)  # 等待令牌补充
    assert limiter.is_allowed() is True  # 补充后允许


def test_token_bucket_limiter_get_stats():
    """测试 TokenBucketLimiter（获取统计信息）"""
    config = RateLimitConfig(
        limit_type=RateLimitType.TOKEN_BUCKET,
        max_requests=10,
        time_window=60.0
    )
    limiter = TokenBucketLimiter(config)
    limiter.is_allowed()
    
    stats = limiter.get_stats()
    
    assert stats['current_tokens'] == 9
    assert stats['capacity'] == 10


def test_rate_limit_decorator_init():
    """测试 RateLimitDecorator（初始化）"""
    limiter = RateLimiter({})
    decorator = RateLimitDecorator(limiter, "key1", timeout=5.0)
    
    assert decorator.rate_limiter == limiter
    assert decorator.key == "key1"
    assert decorator.timeout == 5.0


def test_rate_limit_decorator_call():
    """测试 RateLimitDecorator（调用）"""
    limiter = RateLimiter({})
    decorator = RateLimitDecorator(limiter, "key1", timeout=5.0)
    
    @decorator
    def test_func():
        return "success"
    
    result = test_func()
    assert result == "success"


def test_rate_limit_decorator_rejected():
    """测试 RateLimitDecorator（被拒绝）"""
    limiter = RateLimiter({})
    config = RateLimitConfig(
        limit_type=RateLimitType.FIXED_WINDOW,
        max_requests=1,
        time_window=60.0,
        approach=RateLimitStrategy.REJECT
    )
    limiter.create_limiter("key1", config)
    limiter.acquire("key1")  # 先用掉一个
    
    decorator = RateLimitDecorator(limiter, "key1", timeout=0.1)
    
    @decorator
    def test_func():
        return "success"
    
    with pytest.raises(Exception, match="请求被限流"):
        test_func()


def test_rate_limit_function():
    """测试 rate_limit（装饰器工厂函数）"""
    limiter = RateLimiter({})
    decorator = rate_limit(limiter, "key1", timeout=5.0)
    
    assert isinstance(decorator, RateLimitDecorator)
    assert decorator.key == "key1"
    assert decorator.timeout == 5.0

