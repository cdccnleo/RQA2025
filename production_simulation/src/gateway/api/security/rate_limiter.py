"""限流器模块"""

import logging
import time
import threading
from typing import Dict, Optional
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

# 检查Redis可用性
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis不可用，将使用本地限流")


class RateLimiter:
    """限流器
    
    支持基于令牌桶算法的限流
    可以使用Redis实现分布式限流，或使用本地内存限流
    """
    
    def __init__(self, redis_client: Optional['redis.Redis'] = None):
        """初始化限流器
        
        Args:
            redis_client: Redis客户端，如果提供则使用分布式限流
        """
        self.redis_client = redis_client
        self.local_limits: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.lock = threading.Lock()
    
    def is_allowed(self, rule, key: str) -> bool:
        """检查是否允许请求
        
        Args:
            rule: 限流规则对象（RateLimitRule）
            key: 限流键（如IP地址、用户ID）
            
        Returns:
            如果允许请求返回True，否则返回False
        """
        current_time = int(time.time())
        
        if self.redis_client and REDIS_AVAILABLE:
            return self._check_redis_limit(rule, key, current_time)
        else:
            return self._check_local_limit(rule, key, current_time)
    
    def _check_redis_limit(self, rule, key: str, current_time: int) -> bool:
        """使用Redis检查限流
        
        Args:
            rule: 限流规则
            key: 限流键
            current_time: 当前时间戳
            
        Returns:
            是否允许请求
        """
        try:
            redis_key = f"ratelimit:{rule.limit_type.value}:{key}:{current_time // rule.window}"
            
            # 使用Redis管道执行原子操作
            with self.redis_client.pipeline() as pipe:
                pipe.incr(redis_key)
                pipe.expire(redis_key, rule.window)
                count = pipe.execute()[0]
            
            return count <= rule.limit
        
        except Exception as e:
            logger.error(f"Redis限流检查失败: {e}")
            return True  # 出错时允许请求
    
    def _check_local_limit(self, rule, key: str, current_time: int) -> bool:
        """使用本地内存检查限流
        
        Args:
            rule: 限流规则
            key: 限流键
            current_time: 当前时间戳
            
        Returns:
            是否允许请求
        """
        with self.lock:
            request_times = self.local_limits[key]
            
            # 清理过期请求
            while request_times and request_times[0] < current_time - rule.window:
                request_times.popleft()
            
            # 检查是否超过限制
            if len(request_times) >= rule.limit:
                return False
            
            # 添加当前请求
            request_times.append(current_time)
            return True
    
    def reset(self, key: str):
        """重置指定键的限流计数
        
        Args:
            key: 限流键
        """
        with self.lock:
            if key in self.local_limits:
                self.local_limits[key].clear()
    
    def get_remaining(self, rule, key: str) -> int:
        """获取剩余可用请求数
        
        Args:
            rule: 限流规则
            key: 限流键
            
        Returns:
            剩余可用请求数
        """
        with self.lock:
            current_time = int(time.time())
            request_times = self.local_limits[key]
            
            # 清理过期请求
            while request_times and request_times[0] < current_time - rule.window:
                request_times.popleft()
            
            return max(0, rule.limit - len(request_times))


__all__ = ['RateLimiter']

