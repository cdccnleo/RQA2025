"""
Redis缓存管理器
用于缓存策略性能指标计算结果，提高API响应速度
"""

import logging
import json
import redis
from typing import Dict, Any, Optional
from datetime import timedelta

# 使用统一日志系统
try:
    from src.infrastructure.logging.core.unified_logger import get_unified_logger
    logger = get_unified_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

# Redis配置
REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0,
    'decode_responses': True
}

# 缓存键前缀
CACHE_PREFIX = {
    'STRATEGY_COMPARISON': 'strategy:comparison:',
    'PERFORMANCE_METRICS': 'strategy:metrics:',
    'STRATEGY_DETAIL': 'strategy:detail:',
    'EXECUTION_STATUS': 'execution:status:',
    'EXECUTION_METRICS': 'execution:metrics:',
    'REALTIME_SIGNALS': 'realtime:signals:'
}

# 缓存过期时间
CACHE_EXPIRY = {
    'STRATEGY_COMPARISON': timedelta(minutes=5),
    'PERFORMANCE_METRICS': timedelta(minutes=5),
    'STRATEGY_DETAIL': timedelta(minutes=10),
    'EXECUTION_STATUS': timedelta(seconds=30),  # 执行状态需要更频繁更新
    'EXECUTION_METRICS': timedelta(seconds=30),  # 性能指标需要更频繁更新
    'REALTIME_SIGNALS': timedelta(minutes=1)    # 信号数据更新频率适中
}

class RedisCacheManager:
    """
    Redis缓存管理器
    用于缓存策略性能指标计算结果
    """
    
    def __init__(self):
        """
        初始化Redis连接
        """
        self.redis_client = None
        self._connect()
    
    def _connect(self):
        """
        连接Redis服务器
        """
        try:
            self.redis_client = redis.Redis(**REDIS_CONFIG)
            # 测试连接
            self.redis_client.ping()
            logger.info("Redis连接成功")
        except Exception as e:
            logger.warning(f"Redis连接失败: {e}")
            self.redis_client = None
    
    def is_available(self) -> bool:
        """
        检查Redis是否可用
        
        Returns:
            bool: Redis是否可用
        """
        if not self.redis_client:
            return False
        
        try:
            self.redis_client.ping()
            return True
        except Exception:
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存数据
        
        Args:
            key: 缓存键
            
        Returns:
            Any: 缓存的数据，如果不存在返回None
        """
        if not self.is_available():
            return None
        
        try:
            data = self.redis_client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"获取缓存失败: {e}")
            return None
    
    def set(self, key: str, value: Any, expiry: Optional[timedelta] = None) -> bool:
        """
        设置缓存数据
        
        Args:
            key: 缓存键
            value: 缓存值
            expiry: 过期时间
            
        Returns:
            bool: 是否成功设置
        """
        if not self.is_available():
            return False
        
        try:
            data = json.dumps(value)
            if expiry:
                self.redis_client.setex(key, int(expiry.total_seconds()), data)
            else:
                self.redis_client.set(key, data)
            return True
        except Exception as e:
            logger.error(f"设置缓存失败: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        删除缓存数据
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 是否成功删除
        """
        if not self.is_available():
            return False
        
        try:
            self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"删除缓存失败: {e}")
            return False
    
    def delete_pattern(self, pattern: str) -> bool:
        """
        删除匹配模式的所有缓存
        
        Args:
            pattern: 键模式，如 "strategy:*"
            
        Returns:
            bool: 是否成功删除
        """
        if not self.is_available():
            return False
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
            return True
        except Exception as e:
            logger.error(f"删除缓存失败: {e}")
            return False
    
    def clear_strategy_cache(self) -> bool:
        """
        清除所有策略相关的缓存
        
        Returns:
            bool: 是否成功清除
        """
        patterns = [
            f"{CACHE_PREFIX['STRATEGY_COMPARISON']}*",
            f"{CACHE_PREFIX['PERFORMANCE_METRICS']}*",
            f"{CACHE_PREFIX['STRATEGY_DETAIL']}*"
        ]
        
        success = True
        for pattern in patterns:
            if not self.delete_pattern(pattern):
                success = False
        
        return success

# 全局Redis缓存管理器实例
redis_cache_manager = RedisCacheManager()

# 便捷函数
def get_strategy_comparison_cache() -> Optional[Any]:
    """
    获取策略对比缓存
    
    Returns:
        Any: 缓存的策略对比数据
    """
    key = f"{CACHE_PREFIX['STRATEGY_COMPARISON']}all"
    return redis_cache_manager.get(key)

def set_strategy_comparison_cache(data: Any) -> bool:
    """
    设置策略对比缓存
    
    Args:
        data: 策略对比数据
        
    Returns:
        bool: 是否成功设置
    """
    key = f"{CACHE_PREFIX['STRATEGY_COMPARISON']}all"
    return redis_cache_manager.set(
        key, 
        data, 
        CACHE_EXPIRY['STRATEGY_COMPARISON']
    )

def get_performance_metrics_cache() -> Optional[Any]:
    """
    获取性能指标缓存
    
    Returns:
        Any: 缓存的性能指标数据
    """
    key = f"{CACHE_PREFIX['PERFORMANCE_METRICS']}all"
    return redis_cache_manager.get(key)

def set_performance_metrics_cache(data: Any) -> bool:
    """
    设置性能指标缓存
    
    Args:
        data: 性能指标数据
        
    Returns:
        bool: 是否成功设置
    """
    key = f"{CACHE_PREFIX['PERFORMANCE_METRICS']}all"
    return redis_cache_manager.set(
        key, 
        data, 
        CACHE_EXPIRY['PERFORMANCE_METRICS']
    )

def get_strategy_detail_cache(strategy_id: str) -> Optional[Any]:
    """
    获取策略详情缓存
    
    Args:
        strategy_id: 策略ID
        
    Returns:
        Any: 缓存的策略详情数据
    """
    key = f"{CACHE_PREFIX['STRATEGY_DETAIL']}{strategy_id}"
    return redis_cache_manager.get(key)

def set_strategy_detail_cache(strategy_id: str, data: Any) -> bool:
    """
    设置策略详情缓存
    
    Args:
        strategy_id: 策略ID
        data: 策略详情数据
        
    Returns:
        bool: 是否成功设置
    """
    key = f"{CACHE_PREFIX['STRATEGY_DETAIL']}{strategy_id}"
    return redis_cache_manager.set(
        key, 
        data, 
        CACHE_EXPIRY['STRATEGY_DETAIL']
    )

def clear_strategy_caches() -> bool:
    """
    清除所有策略相关的缓存
    
    Returns:
        bool: 是否成功清除
    """
    return redis_cache_manager.clear_strategy_cache()

# 执行状态相关缓存函数
def get_execution_status_cache() -> Optional[Any]:
    """
    获取执行状态缓存
    
    Returns:
        Any: 缓存的执行状态数据
    """
    key = f"{CACHE_PREFIX['EXECUTION_STATUS']}all"
    return redis_cache_manager.get(key)

def set_execution_status_cache(data: Any) -> bool:
    """
    设置执行状态缓存
    
    Args:
        data: 执行状态数据
        
    Returns:
        bool: 是否成功设置
    """
    key = f"{CACHE_PREFIX['EXECUTION_STATUS']}all"
    return redis_cache_manager.set(
        key, 
        data, 
        CACHE_EXPIRY['EXECUTION_STATUS']
    )

# 执行指标相关缓存函数
def get_execution_metrics_cache() -> Optional[Any]:
    """
    获取执行指标缓存
    
    Returns:
        Any: 缓存的执行指标数据
    """
    key = f"{CACHE_PREFIX['EXECUTION_METRICS']}all"
    return redis_cache_manager.get(key)

def set_execution_metrics_cache(data: Any) -> bool:
    """
    设置执行指标缓存
    
    Args:
        data: 执行指标数据
        
    Returns:
        bool: 是否成功设置
    """
    key = f"{CACHE_PREFIX['EXECUTION_METRICS']}all"
    return redis_cache_manager.set(
        key, 
        data, 
        CACHE_EXPIRY['EXECUTION_METRICS']
    )

# 实时信号相关缓存函数
def get_realtime_signals_cache() -> Optional[Any]:
    """
    获取实时信号缓存
    
    Returns:
        Any: 缓存的实时信号数据
    """
    key = f"{CACHE_PREFIX['REALTIME_SIGNALS']}recent"
    return redis_cache_manager.get(key)

def set_realtime_signals_cache(data: Any) -> bool:
    """
    设置实时信号缓存
    
    Args:
        data: 实时信号数据
        
    Returns:
        bool: 是否成功设置
    """
    key = f"{CACHE_PREFIX['REALTIME_SIGNALS']}recent"
    return redis_cache_manager.set(
        key, 
        data, 
        CACHE_EXPIRY['REALTIME_SIGNALS']
    )

# 清除执行相关缓存
def clear_execution_cache() -> bool:
    """
    清除所有执行相关的缓存
    
    Returns:
        bool: 是否成功清除
    """
    patterns = [
        f"{CACHE_PREFIX['EXECUTION_STATUS']}*",
        f"{CACHE_PREFIX['EXECUTION_METRICS']}*",
        f"{CACHE_PREFIX['REALTIME_SIGNALS']}*"
    ]
    
    success = True
    for pattern in patterns:
        if not redis_cache_manager.delete_pattern(pattern):
            success = False
    
    return success
