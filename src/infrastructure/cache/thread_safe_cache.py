import datetime
import pickle
import sys
import time
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from cachetools import TTLCache
import threading
from typing import Any, Optional, Callable, Dict, Tuple, List, OrderedDict

from concurrent.futures import as_completed
from collections import defaultdict
from typing import Dict, Any
from src.infrastructure.error.exceptions import CacheError

class CacheMonitor:
    """缓存监控类，记录缓存使用指标"""
    def __init__(self, max_memory: int = 0):
        self._hit_count = 0
        self._miss_count = 0
        self._compression_count = 0
        self._eviction_count = 0
        self._memory_usage = 0
        self._write_count = 0
        self._max_memory = max_memory
        # 添加兼容属性
        self._hits = 0
        self._misses = 0
        self._compressions = 0
        self._evictions = 0
        self._writes = 0

    def record_hit(self):
        """记录缓存命中"""
        self._hit_count += 1
        self._hits += 1

    def record_miss(self):
        """记录缓存未命中"""
        self._miss_count += 1
        self._misses += 1

    def record_compression(self):
        """记录压缩操作"""
        self._compression_count += 1
        self._compressions += 1

    def record_eviction(self):
        """记录淘汰操作"""
        self._eviction_count += 1
        self._evictions += 1

    def record_write(self):
        """记录写入操作"""
        self._write_count += 1
        self._writes += 1

    def update_memory_usage(self, usage: int):
        """更新内存使用量"""
        self._memory_usage = usage

    @property
    def hit_rate(self) -> float:
        """计算命中率"""
        total = self._hit_count + self._miss_count
        return self._hit_count / total if total > 0 else 0.0

    @property
    def eviction_rate(self) -> float:
        """计算淘汰率"""
        total = self._hit_count + self._miss_count
        return self._eviction_count / total if total > 0 else 0.0

    @property
    def memory_usage_percent(self) -> float:
        """计算内存使用百分比"""
        if self._max_memory > 0:
            return (self._memory_usage / self._max_memory) * 100
        return 0.0

    def get_metrics(self) -> Dict[str, Any]:
        """获取监控指标"""
        return {
            'hit_count': self._hit_count,
            'miss_count': self._miss_count,
            'compression_count': self._compression_count,
            'eviction_count': self._eviction_count,
            'write_count': self._write_count,
            'memory_usage': self._memory_usage,
            'memory_usage_percentage': self.memory_usage_percent,
            'hit_rate': self.hit_rate,
            'eviction_rate': self.eviction_rate,
            'max_memory': self._max_memory
        }

class ThreadSafeTTLCache:
    """线程安全的TTL缓存实现"""

    def __init__(self, maxsize=1000, ttl=300, max_memory=None, max_memory_mb=None, **kwargs):
        """
        初始化缓存
        
        Args:
            maxsize: 最大缓存项数
            ttl: 默认过期时间(秒)
            max_memory: 最大内存使用量(字节)
            compression_threshold: 压缩阈值(字节)
            enable_lru: 是否启用LRU淘汰
            timer: 时间函数，默认使用time.time
        """
        # 兼容max_memory_mb和max_memory
        if max_memory_mb is not None:
            self._max_memory = int(max_memory_mb * 1024 * 1024)
        elif max_memory is not None:
            self._max_memory = int(max_memory)
        else:
            self._max_memory = 0
        self._compression_threshold = kwargs.get('compression_threshold', 1024 * 1024)
        self._enable_lru = kwargs.get('enable_lru', True)
        self._timer = kwargs.get('timer', time.time)
        self._lock = threading.RLock()
        
        # 初始化TTLCache
        self._cache = TTLCache(maxsize=maxsize, ttl=ttl, timer=self._timer)
        
        # 监控指标
        self._monitor = CacheMonitor(self._max_memory)
        
        # 兼容属性
        self._hit_count = 0
        self._miss_count = 0
        self._compression_count = 0
        self._eviction_count = 0
        self._memory_usage = 0
        self._write_count = 0

    def __getitem__(self, key: str) -> Any:
        """获取缓存项"""
        with self._lock:
            try:
                value = self._cache[key]
                
                # 检查是否为带过期时间的元组
                if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], (int, float)):
                    data, expire_at = value
                    # 检查是否过期
                    if self._timer() > expire_at:
                        # 过期，删除并抛出KeyError
                        del self._cache[key]
                        self._monitor.record_miss()
                        self._miss_count += 1
                        raise KeyError(key)
                    # 未过期，返回数据部分
                    self._monitor.record_hit()
                    self._hit_count += 1
                    return self._decompress(data) if isinstance(data, bytes) else data
                else:
                    # 普通缓存项
                    self._monitor.record_hit()
                    self._hit_count += 1
                    return self._decompress(value) if isinstance(value, bytes) else value
            except KeyError:
                self._monitor.record_miss()
                self._miss_count += 1
                raise

    def __setitem__(self, key: str, value: Any) -> None:
        """设置缓存项"""
        with self._lock:
            # 计算内存使用量
            size = self._get_size(value)
            
            # 检查内存限制
            if self._max_memory > 0 and self._memory_usage + size > self._max_memory:
                self._ensure_memory(size)
            
            # 压缩大对象
            if size > self._compression_threshold:
                value = self._compress(value)
                self._monitor.record_compression()
                self._compression_count += 1
            
            self._cache[key] = value
            self._monitor.record_write()
            self._write_count += 1
            self._update_memory_usage(size)

    def __delitem__(self, key: str) -> None:
        """删除缓存项"""
        with self._lock:
            if key in self._cache:
                size = self._get_size(self._cache[key])
                del self._cache[key]
                self._monitor.record_eviction()
                self._eviction_count += 1
                self._update_memory_usage(-size)

    def __contains__(self, key: str) -> bool:
        """检查键是否存在"""
        return key in self._cache

    def __len__(self) -> int:
        """返回缓存项数量"""
        return len(self._cache)

    def get(self, key, default=None):
        with self._lock:
            value = self._cache.get(key, default)
            # 兼容set_with_ttl写入的元组结构
            if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], (int, float)):
                return value[0]
            return value

    def set(self, key: str, value: Any) -> None:
        """设置缓存项"""
        self[key] = value

    def delete(self, key: str) -> bool:
        """删除缓存项"""
        try:
            del self[key]
            return True
        except KeyError:
            return False

    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._memory_usage = 0

    def keys(self) -> List[str]:
        """获取所有键"""
        return list(self._cache.keys())

    def values(self) -> List[Any]:
        """获取所有值"""
        return [self._decompress(v) if isinstance(v, bytes) else v for v in self._cache.values()]

    def items(self) -> List[Tuple[str, Any]]:
        """获取所有键值对"""
        return [(k, self._decompress(v) if isinstance(v, bytes) else v) for k, v in self._cache.items()]

    def bulk_set(self, items: Dict[str, Any]) -> None:
        """批量设置缓存项"""
        with self._lock:
            for key, value in items.items():
                self[key] = value

    def bulk_get(self, keys: List[str]) -> Dict[str, Any]:
        """批量获取缓存项，部分key不存在时抛KeyError"""
        result = {}
        missing = []
        for key in keys:
            try:
                result[key] = self[key]
            except KeyError:
                missing.append(key)
        if missing:
            raise KeyError(f"Keys not found: {missing}")
        return result

    def bulk_delete(self, keys: List[str]) -> int:
        """批量删除缓存项"""
        deleted_count = 0
        for key in keys:
            if self.delete(key):
                deleted_count += 1
        return deleted_count

    # 兼容属性和方法调用
    @property
    def hit_rate_property(self) -> float:
        return self._monitor.hit_rate

    def hit_rate(self):
        """计算命中率：hits / (hits + misses)"""
        total = self._hit_count + self._miss_count
        return self._hit_count / total if total > 0 else 0.0

    @property
    def eviction_rate_property(self) -> float:
        return self._monitor.eviction_rate

    def eviction_rate(self):
        """计算淘汰率：evictions / (hits + misses)"""
        total = self._hit_count + self._miss_count
        return self._eviction_count / total if total > 0 else 0.0

    def miss_rate(self):
        """计算未命中率：misses / (hits + misses)"""
        total = self._hit_count + self._miss_count
        return self._miss_count / total if total > 0 else 0.0

    def record_compression(self):
        self._compression_count += 1
        self._monitor.record_compression()

    @property
    def memory_usage(self) -> int:
        return self._memory_usage

    @property
    def memory_usage_percent(self) -> float:
        if self._max_memory > 0:
            return (self._memory_usage / self._max_memory) * 100
        return 0.0

    def get_metrics(self):
        """获取监控指标，支持字典和属性两种访问方式"""
        from types import SimpleNamespace
        
        metrics_dict = {
            'hit_count': self._hit_count,
            'miss_count': self._miss_count,
            'compression_count': self._compression_count,
            'eviction_count': self._eviction_count,
            'write_count': self._write_count,
            'memory_usage': self._memory_usage,
            'memory_usage_percentage': self.memory_usage_percent,
            'hit_rate': self.hit_rate(),
            'eviction_rate': self.eviction_rate(),
            'miss_rate': self.miss_rate(),
            'max_memory': self._max_memory
        }
        
        # 创建支持字典访问的对象
        class MetricsDict(dict):
            def __getattr__(self, name):
                return self.get(name)
            
            def __setattr__(self, name, value):
                self[name] = value
        
        return MetricsDict(**metrics_dict)

    # 添加测试期望的方法
    def record_hit(self):
        """记录缓存命中"""
        self._monitor.record_hit()
        self._hit_count += 1

    def record_miss(self):
        """记录缓存未命中"""
        self._monitor.record_miss()
        self._miss_count += 1

    def record_eviction(self):
        """记录淘汰操作"""
        self._monitor.record_eviction()
        self._eviction_count += 1

    def record_set(self):
        """记录设置操作"""
        self._monitor.record_write()
        self._write_count += 1

    def record_delete(self):
        """记录删除操作"""
        self._monitor.record_eviction()
        self._eviction_count += 1

    def update_memory_usage(self, usage: int):
        """更新内存使用量"""
        self._memory_usage = usage
        self._monitor.update_memory_usage(usage)

    def set_with_ttl(self, key, value, ttl):
        """设置带有自定义过期时间的缓存项，ttl非法抛异常"""
        if not isinstance(ttl, (int, float)) or ttl <= 0:
            raise ValueError("ttl必须为正数")
        with self._lock:
            expire_at = self._timer() + ttl
            self._cache[key] = (value, expire_at)
            # 统计写入
            self._write_count += 1
            self._monitor._write_count += 1

    def _get_size(self, obj: Any) -> int:
        """计算对象大小"""
        try:
            return sys.getsizeof(obj)
        except:
            return 1024  # 默认大小

    def _compress(self, data: Any) -> bytes:
        """压缩数据，异常时直接raise"""
        try:
            serialized = pickle.dumps(data)
            return zlib.compress(serialized)
        except Exception as e:
            raise e

    def _decompress(self, data: bytes) -> Any:
        """解压数据，异常时直接raise"""
        try:
            decompressed = zlib.decompress(data)
            return pickle.loads(decompressed)
        except Exception as e:
            raise e

    def _ensure_memory(self, required_size: int) -> None:
        """确保有足够内存"""
        if self._max_memory <= 0:
            return
        
        while self._memory_usage + required_size > self._max_memory:
            if len(self._cache) == 0:
                raise CacheError("无法释放足够内存")
            
            # 淘汰最旧的项
            oldest_key = next(iter(self._cache))
            del self[oldest_key]

    def _update_memory_usage(self, size_delta: int) -> None:
        """更新内存使用量"""
        self._memory_usage = max(0, self._memory_usage + size_delta)
        self._monitor.update_memory_usage(self._memory_usage)

    def update_config(self, **kwargs) -> None:
        """更新缓存配置，非法参数或只读参数抛异常"""
        readonly = {'maxsize', 'ttl'}
        with self._lock:
            for key, value in kwargs.items():
                if key in readonly:
                    raise ValueError(f"参数{key}为只读，禁止修改")
                if key == 'compression_threshold' and not isinstance(value, int):
                    raise TypeError("compression_threshold必须为int类型")
                if hasattr(self, f'_{key}'):
                    setattr(self, f'_{key}', value)
                elif hasattr(self._cache, key):
                    setattr(self._cache, key, value)
                else:
                    raise ValueError(f"未知参数: {key}")

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            'status': 'healthy',
            'cache_size': len(self._cache),
            'memory_usage': self._memory_usage,
            'memory_limit': self._max_memory,
            'hit_rate': self.hit_rate(),
            'eviction_rate': self.eviction_rate()
        }
