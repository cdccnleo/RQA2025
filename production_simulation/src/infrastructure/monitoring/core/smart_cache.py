#!/usr/bin/env python3
"""
RQA2025 基础设施层智能缓存策略

提供基于访问模式的智能缓存管理，包括缓存预热、失效策略优化和性能监控。
这是Phase 3高级功能实现的一部分。
"""

import logging
from typing import Dict, Any, Optional, List, Callable, Set, Tuple
import threading
import time
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict
import hashlib
import json

logger = logging.getLogger(__name__)


class CacheEntry:
    """缓存条目"""

    def __init__(self, key: str, value: Any, ttl_seconds: Optional[int] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        初始化缓存条目

        Args:
            key: 缓存键
            value: 缓存值
            ttl_seconds: 生存时间（秒）
            metadata: 元数据
        """
        self.key = key
        self.value = value
        self.created_at = datetime.now()
        self.accessed_at = datetime.now()
        self.access_count = 0
        self.ttl_seconds = ttl_seconds
        self.metadata = metadata or {}
        self.size_bytes = self._estimate_size()

    def _estimate_size(self) -> int:
        """估算条目大小（字节）"""
        try:
            # 简单的JSON序列化大小估算
            data = json.dumps({
                'key': self.key,
                'value': self.value,
                'metadata': self.metadata
            }, default=str)
            return len(data.encode('utf-8'))
        except:
            return 1024  # 默认大小

    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds

    def access(self):
        """记录访问"""
        self.accessed_at = datetime.now()
        self.access_count += 1

    def get_age_seconds(self) -> float:
        """获取年龄（秒）"""
        return (datetime.now() - self.created_at).total_seconds()

    def get_idle_seconds(self) -> float:
        """获取空闲时间（秒）"""
        return (datetime.now() - self.accessed_at).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'key': self.key,
            'value': self.value,
            'created_at': self.created_at.isoformat(),
            'accessed_at': self.accessed_at.isoformat(),
            'access_count': self.access_count,
            'ttl_seconds': self.ttl_seconds,
            'metadata': self.metadata,
            'size_bytes': self.size_bytes
        }


class SmartCache:
    """
    智能缓存

    基于访问模式的智能缓存管理，支持多种缓存策略和性能监控。
    """

    def __init__(self, max_size_mb: float = 100, default_ttl_seconds: int = 3600,
                 enable_monitoring: bool = True):
        """
        初始化智能缓存

        Args:
            max_size_mb: 最大缓存大小（MB）
            default_ttl_seconds: 默认TTL（秒）
            enable_monitoring: 是否启用监控
        """
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.default_ttl_seconds = default_ttl_seconds
        self.enable_monitoring = enable_monitoring

        # 缓存存储
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        # 访问模式分析
        self.access_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self.pattern_lock = threading.RLock()

        # 统计信息
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'sets': 0,
            'deletes': 0,
            'size_bytes': 0,
            'start_time': datetime.now()
        }

        # 预热和清理
        self.warmup_data: Dict[str, Any] = {}
        self.cleanup_thread = None
        self.stop_event = threading.Event()

        # 启动清理线程
        self._start_cleanup_thread()

        logger.info(f"智能缓存初始化完成，最大大小: {max_size_mb}MB")

    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值

        Args:
            key: 缓存键

        Returns:
            Optional[Any]: 缓存值，如果不存在则返回None
        """
        with self._lock:
            if key in self.cache:
                entry = self.cache[key]

                # 检查过期
                if entry.is_expired():
                    self._remove_entry(key)
                    self.stats['misses'] += 1
                    return None

                # 记录访问
                entry.access()
                self._record_access_pattern(key)

                # 移动到最近使用位置（LRU）
                self.cache.move_to_end(key)

                self.stats['hits'] += 1
                logger.debug(f"缓存命中: {key}")
                return entry.value
            else:
                self.stats['misses'] += 1
                logger.debug(f"缓存未命中: {key}")
                return None

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None,
            metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        设置缓存值

        Args:
            key: 缓存键
            value: 缓存值
            ttl_seconds: TTL（秒），如果为None则使用默认值
            metadata: 元数据

        Returns:
            bool: 是否设置成功
        """
        with self._lock:
            try:
                ttl = ttl_seconds or self.default_ttl_seconds
                entry = CacheEntry(key, value, ttl, metadata)

                # 检查大小限制
                if entry.size_bytes + self.stats['size_bytes'] > self.max_size_bytes:
                    self._evict_entries(entry.size_bytes)

                # 设置缓存
                self.cache[key] = entry
                self.cache.move_to_end(key)  # LRU位置
                self.stats['sets'] += 1
                self.stats['size_bytes'] += entry.size_bytes

                logger.debug(f"设置缓存: {key}, 大小: {entry.size_bytes} bytes")
                return True

            except Exception as e:
                logger.error(f"设置缓存失败 {key}: {e}")
                return False

    def delete(self, key: str) -> bool:
        """
        删除缓存条目

        Args:
            key: 缓存键

        Returns:
            bool: 是否删除成功
        """
        with self._lock:
            return self._remove_entry(key)

    def clear(self):
        """清空缓存"""
        with self._lock:
            self.cache.clear()
            self.stats['size_bytes'] = 0
            logger.info("缓存已清空")

    def contains(self, key: str) -> bool:
        """
        检查键是否存在

        Args:
            key: 缓存键

        Returns:
            bool: 是否存在
        """
        with self._lock:
            return key in self.cache and not self.cache[key].is_expired()

    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        with self._lock:
            current_time = datetime.now()
            uptime = (current_time - self.stats['start_time']).total_seconds()

            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / max(total_requests, 1)

            return {
                'entries': len(self.cache),
                'size_bytes': self.stats['size_bytes'],
                'size_mb': round(self.stats['size_bytes'] / (1024 * 1024), 2),
                'max_size_mb': round(self.max_size_bytes / (1024 * 1024), 2),
                'utilization_percent': round(self.stats['size_bytes'] / self.max_size_bytes * 100, 2),
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'hit_rate': round(hit_rate * 100, 2),
                'evictions': self.stats['evictions'],
                'sets': self.stats['sets'],
                'deletes': self.stats['deletes'],
                'uptime_seconds': uptime,
                'requests_per_second': round(total_requests / max(uptime, 1), 2)
            }

    def warmup(self, data: Dict[str, Any], ttl_seconds: Optional[int] = None):
        """
        缓存预热

        Args:
            data: 预热数据字典
            ttl_seconds: TTL（秒）
        """
        logger.info(f"开始缓存预热，数据量: {len(data)}")

        for key, value in data.items():
            self.set(key, value, ttl_seconds)

        logger.info("缓存预热完成")

    def get_access_patterns(self) -> Dict[str, Any]:
        """
        获取访问模式分析

        Returns:
            Dict[str, Any]: 访问模式信息
        """
        with self.pattern_lock:
            patterns = {}
            current_time = datetime.now()

            for key, access_times in self.access_patterns.items():
                if not access_times:
                    continue

                # 计算访问频率
                time_span = (current_time - access_times[0]).total_seconds()
                frequency = len(access_times) / max(time_span, 1)

                patterns[key] = {
                    'access_count': len(access_times),
                    'frequency_per_hour': round(frequency * 3600, 2),
                    'last_access': access_times[-1].isoformat(),
                    'time_span_hours': round(time_span / 3600, 2)
                }

            return {
                'total_patterns': len(patterns),
                'patterns': patterns
            }

    def get_recommendations(self) -> List[Dict[str, Any]]:
        """
        获取缓存优化建议

        Returns:
            List[Dict[str, Any]]: 优化建议列表
        """
        stats = self.get_stats()
        patterns = self.get_access_patterns()
        recommendations = []

        # 命中率建议
        if stats['hit_rate'] < 50:
            recommendations.append({
                'type': 'hit_rate_optimization',
                'priority': 'high',
                'title': '缓存命中率偏低',
                'description': f'当前命中率 {stats["hit_rate"]}%，建议优化缓存策略',
                'actions': [
                    '增加缓存大小',
                    '调整TTL策略',
                    '预热热点数据'
                ]
            })

        # 大小利用率建议
        if stats['utilization_percent'] > 90:
            recommendations.append({
                'type': 'size_optimization',
                'priority': 'medium',
                'title': '缓存空间利用率过高',
                'description': f'缓存利用率 {stats["utilization_percent"]}%，可能导致频繁淘汰',
                'actions': [
                    '增加缓存大小',
                    '优化淘汰策略',
                    '清理过期数据'
                ]
            })

        # 驱逐频率建议
        if stats['evictions'] > stats['sets'] * 0.1:  # 驱逐超过设置的10%
            recommendations.append({
                'type': 'eviction_optimization',
                'priority': 'medium',
                'title': '缓存驱逐过于频繁',
                'description': f'驱逐次数 {stats["evictions"]}，建议调整缓存策略',
                'actions': [
                    '增加缓存容量',
                    '优化TTL设置',
                    '改进访问模式'
                ]
            })

        # 访问模式建议
        hot_keys = [k for k, v in patterns.get('patterns', {}).items()
                   if v['frequency_per_hour'] > 10]  # 每小时访问超过10次

        if hot_keys:
            recommendations.append({
                'type': 'access_pattern',
                'priority': 'low',
                'title': '发现热点访问模式',
                'description': f'发现 {len(hot_keys)} 个热点键，建议优化这些数据的缓存策略',
                'actions': [
                    '为热点数据设置更长的TTL',
                    '考虑内存预留',
                    '监控热点数据变化'
                ]
            })

        return recommendations

    def _remove_entry(self, key: str) -> bool:
        """
        移除缓存条目

        Args:
            key: 缓存键

        Returns:
            bool: 是否移除成功
        """
        if key in self.cache:
            entry = self.cache[key]
            self.stats['size_bytes'] -= entry.size_bytes
            del self.cache[key]
            self.stats['deletes'] += 1
            return True
        return False

    def _evict_entries(self, required_space: int):
        """
        淘汰条目以释放空间

        Args:
            required_space: 需要释放的空间
        """
        freed_space = 0
        evicted_count = 0

        # LRU淘汰策略
        while freed_space < required_space and self.cache:
            # 淘汰最少使用的条目
            key, entry = next(iter(self.cache.items()))
            freed_space += entry.size_bytes
            self._remove_entry(key)
            evicted_count += 1

        self.stats['evictions'] += evicted_count

        if evicted_count > 0:
            logger.debug(f"淘汰了 {evicted_count} 个缓存条目，释放 {freed_space} bytes")

    def _record_access_pattern(self, key: str):
        """
        记录访问模式

        Args:
            key: 缓存键
        """
        with self.pattern_lock:
            self.access_patterns[key].append(datetime.now())

            # 限制历史记录数量，避免内存溢出
            if len(self.access_patterns[key]) > 1000:
                self.access_patterns[key] = self.access_patterns[key][-500:]

    def _start_cleanup_thread(self):
        """启动清理线程"""
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            return

        self.cleanup_thread = threading.Thread(
            target=self._cleanup_expired_entries,
            name="SmartCache-Cleanup",
            daemon=True
        )
        self.cleanup_thread.start()

    def _cleanup_expired_entries(self):
        """清理过期条目"""
        while not self.stop_event.is_set():
            try:
                time.sleep(60)  # 每分钟清理一次

                with self._lock:
                    expired_keys = [
                        key for key, entry in self.cache.items()
                        if entry.is_expired()
                    ]

                    for key in expired_keys:
                        self._remove_entry(key)

                    if expired_keys:
                        logger.debug(f"清理了 {len(expired_keys)} 个过期缓存条目")

            except Exception as e:
                logger.error(f"清理过期条目异常: {e}")

    def shutdown(self):
        """关闭缓存"""
        logger.info("正在关闭智能缓存...")

        self.stop_event.set()

        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5)

        self.clear()

        logger.info("智能缓存已关闭")

    def get_health_status(self) -> Dict[str, Any]:
        """
        获取缓存健康状态

        Returns:
            Dict[str, Any]: 健康状态信息
        """
        try:
            stats = self.get_stats()

            issues = []

            # 检查命中率
            if stats['hit_rate'] < 30:
                issues.append(f"缓存命中率过低: {stats['hit_rate']:.1%}")

            # 检查空间利用率
            if stats['utilization_percent'] > 95:
                issues.append(f"缓存空间利用率过高: {stats['utilization_percent']:.1f}%")
            # 检查驱逐率
            if stats['evictions'] > stats['sets']:
                issues.append(f"驱逐过于频繁: {stats['evictions']} 次驱逐 vs {stats['sets']} 次设置")

            # 检查清理线程状态
            if not self.cleanup_thread or not self.cleanup_thread.is_alive():
                issues.append("清理线程未运行")

            return {
                'status': 'healthy' if not issues else 'warning',
                'stats': stats,
                'issues': issues,
                'last_check': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"获取健康状态失败: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }


# 全局智能缓存实例
global_smart_cache = SmartCache()
