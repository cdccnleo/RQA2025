#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 智能缓存优化器

实现智能缓存失效、预加载和性能优化功能
"""

import time
import threading
import asyncio
from typing import Any, Dict, List, Optional, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import logging

from ..interfaces.standard_interfaces import DataSourceType, IDataCache


@dataclass
class CacheEntry:

    """缓存条目"""
    key: str
    data_type: DataSourceType
    value: Any
    timestamp: datetime
    access_count: int = 0
    last_access: datetime = field(default_factory=datetime.now)
    priority: int = 2
    ttl_seconds: int = 3600
    expiry_time: datetime = field(default_factory=lambda: datetime.now() + timedelta(hours=1))

    def is_expired(self) -> bool:
        """检查是否过期"""
        return datetime.now() > self.expiry_time

    def touch(self):
        """更新访问信息"""
        self.access_count += 1
        self.last_access = datetime.now()


@dataclass
class PreloadRule:

    """预加载规则"""
    name: str
    data_type: DataSourceType
    condition: Callable[[Dict[str, Any]], bool]
    preload_func: Callable[[], Any]
    priority: int = 2
    interval_seconds: int = 300
    last_execution: Optional[datetime] = None

    def should_execute(self, context: Dict[str, Any]) -> bool:
        """检查是否应该执行预加载"""
        if not self.condition(context):
            return False

        if self.last_execution is None:
            return True

        time_since_last = (datetime.now() - self.last_execution).total_seconds()
        return time_since_last >= self.interval_seconds

    def execute(self, context: Dict[str, Any]) -> Any:
        """执行预加载"""
        try:
            result = self.preload_func()
            self.last_execution = datetime.now()
            return result
        except Exception as e:
            logging.error(f"预加载执行失败 {self.name}: {e}")
            return None


class SmartCacheOptimizer:

    """
    智能缓存优化器

    提供智能缓存失效、预加载和性能优化功能
    """

    def __init__(self, cache: IDataCache, config: Optional[Any] = None):
        """
        初始化智能缓存优化器

        Args:
            cache: 缓存实例
            config: 配置
        """
        self.cache = cache
        self.config = config

        # 缓存条目管理
        self._cache_entries: Dict[str, CacheEntry] = {}
        self._access_history: deque = deque(maxlen=10000)

        # 预加载规则
        self._preload_rules: List[PreloadRule] = []
        self._preload_tasks: Set[str] = set()

        # 预加载调度器
        self._scheduler_thread: Optional[threading.Thread] = None
        self._stop_scheduler = threading.Event()

        # 初始化
        self._initialize_optimizer()
        self._start_preload_scheduler()

        logging.info("智能缓存优化器初始化完成")

    def _initialize_optimizer(self):
        """初始化优化器"""
        # 初始化预加载规则
        self._initialize_preload_rules()

    def _initialize_preload_rules(self):
        """初始化预加载规则"""
        # 高频数据预加载规则
        self.add_preload_rule(
            name="frequent_stock_data",
            data_type=DataSourceType.STOCK,
            condition=lambda ctx: ctx.get('hour') in [9, 10, 14, 15],
            preload_func=self._preload_frequent_stock_data,
            priority=3,
            interval_seconds=180
        )

    def _start_preload_scheduler(self):
        """启动预加载调度器"""
        self._stop_scheduler.clear()
        self._scheduler_thread = threading.Thread(
            target=self._preload_scheduler_loop,
            daemon=True
        )
        self._scheduler_thread.start()

    def _preload_scheduler_loop(self):
        """预加载调度器循环"""
        while not self._stop_scheduler.is_set():
            try:
                # 获取当前上下文
                context = self._get_current_context()

                # 检查和执行预加载规则
                for rule in self._preload_rules:
                    if rule.should_execute(context):
                        asyncio.run(self._execute_preload_async(rule, context))

                # 等待下一次检查
                time.sleep(30)

            except Exception as e:
                logging.error(f"预加载调度器错误: {e}")
                time.sleep(60)

    def _get_current_context(self) -> Dict[str, Any]:
        """获取当前上下文"""
        now = datetime.now()
        return {
            'hour': now.hour,
            'day_of_week': now.weekday(),
            'market_active': self._is_market_active()
        }

    def _is_market_active(self) -> bool:
        """检查市场是否活跃"""
        now = datetime.now()
        hour = now.hour

        # 亚洲市场时段 (0 - 8 UTC)
        if 0 <= hour <= 8:
            return True
        # 欧洲市场时段 (8 - 16 UTC)
        elif 8 <= hour <= 16:
            return True
        # 美国市场时段 (14 - 21 UTC)
        elif 14 <= hour <= 21:
            return True

        return False

    async def _execute_preload_async(self, rule: PreloadRule, context: Dict[str, Any]):
        """异步执行预加载"""
        task_id = f"{rule.name}_{datetime.now().isoformat()}"

        if task_id in self._preload_tasks:
            return

        self._preload_tasks.add(task_id)

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                rule.execute,
                context
            )

            if result is not None:
                logging.info(f"预加载完成: {rule.name}")

        except Exception as e:
            logging.error(f"预加载执行失败: {rule.name}, {e}")
        finally:
            self._preload_tasks.discard(task_id)

    # =========================================================================
    # 核心方法
    # =========================================================================

    def smart_get(self, key: str, data_type: DataSourceType) -> Optional[Any]:
        """
        智能缓存获取

        Args:
            key: 缓存键
            data_type: 数据类型

        Returns:
            缓存数据
        """
        start_time = time.time()

        try:
            # 检查是否需要智能失效
            if self._should_invalidate_smart(key, data_type):
                self.invalidate_cache_entry(key, data_type)
                return None

            # 获取缓存数据
            value = self.cache.get(key)

            if value is not None:
                # 更新缓存条目统计
                self._update_cache_entry_stats(key, data_type)
                return value

            return None

        except Exception as e:
            logging.error(f"智能缓存获取失败: {key}, {e}")
            return None

    def smart_set(self, key: str, value: Any, data_type: DataSourceType,


                  ttl_seconds: Optional[int] = None) -> bool:
        """
        智能缓存设置

        Args:
            key: 缓存键
            value: 缓存值
            data_type: 数据类型
            ttl_seconds: TTL秒数

        Returns:
            是否设置成功
        """
        try:
            # 获取配置的TTL
            if ttl_seconds is None:
                ttl_seconds = self._get_configured_ttl(data_type)

            # 设置缓存
            success = self.cache.set(key, value)

            if success:
                # 创建或更新缓存条目
                entry = CacheEntry(
                    key=key,
                    data_type=data_type,
                    value=value,
                    timestamp=datetime.now(),
                    priority=self._get_configured_priority(data_type),
                    ttl_seconds=ttl_seconds,
                    expiry_time=datetime.now() + timedelta(seconds=ttl_seconds)
                )

                self._cache_entries[key] = entry

            return success

        except Exception as e:
            logging.error(f"智能缓存设置失败: {key}, {e}")
            return False

    # =========================================================================
    # 智能失效策略
    # =========================================================================

    def _should_invalidate_smart(self, key: str, data_type: DataSourceType) -> bool:
        """检查是否需要智能失效"""
        entry = self._cache_entries.get(key)
        if not entry:
            return False

        # 检查基础过期
        if entry.is_expired():
            return True

        # 基于访问模式的失效
        if self._should_invalidate_by_access_pattern(entry):
            return True

        # 基于数据新鲜度的失效
        if self._should_invalidate_by_freshness(entry):
            return True

        return False

    def _should_invalidate_by_access_pattern(self, entry: CacheEntry) -> bool:
        """基于访问模式的失效"""
        time_since_access = (datetime.now() - entry.last_access).total_seconds()

        # 超过24小时未访问的低频数据
        if time_since_access > 86400 and entry.access_count < 5:
            return True

        return False

    def _should_invalidate_by_freshness(self, entry: CacheEntry) -> bool:
        """基于数据新鲜度的失效"""
        freshness_requirements = {
            DataSourceType.CRYPTO: 300,   # 5分钟
            DataSourceType.STOCK: 600,    # 10分钟
            DataSourceType.NEWS: 1800,    # 30分钟
            DataSourceType.MACRO: 3600,   # 1小时
        }

        required_freshness = freshness_requirements.get(entry.data_type, 3600)
        age = (datetime.now() - entry.timestamp).total_seconds()

        return age > required_freshness * 1.2

    def _update_cache_entry_stats(self, key: str, data_type: DataSourceType):
        """更新缓存条目统计"""
        if key in self._cache_entries:
            entry = self._cache_entries[key]
            entry.touch()
            self._access_history.append((key, datetime.now()))

    def invalidate_cache_entry(self, key: str, data_type: DataSourceType) -> bool:
        """使缓存条目失效"""
        try:
            success = self.cache.delete(key)

            if key in self._cache_entries:
                del self._cache_entries[key]

            return success

        except Exception as e:
            logging.error(f"缓存失效失败: {key}, {e}")
            return False

    # =========================================================================
    # 预加载机制
    # =========================================================================

    def add_preload_rule(self, name: str, data_type: DataSourceType,


                         condition: Callable[[Dict[str, Any]], bool],
                         preload_func: Callable[[], Any],
                         priority: int = 2, interval_seconds: int = 300):
        """添加预加载规则"""
        rule = PreloadRule(
            name=name,
            data_type=data_type,
            condition=condition,
            preload_func=preload_func,
            priority=priority,
            interval_seconds=interval_seconds
        )

        self._preload_rules.append(rule)
        logging.info(f"预加载规则已添加: {name}")

    def _preload_frequent_stock_data(self) -> Dict[str, Any]:
        """预加载高频股票数据"""
        logging.info("执行高频股票数据预加载")
        return {"status": "success", "data_type": "stock", "count": 100}

    # =========================================================================
    # 工具方法
    # =========================================================================

    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return {
            'cache_status': {
                'current_entries': len(self._cache_entries)
            },
            'preload_status': {
                'active_rules': len(self._preload_rules),
                'running_tasks': len(self._preload_tasks)
            },
            'timestamp': datetime.now().isoformat()
        }

    def _get_configured_ttl(self, data_type: DataSourceType) -> int:
        """获取配置的TTL"""
        ttl_configs = {
            DataSourceType.STOCK: 3600,
            DataSourceType.CRYPTO: 300,
            DataSourceType.NEWS: 1800,
            DataSourceType.MACRO: 86400,
        }
        return ttl_configs.get(data_type, 3600)

    def _get_configured_priority(self, data_type: DataSourceType) -> int:
        """获取配置的优先级"""
        priority_configs = {
            DataSourceType.STOCK: 3,
            DataSourceType.CRYPTO: 3,
            DataSourceType.NEWS: 2,
            DataSourceType.MACRO: 1,
        }
        return priority_configs.get(data_type, 2)

    def shutdown(self):
        """关闭优化器"""
        self._stop_scheduler.set()
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            self._scheduler_thread.join(timeout=5)

        logging.info("智能缓存优化器已关闭")


class _InMemoryDataCache:
    """默认的内存缓存实现，用于工厂方法回退"""

    def __init__(self):
        self._store: Dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self._store.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        self._store[key] = value
        return True

    def delete(self, key: str) -> bool:
        return self._store.pop(key, None) is not None

    def clear(self) -> bool:
        self._store.clear()
        return True

    def get_stats(self) -> Dict[str, Any]:
        return {
            "size": len(self._store),
            "ttl_support": False
        }


# =============================================================================
# 工厂函数
# =============================================================================


def get_smart_cache_optimizer(
    cache: Optional[IDataCache] = None,
    config: Optional[Dict[str, Any]] = None
) -> SmartCacheOptimizer:
    """
    获取智能缓存优化器实例

    Args:
        cache: 外部注入的缓存实例，默认使用内存缓存
        config: 配置字典

    Returns:
        SmartCacheOptimizer实例
    """
    if config is None:
        config = {}

    cache_instance = cache or _InMemoryDataCache()
    return SmartCacheOptimizer(cache=cache_instance, config=config)
