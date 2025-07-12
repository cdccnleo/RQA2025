"""
数据预加载机制实现
"""
from typing import Dict, List, Any, Optional, Set, Union
import logging
import threading
import time
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import json
import os

from src.infrastructure.utils.exceptions import DataLoaderError
from src.data.interfaces import IDataModel
from src.data.base_loader import BaseDataLoader
from src.data.registry import DataRegistry

logger = logging.getLogger(__name__)


class DataPreloader:
    """
    数据预加载器，负责在后台预先加载可能需要的数据
    """
    def __init__(self, registry: DataRegistry, cache_dir: Union[str, Path]):
        """
        初始化数据预加载器

        Args:
            registry: 数据注册中心
            cache_dir: 缓存目录
        """
        self.registry = registry
        self.cache_dir = Path(cache_dir)
        self.preload_config_file = self.cache_dir / "preload_config.json"

        # 预加载配置
        self.preload_config = self._load_preload_config()

        # 预加载线程
        self.preload_thread = None
        self.stop_event = threading.Event()

        # 已预加载的数据
        self.preloaded_data: Dict[str, IDataModel] = {}

        # 预加载锁
        self.preload_lock = threading.RLock()

        # 预加载统计
        self.stats = {
            'total_preloaded': 0,
            'hits': 0,
            'misses': 0,
            'last_preload_time': None,
            'preload_duration': 0
        }

        logger.info(f"DataPreloader initialized with cache directory: {self.cache_dir}")

    def _load_preload_config(self) -> Dict[str, Any]:
        """
        加载预加载配置

        Returns:
            Dict[str, Any]: 预加载配置
        """
        default_config = {
            'enabled': True,
            'patterns': [
                {
                    'source_type': 'stock',
                    'symbols': ['000001.SZ', '600000.SH'],  # 常用股票
                    'days_back': 30,  # 预加载最近30天数据
                    'frequency': '1d',
                    'priority': 'high'
                },
                {
                    'source_type': 'index',
                    'symbols': ['000300.SH', '000905.SH'],  # 常用指数
                    'days_back': 30,
                    'frequency': '1d',
                    'priority': 'high'
                }
            ],
            'schedule': {
                'market_open_preload': True,  # 市场开盘前预加载
                'daily_preload_time': '08:30',  # 每日预加载时间
                'weekend_preload': True,  # 周末预加载
                'weekend_preload_time': '10:00'  # 周末预加载时间
            },
            'limits': {
                'max_preload_items': 100,  # 最大预加载项数
                'max_memory_usage': 500 * 1024 * 1024,  # 最大内存使用（500MB）
                'preload_thread_priority': 'low'  # 预加载线程优先级
            }
        }

        if not self.preload_config_file.exists():
            # 创建默认配置
            os.makedirs(os.path.dirname(self.preload_config_file), exist_ok=True)
            with open(self.preload_config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config

        try:
            with open(self.preload_config_file, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            logger.warning(f"Failed to load preload config: {e}, using default")
            return default_config

    def _save_preload_config(self) -> None:
        """保存预加载配置"""
        try:
            with open(self.preload_config_file, 'w') as f:
                json.dump(self.preload_config, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save preload config: {e}")

    def start_preload_thread(self) -> None:
        """启动预加载线程"""
        if not self.preload_config.get('enabled', True):
            logger.info("Preloading is disabled in config")
            return

        if self.preload_thread is not None and self.preload_thread.is_alive():
            logger.warning("Preload thread is already running")
            return

        self.stop_event.clear()
        self.preload_thread = threading.Thread(
            target=self._preload_worker,
            name="DataPreloaderThread",
            daemon=True
        )
        self.preload_thread.start()
        logger.info("Preload thread started")

    def stop_preload_thread(self) -> None:
        """停止预加载线程"""
        if self.preload_thread is None or not self.preload_thread.is_alive():
            return

        self.stop_event.set()
        self.preload_thread.join(timeout=5.0)
        logger.info("Preload thread stopped")

    def _preload_worker(self) -> None:
        """预加载工作线程"""
        logger.info("Preload worker started")

        while not self.stop_event.is_set():
            try:
                # 检查是否应该执行预加载
                if self._should_run_preload():
                    start_time = time.time()
                    self.stats['last_preload_time'] = datetime.now().isoformat()

                    # 执行预加载
                    count = self._execute_preload()

                    # 更新统计信息
                    self.stats['total_preloaded'] += count
                    self.stats['preload_duration'] = time.time() - start_time
                    logger.info(f"Preloaded {count} items in {self.stats['preload_duration']:.2f} seconds")

                # 休眠一段时间
                for _ in range(300):  # 5分钟检查一次
                    if self.stop_event.is_set():
                        break
                    time.sleep(1)

            except Exception as e:
                logger.error(f"Error in preload worker: {e}")
                time.sleep(60)  # 出错后等待60秒再继续

        logger.info("Preload worker stopped")

    def _should_run_preload(self) -> bool:
        """
        判断是否应该执行预加载

        Returns:
            bool: 是否应该执行预加载
        """
        now = datetime.now()
        schedule = self.preload_config.get('schedule', {})

        # 检查每日预加载时间
        daily_time = schedule.get('daily_preload_time', '08:30')
        try:
            hour, minute = map(int, daily_time.split(':'))
            if now.hour == hour and now.minute >= minute and now.minute < minute + 5:
                return True
        except ValueError:
            pass

        # 检查周末预加载
        if schedule.get('weekend_preload', True) and now.weekday() >= 5:  # 5=周六, 6=周日
            weekend_time = schedule.get('weekend_preload_time', '10:00')
            try:
                hour, minute = map(int, weekend_time.split(':'))
                if now.hour == hour and now.minute >= minute and now.minute < minute + 5:
                    return True
            except ValueError:
                pass

        # 检查市场开盘前预加载
        if schedule.get('market_open_preload', True):
            # 中国A股市场开盘时间为9:30
            if now.weekday() < 5 and now.hour == 9 and now.minute >= 0 and now.minute < 20:
                return True

        return False

    def _execute_preload(self) -> int:
        """
        执行预加载

        Returns:
            int: 预加载的数据项数
        """
        with self.preload_lock:
            # 清理过期的预加载数据
            self._clean_expired_preloads()

            # 获取预加载模式
            patterns = self.preload_config.get('patterns', [])

            # 获取限制
            limits = self.preload_config.get('limits', {})
            max_items = limits.get('max_preload_items', 100)

            # 按优先级排序模式
            priority_map = {'high': 3, 'medium': 2, 'low': 1}
            sorted_patterns = sorted(
                patterns,
                key=lambda p: priority_map.get(p.get('priority', 'medium'), 2),
                reverse=True
            )

            # 执行预加载
            count = 0
            for pattern in sorted_patterns:
                if count >= max_items:
                    break

                source_type = pattern.get('source_type')
                if not source_type:
                    continue

                # 获取加载器
                loader = self.registry.get_loader(source_type)
                if loader is None:
                    logger.warning(f"Loader not found for source_type: {source_type}")
                    continue

                # 计算日期范围
                days_back = pattern.get('days_back', 30)
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

                # 获取频率
                frequency = pattern.get('frequency', '1d')

                # 获取符号列表
                symbols = pattern.get('symbols', [])
                if not symbols:
                    continue

                # 预加载每个符号的数据
                for symbol in symbols:
                    if count >= max_items:
                        break

                    # 生成缓存键
                    cache_key = f"{source_type}_{symbol}_{start_date}_{end_date}_{frequency}"

                    # 检查是否已预加载
                    if cache_key in self.preloaded_data:
                        continue

                    try:
                        # 加载数据
                        data_model = loader.load(start_date, end_date, frequency, symbol=symbol)

                        # 保存到预加载缓存
                        self.preloaded_data[cache_key] = data_model
                        count += 1

                        logger.debug(f"Preloaded: {cache_key}")
                    except Exception as e:
                        logger.warning(f"Failed to preload {cache_key}: {e}")

            return count

    def _clean_expired_preloads(self) -> None:
        """清理过期的预加载数据"""
        now = time.time()
        expired_keys = []

        for key, data_model in self.preloaded_data.items():
            metadata = data_model.get_metadata()
            created_at = metadata.get('created_at', 0)

            # 默认TTL为1天
            ttl = metadata.get('ttl', 86400)

            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at).timestamp()
                except ValueError:
                    created_at = 0

            if now - created_at > ttl:
                expired_keys.append(key)

        # 移除过期数据
        for key in expired_keys:
            del self.preloaded_data[key]

        if expired_keys:
            logger.debug(f"Cleaned {len(expired_keys)} expired preloaded items")

    def get_preloaded_data(self, source_type: str, start_date: str, end_date: str, frequency: str, **kwargs) -> Optional[IDataModel]:
        """
        获取预加载的数据

        Args:
            source_type: 数据源类型
            start_date: 开始日期
            end_date: 结束日期
            frequency: 数据频率
            **kwargs: 其他参数

        Returns:
            Optional[IDataModel]: 预加载的数据模型，如果不存在则返回None
        """
        with self.preload_lock:
            symbol = kwargs.get('symbol', '')

            # 生成缓存键
            cache_key = f"{source_type}_{symbol}_{start_date}_{end_date}_{frequency}"

            # 检查是否存在预加载数据
            if cache_key in self.preloaded_data:
                self.stats['hits'] += 1
                logger.debug(f"Preload hit: {cache_key}")
                return self.preloaded_data[cache_key]

            # 检查是否有部分匹配的预加载数据
            for key, data_model in self.preloaded_data.items():
                parts = key.split('_')
                if len(parts) >= 5:
                    pre_source, pre_symbol, pre_start, pre_end, pre_freq = parts[0], parts[1], parts[2], parts[3], parts[4]

                    # 检查是否匹配
                    if (pre_source == source_type and
                        pre_symbol == symbol and
                        pre_freq == frequency and
                        pre_start <= start_date and
                        pre_end >= end_date):

                        self.stats['hits'] += 1
                        logger.debug(f"Preload partial hit: {key}")
                        return data_model

            self.stats['misses'] += 1
            return None

    def add_to_preload(self, source_type: str, symbols: List[str], days_back: int = 30, frequency: str = '1d', priority: str = 'medium') -> None:
        """
        添加预加载配置

        Args:
            source_type: 数据源类型
            symbols: 符号列表
            days_back: 回溯天数
            frequency: 数据频率
            priority: 优先级（'high', 'medium', 'low'）
        """
        patterns = self.preload_config.get('patterns', [])

        # 检查是否已存在相同配置
        for pattern in patterns:
            if (pattern.get('source_type') == source_type and
                pattern.get('frequency') == frequency):
                # 更新现有配置
                pattern['symbols'] = list(set(pattern['symbols'] + symbols))
                pattern['days_back'] = max(pattern['days_back'], days_back)
                pattern['priority'] = max(pattern['priority'], priority, key=lambda x: {'high': 3, 'medium': 2, 'low': 1}[x])
                self._save_preload_config()
                logger.info(f"Updated preload pattern for {source_type}")
                return

        # 添加新配置
        patterns.append({
            'source_type': source_type,
            'symbols': symbols,
            'days_back': days_back,
            'frequency': frequency,
            'priority': priority
        })

        self.preload_config['patterns'] = patterns
        self._save_preload_config()
        logger.info(f"Added new preload pattern for {source_type}")

    def remove_from_preload(self, source_type: str, symbols: Optional[List[str]] = None) -> None:
        """
        移除预加载配置

        Args:
            source_type: 数据源类型
            symbols: 要移除的符号列表，如果为None则移除整个source_type的配置
        """
        patterns = self.preload_config.get('patterns', [])

        if symbols is None:
            # 移除整个source_type的配置
            self.preload_config['patterns'] = [p for p in patterns if p.get('source_type') != source_type]
        else:
            # 移除指定的symbols
            for pattern in patterns:
                if pattern.get('source_type') == source_type:
                    pattern['symbols'] = list(set(pattern['symbols']) - set(symbols))

        self._save_preload_config()
        logger.info(f"Removed preload pattern for {source_type}")

    def get_preload_stats(self) -> Dict[str, Any]:
        """
        获取预加载统计信息

        Returns:
            Dict[str, Any]: 预加载统计信息
        """
        total = self.stats['hits'] + self.stats['misses']
        hit_ratio = self.stats['hits'] / total if total > 0 else 0

        return {
            'total_preloaded': self.stats['total_preloaded'],
            'current_preloaded': len(self.preloaded_data),
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'hit_ratio': hit_ratio,
            'last_preload_time': self.stats['last_preload_time'],
            'preload_duration': self.stats['preload_duration'],
            'memory_usage': sum(sys.getsizeof(data.data) for data in self.preloaded_data.values()),
            'patterns': len(self.preload_config.get('patterns', []))
        }

    def clear_preloaded_data(self) -> None:
        """清理所有预加载的数据"""
        with self.preload_lock:
            self.preloaded_data.clear()
            logger.info("Cleared all preloaded data")

    def shutdown(self) -> None:
        """关闭预加载器"""
        self.stop_preload_thread()
        self.clear_preloaded_data()
        logger.info("DataPreloader shutdown complete")
