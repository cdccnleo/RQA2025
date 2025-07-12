"""缓冲区优化器模块"""

from typing import Dict, List, Any, Optional
from src.infrastructure.monitoring import MetricsCollector
from src.data.market_data import MarketData
import pandas as pd
import numpy as np
import logging
import time
from collections import deque

logger = logging.getLogger(__name__)

class BufferOptimizer:
    """缓冲区优化器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.market_data = MarketData()
        
        # 缓冲区配置
        self.buffer_size = config.get('buffer_size', 1000)
        self.max_memory_mb = config.get('max_memory_mb', 512)
        self.cleanup_interval = config.get('cleanup_interval', 300)  # 5分钟
        
        # 性能监控
        self.performance_metrics = {
            'buffer_hits': 0,
            'buffer_misses': 0,
            'memory_usage_mb': 0,
            'cleanup_count': 0
        }
        
        # 初始化缓冲区
        self.data_buffer = {}
        self.access_times = {}
        self.last_cleanup = time.time()
        
    def get_data(self, key: str, data_type: str = 'market') -> Optional[pd.DataFrame]:
        """从缓冲区获取数据"""
        cache_key = f"{data_type}_{key}"
        
        # 检查缓冲区
        if cache_key in self.data_buffer:
            self.performance_metrics['buffer_hits'] += 1
            self.access_times[cache_key] = time.time()
            logger.debug(f"Buffer hit for {cache_key}")
            return self.data_buffer[cache_key]
        
        # 缓存未命中
        self.performance_metrics['buffer_misses'] += 1
        logger.debug(f"Buffer miss for {cache_key}")
        
        # 从数据源获取数据
        data = self._fetch_data(key, data_type)
        if data is not None:
            self._add_to_buffer(cache_key, data)
        
        return data
    
    def _fetch_data(self, key: str, data_type: str) -> Optional[pd.DataFrame]:
        """从数据源获取数据"""
        try:
            if data_type == 'market':
                return self.market_data.get_data(key)
            elif data_type == 'fundamental':
                return self.market_data.get_fundamental_data(key)
            elif data_type == 'technical':
                return self.market_data.get_technical_data(key)
            else:
                logger.warning(f"Unknown data type: {data_type}")
                return None
        except Exception as e:
            logger.error(f"Error fetching data for {key}: {e}")
            return None
    
    def _add_to_buffer(self, key: str, data: pd.DataFrame):
        """添加数据到缓冲区"""
        # 检查内存使用
        if self._should_cleanup():
            self._cleanup_buffer()
        
        # 添加到缓冲区
        self.data_buffer[key] = data
        self.access_times[key] = time.time()
        
        # 更新内存使用统计
        self._update_memory_usage()
        
        logger.debug(f"Added {key} to buffer")
    
    def _should_cleanup(self) -> bool:
        """判断是否需要清理缓冲区"""
        current_time = time.time()
        
        # 检查时间间隔
        if current_time - self.last_cleanup > self.cleanup_interval:
            return True
        
        # 检查内存使用
        if self.performance_metrics['memory_usage_mb'] > self.max_memory_mb:
            return True
        
        # 检查缓冲区大小
        if len(self.data_buffer) > self.buffer_size:
            return True
        
        return False
    
    def _cleanup_buffer(self):
        """清理缓冲区"""
        logger.info("Starting buffer cleanup")
        
        current_time = time.time()
        items_to_remove = []
        
        # 按访问时间排序
        sorted_items = sorted(
            self.access_times.items(),
            key=lambda x: x[1]
        )
        
        # 移除最旧的项，直到内存使用降到阈值以下
        for key, access_time in sorted_items:
            if (self.performance_metrics['memory_usage_mb'] > self.max_memory_mb * 0.8 or
                len(self.data_buffer) > self.buffer_size * 0.8):
                
                items_to_remove.append(key)
            else:
                break
        
        # 执行清理
        for key in items_to_remove:
            if key in self.data_buffer:
                del self.data_buffer[key]
            if key in self.access_times:
                del self.access_times[key]
        
        self.last_cleanup = current_time
        self.performance_metrics['cleanup_count'] += 1
        
        # 更新内存使用
        self._update_memory_usage()
        
        logger.info(f"Buffer cleanup completed, removed {len(items_to_remove)} items")
    
    def _update_memory_usage(self):
        """更新内存使用统计"""
        total_memory = 0
        
        for key, data in self.data_buffer.items():
            # 估算DataFrame内存使用
            if isinstance(data, pd.DataFrame):
                memory_usage = data.memory_usage(deep=True).sum()
                total_memory += memory_usage
        
        # 转换为MB
        self.performance_metrics['memory_usage_mb'] = total_memory / (1024 * 1024)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        metrics = self.performance_metrics.copy()
        
        # 计算命中率
        total_requests = metrics['buffer_hits'] + metrics['buffer_misses']
        if total_requests > 0:
            metrics['hit_rate'] = metrics['buffer_hits'] / total_requests
        else:
            metrics['hit_rate'] = 0.0
        
        # 添加缓冲区统计
        metrics['buffer_size'] = len(self.data_buffer)
        metrics['buffer_keys'] = list(self.data_buffer.keys())
        
        return metrics
    
    def clear_buffer(self):
        """清空缓冲区"""
        self.data_buffer.clear()
        self.access_times.clear()
        self.performance_metrics['memory_usage_mb'] = 0
        logger.info("Buffer cleared")
    
    def preload_data(self, keys: List[str], data_type: str = 'market'):
        """预加载数据到缓冲区"""
        logger.info(f"Preloading {len(keys)} items to buffer")
        
        for key in keys:
            if f"{data_type}_{key}" not in self.data_buffer:
                data = self._fetch_data(key, data_type)
                if data is not None:
                    self._add_to_buffer(f"{data_type}_{key}", data)
    
    def optimize_for_query_pattern(self, query_pattern: Dict[str, Any]):
        """根据查询模式优化缓冲区"""
        # 分析查询模式
        frequently_accessed = query_pattern.get('frequent_keys', [])
        data_types = query_pattern.get('data_types', ['market'])
        
        # 预加载常用数据
        for data_type in data_types:
            self.preload_data(frequently_accessed, data_type)
        
        # 调整缓冲区大小
        if 'buffer_size' in query_pattern:
            self.buffer_size = query_pattern['buffer_size']
        
        # 调整清理间隔
        if 'cleanup_interval' in query_pattern:
            self.cleanup_interval = query_pattern['cleanup_interval']
        
        logger.info(f"Buffer optimized for query pattern: {query_pattern}")
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """获取缓冲区统计信息"""
        return {
            'total_items': len(self.data_buffer),
            'memory_usage_mb': self.performance_metrics['memory_usage_mb'],
            'hit_rate': self.get_performance_metrics()['hit_rate'],
            'last_cleanup': self.last_cleanup,
            'cleanup_count': self.performance_metrics['cleanup_count'],
            'keys': list(self.data_buffer.keys())
        }
