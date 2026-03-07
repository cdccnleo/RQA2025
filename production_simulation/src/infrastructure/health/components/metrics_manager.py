"""
监控指标管理器

负责监控指标的存储、查询、清理等管理功能。
"""

import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable

from src.infrastructure.logging.core.unified_logger import get_unified_logger

logger = get_unified_logger(__name__)


@dataclass
class MetricType:
    """指标类型枚举的简化版本"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Metric:
    """监控指标"""
    name: str
    value: float
    metric_type: str
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    description: str = ""


class MetricsManager:
    """
    监控指标管理器
    
    职责：
    - 指标数据存储和管理
    - 指标查询和过滤
    - 指标数据清理
    - 指标统计计算
    """
    
    def __init__(self, max_metrics: int = 10000, retention_days: int = 30):
        """
        初始化指标管理器
        
        Args:
            max_metrics: 最大指标数量
            retention_days: 数据保留天数
        """
        self._max_metrics = max_metrics
        self._retention_days = retention_days
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_metrics))
        self._callbacks: List[Callable[[Metric], None]] = []
        self._lock = threading.Lock()
        
    def add_metric(self, metric: Metric) -> None:
        """
        添加监控指标
        
        Args:
            metric: 监控指标对象
        """
        try:
            with self._lock:
                self._metrics[metric.name].append(metric)
                
                # 触发回调
                for callback in self._callbacks:
                    try:
                        callback(metric)
                    except Exception as e:
                        logger.error(f"指标回调执行失败: {e}")
                        
                logger.debug(f"添加指标: {metric.name} = {metric.value}")
                
        except Exception as e:
            logger.error(f"添加指标失败: {e}")
    
    def get_metrics(self, 
                   name: str = None, 
                   start_time: float = None, 
                   end_time: float = None) -> List[Metric]:
        """
        获取监控指标
        
        Args:
            name: 指标名称，None则获取所有
            start_time: 开始时间戳
            end_time: 结束时间戳
            
        Returns:
            指标列表
        """
        try:
            with self._lock:
                if name is None:
                    # 获取所有指标
                    all_metrics = []
                    for metric_deque in self._metrics.values():
                        all_metrics.extend(metric_deque)
                    metrics_list = list(all_metrics)
                else:
                    # 获取指定名称的指标
                    metrics_list = list(self._metrics.get(name, deque()))
                
                # 时间过滤
                if start_time or end_time:
                    filtered_metrics = []
                    for metric in metrics_list:
                        if start_time and metric.timestamp < start_time:
                            continue
                        if end_time and metric.timestamp > end_time:
                            continue
                        filtered_metrics.append(metric)
                    return filtered_metrics
                
                return metrics_list
                
        except Exception as e:
            logger.error(f"获取指标失败: {e}")
            return []
    
    def get_latest_metric(self, name: str) -> Optional[Metric]:
        """
        获取最新的指标值
        
        Args:
            name: 指标名称
            
        Returns:
            最新指标，如果不存在返回None
        """
        try:
            with self._lock:
                metric_deque = self._metrics.get(name)
                if metric_deque and len(metric_deque) > 0:
                    return metric_deque[-1]
                return None
        except Exception as e:
            logger.error(f"获取最新指标失败 {name}: {e}")
            return None
    
    def get_metric_names(self) -> List[str]:
        """获取所有指标名称"""
        try:
            with self._lock:
                return list(self._metrics.keys())
        except Exception as e:
            logger.error(f"获取指标名称失败: {e}")
            return []
    
    def get_metric_count(self) -> Dict[str, int]:
        """获取各指标的计数统计"""
        try:
            with self._lock:
                return {name: len(deque_obj) for name, deque_obj in self._metrics.items()}
        except Exception as e:
            logger.error(f"获取指标计数失败: {e}")
            return {}
    
    def add_metric_callback(self, callback: Callable[[Metric], None]) -> None:
        """
        添加指标回调函数
        
        Args:
            callback: 回调函数
        """
        if callback not in self._callbacks:
            self._callbacks.append(callback)
    
    def remove_metric_callback(self, callback: Callable[[Metric], None]) -> None:
        """
        移除指标回调函数
        
        Args:
            callback: 回调函数
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def cleanup_old_data(self) -> int:
        """
        清理过期数据
        
        Returns:
            清理的指标数量
        """
        try:
            current_time = time.time()
            cutoff_time = current_time - (self._retention_days * 24 * 60 * 60)
            cleaned_count = 0
            
            with self._lock:
                for name, metric_deque in list(self._metrics.items()):
                    original_count = len(metric_deque)
                    
                    # 从右侧开始清理过期数据（因为deque是有序的）
                    while metric_deque and metric_deque[0].timestamp < cutoff_time:
                        metric_deque.popleft()
                        
                    cleaned_count += original_count - len(metric_deque)
                    
                    # 如果deque为空，删除该指标
                    if len(metric_deque) == 0:
                        del self._metrics[name]
            
            if cleaned_count > 0:
                logger.info(f"清理了 {cleaned_count} 个过期指标")
                
            return cleaned_count
            
        except Exception as e:
            logger.error(f"清理过期数据失败: {e}")
            return 0
    
    def clear_metrics(self, name: str = None) -> int:
        """
        清空指标数据
        
        Args:
            name: 指标名称，None则清空所有
            
        Returns:
            清空的指标数量
        """
        try:
            with self._lock:
                if name is None:
                    # 清空所有指标
                    total_count = sum(len(deque_obj) for deque_obj in self._metrics.values())
                    self._metrics.clear()
                    return total_count
                else:
                    # 清空指定指标
                    if name in self._metrics:
                        count = len(self._metrics[name])
                        del self._metrics[name]
                        return count
                    return 0
                    
        except Exception as e:
            logger.error(f"清空指标失败: {e}")
            return 0
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要信息"""
        try:
            with self._lock:
                total_metrics = sum(len(deque_obj) for deque_obj in self._metrics.values())
                metric_names = list(self._metrics.keys())
                
                return {
                    "total_metrics": total_metrics,
                    "unique_metric_names": len(metric_names),
                    "metric_names": metric_names,
                    "max_metrics_per_name": self._max_metrics,
                    "retention_days": self._retention_days
                }
        except Exception as e:
            logger.error(f"获取指标摘要失败: {e}")
            return {}
    
    def calculate_statistics(self, name: str, window_minutes: int = 60) -> Dict[str, Any]:
        """
        计算指标统计信息
        
        Args:
            name: 指标名称
            window_minutes: 统计窗口（分钟）
            
        Returns:
            统计信息
        """
        try:
            end_time = time.time()
            start_time = end_time - (window_minutes * 60)
            
            metrics = self.get_metrics(name, start_time, end_time)
            
            if not metrics:
                return {"count": 0, "error": "没有足够的数据"}
            
            values = [m.value for m in metrics]
            
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "latest": values[-1] if values else None,
                "window_minutes": window_minutes
            }
            
        except Exception as e:
            logger.error(f"计算统计信息失败 {name}: {e}")
            return {"error": str(e)}
