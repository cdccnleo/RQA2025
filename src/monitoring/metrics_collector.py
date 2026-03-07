"""
指标收集器模块

负责收集和聚合模型性能指标
"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
import threading
import time
import logging


@dataclass
class MetricValue:
    """指标值"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels
        }


class MetricsCollector:
    """
    指标收集器
    
    功能：
    - 收集技术指标（Accuracy, F1, ROC-AUC等）
    - 收集业务指标（Sharpe Ratio, Max Drawdown等）
    - 收集数据质量指标（漂移分数、缺失率等）
    - 收集资源指标（延迟、吞吐量、CPU/内存使用率等）
    """
    
    def __init__(self):
        self.logger = logging.getLogger("monitoring.metrics_collector")
        self._metrics: Dict[str, List[MetricValue]] = defaultdict(list)
        self._collectors: Dict[str, Callable] = {}
        self._lock = threading.Lock()
        self._running = False
        self._collection_thread: Optional[threading.Thread] = None
    
    def register_collector(self, name: str, collector_func: Callable[[], Dict[str, float]]) -> None:
        """
        注册指标收集函数
        
        Args:
            name: 收集器名称
            collector_func: 收集函数，返回指标字典
        """
        self._collectors[name] = collector_func
        self.logger.info(f"注册指标收集器: {name}")
    
    def collect(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        收集单个指标
        
        Args:
            metric_name: 指标名称
            value: 指标值
            labels: 标签
        """
        metric = MetricValue(
            name=metric_name,
            value=value,
            timestamp=datetime.now(),
            labels=labels or {}
        )
        
        with self._lock:
            self._metrics[metric_name].append(metric)
            # 限制历史数据量
            if len(self._metrics[metric_name]) > 10000:
                self._metrics[metric_name] = self._metrics[metric_name][-5000:]
    
    def collect_batch(self, metrics: Dict[str, float], labels: Optional[Dict[str, str]] = None) -> None:
        """
        批量收集指标
        
        Args:
            metrics: 指标字典
            labels: 标签
        """
        for name, value in metrics.items():
            self.collect(name, value, labels)
    
    def get_metrics(
        self,
        metric_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        labels_filter: Optional[Dict[str, str]] = None
    ) -> List[MetricValue]:
        """
        获取指标数据
        
        Args:
            metric_name: 指标名称（None表示所有）
            start_time: 开始时间
            end_time: 结束时间
            labels_filter: 标签过滤条件
            
        Returns:
            指标值列表
        """
        with self._lock:
            if metric_name:
                metrics = self._metrics.get(metric_name, [])
            else:
                metrics = [m for metrics in self._metrics.values() for m in metrics]
            
            # 时间过滤
            if start_time:
                metrics = [m for m in metrics if m.timestamp >= start_time]
            if end_time:
                metrics = [m for m in metrics if m.timestamp <= end_time]
            
            # 标签过滤
            if labels_filter:
                metrics = [
                    m for m in metrics
                    if all(m.labels.get(k) == v for k, v in labels_filter.items())
                ]
            
            return metrics
    
    def get_latest(self, metric_name: str) -> Optional[MetricValue]:
        """获取最新指标值"""
        with self._lock:
            metrics = self._metrics.get(metric_name, [])
            return metrics[-1] if metrics else None
    
    def get_statistics(self, metric_name: str, window_minutes: int = 60) -> Dict[str, float]:
        """
        获取指标统计信息
        
        Args:
            metric_name: 指标名称
            window_minutes: 时间窗口（分钟）
            
        Returns:
            统计信息字典
        """
        end_time = datetime.now()
        start_time = end_time - __import__('datetime').timedelta(minutes=window_minutes)
        
        metrics = self.get_metrics(metric_name, start_time, end_time)
        values = [m.value for m in metrics]
        
        if not values:
            return {}
        
        import numpy as np
        
        return {
            "count": len(values),
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "p50": np.percentile(values, 50),
            "p95": np.percentile(values, 95),
            "p99": np.percentile(values, 99)
        }
    
    def start_collection(self, interval_seconds: int = 60) -> None:
        """
        启动自动收集
        
        Args:
            interval_seconds: 收集间隔（秒）
        """
        if self._running:
            return
        
        self._running = True
        self._collection_thread = threading.Thread(
            target=self._collection_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._collection_thread.start()
        self.logger.info(f"启动自动收集，间隔: {interval_seconds}秒")
    
    def stop_collection(self) -> None:
        """停止自动收集"""
        self._running = False
        if self._collection_thread:
            self._collection_thread.join(timeout=5)
        self.logger.info("停止自动收集")
    
    def _collection_loop(self, interval_seconds: int) -> None:
        """收集循环"""
        while self._running:
            try:
                for name, collector in self._collectors.items():
                    try:
                        metrics = collector()
                        self.collect_batch(metrics, {"source": name})
                    except Exception as e:
                        self.logger.error(f"收集器 {name} 失败: {e}")
                
                time.sleep(interval_seconds)
            except Exception as e:
                self.logger.error(f"收集循环异常: {e}")
                time.sleep(1)
    
    def clear(self, metric_name: Optional[str] = None) -> None:
        """清除指标数据"""
        with self._lock:
            if metric_name:
                self._metrics[metric_name] = []
            else:
                self._metrics.clear()
    
    def export_to_dict(self) -> Dict[str, Any]:
        """导出所有指标为字典"""
        with self._lock:
            return {
                name: [m.to_dict() for m in metrics]
                for name, metrics in self._metrics.items()
            }


# 全局指标收集器实例
_global_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """获取全局指标收集器"""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def reset_metrics_collector() -> None:
    """重置全局指标收集器"""
    global _global_collector
    _global_collector = None
