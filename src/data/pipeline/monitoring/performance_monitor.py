"""
模型性能监控模块

提供模型部署后的实时性能监控和指标收集功能
"""

import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from threading import Lock, Thread


class MetricType(Enum):
    """指标类型枚举"""
    TECHNICAL = "technical"      # 技术指标（准确率、F1等）
    BUSINESS = "business"        # 业务指标（收益率、夏普比率等）
    DATA_QUALITY = "data_quality"  # 数据质量指标
    RESOURCE = "resource"        # 资源指标（延迟、吞吐量等）


@dataclass
class PerformanceMetrics:
    """
    性能指标数据类
    
    Attributes:
        timestamp: 指标收集时间
        metric_type: 指标类型
        metric_name: 指标名称
        value: 指标值
        threshold: 阈值
        status: 状态（正常/警告/严重）
        metadata: 额外元数据
    """
    timestamp: datetime
    metric_type: MetricType
    metric_name: str
    value: float
    threshold: Optional[float] = None
    status: str = "normal"  # normal, warning, critical
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "metric_type": self.metric_type.value,
            "metric_name": self.metric_name,
            "value": self.value,
            "threshold": self.threshold,
            "status": self.status,
            "metadata": self.metadata
        }


@dataclass
class MetricsSnapshot:
    """
    指标快照
    
    某一时刻的所有指标集合
    """
    timestamp: datetime
    metrics: Dict[str, PerformanceMetrics] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()}
        }


class MetricsCollector(ABC):
    """
    指标收集器抽象基类
    
    所有指标收集器必须继承此类
    """
    
    def __init__(self, name: str, metric_type: MetricType):
        """
        初始化收集器
        
        Args:
            name: 收集器名称
            metric_type: 指标类型
        """
        self.name = name
        self.metric_type = metric_type
        self.logger = logging.getLogger(f"monitoring.collector.{name}")
    
    @abstractmethod
    def collect(self) -> Dict[str, PerformanceMetrics]:
        """
        收集指标
        
        Returns:
            指标字典
        """
        pass
    
    def validate(self, metrics: Dict[str, PerformanceMetrics]) -> bool:
        """
        验证指标有效性
        
        Args:
            metrics: 指标字典
            
        Returns:
            是否有效
        """
        return all(isinstance(m, PerformanceMetrics) for m in metrics.values())


class TechnicalMetricsCollector(MetricsCollector):
    """
    技术指标收集器
    
    收集模型预测准确率、F1分数、ROC-AUC等技术指标
    """
    
    def __init__(self, model: Any, data_source: Callable[[], pd.DataFrame]):
        """
        初始化技术指标收集器
        
        Args:
            model: 模型实例
            data_source: 数据源函数
        """
        super().__init__("technical", MetricType.TECHNICAL)
        self.model = model
        self.data_source = data_source
        self._history: deque = deque(maxlen=1000)
    
    def collect(self) -> Dict[str, PerformanceMetrics]:
        """
        收集技术指标
        
        Returns:
            技术指标字典
        """
        try:
            # 获取最新数据
            data = self.data_source()
            if data is None or data.empty:
                self.logger.warning("数据源返回空数据")
                return {}
            
            # 准备特征和目标
            feature_cols = [c for c in data.columns if c not in ['timestamp', 'target', 'close']]
            if 'target' not in data.columns and 'close' in data.columns:
                data['target'] = (data['close'].shift(-5) > data['close']).astype(int)
            
            data = data.dropna()
            if len(data) < 10:
                self.logger.warning("数据量不足")
                return {}
            
            X = data[feature_cols]
            y_true = data['target']
            
            # 预测
            y_pred = self.model.predict(X)
            y_prob = self.model.predict_proba(X)[:, 1] if hasattr(self.model, 'predict_proba') else None
            
            # 计算指标
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            metrics = {}
            timestamp = datetime.now()
            
            # 准确率
            accuracy = accuracy_score(y_true, y_pred)
            metrics['accuracy'] = PerformanceMetrics(
                timestamp=timestamp,
                metric_type=MetricType.TECHNICAL,
                metric_name='accuracy',
                value=accuracy,
                threshold=0.7,
                status='warning' if accuracy < 0.7 else 'normal'
            )
            
            # F1分数
            f1 = f1_score(y_true, y_pred, zero_division=0)
            metrics['f1_score'] = PerformanceMetrics(
                timestamp=timestamp,
                metric_type=MetricType.TECHNICAL,
                metric_name='f1_score',
                value=f1,
                threshold=0.65,
                status='warning' if f1 < 0.65 else 'normal'
            )
            
            # 精确率
            precision = precision_score(y_true, y_pred, zero_division=0)
            metrics['precision'] = PerformanceMetrics(
                timestamp=timestamp,
                metric_type=MetricType.TECHNICAL,
                metric_name='precision',
                value=precision,
                threshold=0.6,
                status='warning' if precision < 0.6 else 'normal'
            )
            
            # 召回率
            recall = recall_score(y_true, y_pred, zero_division=0)
            metrics['recall'] = PerformanceMetrics(
                timestamp=timestamp,
                metric_type=MetricType.TECHNICAL,
                metric_name='recall',
                value=recall,
                threshold=0.6,
                status='warning' if recall < 0.6 else 'normal'
            )
            
            # ROC-AUC
            if y_prob is not None:
                try:
                    roc_auc = roc_auc_score(y_true, y_prob)
                    metrics['roc_auc'] = PerformanceMetrics(
                        timestamp=timestamp,
                        metric_type=MetricType.TECHNICAL,
                        metric_name='roc_auc',
                        value=roc_auc,
                        threshold=0.7,
                        status='warning' if roc_auc < 0.7 else 'normal'
                    )
                except:
                    pass
            
            self._history.append(metrics)
            return metrics
            
        except Exception as e:
            self.logger.error(f"收集技术指标失败: {e}")
            return {}


class BusinessMetricsCollector(MetricsCollector):
    """
    业务指标收集器
    
    收集收益率、夏普比率、最大回撤等业务指标
    """
    
    def __init__(self, signal_source: Callable[[], pd.DataFrame]):
        """
        初始化业务指标收集器
        
        Args:
            signal_source: 信号数据源函数
        """
        super().__init__("business", MetricType.BUSINESS)
        self.signal_source = signal_source
        self._returns_history: deque = deque(maxlen=252)  # 一年交易日
    
    def collect(self) -> Dict[str, PerformanceMetrics]:
        """
        收集业务指标
        
        Returns:
            业务指标字典
        """
        try:
            # 获取信号数据
            data = self.signal_source()
            if data is None or data.empty:
                return {}
            
            metrics = {}
            timestamp = datetime.now()
            
            # 计算收益率
            if 'returns' in data.columns:
                returns = data['returns'].dropna()
                self._returns_history.extend(returns.tolist())
                
                if len(returns) > 0:
                    # 总收益率
                    total_return = (1 + returns).prod() - 1
                    metrics['total_return'] = PerformanceMetrics(
                        timestamp=timestamp,
                        metric_type=MetricType.BUSINESS,
                        metric_name='total_return',
                        value=total_return,
                        threshold=0.0,
                        status='warning' if total_return < 0 else 'normal'
                    )
                    
                    # 年化收益率
                    days = len(returns)
                    annualized_return = (1 + total_return) ** (252 / max(days, 1)) - 1
                    metrics['annualized_return'] = PerformanceMetrics(
                        timestamp=timestamp,
                        metric_type=MetricType.BUSINESS,
                        metric_name='annualized_return',
                        value=annualized_return,
                        threshold=0.1,
                        status='warning' if annualized_return < 0.1 else 'normal'
                    )
                    
                    # 夏普比率
                    if returns.std() > 0:
                        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
                        metrics['sharpe_ratio'] = PerformanceMetrics(
                            timestamp=timestamp,
                            metric_type=MetricType.BUSINESS,
                            metric_name='sharpe_ratio',
                            value=sharpe,
                            threshold=1.0,
                            status='warning' if sharpe < 1.0 else 'normal'
                        )
                    
                    # 最大回撤
                    cumulative = (1 + returns).cumprod()
                    running_max = cumulative.expanding().max()
                    drawdown = (cumulative - running_max) / running_max
                    max_drawdown = drawdown.min()
                    metrics['max_drawdown'] = PerformanceMetrics(
                        timestamp=timestamp,
                        metric_type=MetricType.BUSINESS,
                        metric_name='max_drawdown',
                        value=abs(max_drawdown),
                        threshold=0.15,
                        status='critical' if abs(max_drawdown) > 0.15 else 'warning' if abs(max_drawdown) > 0.1 else 'normal'
                    )
                    
                    # 胜率
                    win_rate = (returns > 0).mean()
                    metrics['win_rate'] = PerformanceMetrics(
                        timestamp=timestamp,
                        metric_type=MetricType.BUSINESS,
                        metric_name='win_rate',
                        value=win_rate,
                        threshold=0.5,
                        status='warning' if win_rate < 0.5 else 'normal'
                    )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"收集业务指标失败: {e}")
            return {}


class ResourceMetricsCollector(MetricsCollector):
    """
    资源指标收集器
    
    收集模型推理延迟、吞吐量、错误率等资源指标
    """
    
    def __init__(self, model: Any, sample_data: pd.DataFrame):
        """
        初始化资源指标收集器
        
        Args:
            model: 模型实例
            sample_data: 样本数据用于测试
        """
        super().__init__("resource", MetricType.RESOURCE)
        self.model = model
        self.sample_data = sample_data
        self._latency_history: deque = deque(maxlen=1000)
        self._error_count = 0
        self._request_count = 0
    
    def collect(self) -> Dict[str, PerformanceMetrics]:
        """
        收集资源指标
        
        Returns:
            资源指标字典
        """
        try:
            metrics = {}
            timestamp = datetime.now()
            
            # 测试推理延迟
            feature_cols = [c for c in self.sample_data.columns if c not in ['timestamp', 'target']]
            X_sample = self.sample_data[feature_cols].head(100)
            
            latencies = []
            for _ in range(10):  # 测试10次
                start = time.time()
                try:
                    _ = self.model.predict(X_sample)
                    self._request_count += 1
                except:
                    self._error_count += 1
                latencies.append((time.time() - start) * 1000)  # 转换为毫秒
            
            avg_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            
            self._latency_history.extend(latencies)
            
            metrics['avg_latency_ms'] = PerformanceMetrics(
                timestamp=timestamp,
                metric_type=MetricType.RESOURCE,
                metric_name='avg_latency_ms',
                value=avg_latency,
                threshold=100,
                status='critical' if avg_latency > 200 else 'warning' if avg_latency > 100 else 'normal'
            )
            
            metrics['p95_latency_ms'] = PerformanceMetrics(
                timestamp=timestamp,
                metric_type=MetricType.RESOURCE,
                metric_name='p95_latency_ms',
                value=p95_latency,
                threshold=200,
                status='critical' if p95_latency > 300 else 'warning' if p95_latency > 200 else 'normal'
            )
            
            metrics['p99_latency_ms'] = PerformanceMetrics(
                timestamp=timestamp,
                metric_type=MetricType.RESOURCE,
                metric_name='p99_latency_ms',
                value=p99_latency,
                threshold=300,
                status='critical' if p99_latency > 500 else 'warning' if p99_latency > 300 else 'normal'
            )
            
            # 错误率
            error_rate = self._error_count / max(self._request_count, 1)
            metrics['error_rate'] = PerformanceMetrics(
                timestamp=timestamp,
                metric_type=MetricType.RESOURCE,
                metric_name='error_rate',
                value=error_rate,
                threshold=0.05,
                status='critical' if error_rate > 0.1 else 'warning' if error_rate > 0.05 else 'normal'
            )
            
            # 吞吐量（每秒请求数）
            throughput = 1000 / avg_latency if avg_latency > 0 else 0
            metrics['throughput_rps'] = PerformanceMetrics(
                timestamp=timestamp,
                metric_type=MetricType.RESOURCE,
                metric_name='throughput_rps',
                value=throughput,
                threshold=10,
                status='warning' if throughput < 10 else 'normal'
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"收集资源指标失败: {e}")
            return {}


class ModelPerformanceMonitor:
    """
    模型性能监控器
    
    统一管理多个指标收集器，提供实时监控功能
    
    Attributes:
        model_id: 模型ID
        collectors: 指标收集器列表
        metrics_history: 指标历史记录
        monitoring_interval: 监控间隔（秒）
        is_monitoring: 是否正在监控
    """
    
    def __init__(
        self,
        model_id: str,
        monitoring_interval: int = 60,
        max_history: int = 10000
    ):
        """
        初始化性能监控器
        
        Args:
            model_id: 模型ID
            monitoring_interval: 监控间隔（秒）
            max_history: 最大历史记录数
        """
        self.model_id = model_id
        self.monitoring_interval = monitoring_interval
        self.max_history = max_history
        
        self._collectors: List[MetricsCollector] = []
        self._metrics_history: deque = deque(maxlen=max_history)
        self._snapshots: deque = deque(maxlen=1000)
        
        self._is_monitoring = False
        self._monitor_thread: Optional[Thread] = None
        self._lock = Lock()
        
        self.logger = logging.getLogger(f"monitoring.performance.{model_id}")
    
    def register_collector(self, collector: MetricsCollector) -> None:
        """
        注册指标收集器
        
        Args:
            collector: 指标收集器实例
        """
        self._collectors.append(collector)
        self.logger.info(f"注册指标收集器: {collector.name}")
    
    def unregister_collector(self, collector_name: str) -> bool:
        """
        注销指标收集器
        
        Args:
            collector_name: 收集器名称
            
        Returns:
            是否成功注销
        """
        for i, collector in enumerate(self._collectors):
            if collector.name == collector_name:
                self._collectors.pop(i)
                self.logger.info(f"注销指标收集器: {collector_name}")
                return True
        return False
    
    def start_monitoring(self) -> None:
        """启动监控"""
        if self._is_monitoring:
            self.logger.warning("监控已经在运行中")
            return
        
        self._is_monitoring = True
        self._monitor_thread = Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info(f"启动性能监控，间隔: {self.monitoring_interval}秒")
    
    def stop_monitoring(self) -> None:
        """停止监控"""
        self._is_monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self.logger.info("停止性能监控")
    
    def _monitoring_loop(self) -> None:
        """监控循环"""
        while self._is_monitoring:
            try:
                self.collect_metrics()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"监控循环异常: {e}")
                time.sleep(1)
    
    def collect_metrics(self) -> MetricsSnapshot:
        """
        收集所有指标
        
        Returns:
            指标快照
        """
        snapshot = MetricsSnapshot(timestamp=datetime.now())
        
        for collector in self._collectors:
            try:
                metrics = collector.collect()
                snapshot.metrics.update(metrics)
            except Exception as e:
                self.logger.error(f"收集器 {collector.name} 异常: {e}")
        
        with self._lock:
            self._snapshots.append(snapshot)
            for metric in snapshot.metrics.values():
                self._metrics_history.append(metric)
        
        return snapshot
    
    def get_latest_metrics(self) -> Optional[MetricsSnapshot]:
        """
        获取最新指标
        
        Returns:
            最新指标快照
        """
        with self._lock:
            return self._snapshots[-1] if self._snapshots else None
    
    def get_metrics_history(
        self,
        metric_type: Optional[MetricType] = None,
        metric_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[PerformanceMetrics]:
        """
        获取指标历史
        
        Args:
            metric_type: 指标类型过滤
            metric_name: 指标名称过滤
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            指标列表
        """
        with self._lock:
            metrics = list(self._metrics_history)
        
        # 应用过滤
        if metric_type:
            metrics = [m for m in metrics if m.metric_type == metric_type]
        if metric_name:
            metrics = [m for m in metrics if m.metric_name == metric_name]
        if start_time:
            metrics = [m for m in metrics if m.timestamp >= start_time]
        if end_time:
            metrics = [m for m in metrics if m.timestamp <= end_time]
        
        return metrics
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            snapshots = list(self._snapshots)
        
        if not snapshots:
            return {}
        
        # 按类型分组统计
        stats = {
            "model_id": self.model_id,
            "monitoring_duration": (snapshots[-1].timestamp - snapshots[0].timestamp).total_seconds() if len(snapshots) > 1 else 0,
            "total_snapshots": len(snapshots),
            "collectors_count": len(self._collectors),
            "is_monitoring": self._is_monitoring
        }
        
        # 各类型指标统计
        for metric_type in MetricType:
            type_metrics = [
                m for snapshot in snapshots
                for m in snapshot.metrics.values()
                if m.metric_type == metric_type
            ]
            
            if type_metrics:
                values = [m.value for m in type_metrics]
                stats[f"{metric_type.value}_metrics"] = {
                    "count": len(type_metrics),
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "latest": values[-1]
                }
        
        return stats
    
    def export_metrics(self, file_path: Union[str, Path]) -> bool:
        """
        导出指标到文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否成功
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with self._lock:
                snapshots = list(self._snapshots)
            
            data = [s.to_dict() for s in snapshots]
            
            import json
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"指标已导出: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出指标失败: {e}")
            return False
    
    def clear_history(self) -> None:
        """清空历史记录"""
        with self._lock:
            self._metrics_history.clear()
            self._snapshots.clear()
        self.logger.info("历史记录已清空")
