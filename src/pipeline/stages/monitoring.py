"""
监控阶段模块

负责实时性能监控、数据漂移检测和业务指标跟踪
"""

from typing import Dict, Any, List, Optional
import time
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque

from .base import PipelineStage
from ..exceptions import StageExecutionException
from ..config import StageConfig


@dataclass
class MonitoringMetrics:
    """监控指标"""
    timestamp: datetime
    request_count: int = 0
    error_count: int = 0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    accuracy: float = 0.0
    drift_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / self.request_count if self.request_count > 0 else 0,
            "latency_p50": self.latency_p50,
            "latency_p95": self.latency_p95,
            "latency_p99": self.latency_p99,
            "accuracy": self.accuracy,
            "drift_score": self.drift_score
        }


class MonitoringStage(PipelineStage):
    """
    监控阶段
    
    功能：
    - 实时性能监控（延迟、吞吐量、错误率）
    - 数据漂移检测
    - 业务指标跟踪
    - 告警触发
    """
    
    def __init__(self, config: Optional[StageConfig] = None):
        super().__init__("monitoring", config)
        self._metrics_history: deque = deque(maxlen=1000)
        self._monitoring_active: bool = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._alert_thresholds: Dict[str, float] = {}
        self._alerts: List[Dict[str, Any]] = []
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行监控阶段
        
        Args:
            context: 包含deployment_id, model的上下文
            
        Returns:
            包含monitoring_status, alerts的输出
        """
        self.logger.info("开始监控阶段")
        
        # 获取配置
        metrics_interval = self.config.config.get("metrics_interval_seconds", 60)
        drift_detection = self.config.config.get("drift_detection", True)
        
        # 获取告警阈值
        self._alert_thresholds = context.get("alert_thresholds", {
            "error_rate": 0.05,
            "latency_p95": 200,
            "accuracy": 0.55,
            "drift_score": 0.5
        })
        
        deployment_id = context.get("deployment_id")
        
        self.logger.info(f"启动监控，部署ID: {deployment_id}, 间隔: {metrics_interval}秒")
        
        # 启动监控（实际环境中应该在后台持续运行）
        # 这里模拟一段时间的监控
        monitoring_duration = self.config.config.get("monitoring_duration_minutes", 5)
        
        for i in range(monitoring_duration):
            # 收集指标
            metrics = self._collect_metrics(context)
            self._metrics_history.append(metrics)
            
            # 检查告警
            alerts = self._check_alerts(metrics)
            self._alerts.extend(alerts)
            
            # 数据漂移检测
            if drift_detection and len(self._metrics_history) >= 10:
                drift_score = self._detect_drift()
                metrics.drift_score = drift_score
            
            self.logger.info(f"监控周期 {i+1}/{monitoring_duration}: {metrics.to_dict()}")
            
            if i < monitoring_duration - 1:
                time.sleep(metrics_interval)
        
        # 生成监控报告
        monitoring_report = self._generate_monitoring_report()
        
        self.logger.info("监控阶段完成")
        
        return {
            "monitoring_status": "active",
            "deployment_id": deployment_id,
            "metrics_history": [m.to_dict() for m in self._metrics_history],
            "alerts": self._alerts,
            "monitoring_report": monitoring_report
        }
    
    def _collect_metrics(self, context: Dict[str, Any]) -> MonitoringMetrics:
        """收集监控指标"""
        import random
        
        # 模拟指标收集（实际应从监控系统获取）
        metrics = MonitoringMetrics(
            timestamp=datetime.now(),
            request_count=random.randint(100, 500),
            error_count=random.randint(0, 10),
            latency_p50=random.uniform(10, 30),
            latency_p95=random.uniform(30, 80),
            latency_p99=random.uniform(50, 120),
            accuracy=random.uniform(0.6, 0.75)
        )
        
        return metrics
    
    def _check_alerts(self, metrics: MonitoringMetrics) -> List[Dict[str, Any]]:
        """检查告警条件"""
        alerts = []
        
        # 检查错误率
        error_rate = metrics.error_count / metrics.request_count if metrics.request_count > 0 else 0
        if error_rate > self._alert_thresholds.get("error_rate", 0.05):
            alerts.append({
                "timestamp": datetime.now().isoformat(),
                "severity": "high",
                "metric": "error_rate",
                "value": error_rate,
                "threshold": self._alert_thresholds.get("error_rate", 0.05),
                "message": f"错误率 {error_rate:.2%} 超过阈值"
            })
        
        # 检查延迟
        if metrics.latency_p95 > self._alert_thresholds.get("latency_p95", 200):
            alerts.append({
                "timestamp": datetime.now().isoformat(),
                "severity": "medium",
                "metric": "latency_p95",
                "value": metrics.latency_p95,
                "threshold": self._alert_thresholds.get("latency_p95", 200),
                "message": f"P95延迟 {metrics.latency_p95:.2f}ms 超过阈值"
            })
        
        # 检查准确率
        if metrics.accuracy < self._alert_thresholds.get("accuracy", 0.55):
            alerts.append({
                "timestamp": datetime.now().isoformat(),
                "severity": "high",
                "metric": "accuracy",
                "value": metrics.accuracy,
                "threshold": self._alert_thresholds.get("accuracy", 0.55),
                "message": f"准确率 {metrics.accuracy:.2%} 低于阈值"
            })
        
        # 检查数据漂移
        if metrics.drift_score > self._alert_thresholds.get("drift_score", 0.5):
            alerts.append({
                "timestamp": datetime.now().isoformat(),
                "severity": "medium",
                "metric": "drift_score",
                "value": metrics.drift_score,
                "threshold": self._alert_thresholds.get("drift_score", 0.5),
                "message": f"数据漂移分数 {metrics.drift_score:.4f} 超过阈值"
            })
        
        for alert in alerts:
            self.logger.warning(f"告警: {alert['message']}")
        
        return alerts
    
    def _detect_drift(self) -> float:
        """检测数据漂移"""
        # 简化的漂移检测：比较最近两个时间窗口的分布差异
        if len(self._metrics_history) < 20:
            return 0.0
        
        # 获取最近两个窗口的准确率
        recent_window = list(self._metrics_history)[-10:]
        previous_window = list(self._metrics_history)[-20:-10]
        
        recent_accuracies = [m.accuracy for m in recent_window]
        previous_accuracies = [m.accuracy for m in previous_window]
        
        # 计算均值差异
        recent_mean = sum(recent_accuracies) / len(recent_accuracies)
        previous_mean = sum(previous_accuracies) / len(previous_accuracies)
        
        # 简单的漂移分数
        drift_score = abs(recent_mean - previous_mean)
        
        return drift_score
    
    def _generate_monitoring_report(self) -> Dict[str, Any]:
        """生成监控报告"""
        if not self._metrics_history:
            return {}
        
        # 计算统计信息
        accuracies = [m.accuracy for m in self._metrics_history]
        latencies_p95 = [m.latency_p95 for m in self._metrics_history]
        error_counts = [m.error_count for m in self._metrics_history]
        request_counts = [m.request_count for m in self._metrics_history]
        
        total_requests = sum(request_counts)
        total_errors = sum(error_counts)
        
        report = {
            "monitoring_period": {
                "start": self._metrics_history[0].timestamp.isoformat() if self._metrics_history else None,
                "end": self._metrics_history[-1].timestamp.isoformat() if self._metrics_history else None
            },
            "summary": {
                "total_requests": total_requests,
                "total_errors": total_errors,
                "avg_error_rate": total_errors / total_requests if total_requests > 0 else 0,
                "avg_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0,
                "avg_latency_p95": sum(latencies_p95) / len(latencies_p95) if latencies_p95 else 0,
                "alert_count": len(self._alerts)
            },
            "alerts_by_severity": {
                "high": len([a for a in self._alerts if a["severity"] == "high"]),
                "medium": len([a for a in self._alerts if a["severity"] == "medium"]),
                "low": len([a for a in self._alerts if a["severity"] == "low"])
            }
        }
        
        return report
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取阶段指标"""
        if not self._metrics_history:
            return {}
        
        latest = self._metrics_history[-1]
        
        return {
            "latest_accuracy": latest.accuracy,
            "latest_latency_p95": latest.latency_p95,
            "latest_error_rate": latest.error_count / latest.request_count if latest.request_count > 0 else 0,
            "total_alerts": len(self._alerts),
            "monitoring_samples": len(self._metrics_history)
        }
    
    def rollback(self, context: Dict[str, Any]) -> bool:
        """回滚监控阶段"""
        self.logger.info("回滚监控阶段")
        self._monitoring_active = False
        self._metrics_history.clear()
        self._alerts = []
        return True
