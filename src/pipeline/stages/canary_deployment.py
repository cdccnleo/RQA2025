"""
金丝雀部署阶段模块

负责5%流量切换到新模型，实时监控和自动回滚触发
"""

from typing import Dict, Any, Optional
import time
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from .base import PipelineStage
from ..exceptions import DeploymentException, StageExecutionException
from ..config import StageConfig


@dataclass
class CanaryMetrics:
    """金丝雀部署指标"""
    total_requests: int = 0
    error_count: int = 0
    avg_latency_ms: float = 0.0
    accuracy: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def error_rate(self) -> float:
        return self.error_count / self.total_requests if self.total_requests > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "error_count": self.error_count,
            "error_rate": self.error_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "accuracy": self.accuracy,
            "duration_minutes": (
                (self.end_time - self.start_time).total_seconds() / 60
                if self.start_time and self.end_time else 0
            )
        }


class CanaryDeploymentStage(PipelineStage):
    """
    金丝雀部署阶段
    
    功能：
    - 5%流量切换到新模型
    - 实时性能监控
    - 异常检测
    - 自动回滚触发
    """
    
    def __init__(self, config: Optional[StageConfig] = None):
        super().__init__("canary_deployment", config)
        self._canary_metrics: Optional[CanaryMetrics] = None
        self._deployment_info: Dict[str, Any] = {}
        self._monitoring_active: bool = False
        self._monitoring_thread: Optional[threading.Thread] = None
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行金丝雀部署
        
        Args:
            context: 包含model, model_path, validation_passed的上下文
            
        Returns:
            包含deployment_info, canary_metrics的输出
        """
        self.logger.info("开始金丝雀部署阶段")
        
        # 检查验证结果
        validation_passed = context.get("validation_passed", False)
        if not validation_passed:
            raise StageExecutionException(
                message="模型验证未通过，无法进行金丝雀部署",
                stage_name=self.name
            )
        
        # 获取配置
        traffic_percentage = self.config.config.get("traffic_percentage", 5)
        duration_minutes = self.config.config.get("duration_minutes", 30)
        
        model_path = context.get("model_path")
        model_info = context.get("model_info", {})
        
        self.logger.info(f"金丝雀部署配置: 流量{traffic_percentage}%, 持续{duration_minutes}分钟")
        
        # 1. 部署模型（5%流量）
        self.logger.info(f"部署模型到金丝雀环境，流量比例: {traffic_percentage}%")
        deployment_id = self._deploy_canary(model_path, traffic_percentage, context)
        
        # 2. 监控金丝雀部署
        self.logger.info("开始监控金丝雀部署")
        canary_metrics = self._monitor_canary(deployment_id, duration_minutes)
        self._canary_metrics = canary_metrics
        
        # 3. 评估金丝雀结果
        passed = self._evaluate_canary_result(canary_metrics)
        
        # 4. 记录部署信息
        self._deployment_info = {
            "deployment_id": deployment_id,
            "model_path": model_path,
            "model_version": model_info.get("training_timestamp"),
            "traffic_percentage": traffic_percentage,
            "duration_minutes": duration_minutes,
            "deployment_timestamp": datetime.now().isoformat(),
            "canary_metrics": canary_metrics.to_dict(),
            "passed": passed
        }
        
        if passed:
            self.logger.info("金丝雀部署通过，可以进行全面部署")
        else:
            self.logger.warning("金丝雀部署未通过，建议回滚")
            raise DeploymentException(
                message="金丝雀部署监控未通过",
                deployment_type="canary",
                context={"metrics": canary_metrics.to_dict()}
            )
        
        return {
            "deployment_info": self._deployment_info,
            "canary_metrics": canary_metrics.to_dict(),
            "canary_passed": passed,
            "deployment_id": deployment_id
        }
    
    def _deploy_canary(self, model_path: str, traffic_percentage: int, context: Dict[str, Any]) -> str:
        """
        部署模型到金丝雀环境
        
        Args:
            model_path: 模型文件路径
            traffic_percentage: 流量百分比
            context: 上下文
            
        Returns:
            部署ID
        """
        import uuid
        deployment_id = f"canary_{uuid.uuid4().hex[:8]}"
        
        # 实际部署逻辑（这里模拟）
        self.logger.info(f"模型已部署到金丝雀环境，部署ID: {deployment_id}")
        
        # 模拟部署延迟
        time.sleep(2)
        
        return deployment_id
    
    def _monitor_canary(self, deployment_id: str, duration_minutes: int) -> CanaryMetrics:
        """
        监控金丝雀部署
        
        Args:
            deployment_id: 部署ID
            duration_minutes: 监控持续时间
            
        Returns:
            金丝雀指标
        """
        metrics = CanaryMetrics(start_time=datetime.now())
        
        # 模拟监控（实际应连接到监控系统）
        self.logger.info(f"开始{duration_minutes}分钟的监控...")
        
        # 简化的监控循环
        check_interval = 60  # 每分钟检查一次
        total_checks = duration_minutes
        
        for i in range(total_checks):
            # 模拟收集指标
            self._collect_metrics_sample(metrics)
            
            # 检查是否需要提前终止
            if self._should_rollback_early(metrics):
                self.logger.warning("触发提前回滚条件")
                break
            
            if i < total_checks - 1:
                time.sleep(check_interval)
        
        metrics.end_time = datetime.now()
        
        self.logger.info(f"监控完成: {metrics.to_dict()}")
        
        return metrics
    
    def _collect_metrics_sample(self, metrics: CanaryMetrics) -> None:
        """收集指标样本（模拟）"""
        # 模拟请求数据
        import random
        sample_requests = random.randint(100, 200)
        sample_errors = random.randint(0, 5)
        sample_latency = random.uniform(10, 50)
        sample_accuracy = random.uniform(0.6, 0.8)
        
        # 更新累计指标
        metrics.total_requests += sample_requests
        metrics.error_count += sample_errors
        
        # 移动平均延迟
        if metrics.total_requests > 0:
            metrics.avg_latency_ms = (
                metrics.avg_latency_ms * (metrics.total_requests - sample_requests) +
                sample_latency * sample_requests
            ) / metrics.total_requests
        
        # 更新准确率（简单平均）
        metrics.accuracy = sample_accuracy
        
        self.logger.debug(f"指标样本 - 请求: {sample_requests}, 错误: {sample_errors}, 延迟: {sample_latency:.2f}ms")
    
    def _should_rollback_early(self, metrics: CanaryMetrics) -> bool:
        """检查是否应该提前回滚"""
        # 错误率超过5%
        if metrics.error_rate > 0.05:
            self.logger.warning(f"错误率 {metrics.error_rate:.2%} 超过阈值 5%")
            return True
        
        # 延迟超过200ms
        if metrics.avg_latency_ms > 200:
            self.logger.warning(f"平均延迟 {metrics.avg_latency_ms:.2f}ms 超过阈值 200ms")
            return True
        
        # 准确率低于50%
        if metrics.accuracy < 0.5 and metrics.total_requests > 500:
            self.logger.warning(f"准确率 {metrics.accuracy:.2%} 低于阈值 50%")
            return True
        
        return False
    
    def _evaluate_canary_result(self, metrics: CanaryMetrics) -> bool:
        """评估金丝雀结果"""
        # 获取阈值配置
        max_error_rate = self.config.config.get("max_error_rate", 0.05)
        max_latency_ms = self.config.config.get("max_latency_ms", 200)
        min_accuracy = self.config.config.get("min_accuracy", 0.55)
        
        # 检查各项指标
        if metrics.error_rate > max_error_rate:
            self.logger.warning(f"错误率 {metrics.error_rate:.2%} 超过阈值 {max_error_rate}")
            return False
        
        if metrics.avg_latency_ms > max_latency_ms:
            self.logger.warning(f"延迟 {metrics.avg_latency_ms:.2f}ms 超过阈值 {max_latency_ms}ms")
            return False
        
        if metrics.accuracy < min_accuracy:
            self.logger.warning(f"准确率 {metrics.accuracy:.2%} 低于阈值 {min_accuracy}")
            return False
        
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取阶段指标"""
        if self._canary_metrics is None:
            return {}
        
        return {
            "total_requests": self._canary_metrics.total_requests,
            "error_rate": self._canary_metrics.error_rate,
            "avg_latency_ms": self._canary_metrics.avg_latency_ms,
            "accuracy": self._canary_metrics.accuracy,
            "deployment_id": self._deployment_info.get("deployment_id")
        }
    
    def rollback(self, context: Dict[str, Any]) -> bool:
        """回滚金丝雀部署"""
        self.logger.info("回滚金丝雀部署")
        
        deployment_id = self._deployment_info.get("deployment_id")
        if deployment_id:
            # 实际回滚逻辑
            self.logger.info(f"撤销金丝雀部署: {deployment_id}")
            time.sleep(1)
        
        self._canary_metrics = None
        self._deployment_info = {}
        return True
