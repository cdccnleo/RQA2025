"""
自动回滚管理模块

提供模型自动回滚决策、执行和验证功能
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import joblib

from .performance_monitor import ModelPerformanceMonitor, PerformanceMetrics
from .alert_manager import AlertManager, Alert, AlertSeverity
from .drift_detector import DriftDetector, DriftSeverity


class RollbackTrigger(Enum):
    """回滚触发条件枚举"""
    ACCURACY_DROP = "accuracy_drop"          # 准确率下降
    DRAWDOWN_EXCEEDED = "drawdown_exceeded"  # 回撤超限
    DRIFT_DETECTED = "drift_detected"        # 数据漂移
    ERROR_RATE_HIGH = "error_rate_high"      # 错误率过高
    LATENCY_HIGH = "latency_high"            # 延迟过高
    MANUAL = "manual"                        # 手动触发


class RollbackStatus(Enum):
    """回滚状态枚举"""
    PENDING = "pending"          # 等待执行
    IN_PROGRESS = "in_progress"  # 执行中
    COMPLETED = "completed"      # 已完成
    FAILED = "failed"            # 失败
    CANCELLED = "cancelled"      # 已取消


@dataclass
class RollbackDecision:
    """
    回滚决策
    
    Attributes:
        should_rollback: 是否应该回滚
        trigger: 触发条件
        confidence: 置信度（0-1）
        reasons: 回滚原因列表
        recommended_action: 建议操作
        metrics_snapshot: 触发时的指标快照
    """
    should_rollback: bool
    trigger: Optional[RollbackTrigger]
    confidence: float
    reasons: List[str] = field(default_factory=list)
    recommended_action: str = ""
    metrics_snapshot: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "should_rollback": self.should_rollback,
            "trigger": self.trigger.value if self.trigger else None,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "recommended_action": self.recommended_action,
            "metrics_snapshot": self.metrics_snapshot,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class RollbackResult:
    """
    回滚结果
    
    Attributes:
        success: 是否成功
        status: 状态
        previous_version: 回滚到的版本
        current_version: 当前版本
        start_time: 开始时间
        end_time: 结束时间
        error_message: 错误信息
    """
    success: bool
    status: RollbackStatus
    previous_version: Optional[str] = None
    current_version: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "status": self.status.value,
            "previous_version": self.previous_version,
            "current_version": self.current_version,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "error_message": self.error_message
        }


class RollbackManager:
    """
    回滚管理器
    
    管理模型自动回滚决策和执行
    
    Attributes:
        model_id: 模型ID
        model_path: 当前模型路径
        backup_path: 备份模型路径
        performance_monitor: 性能监控器
        alert_manager: 告警管理器
        drift_detector: 漂移检测器
    """
    
    def __init__(
        self,
        model_id: str,
        model_path: str,
        backup_path: Optional[str] = None,
        performance_monitor: Optional[ModelPerformanceMonitor] = None,
        alert_manager: Optional[AlertManager] = None,
        drift_detector: Optional[DriftDetector] = None
    ):
        """
        初始化回滚管理器
        
        Args:
            model_id: 模型ID
            model_path: 当前模型路径
            backup_path: 备份模型路径
            performance_monitor: 性能监控器
            alert_manager: 告警管理器
            drift_detector: 漂移检测器
        """
        self.model_id = model_id
        self.model_path = model_path
        self.backup_path = backup_path
        
        self._performance_monitor = performance_monitor
        self._alert_manager = alert_manager
        self._drift_detector = drift_detector
        
        # 回滚配置
        self._thresholds = {
            "accuracy_drop": 0.10,      # 准确率下降10%
            "max_drawdown": 0.15,       # 最大回撤15%
            "drift_score": 0.5,         # 漂移分数0.5
            "error_rate": 0.05,         # 错误率5%
            "latency_p95": 200,         # P95延迟200ms
        }
        
        # 基线指标
        self._baseline_metrics: Dict[str, float] = {}
        
        # 回滚历史
        self._rollback_history: List[Dict[str, Any]] = []
        
        # 状态
        self._is_rollback_in_progress = False
        
        # 回调函数
        self._pre_rollback_callbacks: List[Callable[[], bool]] = []
        self._post_rollback_callbacks: List[Callable[[RollbackResult], None]] = []
        
        self.logger = logging.getLogger(f"monitoring.rollback.{model_id}")
    
    def set_thresholds(self, thresholds: Dict[str, float]) -> None:
        """
        设置回滚阈值
        
        Args:
            thresholds: 阈值字典
        """
        self._thresholds.update(thresholds)
        self.logger.info(f"更新回滚阈值: {thresholds}")
    
    def set_baseline_metrics(self, metrics: Dict[str, float]) -> None:
        """
        设置基线指标
        
        Args:
            metrics: 基线指标字典
        """
        self._baseline_metrics = metrics.copy()
        self.logger.info(f"设置基线指标: {metrics}")
    
    def register_pre_rollback_callback(self, callback: Callable[[], bool]) -> None:
        """
        注册回滚前回调
        
        Args:
            callback: 回调函数，返回False则取消回滚
        """
        self._pre_rollback_callbacks.append(callback)
    
    def register_post_rollback_callback(self, callback: Callable[[RollbackResult], None]) -> None:
        """
        注册回滚后回调
        
        Args:
            callback: 回调函数
        """
        self._post_rollback_callbacks.append(callback)
    
    def evaluate_rollback_need(self) -> RollbackDecision:
        """
        评估是否需要回滚
        
        Returns:
            回滚决策
        """
        reasons = []
        triggers = []
        confidence_scores = []
        metrics_snapshot = {}
        
        # 1. 检查性能监控指标
        if self._performance_monitor:
            latest = self._performance_monitor.get_latest_metrics()
            if latest:
                metrics_snapshot = {
                    k: v.to_dict() if hasattr(v, 'to_dict') else v
                    for k, v in latest.metrics.items()
                }
                
                for metric_name, metric in latest.metrics.items():
                    value = metric.value
                    
                    # 检查准确率下降
                    if metric_name == "accuracy":
                        baseline = self._baseline_metrics.get("accuracy", value)
                        if baseline > 0:
                            drop = (baseline - value) / baseline
                            if drop > self._thresholds["accuracy_drop"]:
                                reasons.append(f"准确率下降 {drop*100:.1f}%")
                                triggers.append(RollbackTrigger.ACCURACY_DROP)
                                confidence_scores.append(min(drop * 5, 1.0))
                    
                    # 检查最大回撤
                    if metric_name == "max_drawdown":
                        if value > self._thresholds["max_drawdown"]:
                            reasons.append(f"最大回撤 {value*100:.1f}%")
                            triggers.append(RollbackTrigger.DRAWDOWN_EXCEEDED)
                            confidence_scores.append(min(value * 3, 1.0))
                    
                    # 检查错误率
                    if metric_name == "error_rate":
                        if value > self._thresholds["error_rate"]:
                            reasons.append(f"错误率 {value*100:.1f}%")
                            triggers.append(RollbackTrigger.ERROR_RATE_HIGH)
                            confidence_scores.append(min(value * 10, 1.0))
                    
                    # 检查延迟
                    if metric_name == "p95_latency_ms":
                        if value > self._thresholds["latency_p95"]:
                            reasons.append(f"P95延迟 {value:.0f}ms")
                            triggers.append(RollbackTrigger.LATENCY_HIGH)
                            confidence_scores.append(min(value / 500, 1.0))
        
        # 2. 检查漂移检测
        if self._drift_detector:
            drift_summary = self._drift_detector.get_drift_summary()
            if drift_summary.get("has_high_severity"):
                reasons.append("检测到严重数据漂移")
                triggers.append(RollbackTrigger.DRIFT_DETECTED)
                confidence_scores.append(0.8)
        
        # 3. 检查告警
        if self._alert_manager:
            critical_alerts = self._alert_manager.get_active_alerts(
                severity=AlertSeverity.CRITICAL
            )
            if len(critical_alerts) >= 2:
                reasons.append(f"存在 {len(critical_alerts)} 个严重告警")
                confidence_scores.append(min(len(critical_alerts) * 0.2, 1.0))
        
        # 计算综合置信度
        if confidence_scores:
            confidence = max(confidence_scores)
        else:
            confidence = 0.0
        
        # 决策
        should_rollback = confidence > 0.7 or len([t for t in triggers if t in [
            RollbackTrigger.ACCURACY_DROP,
            RollbackTrigger.DRAWDOWN_EXCEEDED
        ]]) >= 1
        
        # 确定主要触发条件
        primary_trigger = triggers[0] if triggers else None
        
        # 建议操作
        if should_rollback:
            recommended_action = "立即执行回滚到上一版本"
        elif confidence > 0.4:
            recommended_action = "加强监控，准备回滚"
        else:
            recommended_action = "继续监控"
        
        decision = RollbackDecision(
            should_rollback=should_rollback,
            trigger=primary_trigger,
            confidence=confidence,
            reasons=reasons,
            recommended_action=recommended_action,
            metrics_snapshot=metrics_snapshot
        )
        
        if should_rollback:
            self.logger.warning(
                f"建议回滚: 置信度={confidence:.2f}, 原因={reasons}"
            )
        
        return decision
    
    def execute_rollback(
        self,
        force: bool = False,
        target_version: Optional[str] = None
    ) -> RollbackResult:
        """
        执行回滚
        
        Args:
            force: 是否强制回滚
            target_version: 目标版本（None则回滚到备份）
            
        Returns:
            回滚结果
        """
        if self._is_rollback_in_progress:
            return RollbackResult(
                success=False,
                status=RollbackStatus.FAILED,
                error_message="回滚已在进行中"
            )
        
        # 评估是否需要回滚
        if not force:
            decision = self.evaluate_rollback_need()
            if not decision.should_rollback:
                return RollbackResult(
                    success=False,
                    status=RollbackStatus.CANCELLED,
                    error_message="不满足回滚条件"
                )
        
        # 执行回滚前回调
        for callback in self._pre_rollback_callbacks:
            try:
                if not callback():
                    return RollbackResult(
                        success=False,
                        status=RollbackStatus.CANCELLED,
                        error_message="回滚被前置检查取消"
                    )
            except Exception as e:
                self.logger.error(f"回滚前回调异常: {e}")
        
        self._is_rollback_in_progress = True
        start_time = datetime.now()
        
        try:
            self.logger.info("开始执行回滚")
            
            # 确定回滚目标
            if target_version:
                backup_path = self._find_version(target_version)
            else:
                backup_path = self.backup_path
            
            if not backup_path or not Path(backup_path).exists():
                raise FileNotFoundError(f"备份模型不存在: {backup_path}")
            
            # 保存当前版本信息
            current_version = self._get_current_version()
            
            # 执行回滚
            self._perform_rollback(backup_path)
            
            # 验证回滚
            if not self._verify_rollback():
                raise RuntimeError("回滚验证失败")
            
            result = RollbackResult(
                success=True,
                status=RollbackStatus.COMPLETED,
                previous_version=backup_path,
                current_version=current_version,
                start_time=start_time,
                end_time=datetime.now()
            )
            
            self.logger.info("回滚成功完成")
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"回滚失败: {error_msg}")
            
            result = RollbackResult(
                success=False,
                status=RollbackStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(),
                error_message=error_msg
            )
        
        finally:
            self._is_rollback_in_progress = False
            self._rollback_history.append(result.to_dict())
            
            # 执行回滚后回调
            for callback in self._post_rollback_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    self.logger.error(f"回滚后回调异常: {e}")
        
        return result
    
    def _find_version(self, version: str) -> Optional[str]:
        """查找指定版本"""
        # 从模型存储中查找
        model_dir = Path(self.model_path).parent
        version_file = model_dir / f"model_{version}.joblib"
        if version_file.exists():
            return str(version_file)
        return None
    
    def _get_current_version(self) -> str:
        """获取当前版本"""
        return Path(self.model_path).name
    
    def _perform_rollback(self, backup_path: str) -> None:
        """执行回滚操作"""
        import shutil
        
        # 备份当前模型
        current_backup = f"{self.model_path}.failed.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(self.model_path, current_backup)
        
        # 恢复备份
        shutil.copy2(backup_path, self.model_path)
        
        self.logger.info(f"模型已回滚: {backup_path} -> {self.model_path}")
    
    def _verify_rollback(self) -> bool:
        """验证回滚"""
        try:
            # 尝试加载模型
            model = joblib.load(self.model_path)
            
            # 简单验证
            if model is None:
                return False
            
            self.logger.info("回滚验证通过")
            return True
            
        except Exception as e:
            self.logger.error(f"回滚验证失败: {e}")
            return False
    
    def get_rollback_history(self) -> List[Dict[str, Any]]:
        """
        获取回滚历史
        
        Returns:
            回滚历史列表
        """
        return self._rollback_history.copy()
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取状态
        
        Returns:
            状态字典
        """
        return {
            "model_id": self.model_id,
            "is_rollback_in_progress": self._is_rollback_in_progress,
            "thresholds": self._thresholds,
            "baseline_metrics": self._baseline_metrics,
            "rollback_count": len(self._rollback_history),
            "backup_available": self.backup_path is not None and Path(self.backup_path).exists()
        }


def create_rollback_manager_with_defaults(
    model_id: str,
    model_path: str,
    backup_path: Optional[str] = None
) -> RollbackManager:
    """
    创建带有默认配置的回滚管理器
    
    Args:
        model_id: 模型ID
        model_path: 模型路径
        backup_path: 备份路径
        
    Returns:
        回滚管理器实例
    """
    manager = RollbackManager(
        model_id=model_id,
        model_path=model_path,
        backup_path=backup_path
    )
    
    # 设置默认阈值
    manager.set_thresholds({
        "accuracy_drop": 0.10,      # 准确率下降10%
        "max_drawdown": 0.15,       # 最大回撤15%
        "drift_score": 0.5,         # 漂移分数0.5
        "error_rate": 0.05,         # 错误率5%
        "latency_p95": 200,         # P95延迟200ms
    })
    
    return manager
