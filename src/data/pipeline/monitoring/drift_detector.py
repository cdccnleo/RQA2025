"""
数据漂移检测模块

提供数据分布漂移检测和概念漂移检测功能
"""

import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


class DriftType(Enum):
    """漂移类型枚举"""
    DATA_DRIFT = "data_drift"          # 数据漂移（特征分布变化）
    CONCEPT_DRIFT = "concept_drift"    # 概念漂移（目标变量关系变化）
    FEATURE_DRIFT = "feature_drift"    # 特征漂移（单个特征变化）


class DriftSeverity(Enum):
    """漂移严重程度枚举"""
    NONE = "none"          # 无漂移
    LOW = "low"            # 轻度漂移
    MEDIUM = "medium"      # 中度漂移
    HIGH = "high"          # 严重漂移


@dataclass
class DriftReport:
    """
    漂移检测报告
    
    Attributes:
        timestamp: 检测时间
        drift_type: 漂移类型
        severity: 严重程度
        drift_score: 漂移分数（0-1）
        affected_features: 受影响的特征列表
        statistics: 统计信息
        recommendations: 建议措施
    """
    timestamp: datetime
    drift_type: DriftType
    severity: DriftSeverity
    drift_score: float
    affected_features: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "drift_type": self.drift_type.value,
            "severity": self.severity.value,
            "drift_score": self.drift_score,
            "affected_features": self.affected_features,
            "statistics": self.statistics,
            "recommendations": self.recommendations
        }


class DriftDetectorBase(ABC):
    """
    漂移检测器基类
    
    所有漂移检测器必须继承此类
    """
    
    def __init__(self, name: str, drift_type: DriftType):
        """
        初始化检测器
        
        Args:
            name: 检测器名称
            drift_type: 漂移类型
        """
        self.name = name
        self.drift_type = drift_type
        self.logger = logging.getLogger(f"monitoring.drift.{name}")
    
    @abstractmethod
    def detect(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> DriftReport:
        """
        检测漂移
        
        Args:
            reference_data: 参考数据（基准分布）
            current_data: 当前数据
            
        Returns:
            漂移检测报告
        """
        pass
    
    def _calculate_severity(self, drift_score: float) -> DriftSeverity:
        """
        根据漂移分数计算严重程度
        
        Args:
            drift_score: 漂移分数（0-1）
            
        Returns:
            严重程度
        """
        if drift_score < 0.1:
            return DriftSeverity.NONE
        elif drift_score < 0.3:
            return DriftSeverity.LOW
        elif drift_score < 0.5:
            return DriftSeverity.MEDIUM
        else:
            return DriftSeverity.HIGH


class KSDriftDetector(DriftDetectorBase):
    """
    KS检验漂移检测器
    
    使用Kolmogorov-Smirnov检验检测特征分布漂移
    """
    
    def __init__(self, threshold: float = 0.05):
        """
        初始化KS检测器
        
        Args:
            threshold: 显著性水平阈值
        """
        super().__init__("ks_drift", DriftType.DATA_DRIFT)
        self.threshold = threshold
    
    def detect(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> DriftReport:
        """
        使用KS检验检测漂移
        
        Args:
            reference_data: 参考数据
            current_data: 当前数据
            
        Returns:
            漂移检测报告
        """
        try:
            # 获取数值列
            numeric_cols = reference_data.select_dtypes(include=[np.number]).columns
            
            drift_scores = {}
            affected_features = []
            
            for col in numeric_cols:
                if col in current_data.columns:
                    ref_values = reference_data[col].dropna()
                    cur_values = current_data[col].dropna()
                    
                    if len(ref_values) > 0 and len(cur_values) > 0:
                        # KS检验
                        statistic, p_value = stats.ks_2samp(ref_values, cur_values)
                        
                        # 漂移分数 = 1 - p_value
                        drift_scores[col] = 1 - p_value
                        
                        # 如果p值小于阈值，认为存在漂移
                        if p_value < self.threshold:
                            affected_features.append(col)
            
            # 计算整体漂移分数
            if drift_scores:
                avg_drift_score = np.mean(list(drift_scores.values()))
                max_drift_score = np.max(list(drift_scores.values()))
            else:
                avg_drift_score = 0.0
                max_drift_score = 0.0
            
            severity = self._calculate_severity(max_drift_score)
            
            # 生成建议
            recommendations = self._generate_recommendations(
                severity, affected_features, drift_scores
            )
            
            return DriftReport(
                timestamp=datetime.now(),
                drift_type=self.drift_type,
                severity=severity,
                drift_score=max_drift_score,
                affected_features=affected_features,
                statistics={
                    "avg_drift_score": avg_drift_score,
                    "max_drift_score": max_drift_score,
                    "feature_scores": drift_scores,
                    "total_features": len(numeric_cols),
                    "drifted_features": len(affected_features)
                },
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"KS漂移检测失败: {e}")
            return DriftReport(
                timestamp=datetime.now(),
                drift_type=self.drift_type,
                severity=DriftSeverity.NONE,
                drift_score=0.0,
                recommendations=["检测过程发生错误，请检查数据"]
            )
    
    def _generate_recommendations(
        self,
        severity: DriftSeverity,
        affected_features: List[str],
        drift_scores: Dict[str, float]
    ) -> List[str]:
        """生成建议"""
        recommendations = []
        
        if severity == DriftSeverity.NONE:
            recommendations.append("数据分布稳定，无需采取措施")
        elif severity == DriftSeverity.LOW:
            recommendations.append("检测到轻微数据漂移，建议持续监控")
        elif severity == DriftSeverity.MEDIUM:
            recommendations.append("检测到中度数据漂移，建议：")
            recommendations.append("1. 检查数据源是否有变化")
            recommendations.append("2. 考虑重新训练模型")
            recommendations.append(f"3. 重点关注特征: {', '.join(affected_features[:3])}")
        else:
            recommendations.append("检测到严重数据漂移，建议立即：")
            recommendations.append("1. 暂停模型服务")
            recommendations.append("2. 检查数据管道")
            recommendations.append("3. 使用备用模型")
            recommendations.append(f"4. 漂移最严重的特征: {', '.join(affected_features[:5])}")
        
        return recommendations


class PSI_DriftDetector(DriftDetectorBase):
    """
    PSI（Population Stability Index）漂移检测器
    
    常用于信用评分模型，检测分数分布变化
    """
    
    def __init__(self, threshold: float = 0.25):
        """
        初始化PSI检测器
        
        Args:
            threshold: PSI阈值（通常0.1-0.25）
        """
        super().__init__("psi_drift", DriftType.DATA_DRIFT)
        self.threshold = threshold
    
    def detect(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> DriftReport:
        """
        使用PSI检测漂移
        
        Args:
            reference_data: 参考数据
            current_data: 当前数据
            
        Returns:
            漂移检测报告
        """
        try:
            numeric_cols = reference_data.select_dtypes(include=[np.number]).columns
            
            psi_scores = {}
            affected_features = []
            
            for col in numeric_cols:
                if col in current_data.columns:
                    psi = self._calculate_psi(
                        reference_data[col].dropna(),
                        current_data[col].dropna()
                    )
                    psi_scores[col] = psi
                    
                    if psi > self.threshold:
                        affected_features.append(col)
            
            if psi_scores:
                max_psi = max(psi_scores.values())
                avg_psi = np.mean(list(psi_scores.values()))
            else:
                max_psi = 0.0
                avg_psi = 0.0
            
            # PSI分数归一化到0-1
            drift_score = min(max_psi / 0.5, 1.0)
            severity = self._calculate_severity(drift_score)
            
            recommendations = self._generate_psi_recommendations(
                severity, affected_features, psi_scores
            )
            
            return DriftReport(
                timestamp=datetime.now(),
                drift_type=self.drift_type,
                severity=severity,
                drift_score=drift_score,
                affected_features=affected_features,
                statistics={
                    "avg_psi": avg_psi,
                    "max_psi": max_psi,
                    "feature_psi": psi_scores,
                    "total_features": len(numeric_cols),
                    "drifted_features": len(affected_features)
                },
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"PSI漂移检测失败: {e}")
            return DriftReport(
                timestamp=datetime.now(),
                drift_type=self.drift_type,
                severity=DriftSeverity.NONE,
                drift_score=0.0
            )
    
    def _calculate_psi(self, expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
        """
        计算PSI值
        
        Args:
            expected: 期望分布
            actual: 实际分布
            bins: 分箱数
            
        Returns:
            PSI值
        """
        # 创建分箱
        min_val = min(expected.min(), actual.min())
        max_val = max(expected.max(), actual.max())
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        # 计算频率
        expected_counts, _ = np.histogram(expected, bins=bin_edges)
        actual_counts, _ = np.histogram(actual, bins=bin_edges)
        
        # 转换为百分比
        expected_pct = expected_counts / len(expected)
        actual_pct = actual_counts / len(actual)
        
        # 避免除零
        expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
        actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)
        
        # 计算PSI
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        
        return psi
    
    def _generate_psi_recommendations(
        self,
        severity: DriftSeverity,
        affected_features: List[str],
        psi_scores: Dict[str, float]
    ) -> List[str]:
        """生成PSI建议"""
        recommendations = []
        
        if severity == DriftSeverity.NONE:
            recommendations.append("PSI检测通过，数据分布稳定")
        elif severity == DriftSeverity.LOW:
            recommendations.append("PSI显示轻微漂移，建议关注")
        elif severity == DriftSeverity.MEDIUM:
            recommendations.append("PSI显示中度漂移，建议重新评估模型")
        else:
            recommendations.append("PSI显示严重漂移，建议立即重新训练模型")
        
        return recommendations


class ConceptDriftDetector(DriftDetectorBase):
    """
    概念漂移检测器
    
    检测目标变量与特征之间关系的变化
    """
    
    def __init__(self, model: Any, threshold: float = 0.1):
        """
        初始化概念漂移检测器
        
        Args:
            model: 模型实例
            threshold: 性能下降阈值
        """
        super().__init__("concept_drift", DriftType.CONCEPT_DRIFT)
        self.model = model
        self.threshold = threshold
        self._baseline_performance: Optional[float] = None
    
    def set_baseline(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        设置基线性能
        
        Args:
            X: 特征数据
            y: 目标数据
        """
        from sklearn.metrics import accuracy_score
        y_pred = self.model.predict(X)
        self._baseline_performance = accuracy_score(y, y_pred)
        self.logger.info(f"基线性能设置: {self._baseline_performance:.4f}")
    
    def detect(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> DriftReport:
        """
        检测概念漂移
        
        Args:
            reference_data: 参考数据（包含target列）
            current_data: 当前数据（包含target列）
            
        Returns:
            漂移检测报告
        """
        try:
            if self._baseline_performance is None:
                self.logger.warning("未设置基线性能，无法检测概念漂移")
                return DriftReport(
                    timestamp=datetime.now(),
                    drift_type=self.drift_type,
                    severity=DriftSeverity.NONE,
                    drift_score=0.0,
                    recommendations=["请先设置基线性能"]
                )
            
            if 'target' not in current_data.columns:
                self.logger.warning("当前数据缺少target列")
                return DriftReport(
                    timestamp=datetime.now(),
                    drift_type=self.drift_type,
                    severity=DriftSeverity.NONE,
                    drift_score=0.0
                )
            
            # 准备数据
            feature_cols = [c for c in current_data.columns if c not in ['target', 'timestamp']]
            X_current = current_data[feature_cols].dropna()
            y_current = current_data.loc[X_current.index, 'target']
            
            if len(X_current) < 10:
                return DriftReport(
                    timestamp=datetime.now(),
                    drift_type=self.drift_type,
                    severity=DriftSeverity.NONE,
                    drift_score=0.0,
                    recommendations=["数据量不足，跳过概念漂移检测"]
                )
            
            # 计算当前性能
            from sklearn.metrics import accuracy_score
            y_pred = self.model.predict(X_current)
            current_performance = accuracy_score(y_current, y_pred)
            
            # 计算性能下降
            performance_degradation = self._baseline_performance - current_performance
            drift_score = max(0, performance_degradation / self._baseline_performance) if self._baseline_performance > 0 else 0
            
            severity = self._calculate_severity(drift_score)
            
            recommendations = []
            if severity == DriftSeverity.HIGH:
                recommendations.append("检测到严重概念漂移，模型性能显著下降")
                recommendations.append(f"基线性能: {self._baseline_performance:.4f}, 当前性能: {current_performance:.4f}")
                recommendations.append("建议立即重新训练模型")
            elif severity == DriftSeverity.MEDIUM:
                recommendations.append("检测到中度概念漂移")
                recommendations.append("建议准备模型更新")
            elif severity == DriftSeverity.LOW:
                recommendations.append("检测到轻微概念漂移")
                recommendations.append("建议持续监控")
            else:
                recommendations.append("模型性能稳定")
            
            return DriftReport(
                timestamp=datetime.now(),
                drift_type=self.drift_type,
                severity=severity,
                drift_score=drift_score,
                statistics={
                    "baseline_performance": self._baseline_performance,
                    "current_performance": current_performance,
                    "performance_degradation": performance_degradation,
                    "degradation_percentage": drift_score * 100
                },
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"概念漂移检测失败: {e}")
            return DriftReport(
                timestamp=datetime.now(),
                drift_type=self.drift_type,
                severity=DriftSeverity.NONE,
                drift_score=0.0
            )


class DriftDetector:
    """
    漂移检测管理器
    
    统一管理多个漂移检测器
    """
    
    def __init__(self):
        """初始化漂移检测管理器"""
        self._detectors: List[DriftDetectorBase] = []
        self._reference_data: Optional[pd.DataFrame] = None
        self._drift_history: deque = deque(maxlen=100)
        self.logger = logging.getLogger("monitoring.drift_detector")
    
    def register_detector(self, detector: DriftDetectorBase) -> None:
        """
        注册漂移检测器
        
        Args:
            detector: 检测器实例
        """
        self._detectors.append(detector)
        self.logger.info(f"注册漂移检测器: {detector.name}")
    
    def set_reference_data(self, data: pd.DataFrame) -> None:
        """
        设置参考数据
        
        Args:
            data: 参考数据
        """
        self._reference_data = data.copy()
        self.logger.info(f"设置参考数据: {len(data)} 行")
    
    def detect(self, current_data: pd.DataFrame) -> List[DriftReport]:
        """
        执行所有漂移检测
        
        Args:
            current_data: 当前数据
            
        Returns:
            漂移检测报告列表
        """
        if self._reference_data is None:
            self.logger.warning("未设置参考数据")
            return []
        
        reports = []
        
        for detector in self._detectors:
            try:
                report = detector.detect(self._reference_data, current_data)
                reports.append(report)
                self._drift_history.append(report)
                
                if report.severity in [DriftSeverity.MEDIUM, DriftSeverity.HIGH]:
                    self.logger.warning(
                        f"检测到 {detector.name} 漂移: {report.severity.value}, "
                        f"分数: {report.drift_score:.4f}"
                    )
                
            except Exception as e:
                self.logger.error(f"检测器 {detector.name} 执行失败: {e}")
        
        return reports
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """
        获取漂移汇总信息
        
        Returns:
            汇总信息字典
        """
        if not self._drift_history:
            return {"status": "no_data"}
        
        recent_reports = list(self._drift_history)[-10:]  # 最近10次
        
        # 统计各严重程度次数
        severity_counts = {s.value: 0 for s in DriftSeverity}
        for report in recent_reports:
            severity_counts[report.severity.value] += 1
        
        # 最新报告
        latest = recent_reports[-1]
        
        return {
            "status": "active",
            "total_detections": len(self._drift_history),
            "recent_severity_distribution": severity_counts,
            "latest_drift_score": latest.drift_score,
            "latest_severity": latest.severity.value,
            "has_high_severity": any(
                r.severity == DriftSeverity.HIGH for r in recent_reports
            )
        }
    
    def should_trigger_retraining(self) -> bool:
        """
        判断是否应触发重新训练
        
        Returns:
            是否应该重新训练
        """
        if not self._drift_history:
            return False
        
        # 最近3次检测中有2次中度或以上漂移
        recent = list(self._drift_history)[-3:]
        if len(recent) < 2:
            return False
        
        significant_drifts = sum(
            1 for r in recent
            if r.severity in [DriftSeverity.MEDIUM, DriftSeverity.HIGH]
        )
        
        return significant_drifts >= 2
