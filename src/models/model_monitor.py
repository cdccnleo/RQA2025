import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum, auto
import logging
from abc import ABC, abstractmethod
from scipy.stats import ks_2samp
import warnings
from sklearn.covariance import MinCovDet
from .base_model import BaseModel

logger = logging.getLogger(__name__)

class DriftType(Enum):
    """模型漂移类型枚举"""
    COVARIATE = auto()   # 特征漂移
    TARGET = auto()      # 目标漂移
    CONCEPT = auto()     # 概念漂移
    PRIOR = auto()       # 先验漂移

@dataclass
class ModelPerformance:
    """模型性能指标"""
    accuracy: float
    precision: float
    recall: float
    f1: float
    sharpe: Optional[float] = None
    max_drawdown: Optional[float] = None

@dataclass
class DriftAlert:
    """漂移警报"""
    drift_type: DriftType
    severity: float  # 严重程度 0-1
    test_statistic: float
    p_value: float
    baseline_period: str
    current_period: str

class BaseDriftDetector(ABC):
    """漂移检测基类"""

    @abstractmethod
    def detect(self,
              baseline: pd.DataFrame,
              current: pd.DataFrame) -> DriftAlert:
        """检测数据集漂移"""
        pass

class KSTestDetector(BaseDriftDetector):
    """KS检验漂移检测"""

    def __init__(self,
                 threshold: float = 0.05,
                 min_samples: int = 100):
        """
        Args:
            threshold: 显著性阈值
            min_samples: 最小样本量
        """
        self.threshold = threshold
        self.min_samples = min_samples

    def detect(self,
              baseline: pd.DataFrame,
              current: pd.DataFrame) -> DriftAlert:
        """KS检验检测分布漂移"""
        if len(baseline) < self.min_samples or len(current) < self.min_samples:
            warnings.warn(f"样本量不足({len(baseline)}, {len(current)})")
            return None

        alerts = []
        for col in baseline.columns:
            stat, p = ks_2samp(baseline[col], current[col])
            if p < self.threshold:
                severity = min(1.0, stat * 2)  # 标准化到0-1
                alerts.append(DriftAlert(
                    drift_type=DriftType.COVARIATE,
                    severity=severity,
                    test_statistic=stat,
                    p_value=p,
                    baseline_period=f"{baseline.index.min()}~{baseline.index.max()}",
                    current_period=f"{current.index.min()}~{current.index.max()}"
                ))

        return alerts

class ModelMonitor:
    """模型监控器"""

    def __init__(self,
                 detectors: Dict[str, BaseDriftDetector],
                 rolling_window: int = 30):
        """
        Args:
            detectors: 漂移检测器字典
            rolling_window: 滚动窗口大小(天)
        """
        self.detectors = detectors
        self.window = rolling_window
        self.performance_history = []
        self.drift_alerts = []

    def log_performance(self,
                       model_name: str,
                       performance: ModelPerformance,
                       timestamp: pd.Timestamp):
        """记录模型表现"""
        self.performance_history.append({
            'model': model_name,
            'timestamp': timestamp,
            'metrics': performance
        })

    def check_drift(self,
                   model_name: str,
                   baseline_data: pd.DataFrame,
                   current_data: pd.DataFrame) -> List[DriftAlert]:
        """检查模型漂移"""
        alerts = []
        for name, detector in self.detectors.items():
            try:
                result = detector.detect(baseline_data, current_data)
                if result:
                    if isinstance(result, list):
                        alerts.extend(result)
                    else:
                        alerts.append(result)
            except Exception as e:
                logger.error(f"Detector {name} failed: {str(e)}")

        # 记录漂移警报
        for alert in alerts:
            self.drift_alerts.append({
                'model': model_name,
                'alert': alert,
                'timestamp': pd.Timestamp.now()
            })

        return alerts

    def get_performance_trend(self,
                            model_name: str,
                            metric: str) -> pd.DataFrame:
        """获取模型表现趋势"""
        history = [
            (rec['timestamp'], rec['metrics'].__dict__[metric])
            for rec in self.performance_history
            if rec['model'] == model_name and hasattr(rec['metrics'], metric)
        ]
        return pd.DataFrame(history, columns=['timestamp', metric]).set_index('timestamp')

    def get_recent_alerts(self,
                        model_name: Optional[str] = None,
                        days: int = 7) -> List[Dict]:
        """获取近期警报"""
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
        return [
            alert for alert in self.drift_alerts
            if alert['timestamp'] >= cutoff and
            (model_name is None or alert['model'] == model_name)
        ]

class AdaptiveModelManager:
    """自适应模型管理器"""

    def __init__(self,
                 models: Dict[str, BaseModel],
                 monitor: ModelMonitor,
                 fallback_model: str = 'random_forest'):
        """
        Args:
            models: 模型字典
            monitor: 模型监控器
            fallback_model: 回退模型名
        """
        self.models = models
        self.monitor = monitor
        self.fallback = fallback_model
        self.active_model = None
        self.candidate_models = []

    def switch_model(self, model_name: str) -> bool:
        """切换当前活跃模型"""
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return False

        self.active_model = model_name
        logger.info(f"Switched to model: {model_name}")
        return True

    def evaluate_candidates(self,
                          eval_data: pd.DataFrame,
                          targets: pd.Series) -> Dict[str, ModelPerformance]:
        """评估候选模型"""
        results = {}
        for name, model in self.models.items():
            preds = model.predict(eval_data)
            performance = self._calculate_performance(preds, targets)
            results[name] = performance

            # 记录表现
            self.monitor.log_performance(
                model_name=name,
                performance=performance,
                timestamp=pd.Timestamp.now()
            )

        return results

    def _calculate_performance(self,
                             predictions: pd.Series,
                             targets: pd.Series) -> ModelPerformance:
        """计算模型表现指标"""
        # 分类指标
        if np.all(np.isin(predictions, [0, 1])):
            tp = ((predictions == 1) & (targets == 1)).sum()
            fp = ((predictions == 1) & (targets == 0)).sum()
            fn = ((predictions == 0) & (targets == 1)).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            return ModelPerformance(
                accuracy=(predictions == targets).mean(),
                precision=precision,
                recall=recall,
                f1=f1
            )
        # 回归指标
        else:
            returns = predictions * targets  # 简化计算
            sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            cum_returns = (1 + returns).cumprod()
            max_dd = (cum_returns / cum_returns.cummax() - 1).min()

            return ModelPerformance(
                accuracy=-((predictions - targets) ** 2).mean(),  # 负MSE
                precision=sharpe,
                recall=max_dd,
                f1=sharpe / (1 - max_dd) if max_dd < 0 else sharpe
            )

    def adaptive_update(self,
                       new_data: pd.DataFrame,
                       targets: pd.Series,
                       threshold: float = 0.1) -> bool:
        """自适应模型更新"""
        if not self.active_model:
            self.switch_model(self.fallback)
            return False

        # 检查当前模型表现
        current_perf = self.evaluate_candidates(
            new_data,
            targets
        )[self.active_model]

        # 评估候选模型
        candidate_perf = {}
        for name in self.candidate_models:
            if name != self.active_model:
                candidate_perf[name] = self.evaluate_candidates(
                    new_data,
                    targets
                )[name]

        # 寻找表现提升超过阈值的候选模型
        for name, perf in candidate_perf.items():
            improvement = self._calculate_improvement(current_perf, perf)
            if improvement >= threshold:
                return self.switch_model(name)

        return False

    def _calculate_improvement(self,
                            current: ModelPerformance,
                            candidate: ModelPerformance) -> float:
        """计算模型表现提升"""
        # 使用夏普比率或F1分数作为主要指标
        if current.sharpe is not None and candidate.sharpe is not None:
            return (candidate.sharpe - current.sharpe) / (abs(current.sharpe) + 1e-6)
        else:
            return (candidate.f1 - current.f1) / (current.f1 + 1e-6)

class OnlineLearner:
    """在线学习器"""

    def __init__(self,
                 base_model: BaseModel,
                 learning_rate: float = 0.01,
                 batch_size: int = 1000):
        """
        Args:
            base_model: 基础模型
            learning_rate: 学习率
            batch_size: 批处理大小
        """
        self.model = base_model
        self.lr = learning_rate
        self.batch_size = batch_size
        self.buffer_X = []
        self.buffer_y = []

    def partial_fit(self,
                   X: pd.DataFrame,
                   y: pd.Series):
        """增量学习"""
        self.buffer_X.append(X)
        self.buffer_y.append(y)

        if len(self.buffer_X) >= self.batch_size:
            self._flush_buffer()

    def _flush_buffer(self):
        """刷新缓冲区并更新模型"""
        if not self.buffer_X:
            return

        X = pd.concat(self.buffer_X)
        y = pd.concat(self.buffer_y)

        # 增量训练
        self.model.partial_fit(X, y)

        # 清空缓冲区
        self.buffer_X = []
        self.buffer_y = []
