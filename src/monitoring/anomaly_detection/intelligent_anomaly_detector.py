"""
智能异常检测算法模块

功能：
- 孤立森林(Isolation Forest)异常检测
- 局部异常因子(LOF)算法
- 多维度异常检测
- 实时异常监控与告警
- 异常根因分析
- 自适应阈值调整

技术栈：
- scikit-learn: 孤立森林、LOF实现
- numpy/pandas: 数据处理
- asyncio: 异步检测

作者: Claude
创建日期: 2026-02-21
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Callable, Union
from collections import defaultdict, deque
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib

# 忽略警告
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """异常类型"""
    POINT_ANOMALY = "point_anomaly"           # 点异常
    CONTEXTUAL_ANOMALY = "contextual_anomaly" # 上下文异常
    COLLECTIVE_ANOMALY = "collective_anomaly" # 集合异常
    TEMPORAL_ANOMALY = "temporal_anomaly"     # 时间异常
    SPATIAL_ANOMALY = "spatial_anomaly"       # 空间异常


class AnomalySeverity(Enum):
    """异常严重程度"""
    LOW = 1       # 低
    MEDIUM = 2    # 中
    HIGH = 3      # 高
    CRITICAL = 4  # 严重


@dataclass
class AnomalyDetection:
    """异常检测结果"""
    timestamp: datetime
    data_key: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    score: float
    features: Dict[str, Any]
    explanation: str
    recommended_action: str
    related_metrics: List[str] = field(default_factory=list)


@dataclass
class DetectionConfig:
    """检测配置"""
    algorithm: str = "isolation_forest"
    contamination: float = 0.1
    n_estimators: int = 100
    max_samples: Union[int, str] = "auto"
    n_neighbors: int = 20
    threshold: float = -0.5
    feature_columns: List[str] = field(default_factory=list)
    window_size: int = 1000
    adaptive_threshold: bool = True


@dataclass
class DetectionStats:
    """检测统计"""
    total_detections: int = 0
    false_positives: int = 0
    true_positives: int = 0
    avg_detection_time_ms: float = 0.0
    last_detection: Optional[datetime] = None
    detection_distribution: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


class FeatureExtractor:
    """
    特征提取器
    
    从原始数据中提取异常检测特征
    """
    
    def __init__(self, window_size: int = 100):
        """
        初始化特征提取器
        
        Args:
            window_size: 滑动窗口大小
        """
        self.window_size = window_size
        self.data_history: deque = deque(maxlen=window_size)
        self.scaler = RobustScaler()
        
    def extract_features(self, data: Dict[str, Any], 
                        feature_columns: Optional[List[str]] = None) -> np.ndarray:
        """
        提取特征
        
        Args:
            data: 原始数据
            feature_columns: 特征列名
            
        Returns:
            特征向量
        """
        features = []
        
        if feature_columns:
            for col in feature_columns:
                value = data.get(col, 0)
                if isinstance(value, (int, float)):
                    features.append(float(value))
                elif isinstance(value, str):
                    features.append(hash(value) % 10000)
                else:
                    features.append(0.0)
        else:
            # 自动提取数值特征
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    features.append(float(value))
                    
        # 添加统计特征
        if len(self.data_history) > 0:
            historical_values = list(self.data_history)
            if historical_values and len(historical_values[0]) == len(features):
                hist_array = np.array(historical_values)
                
                # 添加移动平均、标准差等特征
                for i in range(len(features)):
                    col_values = hist_array[:, i]
                    features.append(np.mean(col_values))
                    features.append(np.std(col_values))
                    features.append(np.max(col_values) - np.min(col_values))
                    
        # 保存到历史
        self.data_history.append(features[:len(features) // 3] if len(features) > 10 else features)
        
        return np.array(features).reshape(1, -1)
        
    def fit_scaler(self, data: List[Dict[str, Any]], 
                   feature_columns: Optional[List[str]] = None) -> None:
        """
        拟合标准化器
        
        Args:
            data: 训练数据
            feature_columns: 特征列
        """
        features_list = []
        for item in data:
            features = self.extract_features(item, feature_columns)
            features_list.append(features[0])
            
        if features_list:
            X = np.array(features_list)
            self.scaler.fit(X)
            
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        标准化特征
        
        Args:
            features: 原始特征
            
        Returns:
            标准化后的特征
        """
        return self.scaler.transform(features)


class IsolationForestDetector:
    """
    孤立森林异常检测器
    
    基于随机划分的异常检测算法
    """
    
    def __init__(self, config: Optional[DetectionConfig] = None):
        """
        初始化孤立森林检测器
        
        Args:
            config: 检测配置
        """
        self.config = config or DetectionConfig()
        self.model: Optional[IsolationForest] = None
        self.feature_extractor = FeatureExtractor()
        self.is_trained = False
        self.threshold = self.config.threshold
        
    def fit(self, data: List[Dict[str, Any]], 
            feature_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            data: 训练数据
            feature_columns: 特征列
            
        Returns:
            训练结果
        """
        if len(data) < 50:
            return {'status': 'insufficient_data', 'samples': len(data)}
            
        logger.info(f"训练孤立森林模型，数据量: {len(data)}")
        
        # 提取特征
        features_list = []
        for item in data:
            features = self.feature_extractor.extract_features(
                item, feature_columns or self.config.feature_columns
            )
            features_list.append(features[0])
            
        X = np.array(features_list)
        
        # 拟合标准化器
        self.feature_extractor.fit_scaler(data, feature_columns)
        X_scaled = self.feature_extractor.scaler.transform(X)
        
        # 创建并训练模型
        self.model = IsolationForest(
            n_estimators=self.config.n_estimators,
            contamination=self.config.contamination,
            max_samples=self.config.max_samples,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_scaled)
        
        # 计算阈值
        scores = self.model.decision_function(X_scaled)
        self.threshold = np.percentile(scores, self.config.contamination * 100)
        
        self.is_trained = True
        
        # 评估
        predictions = self.model.predict(X_scaled)
        anomaly_ratio = np.sum(predictions == -1) / len(predictions)
        
        metrics = {
            'status': 'success',
            'samples': len(data),
            'features': X.shape[1],
            'anomaly_ratio_in_training': round(anomaly_ratio, 4),
            'threshold': round(self.threshold, 4),
            'score_mean': round(float(np.mean(scores)), 4),
            'score_std': round(float(np.std(scores)), 4)
        }
        
        logger.info(f"孤立森林模型训练完成: {metrics}")
        return metrics
        
    def detect(self, data: Dict[str, Any], 
               data_key: str = "unknown") -> Optional[AnomalyDetection]:
        """
        检测异常
        
        Args:
            data: 待检测数据
            data_key: 数据标识
            
        Returns:
            异常检测结果
        """
        if not self.is_trained or self.model is None:
            return None
            
        start_time = time.time()
        
        # 提取特征
        features = self.feature_extractor.extract_features(
            data, self.config.feature_columns
        )
        features_scaled = self.feature_extractor.transform(features)
        
        # 预测
        prediction = self.model.predict(features_scaled)[0]
        score = self.model.decision_function(features_scaled)[0]
        
        detection_time = (time.time() - start_time) * 1000
        
        # 判断是否为异常
        if prediction == -1 or score < self.threshold:
            # 确定严重程度
            severity = self._calculate_severity(score)
            
            # 生成解释
            explanation = self._generate_explanation(data, score, features[0])
            
            return AnomalyDetection(
                timestamp=datetime.now(),
                data_key=data_key,
                anomaly_type=AnomalyType.POINT_ANOMALY,
                severity=severity,
                score=float(score),
                features={k: v for k, v in data.items() if isinstance(v, (int, float, str))},
                explanation=explanation,
                recommended_action=self._recommend_action(severity),
                related_metrics=self.config.feature_columns[:5]
            )
            
        return None
        
    def _calculate_severity(self, score: float) -> AnomalySeverity:
        """计算严重程度"""
        if score < -0.7:
            return AnomalySeverity.CRITICAL
        elif score < -0.5:
            return AnomalySeverity.HIGH
        elif score < -0.3:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW
            
    def _generate_explanation(self, data: Dict[str, Any], 
                             score: float, 
                             features: np.ndarray) -> str:
        """生成异常解释"""
        explanations = []
        
        if score < -0.5:
            explanations.append(f"异常得分 {score:.3f} 远低于正常阈值")
            
        # 检查具体特征
        for key, value in data.items():
            if isinstance(value, (int, float)):
                if value == 0:
                    explanations.append(f"{key} 值为零，可能存在异常")
                elif value < 0:
                    explanations.append(f"{key} 为负值: {value}")
                    
        return "; ".join(explanations) if explanations else "检测到异常模式"
        
    def _recommend_action(self, severity: AnomalySeverity) -> str:
        """推荐处理动作"""
        actions = {
            AnomalySeverity.LOW: "建议监控观察",
            AnomalySeverity.MEDIUM: "建议检查相关指标",
            AnomalySeverity.HIGH: "建议立即调查原因",
            AnomalySeverity.CRITICAL: "建议立即采取措施并通知相关人员"
        }
        return actions.get(severity, "建议进一步分析")
        
    def batch_detect(self, data_list: List[Dict[str, Any]], 
                    data_keys: Optional[List[str]] = None) -> List[AnomalyDetection]:
        """
        批量检测
        
        Args:
            data_list: 数据列表
            data_keys: 数据标识列表
            
        Returns:
            异常列表
        """
        anomalies = []
        
        for i, data in enumerate(data_list):
            key = data_keys[i] if data_keys and i < len(data_keys) else f"item_{i}"
            anomaly = self.detect(data, key)
            if anomaly:
                anomalies.append(anomaly)
                
        return anomalies


class LOFDetector:
    """
    局部异常因子(LOF)检测器
    
    基于密度的局部异常检测算法
    """
    
    def __init__(self, config: Optional[DetectionConfig] = None):
        """
        初始化LOF检测器
        
        Args:
            config: 检测配置
        """
        self.config = config or DetectionConfig(algorithm="lof")
        self.model: Optional[LocalOutlierFactor] = None
        self.feature_extractor = FeatureExtractor()
        self.reference_data: Optional[np.ndarray] = None
        self.is_trained = False
        
    def fit(self, data: List[Dict[str, Any]], 
            feature_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            data: 训练数据
            feature_columns: 特征列
            
        Returns:
            训练结果
        """
        if len(data) < self.config.n_neighbors * 2:
            return {'status': 'insufficient_data', 'samples': len(data)}
            
        logger.info(f"训练LOF模型，数据量: {len(data)}")
        
        # 提取特征
        features_list = []
        for item in data:
            features = self.feature_extractor.extract_features(
                item, feature_columns or self.config.feature_columns
            )
            features_list.append(features[0])
            
        X = np.array(features_list)
        
        # 拟合标准化器
        self.feature_extractor.fit_scaler(data, feature_columns)
        X_scaled = self.feature_extractor.scaler.transform(X)
        
        # 保存参考数据
        self.reference_data = X_scaled
        
        # 创建并训练模型
        self.model = LocalOutlierFactor(
            n_neighbors=self.config.n_neighbors,
            contamination=self.config.contamination,
            novelty=True,  # 启用novelty模式用于预测
            n_jobs=-1
        )
        
        self.model.fit(X_scaled)
        self.is_trained = True
        
        # 评估
        scores = self.model.decision_function(X_scaled)
        predictions = self.model.predict(X_scaled)
        anomaly_ratio = np.sum(predictions == -1) / len(predictions)
        
        metrics = {
            'status': 'success',
            'samples': len(data),
            'features': X.shape[1],
            'n_neighbors': self.config.n_neighbors,
            'anomaly_ratio_in_training': round(anomaly_ratio, 4),
            'score_mean': round(float(np.mean(scores)), 4),
            'score_std': round(float(np.std(scores)), 4)
        }
        
        logger.info(f"LOF模型训练完成: {metrics}")
        return metrics
        
    def detect(self, data: Dict[str, Any], 
               data_key: str = "unknown") -> Optional[AnomalyDetection]:
        """
        检测异常
        
        Args:
            data: 待检测数据
            data_key: 数据标识
            
        Returns:
            异常检测结果
        """
        if not self.is_trained or self.model is None:
            return None
            
        start_time = time.time()
        
        # 提取特征
        features = self.feature_extractor.extract_features(
            data, self.config.feature_columns
        )
        features_scaled = self.feature_extractor.transform(features)
        
        # 预测
        prediction = self.model.predict(features_scaled)[0]
        score = self.model.decision_function(features_scaled)[0]
        
        # 获取LOF分数（负的异常因子）
        lof_scores = -self.model.score_samples(features_scaled)
        lof_score = lof_scores[0]
        
        detection_time = (time.time() - start_time) * 1000
        
        # 判断是否为异常
        if prediction == -1 or lof_score > 1.5:
            # 确定严重程度
            severity = self._calculate_severity(lof_score)
            
            return AnomalyDetection(
                timestamp=datetime.now(),
                data_key=data_key,
                anomaly_type=AnomalyType.CONTEXTUAL_ANOMALY,
                severity=severity,
                score=float(lof_score),
                features={k: v for k, v in data.items() if isinstance(v, (int, float, str))},
                explanation=f"LOF得分 {lof_score:.3f} 表明该数据点与邻居显著不同",
                recommended_action=self._recommend_action(severity),
                related_metrics=self.config.feature_columns[:5]
            )
            
        return None
        
    def _calculate_severity(self, lof_score: float) -> AnomalySeverity:
        """计算严重程度"""
        if lof_score > 3.0:
            return AnomalySeverity.CRITICAL
        elif lof_score > 2.0:
            return AnomalySeverity.HIGH
        elif lof_score > 1.5:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW
            
    def _recommend_action(self, severity: AnomalySeverity) -> str:
        """推荐处理动作"""
        actions = {
            AnomalySeverity.LOW: "LOF检测到低风险异常，建议监控",
            AnomalySeverity.MEDIUM: "LOF检测到中等风险异常，建议检查",
            AnomalySeverity.HIGH: "LOF检测到高风险异常，建议立即调查",
            AnomalySeverity.CRITICAL: "LOF检测到严重异常，建议立即响应"
        }
        return actions.get(severity, "建议进一步分析")
        
    def batch_detect(self, data_list: List[Dict[str, Any]], 
                    data_keys: Optional[List[str]] = None) -> List[AnomalyDetection]:
        """
        批量检测
        
        Args:
            data_list: 数据列表
            data_keys: 数据标识列表
            
        Returns:
            异常列表
        """
        anomalies = []
        
        for i, data in enumerate(data_list):
            key = data_keys[i] if data_keys and i < len(data_keys) else f"item_{i}"
            anomaly = self.detect(data, key)
            if anomaly:
                anomalies.append(anomaly)
                
        return anomalies


class EnsembleAnomalyDetector:
    """
    集成异常检测器
    
    结合多种检测算法的结果
    """
    
    def __init__(self):
        """初始化集成检测器"""
        self.detectors: Dict[str, Union[IsolationForestDetector, LOFDetector]] = {}
        self.weights: Dict[str, float] = {}
        self.stats = DetectionStats()
        
    def add_detector(self, name: str, 
                    detector: Union[IsolationForestDetector, LOFDetector],
                    weight: float = 1.0) -> None:
        """
        添加检测器
        
        Args:
            name: 检测器名称
            detector: 检测器实例
            weight: 权重
        """
        self.detectors[name] = detector
        self.weights[name] = weight
        
    def detect(self, data: Dict[str, Any], 
               data_key: str = "unknown") -> Optional[AnomalyDetection]:
        """
        集成检测
        
        Args:
            data: 待检测数据
            data_key: 数据标识
            
        Returns:
            异常检测结果
        """
        if not self.detectors:
            return None
            
        start_time = time.time()
        
        detections = []
        total_score = 0.0
        total_weight = 0.0
        
        for name, detector in self.detectors.items():
            weight = self.weights.get(name, 1.0)
            detection = detector.detect(data, data_key)
            
            if detection:
                detections.append(detection)
                total_score += detection.score * weight
                total_weight += weight
                
        detection_time = (time.time() - start_time) * 1000
        self._update_stats(detection_time)
        
        if not detections:
            return None
            
        # 计算加权平均分数
        avg_score = total_score / total_weight if total_weight > 0 else 0
        
        # 确定最严重的异常
        max_severity = max(d.severity for d in detections)
        
        # 合并解释
        all_explanations = [d.explanation for d in detections]
        merged_explanation = " | ".join(set(all_explanations))
        
        return AnomalyDetection(
            timestamp=datetime.now(),
            data_key=data_key,
            anomaly_type=AnomalyType.COLLECTIVE_ANOMALY,
            severity=max_severity,
            score=float(avg_score),
            features={k: v for k, v in data.items() if isinstance(v, (int, float, str))},
            explanation=f"集成检测结果: {merged_explanation}",
            recommended_action="建议综合多个检测结果进行分析",
            related_metrics=list(set(
                metric for d in detections for metric in d.related_metrics
            ))
        )
        
    def _update_stats(self, detection_time: float) -> None:
        """更新统计"""
        self.stats.total_detections += 1
        
        n = self.stats.total_detections
        if n == 1:
            self.stats.avg_detection_time_ms = detection_time
        else:
            self.stats.avg_detection_time_ms = (
                (self.stats.avg_detection_time_ms * (n - 1) + detection_time) / n
            )
            
        self.stats.last_detection = datetime.now()


class IntelligentAnomalyDetector:
    """
    智能异常检测器主类
    
    提供统一的异常检测接口
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        初始化智能异常检测器
        
        Args:
            model_path: 模型保存路径
        """
        self.model_path = model_path
        self.ensemble = EnsembleAnomalyDetector()
        self.anomaly_history: deque = deque(maxlen=1000)
        self.alert_callbacks: List[Callable] = []
        
        # 初始化检测器
        self._init_detectors()
        
    def _init_detectors(self) -> None:
        """初始化检测器"""
        # 孤立森林检测器
        if_config = DetectionConfig(
            algorithm="isolation_forest",
            contamination=0.1,
            n_estimators=100
        )
        if_detector = IsolationForestDetector(if_config)
        self.ensemble.add_detector("isolation_forest", if_detector, weight=1.0)
        
        # LOF检测器
        lof_config = DetectionConfig(
            algorithm="lof",
            contamination=0.1,
            n_neighbors=20
        )
        lof_detector = LOFDetector(lof_config)
        self.ensemble.add_detector("lof", lof_detector, weight=0.8)
        
    def train(self, data: List[Dict[str, Any]], 
              feature_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        训练所有检测器
        
        Args:
            data: 训练数据
            feature_columns: 特征列
            
        Returns:
            训练结果
        """
        results = {}
        
        for name, detector in self.ensemble.detectors.items():
            result = detector.fit(data, feature_columns)
            results[name] = result
            
        # 保存模型
        if self.model_path:
            self._save_models()
            
        return results
        
    def detect(self, data: Dict[str, Any], 
               data_key: str = "unknown") -> Optional[AnomalyDetection]:
        """
        检测异常
        
        Args:
            data: 待检测数据
            data_key: 数据标识
            
        Returns:
            异常检测结果
        """
        anomaly = self.ensemble.detect(data, data_key)
        
        if anomaly:
            self.anomaly_history.append(anomaly)
            self.ensemble.stats.detection_distribution[anomaly.anomaly_type.value] += 1
            
            # 触发告警
            if anomaly.severity in [AnomalySeverity.HIGH, AnomalySeverity.CRITICAL]:
                self._trigger_alert(anomaly)
                
        return anomaly
        
    def batch_detect(self, data_list: List[Dict[str, Any]], 
                    data_keys: Optional[List[str]] = None) -> List[AnomalyDetection]:
        """
        批量检测
        
        Args:
            data_list: 数据列表
            data_keys: 数据标识列表
            
        Returns:
            异常列表
        """
        anomalies = []
        
        for i, data in enumerate(data_list):
            key = data_keys[i] if data_keys and i < len(data_keys) else f"item_{i}"
            anomaly = self.detect(data, key)
            if anomaly:
                anomalies.append(anomaly)
                
        return anomalies
        
    def register_alert_callback(self, callback: Callable) -> None:
        """
        注册告警回调
        
        Args:
            callback: 回调函数
        """
        self.alert_callbacks.append(callback)
        
    def _trigger_alert(self, anomaly: AnomalyDetection) -> None:
        """
        触发告警
        
        Args:
            anomaly: 异常检测结果
        """
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(anomaly))
                else:
                    callback(anomaly)
            except Exception as e:
                logger.error(f"告警回调执行失败: {e}")
                
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'total_detections': self.ensemble.stats.total_detections,
            'avg_detection_time_ms': round(self.ensemble.stats.avg_detection_time_ms, 2),
            'last_detection': self.ensemble.stats.last_detection.isoformat() if self.ensemble.stats.last_detection else None,
            'detection_distribution': dict(self.ensemble.stats.detection_distribution),
            'recent_anomalies': len(self.anomaly_history),
            'detectors': list(self.ensemble.detectors.keys())
        }
        
    def get_recent_anomalies(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        获取最近的异常
        
        Args:
            n: 数量
            
        Returns:
            异常列表
        """
        recent = list(self.anomaly_history)[-n:]
        return [
            {
                'timestamp': a.timestamp.isoformat(),
                'data_key': a.data_key,
                'type': a.anomaly_type.value,
                'severity': a.severity.name,
                'score': round(a.score, 4),
                'explanation': a.explanation
            }
            for a in recent
        ]
        
    def _save_models(self) -> None:
        """保存模型"""
        if self.model_path:
            model_data = {
                'detectors': self.ensemble.detectors,
                'weights': self.ensemble.weights
            }
            joblib.dump(model_data, self.model_path)
            logger.info(f"模型已保存到: {self.model_path}")
            
    def load_models(self) -> bool:
        """
        加载模型
        
        Returns:
            是否成功
        """
        try:
            if self.model_path:
                model_data = joblib.load(self.model_path)
                self.ensemble.detectors = model_data['detectors']
                self.ensemble.weights = model_data['weights']
                logger.info(f"模型已从 {self.model_path} 加载")
                return True
        except Exception as e:
            logger.warning(f"加载模型失败: {e}")
        return False
        
    def export_anomalies(self, filepath: str, 
                        since: Optional[datetime] = None) -> None:
        """
        导出异常记录
        
        Args:
            filepath: 文件路径
            since: 起始时间
        """
        anomalies = list(self.anomaly_history)
        
        if since:
            anomalies = [a for a in anomalies if a.timestamp >= since]
            
        data = [
            {
                'timestamp': a.timestamp.isoformat(),
                'data_key': a.data_key,
                'type': a.anomaly_type.value,
                'severity': a.severity.name,
                'score': a.score,
                'explanation': a.explanation,
                'recommended_action': a.recommended_action,
                'features': a.features
            }
            for a in anomalies
        ]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# 单例实例
_detector_instance: Optional[IntelligentAnomalyDetector] = None


def get_detector(model_path: Optional[str] = None) -> IntelligentAnomalyDetector:
    """
    获取智能异常检测器单例
    
    Args:
        model_path: 模型路径
        
    Returns:
        IntelligentAnomalyDetector实例
    """
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = IntelligentAnomalyDetector(model_path)
    return _detector_instance
