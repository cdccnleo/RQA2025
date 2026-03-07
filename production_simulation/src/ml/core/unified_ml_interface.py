#!/usr/bin/env python3
"""
统一机器学习算法接口

定义机器学习层算法的统一接口，确保所有ML算法实现统一的API。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd


class MLAlgorithmType(Enum):
    """机器学习算法类型"""
    SUPERVISED_LEARNING = "supervised_learning"
    UNSUPERVISED_LEARNING = "unsupervised_learning"
    SEMI_SUPERVISED_LEARNING = "semi_supervised_learning"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    DEEP_LEARNING = "deep_learning"
    ENSEMBLE_LEARNING = "ensemble_learning"


class MLTaskType(Enum):
    """机器学习任务类型"""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    ANOMALY_DETECTION = "anomaly_detection"
    RECOMMENDATION = "recommendation"
    TIME_SERIES = "time_series"
    NATURAL_LANGUAGE_PROCESSING = "nlp"
    COMPUTER_VISION = "computer_vision"


class ModelState(Enum):
    """模型状态"""
    INITIALIZED = "initialized"
    TRAINING = "training"
    TRAINED = "trained"
    EVALUATING = "evaluating"
    DEPLOYED = "deployed"
    ERROR = "error"
    ARCHIVED = "archived"


class OptimizationMetric(Enum):
    """优化指标"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    AUC = "auc"
    MSE = "mse"
    RMSE = "rmse"
    MAE = "mae"
    R2_SCORE = "r2_score"
    SILHOUETTE_SCORE = "silhouette_score"


@dataclass
class MLModelConfig:
    """
    机器学习模型配置

    定义模型的配置参数。
    """
    algorithm_type: MLAlgorithmType
    task_type: MLTaskType
    hyperparameters: Dict[str, Any]
    feature_columns: Optional[List[str]] = None
    target_column: Optional[str] = None
    validation_split: float = 0.2
    random_state: int = 42
    max_training_time: Optional[int] = None  # 秒
    early_stopping_patience: Optional[int] = None
    cross_validation_folds: int = 5
    optimization_metric: OptimizationMetric = OptimizationMetric.ACCURACY


@dataclass
class MLTrainingResult:
    """
    机器学习训练结果

    包含训练过程和结果的所有信息。
    """
    model_id: str
    algorithm_name: str
    training_start_time: datetime
    training_end_time: datetime
    training_duration: float
    final_score: float
    best_score: float
    hyperparameters: Dict[str, Any]
    feature_importance: Optional[Dict[str, float]] = None
    training_metrics: Dict[str, List[float]] = None
    validation_metrics: Dict[str, List[float]] = None
    model_size: int = 0  # 字节
    training_samples: int = 0
    feature_count: int = 0
    converged: bool = True
    early_stopped: bool = False


@dataclass
class MLPredictionResult:
    """
    机器学习预测结果

    包含预测结果和相关信息。
    """
    predictions: Union[np.ndarray, List[Any]]
    probabilities: Optional[Union[np.ndarray, List[List[float]]]] = None
    prediction_time: float = 0.0
    model_version: str = ""
    confidence_scores: Optional[Union[np.ndarray, List[float]]] = None
    feature_contributions: Optional[Dict[str, Union[np.ndarray, List[float]]]] = None


@dataclass
class MLModelMetadata:
    """
    机器学习模型元数据

    包含模型的描述信息和版本控制。
    """
    model_id: str
    model_name: str
    version: str
    algorithm_type: MLAlgorithmType
    task_type: MLTaskType
    created_at: datetime
    updated_at: datetime
    author: str
    description: str
    tags: List[str]
    framework_version: str
    training_data_info: Dict[str, Any]
    performance_metrics: Dict[str, float]
    model_size: int
    is_active: bool = True


class IMLAlgorithm(ABC):
    """
    机器学习算法统一接口

    所有ML算法实现必须遵循此接口，确保API的一致性。
    """

    @abstractmethod
    def get_algorithm_name(self) -> str:
        """
        获取算法名称

        Returns:
            算法名称
        """

    @abstractmethod
    def get_algorithm_type(self) -> MLAlgorithmType:
        """
        获取算法类型

        Returns:
            算法类型
        """

    @abstractmethod
    def get_supported_tasks(self) -> List[MLTaskType]:
        """
        获取支持的任务类型

        Returns:
            支持的任务类型列表
        """

    @abstractmethod
    def get_hyperparameter_space(self) -> Dict[str, Any]:
        """
        获取超参数搜索空间

        Returns:
            超参数空间字典
        """

    @abstractmethod
    def get_default_hyperparameters(self) -> Dict[str, Any]:
        """
        获取默认超参数

        Returns:
            默认超参数字典
        """

    @abstractmethod
    def validate_hyperparameters(self, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证超参数

        Args:
            hyperparameters: 超参数字典

        Returns:
            验证结果 {'valid': bool, 'errors': List[str], 'warnings': List[str]}
        """

    @abstractmethod
    def preprocess_data(self, X: Union[np.ndarray, pd.DataFrame],
                        y: Optional[Union[np.ndarray, pd.Series]] = None,
                        config: MLModelConfig = None) -> Tuple[Any, Optional[Any]]:
        """
        数据预处理

        Args:
            X: 特征数据
            y: 目标数据（可选）
            config: 模型配置

        Returns:
            预处理后的数据 (X_processed, y_processed)
        """

    @abstractmethod
    def train(self, X: Union[np.ndarray, pd.DataFrame],
              y: Union[np.ndarray, pd.Series],
              config: MLModelConfig,
              validation_data: Optional[Tuple[Any, Any]] = None) -> MLTrainingResult:
        """
        训练模型

        Args:
            X: 训练特征数据
            y: 训练目标数据
            config: 模型配置
            validation_data: 验证数据 (X_val, y_val)

        Returns:
            训练结果
        """

    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> MLPredictionResult:
        """
        进行预测

        Args:
            X: 预测特征数据

        Returns:
            预测结果
        """

    @abstractmethod
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        进行概率预测（分类任务）

        Args:
            X: 预测特征数据

        Returns:
            预测概率数组
        """

    @abstractmethod
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame],
                 y: Union[np.ndarray, pd.Series],
                 metrics: List[OptimizationMetric]) -> Dict[str, float]:
        """
        评估模型性能

        Args:
            X: 测试特征数据
            y: 测试目标数据
            metrics: 评估指标列表

        Returns:
            评估结果字典
        """

    @abstractmethod
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        获取特征重要性

        Returns:
            特征重要性字典
        """

    @abstractmethod
    def save_model(self, path: str) -> bool:
        """
        保存模型

        Args:
            path: 保存路径

        Returns:
            是否保存成功
        """

    @abstractmethod
    def load_model(self, path: str) -> bool:
        """
        加载模型

        Args:
            path: 加载路径

        Returns:
            是否加载成功
        """

    @abstractmethod
    def get_model_size(self) -> int:
        """
        获取模型大小

        Returns:
            模型大小（字节）
        """

    @abstractmethod
    def get_model_state(self) -> ModelState:
        """
        获取模型状态

        Returns:
            模型状态
        """

    @abstractmethod
    def reset_model(self) -> None:
        """重置模型状态"""

    @abstractmethod
    def get_training_info(self) -> Dict[str, Any]:
        """
        获取训练信息

        Returns:
            训练信息字典
        """

    @abstractmethod
    def supports_early_stopping(self) -> bool:
        """
        是否支持早停

        Returns:
            是否支持早停
        """

    @abstractmethod
    def supports_incremental_learning(self) -> bool:
        """
        是否支持增量学习

        Returns:
            是否支持增量学习
        """

    @abstractmethod
    def update_model(self, X: Union[np.ndarray, pd.DataFrame],
                     y: Union[np.ndarray, pd.Series]) -> bool:
        """
        增量更新模型

        Args:
            X: 新特征数据
            y: 新目标数据

        Returns:
            是否更新成功
        """

    @abstractmethod
    def get_model_metadata(self) -> MLModelMetadata:
        """
        获取模型元数据

        Returns:
            模型元数据
        """


class IMLModelManager(ABC):
    """
    机器学习模型管理器接口
    """

    @abstractmethod
    def register_algorithm(self, algorithm: IMLAlgorithm) -> bool:
        """
        注册算法

        Args:
            algorithm: 算法实例

        Returns:
            是否注册成功
        """

    @abstractmethod
    def unregister_algorithm(self, algorithm_name: str) -> bool:
        """
        注销算法

        Args:
            algorithm_name: 算法名称

        Returns:
            是否注销成功
        """

    @abstractmethod
    def get_algorithm(self, algorithm_name: str) -> Optional[IMLAlgorithm]:
        """
        获取算法

        Args:
            algorithm_name: 算法名称

        Returns:
            算法实例
        """

    @abstractmethod
    def list_algorithms(self) -> List[str]:
        """
        列出所有注册的算法

        Returns:
            算法名称列表
        """

    @abstractmethod
    def create_model(self, algorithm_name: str, config: MLModelConfig) -> Optional[IMLAlgorithm]:
        """
        创建模型实例

        Args:
            algorithm_name: 算法名称
            config: 模型配置

        Returns:
            模型实例
        """

    @abstractmethod
    def train_model(self, model: IMLAlgorithm,
                    X: Union[np.ndarray, pd.DataFrame],
                    y: Union[np.ndarray, pd.Series],
                    config: MLModelConfig) -> MLTrainingResult:
        """
        训练模型

        Args:
            model: 模型实例
            X: 训练特征数据
            y: 训练目标数据
            config: 训练配置

        Returns:
            训练结果
        """

    @abstractmethod
    def save_model(self, model: IMLAlgorithm, model_id: str, path: str) -> bool:
        """
        保存模型

        Args:
            model: 模型实例
            model_id: 模型ID
            path: 保存路径

        Returns:
            是否保存成功
        """

    @abstractmethod
    def load_model(self, model_id: str, path: str) -> Optional[IMLAlgorithm]:
        """
        加载模型

        Args:
            model_id: 模型ID
            path: 加载路径

        Returns:
            模型实例
        """

    @abstractmethod
    def delete_model(self, model_id: str) -> bool:
        """
        删除模型

        Args:
            model_id: 模型ID

        Returns:
            是否删除成功
        """

    @abstractmethod
    def list_models(self) -> List[MLModelMetadata]:
        """
        列出所有模型

        Returns:
            模型元数据列表
        """

    @abstractmethod
    def get_model_metadata(self, model_id: str) -> Optional[MLModelMetadata]:
        """
        获取模型元数据

        Args:
            model_id: 模型ID

        Returns:
            模型元数据
        """

    @abstractmethod
    def compare_models(self, model_ids: List[str], metrics: List[str]) -> Dict[str, Any]:
        """
        比较模型性能

        Args:
            model_ids: 模型ID列表
            metrics: 比较指标列表

        Returns:
            比较结果字典
        """


class IMLHyperparameterTuner(ABC):
    """
    机器学习超参数调优器接口
    """

    @abstractmethod
    def tune_hyperparameters(self, algorithm: IMLAlgorithm,
                             X: Union[np.ndarray, pd.DataFrame],
                             y: Union[np.ndarray, pd.Series],
                             search_space: Dict[str, Any],
                             optimization_metric: OptimizationMetric,
                             max_evaluations: int = 100) -> Dict[str, Any]:
        """
        调优超参数

        Args:
            algorithm: 算法实例
            X: 训练特征数据
            y: 训练目标数据
            search_space: 搜索空间
            optimization_metric: 优化指标
            max_evaluations: 最大评估次数

        Returns:
            调优结果字典
        """

    @abstractmethod
    def get_supported_optimizers(self) -> List[str]:
        """
        获取支持的优化器

        Returns:
            优化器名称列表
        """

    @abstractmethod
    def validate_search_space(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证搜索空间

        Args:
            search_space: 搜索空间字典

        Returns:
            验证结果
        """


class IMLFeatureEngineer(ABC):
    """
    机器学习特征工程接口
    """

    @abstractmethod
    def analyze_features(self, X: Union[np.ndarray, pd.DataFrame],
                         y: Optional[Union[np.ndarray, pd.Series]] = None) -> Dict[str, Any]:
        """
        分析特征

        Args:
            X: 特征数据
            y: 目标数据（可选）

        Returns:
            特征分析结果
        """

    @abstractmethod
    def select_features(self, X: Union[np.ndarray, pd.DataFrame],
                        y: Union[np.ndarray, pd.Series],
                        method: str = "importance",
                        k: int = 10) -> List[str]:
        """
        特征选择

        Args:
            X: 特征数据
            y: 目标数据
            method: 选择方法
            k: 选择的特征数量

        Returns:
            选择的特征列表
        """

    @abstractmethod
    def engineer_features(self, X: Union[np.ndarray, pd.DataFrame],
                          method: str = "auto") -> Union[np.ndarray, pd.DataFrame]:
        """
        特征工程

        Args:
            X: 原始特征数据
            method: 工程方法

        Returns:
            工程后的特征数据
        """

    @abstractmethod
    def create_feature_pipeline(self, config: Dict[str, Any]) -> Any:
        """
        创建特征处理流水线

        Args:
            config: 流水线配置

        Returns:
            特征处理流水线
        """


class IMLModelEvaluator(ABC):
    """
    机器学习模型评估器接口
    """

    @abstractmethod
    def evaluate_model(self, model: IMLAlgorithm,
                       X_test: Union[np.ndarray, pd.DataFrame],
                       y_test: Union[np.ndarray, pd.Series],
                       metrics: List[OptimizationMetric]) -> Dict[str, Any]:
        """
        评估模型

        Args:
            model: 模型实例
            X_test: 测试特征数据
            y_test: 测试目标数据
            metrics: 评估指标

        Returns:
            评估结果字典
        """

    @abstractmethod
    def cross_validate(self, model: IMLAlgorithm,
                       X: Union[np.ndarray, pd.DataFrame],
                       y: Union[np.ndarray, pd.Series],
                       cv: int = 5,
                       metrics: List[OptimizationMetric] = None) -> Dict[str, Any]:
        """
        交叉验证

        Args:
            model: 模型实例
            X: 特征数据
            y: 目标数据
            cv: 交叉验证折数
            metrics: 评估指标

        Returns:
            交叉验证结果
        """

    @abstractmethod
    def plot_learning_curve(self, model: IMLAlgorithm,
                            X: Union[np.ndarray, pd.DataFrame],
                            y: Union[np.ndarray, pd.Series]) -> Any:
        """
        绘制学习曲线

        Args:
            model: 模型实例
            X: 特征数据
            y: 目标数据

        Returns:
            学习曲线图表
        """

    @abstractmethod
    def plot_confusion_matrix(self, y_true: Union[np.ndarray, pd.Series],
                              y_pred: Union[np.ndarray, pd.Series]) -> Any:
        """
        绘制混淆矩阵

        Args:
            y_true: 真实标签
            y_pred: 预测标签

        Returns:
            混淆矩阵图表
        """

    @abstractmethod
    def calculate_model_complexity(self, model: IMLAlgorithm) -> Dict[str, Any]:
        """
        计算模型复杂度

        Args:
            model: 模型实例

        Returns:
            复杂度指标字典
        """


class IMLModelMonitor(ABC):
    """
    机器学习模型监控器接口
    """

    @abstractmethod
    def monitor_prediction_drift(self, model: IMLAlgorithm,
                                 reference_data: Union[np.ndarray, pd.DataFrame],
                                 current_data: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """
        监控预测漂移

        Args:
            model: 模型实例
            reference_data: 参考数据
            current_data: 当前数据

        Returns:
            漂移检测结果
        """

    @abstractmethod
    def monitor_performance_decay(self, model: IMLAlgorithm,
                                  X_test: Union[np.ndarray, pd.DataFrame],
                                  y_test: Union[np.ndarray, pd.Series],
                                  threshold: float = 0.05) -> Dict[str, Any]:
        """
        监控性能衰减

        Args:
            model: 模型实例
            X_test: 测试特征数据
            y_test: 测试目标数据
            threshold: 衰减阈值

        Returns:
            性能衰减检测结果
        """

    @abstractmethod
    def detect_data_drift(self, reference_data: Union[np.ndarray, pd.DataFrame],
                          current_data: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """
        检测数据漂移

        Args:
            reference_data: 参考数据
            current_data: 当前数据

        Returns:
            数据漂移检测结果
        """

    @abstractmethod
    def get_monitoring_metrics(self, model_id: str) -> Dict[str, Any]:
        """
        获取监控指标

        Args:
            model_id: 模型ID

        Returns:
            监控指标字典
        """

    @abstractmethod
    def alert_on_anomaly(self, model_id: str, anomaly_config: Dict[str, Any]) -> bool:
        """
        异常告警

        Args:
            model_id: 模型ID
            anomaly_config: 异常检测配置

        Returns:
            是否触发告警
        """
