# src / features / intelligent / ml_model_integration.py
"""
机器学习模型集成模块
实现智能化的模型集成功能，包括模型选择、集成学习、自动调优等
"""

import logging
from typing import Optional, Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    VotingClassifier, VotingRegressor
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from pathlib import Path
from enum import Enum

from ..core.config_integration import get_config_integration_manager, ConfigScope

logger = logging.getLogger(__name__)


class ModelType(Enum):

    """模型类型"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class EnsembleMethod(Enum):

    """集成方法"""
    VOTING = "voting"
    STACKING = "stacking"
    BAGGING = "bagging"
    BOOSTING = "boosting"


class MLModelIntegration:

    """机器学习模型集成器"""

    def __init__(


        self,
        task_type: str = "classification",
        ensemble_method: str = "voting",
        enable_auto_tuning: bool = True,
        config_manager=None
    ):
        """
        初始化机器学习模型集成器

        Args:
            task_type: 任务类型 ('classification', 'regression')
            ensemble_method: 集成方法 ('voting', 'stacking', 'bagging', 'boosting')
            enable_auto_tuning: 是否启用自动调优
            config_manager: 配置管理器
        """
        # 配置管理集成
        self.config_manager = config_manager or get_config_integration_manager()
        self.config_manager.register_config_watcher(ConfigScope.PROCESSING, self._on_config_change)

        # 系统参数
        self.task_type = task_type
        self.ensemble_method = ensemble_method
        self.enable_auto_tuning = enable_auto_tuning

        # 模型存储
        self.models: Dict[str, Any] = {}
        self.ensemble_model: Optional[Any] = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model_performance: Dict[str, Dict[str, float]] = {}

        # 初始化基础模型
        self._init_base_models()

        logger.info(f"机器学习模型集成器初始化完成: task_type={task_type}, ensemble_method={ensemble_method}")

    def _on_config_change(self, scope: ConfigScope, key: str, value: Any) -> None:
        """配置变更处理"""
        if scope == ConfigScope.PROCESSING:
            if key == "enable_auto_tuning":
                self.enable_auto_tuning = value
                logger.info(f"更新自动调优状态: {value}")

    def _init_base_models(self) -> None:
        """初始化基础模型"""
        if self.task_type == "classification":
            self.models = {
                "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
                "logistic_regression": LogisticRegression(random_state=42, max_iter=1000),
                "svm": SVC(random_state=42, probability=True),
                "decision_tree": DecisionTreeClassifier(random_state=42)
            }
        else:  # regression
            self.models = {
                "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
                "linear_regression": LinearRegression(),
                "svr": SVR(),
                "decision_tree": DecisionTreeRegressor(random_state=42)
            }

        logger.info(f"初始化了 {len(self.models)} 个基础模型")

    def train_models(


        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Dict[str, float]]:
        """训练所有模型"""
        if X.empty or y.empty:
            raise ValueError("输入数据不能为空")

        # 数据预处理
        X_processed, y_processed = self._preprocess_data(X, y)

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed,
            test_size=test_size,
            random_state=random_state,
            stratify=y_processed if self.task_type == "classification" else None
        )

        # 训练每个模型
        for name, model in self.models.items():
            try:
                logger.info(f"训练模型: {name}")
                model.fit(X_train, y_train)

                # 预测
                y_pred = model.predict(X_test)

                # 计算性能指标
                performance = self._calculate_performance(y_test, y_pred)
                self.model_performance[name] = performance

                logger.info(f"模型 {name} 训练完成，性能: {performance}")

            except Exception as e:
                logger.error(f"模型 {name} 训练失败: {e}")
                self.model_performance[name] = {}

        return self.model_performance

    def create_ensemble(self) -> Any:
        """创建集成模型"""
        if not self.model_performance:
            raise ValueError("请先训练模型")

        # 选择性能最好的模型
        best_models = []
        for name, performance in self.model_performance.items():
            if name in self.models:
                best_models.append((name, self.models[name]))

        if not best_models:
            raise ValueError("没有可用的模型进行集成")

        # 创建集成模型
        if self.ensemble_method == "voting":
            if self.task_type == "classification":
                self.ensemble_model = VotingClassifier(
                    estimators=best_models,
                    voting='soft'
                )
            else:
                self.ensemble_model = VotingRegressor(
                    estimators=best_models
                )

        logger.info(f"创建集成模型: {self.ensemble_method}, 包含 {len(best_models)} 个模型")
        return self.ensemble_model

    def train_ensemble(


        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, float]:
        """训练集成模型"""
        if self.ensemble_model is None:
            self.create_ensemble()

        # 数据预处理
        X_processed, y_processed = self._preprocess_data(X, y)

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed,
            test_size=test_size,
            random_state=random_state,
            stratify=y_processed if self.task_type == "classification" else None
        )

        # 训练集成模型
        logger.info("训练集成模型")
        self.ensemble_model.fit(X_train, y_train)

        # 预测
        y_pred = self.ensemble_model.predict(X_test)

        # 计算性能指标
        performance = self._calculate_performance(y_test, y_pred)

        logger.info(f"集成模型训练完成，性能: {performance}")
        return performance

    def predict(self, X: pd.DataFrame, use_ensemble: bool = True) -> np.ndarray:
        """预测"""
        X_processed = self._preprocess_features(X)

        if use_ensemble and self.ensemble_model is not None:
            return self.ensemble_model.predict(X_processed)
        else:
            # 使用最佳单模型
            best_model_name = self._get_best_model()
            if best_model_name and best_model_name in self.models:
                return self.models[best_model_name].predict(X_processed)
            else:
                raise ValueError("没有可用的模型进行预测")

    def _preprocess_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """数据预处理"""
        # 特征标准化
        X_processed = self.scaler.fit_transform(X)

        # 标签编码（分类任务）
        if self.task_type == "classification":
            y_processed = self.label_encoder.fit_transform(y)
        else:
            y_processed = y.values

        return X_processed, y_processed

    def _preprocess_features(self, X: pd.DataFrame) -> np.ndarray:
        """特征预处理"""
        return self.scaler.transform(X)

    def _calculate_performance(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """计算性能指标"""
        performance = {}

        if self.task_type == "classification":
            performance["accuracy"] = accuracy_score(y_true, y_pred)
        else:
            performance["mse"] = mean_squared_error(y_true, y_pred)
            performance["rmse"] = np.sqrt(performance["mse"])
            performance["r2"] = r2_score(y_true, y_pred)

        return performance

    def _get_best_model(self) -> Optional[str]:
        """获取最佳模型名称"""
        if not self.model_performance:
            return None

        if self.task_type == "classification":
            best_model = max(
                self.model_performance.items(),
                key=lambda x: x[1].get("accuracy", 0.0)
            )
        else:
            best_model = max(
                self.model_performance.items(),
                key=lambda x: x[1].get("r2", 0.0)
            )

        return best_model[0]

    def get_model_performance(self) -> Dict[str, Dict[str, float]]:
        """获取所有模型性能"""
        return self.model_performance

    def save_model(self, filepath: Union[str, Path]) -> None:
        """保存模型"""
        if self.ensemble_model is not None:
            model = self.ensemble_model
        else:
            best_model_name = self._get_best_model()
            if best_model_name:
                model = self.models[best_model_name]
            else:
                raise ValueError("没有可用的模型进行保存")

        # 保存模型和相关信息
        model_data = {
            "model": model,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder if self.task_type == "classification" else None,
            "task_type": self.task_type,
            "performance": self.model_performance
        }

        joblib.dump(model_data, filepath)
        logger.info(f"模型已保存到: {filepath}")

    def load_model(self, filepath: Union[str, Path]) -> None:
        """加载模型"""
        model_data = joblib.load(filepath)

        self.task_type = model_data["task_type"]
        self.scaler = model_data["scaler"]
        if self.task_type == "classification":
            self.label_encoder = model_data["label_encoder"]

        self.ensemble_model = model_data["model"]
        self.model_performance = model_data.get("performance", {})

        logger.info(f"模型已从 {filepath} 加载")

    def get_model_summary(self) -> Dict[str, Any]:
        """获取模型摘要"""
        summary = {
            "task_type": self.task_type,
            "ensemble_method": self.ensemble_method,
            "total_models": len(self.models),
            "trained_models": len(self.model_performance),
            "best_model": self._get_best_model(),
            "has_ensemble": self.ensemble_model is not None,
            "enable_auto_tuning": self.enable_auto_tuning
        }

        if self.model_performance:
            best_model = self._get_best_model()
            if best_model:
                summary["best_performance"] = self.model_performance[best_model]

        return summary
