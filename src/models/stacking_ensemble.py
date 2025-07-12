import numpy as np
from typing import List, Dict, Optional
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import KFold
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class StackingEnsemble(BaseEstimator):
    """堆叠(Stacking)集成模型"""

    def __init__(
        self,
        base_models: List[BaseEstimator],
        meta_model: BaseEstimator,
        n_folds: int = 5,
        use_probas: bool = False,
        verbose: bool = False
    ):
        """
        初始化堆叠集成模型

        Args:
            base_models: 基模型列表
            meta_model: 元模型
            n_folds: 交叉验证折数
            use_probas: 是否使用概率预测
            verbose: 是否打印详细日志
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.use_probas = use_probas
        self.verbose = verbose
        self.base_models_train_ = None
        self.meta_features_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """训练堆叠模型"""
        # 存储训练好的基模型
        self.base_models_train_ = [list() for _ in self.base_models]

        # 生成元特征
        self.meta_features_ = np.zeros((X.shape[0], len(self.base_models)))
        kf = KFold(n_splits=self.n_folds, shuffle=True)

        for i, model in enumerate(self.base_models):
            if self.verbose:
                logger.info(f"Training base model {i+1}/{len(self.base_models)}")

            for train_idx, val_idx in kf.split(X, y):
                # 训练基模型
                instance = model.__class__()
                instance.fit(X[train_idx], y[train_idx])
                self.base_models_train_[i].append(instance)

                # 生成验证集预测
                if self.use_probas and hasattr(instance, "predict_proba"):
                    preds = instance.predict_proba(X[val_idx])[:, 1]
                else:
                    preds = instance.predict(X[val_idx])

                self.meta_features_[val_idx, i] = preds

        # 训练元模型
        if self.verbose:
            logger.info("Training meta model")
        self.meta_model.fit(self.meta_features_, y)

        # 在完整数据集上重新训练基模型
        for i, model in enumerate(self.base_models):
            if self.verbose:
                logger.info(f"Retraining base model {i+1} on full data")
            instance = model.__class__()
            instance.fit(X, y)
            self.base_models_train_[i].append(instance)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """生成预测"""
        meta_features = self._get_meta_features(X)
        return self.meta_model.predict(meta_features)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """生成概率预测"""
        if not hasattr(self.meta_model, "predict_proba"):
            raise NotImplementedError("Meta model does not support predict_proba")

        meta_features = self._get_meta_features(X)
        return self.meta_model.predict_proba(meta_features)

    def _get_meta_features(self, X: np.ndarray) -> np.ndarray:
        """获取元特征"""
        if self.base_models_train_ is None:
            raise RuntimeError("Please fit the model first")

        meta_features = np.zeros((X.shape[0], len(self.base_models_train_)))

        for i, models in enumerate(self.base_models_train_):
            # 使用所有基模型实例的平均预测
            preds = np.zeros(X.shape[0])
            for model in models:
                if self.use_probas and hasattr(model, "predict_proba"):
                    preds += model.predict_proba(X)[:, 1]
                else:
                    preds += model.predict(X)

            meta_features[:, i] = preds / len(models)

        return meta_features

    def get_model_weights(self) -> Dict[str, float]:
        """获取模型权重(仅当元模型为线性模型时有效)"""
        if not hasattr(self.meta_model, "coef_"):
            raise AttributeError("Meta model does not have coef_ attribute")

        return {
            f"model_{i}": weight
            for i, weight in enumerate(self.meta_model.coef_.flatten())
        }

class WeightedAverageEnsemble(BaseEstimator):
    """加权平均集成模型"""

    def __init__(
        self,
        models: List[BaseEstimator],
        weights: Optional[List[float]] = None,
        use_probas: bool = False
    ):
        """
        初始化加权平均集成

        Args:
            models: 模型列表
            weights: 模型权重(None表示等权重)
            use_probas: 是否使用概率预测
        """
        self.models = models
        self.weights = weights if weights else [1.0/len(models)] * len(models)
        self.use_probas = use_probas
        self.trained_models_ = None

        if len(self.weights) != len(models):
            raise ValueError("Number of weights must match number of models")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """训练集成模型"""
        self.trained_models_ = []

        for model in self.models:
            instance = model.__class__()
            instance.fit(X, y)
            self.trained_models_.append(instance)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """生成预测"""
        if self.trained_models_ is None:
            raise RuntimeError("Please fit the model first")

        preds = np.zeros(X.shape[0])
        total_weight = sum(self.weights)

        for model, weight in zip(self.trained_models_, self.weights):
            if self.use_probas and hasattr(model, "predict_proba"):
                preds += model.predict_proba(X)[:, 1] * weight
            else:
                preds += model.predict(X) * weight

        return preds / total_weight

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """生成概率预测"""
        if not self.use_probas:
            raise NotImplementedError("Model not configured to use probabilities")

        if self.trained_models_ is None:
            raise RuntimeError("Please fit the model first")

        probas = np.zeros((X.shape[0], 2))
        total_weight = sum(self.weights)

        for model, weight in zip(self.trained_models_, self.weights):
            probas += model.predict_proba(X) * weight

        return probas / total_weight

    def update_weights(self, weights: List[float]):
        """更新模型权重"""
        if len(weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
        self.weights = weights
