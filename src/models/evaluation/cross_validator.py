from typing import Dict, List, Union, Optional
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from .model_evaluator import ModelEvaluator

class CrossValidator:
    """交叉验证器，支持k折和分层交叉验证"""

    def __init__(
        self,
        model: object,
        n_splits: int = 5,
        random_state: Optional[int] = None
    ):
        """
        初始化交叉验证器

        Args:
            model: 待验证的模型对象
            n_splits: 交叉验证折数 (默认5)
            random_state: 随机种子 (可选)
        """
        self.model = model
        self.n_splits = n_splits
        self.random_state = random_state
        self.validation_results: List[Dict[str, float]] = []

    def k_fold_validate(
        self,
        X: Union[np.ndarray, list],
        y: Union[np.ndarray, list],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        执行k折交叉验证

        Args:
            X: 特征数据
            y: 目标值
            metrics: 自定义评估指标列表 (可选)

        Returns:
            各折验证的平均分数
        """
        kf = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )
        return self._run_validation(X, y, kf.split(X), metrics)

    def stratified_validate(
        self,
        X: Union[np.ndarray, list],
        y: Union[np.ndarray, list],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        执行分层k折交叉验证 (适用于分类任务)

        Args:
            X: 特征数据
            y: 目标值
            metrics: 自定义评估指标列表 (可选)

        Returns:
            各折验证的平均分数
        """
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )
        return self._run_validation(X, y, skf.split(X, y), metrics)

    def _run_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        indices: iter,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        内部验证逻辑

        Args:
            X: 特征数据
            y: 目标值
            indices: 交叉验证索引生成器
            metrics: 自定义评估指标列表 (可选)

        Returns:
            各折验证的平均分数
        """
        self.validation_results = []
        X, y = np.array(X), np.array(y)

        for train_idx, val_idx in indices:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # 克隆模型以避免污染
            model_clone = self._clone_model()
            model_clone.fit(X_train, y_train)

            # 使用ModelEvaluator进行评估
            evaluator = ModelEvaluator(model_clone)
            evaluator.evaluate(X_val, y_val)

            if metrics:
                scores = {m: evaluator.get_metrics().get(m) for m in metrics}
            else:
                scores = evaluator.get_metrics()

            self.validation_results.append(scores)

        return self.get_mean_scores()

    def _clone_model(self):
        """克隆模型对象 (简化实现)"""
        # 实际项目中应根据模型类型实现深度克隆
        return self.model.__class__()

    def get_mean_scores(self) -> Dict[str, float]:
        """
        计算各折验证的平均分数

        Returns:
            包含各指标平均值的字典
        """
        if not self.validation_results:
            raise RuntimeError("No validation has been performed yet")

        mean_scores = {
            metric: np.mean([res[metric] for res in self.validation_results])
            for metric in self.validation_results[0].keys()
        }
        return mean_scores

    def get_std_scores(self) -> Dict[str, float]:
        """
        计算各折验证的标准差

        Returns:
            包含各指标标准差的字典
        """
        if not self.validation_results:
            raise RuntimeError("No validation has been performed yet")

        std_scores = {
            metric: np.std([res[metric] for res in self.validation_results])
            for metric in self.validation_results[0].keys()
        }
        return std_scores

    def get_full_results(self) -> List[Dict[str, float]]:
        """获取完整的交叉验证结果"""
        return self.validation_results
