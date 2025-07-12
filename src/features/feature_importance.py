import numpy as np
from typing import Dict, List, Optional
import logging
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class FeatureImportanceAnalyzer:
    """特征重要性分析器"""

    def __init__(self, model: BaseEstimator):
        """
        初始化分析器

        Args:
            model: 已训练的模型实例
        """
        self.model = model
        self.importance_scores: Optional[Dict[str, float]] = None
        self.importance_std: Optional[Dict[str, float]] = None

    def calculate_permutation_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        n_repeats: int = 10,
        scoring: str = "accuracy",
        random_state: Optional[int] = None
    ) -> Dict[str, float]:
        """
        计算排列重要性

        Args:
            X: 特征矩阵
            y: 目标变量
            feature_names: 特征名称列表
            n_repeats: 排列重复次数
            scoring: 评估指标
            random_state: 随机种子

        Returns:
            特征重要性字典 {特征名: 重要性得分}
        """
        if X.shape[1] != len(feature_names):
            raise ValueError("特征数量与名称数量不匹配")

        result = permutation_importance(
            self.model, X, y,
            n_repeats=n_repeats,
            scoring=scoring,
            random_state=random_state
        )

        self.importance_scores = {
            name: score for name, score in zip(
                feature_names, result.importances_mean
            )
        }
        self.importance_std = {
            name: std for name, std in zip(
                feature_names, result.importances_std
            )
        }

        return self.importance_scores

    def get_top_features(self, top_n: int = 10) -> List[str]:
        """
        获取最重要的top_n个特征

        Args:
            top_n: 要返回的特征数量

        Returns:
            重要性最高的top_n个特征名
        """
        if self.importance_scores is None:
            raise RuntimeError("请先计算特征重要性")

        sorted_features = sorted(
            self.importance_scores.items(),
            key=lambda x: -x[1]
        )
        return [f[0] for f in sorted_features[:top_n]]

    def plot_importance(
        self,
        title: str = "Feature Importance",
        figsize: tuple = (10, 6),
        top_n: Optional[int] = None
    ) -> plt.Figure:
        """
        绘制特征重要性图

        Args:
            title: 图表标题
            figsize: 图表大小
            top_n: 只显示前top_n个特征

        Returns:
            matplotlib Figure对象
        """
        if self.importance_scores is None:
            raise RuntimeError("请先计算特征重要性")

        # 准备数据
        features = list(self.importance_scores.keys())
        scores = list(self.importance_scores.values())
        stds = list(self.importance_std.values())

        # 筛选top_n特征
        if top_n is not None:
            indices = np.argsort(scores)[-top_n:]
            features = [features[i] for i in indices]
            scores = [scores[i] for i in indices]
            stds = [stds[i] for i in indices]

        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)
        y_pos = np.arange(len(features))

        ax.barh(
            y_pos, scores, xerr=stds,
            align='center', color='skyblue'
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score')
        ax.set_title(title)

        plt.tight_layout()
        return fig

    def calculate_shap_values(
        self,
        X: np.ndarray,
        feature_names: List[str],
        explainer_type: str = "tree"
    ) -> Dict[str, np.ndarray]:
        """
        计算SHAP值 (需要安装shap包)

        Args:
            X: 特征矩阵
            feature_names: 特征名称列表
            explainer_type: 解释器类型 ("tree"或"kernel")

        Returns:
            各样本的SHAP值字典 {特征名: SHAP值数组}
        """
        try:
            import shap
        except ImportError:
            raise ImportError("请先安装shap包: pip install shap")

        if X.shape[1] != len(feature_names):
            raise ValueError("特征数量与名称数量不匹配")

        # 根据模型类型选择解释器
        if explainer_type == "tree":
            explainer = shap.TreeExplainer(self.model)
        else:
            explainer = shap.KernelExplainer(
                self.model.predict, X[:100]
            )

        shap_values = explainer.shap_values(X)

        # 对于多分类问题，取平均SHAP值
        if isinstance(shap_values, list):
            shap_values = np.mean(np.abs(shap_values), axis=0)

        return {
            name: values for name, values in zip(
                feature_names, shap_values.T
            )
        }

    def plot_shap_summary(
        self,
        X: np.ndarray,
        feature_names: List[str],
        plot_type: str = "dot",
        **kwargs
    ) -> plt.Figure:
        """
        绘制SHAP摘要图 (需要安装shap包)

        Args:
            X: 特征矩阵
            feature_names: 特征名称列表
            plot_type: 绘图类型 ("dot"/"bar"/"violin")
            **kwargs: 传递给shap.summary_plot的参数

        Returns:
            matplotlib Figure对象
        """
        try:
            import shap
        except ImportError:
            raise ImportError("请先安装shap包: pip install shap")

        shap_values = self.calculate_shap_values(X, feature_names)

        # 转换为shap需要的格式
        shap_array = np.array(list(shap_values.values())).T

        fig = plt.figure(**kwargs.get("fig_args", {}))
        shap.summary_plot(
            shap_array, X,
            feature_names=feature_names,
            plot_type=plot_type,
            show=False,
            **kwargs
        )

        plt.tight_layout()
        return fig
