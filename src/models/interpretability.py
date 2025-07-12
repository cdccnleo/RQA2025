import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import shap
import lime
import lime.lime_tabular

logger = logging.getLogger(__name__)

@dataclass
class GlobalInterpretation:
    """全局解释结果"""
    feature_importance: Dict[str, float]
    partial_dependence: Dict[str, np.ndarray]
    interaction_strength: Dict[Tuple[str, str], float]

@dataclass
class LocalInterpretation:
    """局部解释结果"""
    prediction: float
    feature_contributions: Dict[str, float]
    decision_path: List[Tuple[str, float]]

class ModelInterpreter:
    """模型解释器"""

    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.Explainer(model)
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.zeros((1, len(feature_names))),
            feature_names=feature_names,
            mode="regression"
        )

    def global_interpret(self, X: pd.DataFrame) -> GlobalInterpretation:
        """全局解释"""
        # SHAP全局重要性
        shap_values = self.explainer(X).values
        if len(shap_values.shape) == 3:  # 多分类情况
            shap_values = np.abs(shap_values).mean(0)
        importance = dict(zip(self.feature_names, np.abs(shap_values).mean(0)))

        # 排列重要性
        perm_importance = permutation_importance(
            self.model, X, X.iloc[0:1], n_repeats=5, random_state=42
        )
        perm_importance = dict(zip(self.feature_names, perm_importance.importances_mean))

        # 合并重要性
        combined_importance = {
            feat: (importance.get(feat, 0) + perm_importance.get(feat, 0)) / 2
            for feat in self.feature_names
        }

        return GlobalInterpretation(
            feature_importance=combined_importance,
            partial_dependence=self._calculate_pd(X),
            interaction_strength=self._calculate_interactions(X)
        )

    def local_interpret(self, instance: pd.DataFrame) -> LocalInterpretation:
        """局部解释"""
        # SHAP解释
        shap_values = self.explainer(instance).values[0]
        if len(shap_values.shape) == 2:  # 多分类情况
            shap_values = shap_values[0]
        contributions = dict(zip(self.feature_names, shap_values))

        # LIME解释
        lime_exp = self.lime_explainer.explain_instance(
            instance.values[0],
            self.model.predict,
            num_features=len(self.feature_names)
        )
        lime_contrib = dict(lime_exp.as_map()[0])

        # 合并解释
        combined_contrib = {
            feat: (contributions.get(feat, 0) + lime_contrib.get(feat, 0)) / 2
            for feat in self.feature_names
        }

        return LocalInterpretation(
            prediction=float(self.model.predict(instance)[0]),
            feature_contributions=combined_contrib,
            decision_path=self._extract_decision_path(instance)
        )

    def plot_feature_importance(self, importance: Dict[str, float]) -> plt.Figure:
        """绘制特征重要性图"""
        fig, ax = plt.subplots(figsize=(10, 6))
        features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        features = features[:20]  # 只显示前20个重要特征
        sns.barplot(
            x=[v for _, v in features],
            y=[k for k, _ in features],
            ax=ax
        )
        ax.set_title("Feature Importance")
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Features")
        return fig

    def plot_contributions(self, contributions: Dict[str, float]) -> plt.Figure:
        """绘制特征贡献图"""
        fig, ax = plt.subplots(figsize=(10, 6))
        contribs = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        contribs = contribs[:15]  # 只显示前15个重要贡献
        sns.barplot(
            x=[v for _, v in contribs],
            y=[k for k, _ in contribs],
            ax=ax
        )
        ax.set_title("Feature Contributions")
        ax.set_xlabel("Contribution Value")
        ax.set_ylabel("Features")
        return fig

    def _calculate_pd(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """计算部分依赖"""
        # 简化实现 - 实际应用中需要更复杂的计算
        return {
            feat: np.linspace(X[feat].min(), X[feat].max(), 10)
            for feat in self.feature_names[:5]  # 只计算前5个特征
        }

    def _calculate_interactions(self, X: pd.DataFrame) -> Dict[Tuple[str, str], float]:
        """计算特征交互强度"""
        # 简化实现 - 实际应用中需要SHAP交互值
        return {
            (f1, f2): np.random.uniform(0, 0.5)
            for i, f1 in enumerate(self.feature_names[:5])
            for f2 in self.feature_names[i+1:5]
        }

    def _extract_decision_path(self, instance: pd.DataFrame) -> List[Tuple[str, float]]:
        """提取决策路径"""
        # 简化实现 - 实际应用中需要根据模型类型实现
        return [
            (feat, float(instance[feat].iloc[0]))
            for feat in self.feature_names[:5]  # 只显示前5个特征
        ]

class ExplanationReport:
    """可解释性报告生成器"""

    def __init__(self, interpreter: ModelInterpreter):
        self.interpreter = interpreter

    def generate_report(self, X: pd.DataFrame,
                       sample_instances: Optional[pd.DataFrame] = None) -> Dict:
        """生成解释报告"""
        if sample_instances is None:
            sample_instances = X.sample(3)

        global_exp = self.interpreter.global_interpret(X)
        local_exps = [
            self.interpreter.local_interpret(pd.DataFrame([row]))
            for _, row in sample_instances.iterrows()
        ]

        return {
            "global_importance": global_exp.feature_importance,
            "sample_explanations": [
                {
                    "prediction": exp.prediction,
                    "top_features": dict(sorted(
                        exp.feature_contributions.items(),
                        key=lambda x: abs(x[1]),
                        reverse=True
                    )[:5])
                }
                for exp in local_exps
            ],
            "partial_dependence": {
                feat: values.tolist()
                for feat, values in global_exp.partial_dependence.items()
            }
        }
