import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
import shap
import lime
import lime.lime_tabular
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import logging
from enum import Enum
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class ExplanationMethod(Enum):
    SHAP = "shap"
    LIME = "lime"
    FEATURE_IMPORTANCE = "feature_importance"

class BaseExplainer(ABC):
    """可解释性分析基类"""

    def __init__(self, model, feature_names: List[str]):
        """
        Args:
            model: 待解释模型
            feature_names: 特征名称列表
        """
        self.model = model
        self.feature_names = feature_names

    @abstractmethod
    def explain(self, X: Union[pd.DataFrame, np.ndarray]) -> Dict:
        """生成解释结果"""
        pass

    @abstractmethod
    def visualize(self, explanation: Dict, sample_idx: int = 0):
        """可视化解释结果"""
        pass

class SHAPExplainer(BaseExplainer):
    """SHAP值解释器"""

    def __init__(self, model, feature_names: List[str], **kwargs):
        super().__init__(model, feature_names)

        # 根据模型类型选择解释器
        if hasattr(model, 'predict_proba'):
            self.explainer = shap.Explainer(model.predict_proba, **kwargs)
        else:
            self.explainer = shap.Explainer(model.predict, **kwargs)

    def explain(self, X: Union[pd.DataFrame, np.ndarray]) -> Dict:
        """计算SHAP值"""
        if isinstance(X, pd.DataFrame):
            X = X.values

        shap_values = self.explainer(X)

        return {
            'values': shap_values.values,
            'base_values': shap_values.base_values,
            'data': shap_values.data,
            'feature_names': self.feature_names
        }

    def visualize(self, explanation: Dict, sample_idx: int = 0):
        """可视化SHAP值"""
        shap_values = shap.Explanation(
            values=explanation['values'][sample_idx],
            base_values=explanation['base_values'][sample_idx],
            data=explanation['data'][sample_idx],
            feature_names=explanation['feature_names']
        )

        plt.figure()
        shap.plots.waterfall(shap_values)
        plt.tight_layout()
        plt.show()

class LIMEExplainer(BaseExplainer):
    """LIME解释器"""

    def __init__(self, model, feature_names: List[str],
                mode: str = 'classification', **kwargs):
        super().__init__(model, feature_names)
        self.mode = mode

        # 初始化LIME解释器
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.zeros((1, len(feature_names))),  # 占位数据
            feature_names=feature_names,
            mode=mode,
            **kwargs
        )

    def explain(self, X: Union[pd.DataFrame, np.ndarray],
               num_features: int = 5) -> Dict:
        """生成LIME解释"""
        if isinstance(X, pd.DataFrame):
            X = X.values

        explanations = []
        for x in X:
            if self.mode == 'classification':
                exp = self.explainer.explain_instance(
                    x,
                    self.model.predict_proba,
                    num_features=num_features
                )
            else:
                exp = self.explainer.explain_instance(
                    x,
                    self.model.predict,
                    num_features=num_features
                )

            explanations.append({
                'features': exp.domain_mapper.feature_names,
                'values': exp.local_exp[1] if self.mode == 'classification' else exp.local_exp,
                'score': exp.score,
                'prediction': exp.predicted_value
            })

        return {
            'explanations': explanations,
            'feature_names': self.feature_names
        }

    def visualize(self, explanation: Dict, sample_idx: int = 0):
        """可视化LIME解释"""
        exp = explanation['explanations'][sample_idx]

        plt.figure(figsize=(10, 6))
        plt.barh(
            [f[0] for f in exp['values']],
            [f[1] for f in exp['values']],
            color='#1f77b4'
        )
        plt.title(f"LIME Explanation (Score: {exp['score']:.2f})")
        plt.xlabel("Feature Importance")
        plt.tight_layout()
        plt.show()

class FeatureImportanceExplainer(BaseExplainer):
    """特征重要性解释器"""

    def __init__(self, model, feature_names: List[str]):
        super().__init__(model, feature_names)

    def explain(self, X: Union[pd.DataFrame, np.ndarray] = None) -> Dict:
        """获取特征重要性"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            raise ValueError("Model does not support feature importance")

        return {
            'importances': importances,
            'feature_names': self.feature_names
        }

    def visualize(self, explanation: Dict, sample_idx: int = 0):
        """可视化特征重要性"""
        importances = explanation['importances']
        features = explanation['feature_names']

        sorted_idx = np.argsort(importances)
        plt.figure(figsize=(10, 6))
        plt.barh(
            np.array(features)[sorted_idx],
            importances[sorted_idx],
            color='#1f77b4'
        )
        plt.title("Feature Importance")
        plt.xlabel("Importance Score")
        plt.tight_layout()
        plt.show()

class ModelExplainer:
    """模型解释统一接口"""

    def __init__(self, model, feature_names: List[str]):
        """
        Args:
            model: 待解释模型
            feature_names: 特征名称列表
        """
        self.model = model
        self.feature_names = feature_names
        self.explainers = {
            ExplanationMethod.SHAP: SHAPExplainer(model, feature_names),
            ExplanationMethod.LIME: LIMEExplainer(model, feature_names),
            ExplanationMethod.FEATURE_IMPORTANCE: FeatureImportanceExplainer(model, feature_names)
        }

    def explain(self,
               X: Union[pd.DataFrame, np.ndarray],
               method: ExplanationMethod = ExplanationMethod.SHAP,
               **kwargs) -> Dict:
        """
        生成模型解释

        Args:
            X: 输入数据
            method: 解释方法
            **kwargs: 解释器参数

        Returns:
            解释结果字典
        """
        if method not in self.explainers:
            raise ValueError(f"Unsupported explanation method: {method}")

        return self.explainers[method].explain(X, **kwargs)

    def visualize(self,
                explanation: Dict,
                method: ExplanationMethod = ExplanationMethod.SHAP,
                sample_idx: int = 0):
        """
        可视化解释结果

        Args:
            explanation: 解释结果字典
            method: 解释方法
            sample_idx: 样本索引
        """
        if method not in self.explainers:
            raise ValueError(f"Unsupported explanation method: {method}")

        self.explainers[method].visualize(explanation, sample_idx)

    def get_feature_contributions(self,
                                X: Union[pd.DataFrame, np.ndarray],
                                method: ExplanationMethod = ExplanationMethod.SHAP) -> pd.DataFrame:
        """
        获取特征贡献度

        Args:
            X: 输入数据
            method: 解释方法

        Returns:
            特征贡献度DataFrame
        """
        explanation = self.explain(X, method)

        if method == ExplanationMethod.SHAP:
            contribs = np.abs(explanation['values']).mean(axis=0)
        elif method == ExplanationMethod.LIME:
            contribs = np.zeros(len(self.feature_names))
            for exp in explanation['explanations']:
                for f, v in exp['values']:
                    contribs[f] += abs(v)
            contribs /= len(explanation['explanations'])
        elif method == ExplanationMethod.FEATURE_IMPORTANCE:
            contribs = explanation['importances']
        else:
            raise ValueError(f"Unsupported method: {method}")

        return pd.DataFrame({
            'feature': self.feature_names,
            'contribution': contribs
        }).sort_values('contribution', ascending=False)

class ModelExplanationSystem:
    """模型解释系统"""

    def __init__(self, config: Dict[str, Dict]):
        """
        Args:
            config: 模型解释配置 {模型名: {feature_names: [...]}}
        """
        self.explainers = {}
        for model_name, params in config.items():
            if 'model' not in params:
                raise ValueError(f"Missing model in config for {model_name}")
            if 'feature_names' not in params:
                raise ValueError(f"Missing feature_names in config for {model_name}")

            self.explainers[model_name] = ModelExplainer(
                params['model'],
                params['feature_names']
            )

    def add_model(self, model_name: str, model: object, feature_names: List[str]):
        """添加模型解释器"""
        if model_name in self.explainers:
            logger.warning(f"Model {model_name} already has explainer")
            return

        self.explainers[model_name] = ModelExplainer(model, feature_names)

    def explain_model(self,
                    model_name: str,
                    X: Union[pd.DataFrame, np.ndarray],
                    method: Union[str, ExplanationMethod] = 'shap',
                    **kwargs) -> Dict:
        """
        解释模型预测

        Args:
            model_name: 模型名称
            X: 输入数据
            method: 解释方法(shap/lime/feature_importance)
            **kwargs: 解释器参数

        Returns:
            解释结果字典
        """
        if model_name not in self.explainers:
            raise ValueError(f"Model {model_name} not found")

        if isinstance(method, str):
            method = ExplanationMethod(method.lower())

        return self.explainers[model_name].explain(X, method, **kwargs)

    def visualize_explanation(self,
                            model_name: str,
                            explanation: Dict,
                            method: Union[str, ExplanationMethod] = 'shap',
                            sample_idx: int = 0):
        """
        可视化模型解释

        Args:
            model_name: 模型名称
            explanation: 解释结果
            method: 解释方法
            sample_idx: 样本索引
        """
        if model_name not in self.explainers:
            raise ValueError(f"Model {model_name} not found")

        if isinstance(method, str):
            method = ExplanationMethod(method.lower())

        self.explainers[model_name].visualize(explanation, method, sample_idx)

    def get_model_contributions(self,
                              model_name: str,
                              X: Union[pd.DataFrame, np.ndarray],
                              method: Union[str, ExplanationMethod] = 'shap') -> pd.DataFrame:
        """
        获取模型特征贡献度

        Args:
            model_name: 模型名称
            X: 输入数据
            method: 解释方法

        Returns:
            特征贡献度DataFrame
        """
        if model_name not in self.explainers:
            raise ValueError(f"Model {model_name} not found")

        if isinstance(method, str):
            method = ExplanationMethod(method.lower())

        return self.explainers[model_name].get_feature_contributions(X, method)
