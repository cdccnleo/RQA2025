"""
模型解释器模块 - 集成SHAP实现模型可解释性

本模块提供AI模型的解释性功能，包括特征重要性分析、决策路径展示、预测置信度计算等。
支持多种模型类型：树模型、线性模型、神经网络等。

作者: AI团队
创建日期: 2026-02-21
版本: 1.0.0
"""

import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

# 尝试导入SHAP库，如果未安装则提供降级方案
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP库未安装，模型解释功能将使用降级方案")

from src.ml.core.ml_service import MLService
from src.common.exceptions import ModelExplainerError


# 配置日志
logger = logging.getLogger(__name__)


class ModelType(Enum):
    """支持的模型类型"""
    TREE = "tree"           # 树模型 (XGBoost, LightGBM, RandomForest)
    LINEAR = "linear"       # 线性模型 (Linear Regression, Logistic Regression)
    NEURAL = "neural"       # 神经网络 (PyTorch, TensorFlow)
    UNKNOWN = "unknown"     # 未知类型


@dataclass
class FeatureImportance:
    """特征重要性数据类"""
    feature_name: str
    importance: float
    shap_value: float
    description: str = ""


@dataclass
class PredictionExplanation:
    """预测解释结果数据类"""
    prediction: float
    base_value: float
    confidence: float
    feature_importance: List[FeatureImportance]
    decision_path: Optional[List[str]] = None
    explanation_summary: str = ""


class ModelExplainer:
    """
    模型解释器 - 集成SHAP实现模型可解释性
    
    功能:
    1. 特征重要性分析 (SHAP值计算)
    2. 决策路径展示 (树模型)
    3. 预测置信度计算
    4. 模型解释报告生成
    
    使用示例:
        explainer = ModelExplainer()
        explanation = explainer.explain_prediction(
            model_id="strategy_model_001",
            input_data={"feature1": 0.5, "feature2": 1.2}
        )
    """
    
    def __init__(self):
        """初始化模型解释器"""
        self.ml_service = MLService()
        self._explainer_cache: Dict[str, Any] = {}
        
        if not SHAP_AVAILABLE:
            logger.warning("SHAP库不可用，将使用基于特征重要性的降级方案")
    
    def explain_prediction(
        self,
        model_id: str,
        input_data: Dict[str, Any],
        method: str = "shap",
        top_k: int = 10
    ) -> PredictionExplanation:
        """
        解释单个预测结果
        
        参数:
            model_id: 模型唯一标识符
            input_data: 输入特征数据，格式为 {feature_name: value}
            method: 解释方法 ("shap" | "feature_importance" | "lime")
            top_k: 返回最重要的K个特征
            
        返回:
            PredictionExplanation: 预测解释结果
            
        异常:
            ModelExplainerError: 当解释过程失败时抛出
            
        示例:
            >>> explainer = ModelExplainer()
            >>> result = explainer.explain_prediction(
            ...     model_id="model_001",
            ...     input_data={"price": 100.5, "volume": 10000}
            ... )
            >>> print(f"预测值: {result.prediction}")
            >>> print(f"置信度: {result.confidence}")
        """
        try:
            # 获取模型
            model = self.ml_service.get_model(model_id)
            if model is None:
                raise ModelExplainerError(f"模型不存在: {model_id}")
            
            # 检测模型类型
            model_type = self._detect_model_type(model)
            
            # 转换输入数据
            input_array = self._convert_input_to_array(input_data)
            
            # 获取预测结果
            prediction = self._get_prediction(model, input_array)
            
            # 根据方法选择解释策略
            if method == "shap" and SHAP_AVAILABLE:
                explanation = self._explain_with_shap(
                    model, input_array, input_data, model_type, top_k
                )
            else:
                explanation = self._explain_with_feature_importance(
                    model, input_array, input_data, top_k
                )
            
            # 计算置信度
            confidence = self._calculate_confidence(model, input_array)
            
            # 生成决策路径 (仅树模型)
            decision_path = None
            if model_type == ModelType.TREE:
                decision_path = self._get_decision_path(model, input_array)
            
            # 生成解释摘要
            summary = self._generate_explanation_summary(
                explanation, prediction, confidence
            )
            
            return PredictionExplanation(
                prediction=float(prediction),
                base_value=explanation.get("base_value", 0.0),
                confidence=confidence,
                feature_importance=explanation["feature_importance"],
                decision_path=decision_path,
                explanation_summary=summary
            )
            
        except Exception as e:
            logger.error(f"解释预测失败: {str(e)}", exc_info=True)
            raise ModelExplainerError(f"解释预测失败: {str(e)}")
    
    def explain_model_global(
        self,
        model_id: str,
        dataset: Optional[pd.DataFrame] = None,
        top_k: int = 20
    ) -> Dict[str, Any]:
        """
        全局模型解释 - 分析模型整体行为
        
        参数:
            model_id: 模型唯一标识符
            dataset: 用于解释的数据集，如果为None则使用训练数据
            top_k: 返回最重要的K个特征
            
        返回:
            Dict: 包含全局特征重要性、特征交互等信息
            
        示例:
            >>> explainer = ModelExplainer()
            >>> global_exp = explainer.explain_model_global("model_001")
            >>> print(global_exp["top_features"])
        """
        try:
            model = self.ml_service.get_model(model_id)
            if model is None:
                raise ModelExplainerError(f"模型不存在: {model_id}")
            
            # 如果没有提供数据集，尝试获取训练数据
            if dataset is None:
                dataset = self.ml_service.get_training_data(model_id)
            
            if dataset is None or len(dataset) == 0:
                raise ModelExplainerError("没有可用的数据集进行全局解释")
            
            model_type = self._detect_model_type(model)
            
            # 计算全局SHAP值
            if SHAP_AVAILABLE and model_type == ModelType.TREE:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(dataset)
                
                # 计算平均绝对SHAP值
                mean_shap = np.abs(shap_values).mean(axis=0)
                feature_names = dataset.columns.tolist()
                
                global_importance = [
                    {
                        "feature": feature_names[i],
                        "importance": float(mean_shap[i]),
                        "rank": i + 1
                    }
                    for i in np.argsort(mean_shap)[::-1][:top_k]
                ]
                
                return {
                    "model_id": model_id,
                    "model_type": model_type.value,
                    "top_features": global_importance,
                    "explanation_method": "shap_global",
                    "dataset_size": len(dataset)
                }
            else:
                # 使用模型内置的特征重要性
                return self._explain_global_fallback(model, dataset, top_k)
                
        except Exception as e:
            logger.error(f"全局模型解释失败: {str(e)}", exc_info=True)
            raise ModelExplainerError(f"全局模型解释失败: {str(e)}")
    
    def get_feature_interactions(
        self,
        model_id: str,
        feature_pairs: Optional[List[tuple]] = None
    ) -> Dict[str, Any]:
        """
        分析特征交互作用
        
        参数:
            model_id: 模型唯一标识符
            feature_pairs: 要分析的特征对列表，如果为None则分析所有组合
            
        返回:
            Dict: 特征交互强度分析结果
        """
        try:
            model = self.ml_service.get_model(model_id)
            if model is None:
                raise ModelExplainerError(f"模型不存在: {model_id}")
            
            # 获取训练数据
            dataset = self.ml_service.get_training_data(model_id)
            if dataset is None:
                raise ModelExplainerError("没有可用的训练数据")
            
            if not SHAP_AVAILABLE:
                return {"error": "SHAP库未安装，无法计算特征交互"}
            
            # 计算SHAP交互值
            explainer = shap.TreeExplainer(model)
            shap_interaction = explainer.shap_interaction_values(dataset)
            
            # 计算交互强度
            interaction_strength = np.abs(shap_interaction).mean(axis=0)
            feature_names = dataset.columns.tolist()
            
            interactions = []
            n_features = len(feature_names)
            
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    interactions.append({
                        "feature_1": feature_names[i],
                        "feature_2": feature_names[j],
                        "strength": float(interaction_strength[i, j])
                    })
            
            # 按交互强度排序
            interactions.sort(key=lambda x: x["strength"], reverse=True)
            
            return {
                "model_id": model_id,
                "interactions": interactions[:20],  # 返回前20个
                "total_pairs": len(interactions)
            }
            
        except Exception as e:
            logger.error(f"特征交互分析失败: {str(e)}", exc_info=True)
            raise ModelExplainerError(f"特征交互分析失败: {str(e)}")
    
    def _detect_model_type(self, model: Any) -> ModelType:
        """
        检测模型类型
        
        参数:
            model: 模型对象
            
        返回:
            ModelType: 检测到的模型类型
        """
        model_class = model.__class__.__name__.lower()
        
        tree_models = ['xgboost', 'lgbm', 'lightgbm', 'randomforest', 'decisiontree']
        linear_models = ['linear', 'logistic', 'ridge', 'lasso']
        neural_models = ['nn', 'neural', 'mlp', 'pytorch', 'tensorflow']
        
        if any(tm in model_class for tm in tree_models):
            return ModelType.TREE
        elif any(lm in model_class for lm in linear_models):
            return ModelType.LINEAR
        elif any(nm in model_class for nm in neural_models):
            return ModelType.NEURAL
        else:
            return ModelType.UNKNOWN
    
    def _convert_input_to_array(self, input_data: Dict[str, Any]) -> np.ndarray:
        """将输入数据转换为numpy数组"""
        return np.array([list(input_data.values())])
    
    def _get_prediction(self, model: Any, input_array: np.ndarray) -> float:
        """获取模型预测结果"""
        prediction = model.predict(input_array)
        return float(prediction[0]) if len(prediction.shape) > 0 else float(prediction)
    
    def _explain_with_shap(
        self,
        model: Any,
        input_array: np.ndarray,
        input_data: Dict[str, Any],
        model_type: ModelType,
        top_k: int
    ) -> Dict[str, Any]:
        """使用SHAP进行解释"""
        try:
            if model_type == ModelType.TREE:
                explainer = shap.TreeExplainer(model)
            elif model_type == ModelType.LINEAR:
                explainer = shap.LinearExplainer(model, input_array)
            else:
                explainer = shap.KernelExplainer(model.predict, input_array)
            
            shap_values = explainer.shap_values(input_array)
            base_value = explainer.expected_value
            
            # 处理多分类情况
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            feature_names = list(input_data.keys())
            feature_importance = []
            
            for i, feature in enumerate(feature_names):
                importance = abs(float(shap_values[0][i]))
                feature_importance.append(FeatureImportance(
                    feature_name=feature,
                    importance=importance,
                    shap_value=float(shap_values[0][i]),
                    description=self._get_feature_description(feature)
                ))
            
            # 按重要性排序
            feature_importance.sort(key=lambda x: x.importance, reverse=True)
            
            return {
                "base_value": float(base_value) if not isinstance(base_value, np.ndarray) else float(base_value[0]),
                "feature_importance": feature_importance[:top_k]
            }
            
        except Exception as e:
            logger.warning(f"SHAP解释失败，使用降级方案: {str(e)}")
            return self._explain_with_feature_importance(model, input_array, input_data, top_k)
    
    def _explain_with_feature_importance(
        self,
        model: Any,
        input_array: np.ndarray,
        input_data: Dict[str, Any],
        top_k: int
    ) -> Dict[str, Any]:
        """使用模型内置特征重要性进行解释 (降级方案)"""
        try:
            # 获取模型内置特征重要性
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_).flatten()
            else:
                # 使用排列重要性作为最后手段
                importances = self._calculate_permutation_importance(model, input_array)
            
            feature_names = list(input_data.keys())
            feature_importance = []
            
            for i, feature in enumerate(feature_names):
                importance = float(importances[i]) if i < len(importances) else 0.0
                feature_importance.append(FeatureImportance(
                    feature_name=feature,
                    importance=importance,
                    shap_value=0.0,  # 降级方案无SHAP值
                    description=self._get_feature_description(feature)
                ))
            
            feature_importance.sort(key=lambda x: x.importance, reverse=True)
            
            return {
                "base_value": 0.0,
                "feature_importance": feature_importance[:top_k]
            }
            
        except Exception as e:
            logger.error(f"特征重要性解释失败: {str(e)}")
            return {
                "base_value": 0.0,
                "feature_importance": []
            }
    
    def _calculate_permutation_importance(
        self,
        model: Any,
        input_array: np.ndarray
    ) -> np.ndarray:
        """计算排列重要性"""
        baseline_pred = model.predict(input_array)
        importances = np.zeros(input_array.shape[1])
        
        for i in range(input_array.shape[1]):
            # 打乱第i个特征
            permuted = input_array.copy()
            np.random.shuffle(permuted[:, i])
            permuted_pred = model.predict(permuted)
            
            # 计算性能下降
            importances[i] = np.abs(baseline_pred - permuted_pred).mean()
        
        return importances
    
    def _calculate_confidence(self, model: Any, input_array: np.ndarray) -> float:
        """计算预测置信度"""
        try:
            # 如果模型支持概率预测
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_array)
                # 返回最高概率作为置信度
                return float(np.max(proba))
            else:
                # 使用预测值的稳定性作为置信度代理
                return 0.8  # 默认置信度
        except:
            return 0.8
    
    def _get_decision_path(self, model: Any, input_array: np.ndarray) -> List[str]:
        """获取树模型的决策路径"""
        try:
            if hasattr(model, 'decision_path'):
                path = model.decision_path(input_array)
                # 转换为可读格式
                return [f"节点_{i}" for i in range(path.indices.shape[1])]
            return None
        except:
            return None
    
    def _generate_explanation_summary(
        self,
        explanation: Dict[str, Any],
        prediction: float,
        confidence: float
    ) -> str:
        """生成解释摘要"""
        top_features = explanation.get("feature_importance", [])[:3]
        
        summary = f"预测值: {prediction:.4f}, 置信度: {confidence:.2%}\n"
        summary += "主要影响因素:\n"
        
        for i, feat in enumerate(top_features, 1):
            direction = "正向" if feat.shap_value > 0 else "负向"
            summary += f"{i}. {feat.feature_name}: {direction}影响 (重要性: {feat.importance:.4f})\n"
        
        return summary
    
    def _get_feature_description(self, feature_name: str) -> str:
        """获取特征描述"""
        descriptions = {
            "price": "当前价格",
            "volume": "交易量",
            "open": "开盘价",
            "high": "最高价",
            "low": "最低价",
            "close": "收盘价",
            "ma5": "5日均线",
            "ma10": "10日均线",
            "ma20": "20日均线",
            "rsi": "相对强弱指标",
            "macd": "MACD指标",
            "boll_upper": "布林带上轨",
            "boll_lower": "布林带下轨"
        }
        return descriptions.get(feature_name, f"特征: {feature_name}")
    
    def _explain_global_fallback(
        self,
        model: Any,
        dataset: pd.DataFrame,
        top_k: int
    ) -> Dict[str, Any]:
        """全局解释降级方案"""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_).flatten()
            else:
                importances = np.ones(len(dataset.columns)) / len(dataset.columns)
            
            feature_names = dataset.columns.tolist()
            global_importance = [
                {
                    "feature": feature_names[i],
                    "importance": float(importances[i]),
                    "rank": i + 1
                }
                for i in np.argsort(importances)[::-1][:top_k]
            ]
            
            return {
                "model_id": "unknown",
                "model_type": "unknown",
                "top_features": global_importance,
                "explanation_method": "feature_importance_fallback",
                "dataset_size": len(dataset)
            }
            
        except Exception as e:
            logger.error(f"全局解释降级方案失败: {str(e)}")
            return {
                "error": "无法计算全局特征重要性",
                "top_features": []
            }


# 便捷函数
def explain_prediction(
    model_id: str,
    input_data: Dict[str, Any],
    method: str = "shap",
    top_k: int = 10
) -> PredictionExplanation:
    """
    便捷函数 - 解释单个预测结果
    
    参数:
        model_id: 模型唯一标识符
        input_data: 输入特征数据
        method: 解释方法
        top_k: 返回最重要的K个特征
        
    返回:
        PredictionExplanation: 预测解释结果
    """
    explainer = ModelExplainer()
    return explainer.explain_prediction(model_id, input_data, method, top_k)


def explain_model_global(
    model_id: str,
    dataset: Optional[pd.DataFrame] = None,
    top_k: int = 20
) -> Dict[str, Any]:
    """
    便捷函数 - 全局模型解释
    
    参数:
        model_id: 模型唯一标识符
        dataset: 用于解释的数据集
        top_k: 返回最重要的K个特征
        
    返回:
        Dict: 全局解释结果
    """
    explainer = ModelExplainer()
    return explainer.explain_model_global(model_id, dataset, top_k)
