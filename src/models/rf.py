# src/models/rf.py
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Union

import shap
from sklearn.ensemble import RandomForestRegressor

from src.models.base_model import BaseModel
from src.infrastructure.utils.logger import get_logger

logger = get_logger(__name__)


class RandomForestModel(BaseModel):
    """随机森林回归模型，继承自BaseModel基类

    属性：
        model_name (str): 模型名称标识，默认"random_forest"
        n_estimators (int): 决策树数量，默认100
        max_depth (Optional[int]): 树的最大深度，None表示不限制
        model (Optional[RandomForestRegressor]): Scikit-learn模型实例
    """

    def __init__(self, model_name: str = "random_forest", n_estimators: int = 100, max_depth: int = None):
        super().__init__(model_name=model_name)
        if n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0")
        if max_depth is not None and max_depth <= 0:
            raise ValueError("max_depth must be greater than 0 or None")
        self.config = {
            "model_name": model_name,
            "n_estimators": n_estimators,
            "max_depth": max_depth
        }
        self.model = None
        self._is_trained = False

    def train(self, features: pd.DataFrame, target: pd.Series):
        if features.empty or target.empty:
            raise ValueError("特征和目标数据不能为空")

        # 确保模型被正确初始化
        if self.model is None:
            self.model = RandomForestRegressor(
                n_estimators=self.config['n_estimators'],
                max_depth=self.config.get('max_depth'),
                random_state=42  # 添加随机种子保证可重复性
            )

        # 实际训练过程
        try:
            self.model.fit(features, target)
            self._is_trained = True
            self.feature_names_ = features.columns.tolist()

        except Exception as e:
            self._is_trained = False
            raise RuntimeError(f"模型训练失败: {str(e)}")

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """生成预测结果"""
        if not self.is_trained or self.model is None:
            logger.error("模型尚未训练，无法进行预测")
            raise RuntimeError("模型尚未训练")

        logger.info("开始预测过程")

        # 校验特征顺序（当输入为DataFrame时）
        logger.info(f"输入特征为 DataFrame，列数: {features.shape[1]}")
        self._validate_feature_order(features)  # 在数据转换前校验特征顺序

        logger.info("特征校验完成")

        logger.info("开始模型预测")
        predictions = self.model.predict(features)
        logger.info(f"预测完成，结果形状: {predictions.shape}")

        return predictions

    def save(self, dir_path: Union[str, Path], model_name: str = None, overwrite: bool = False):
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        model_name = model_name or self.config['model_name']
        model_path = dir_path / f"{model_name}.pkl"

        if model_path.exists() and not overwrite:
            raise FileExistsError(f"模型文件已存在: {model_path}")

        # 保存模型和配置
        joblib.dump({
            'model': self.model,
            'config': self.config,
            'feature_names_': self.feature_names_
        }, model_path)

        self.logger.info(f"模型保存成功: {model_path}")
        return model_path

    @classmethod
    def load(cls, dir_path: Union[str, Path], model_name: str = "random_forest"):
        dir_path = Path(dir_path)
        model_path = dir_path / f"{model_name}.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 加载模型和配置
        checkpoint = joblib.load(model_path)
        instance = cls(model_name=model_name)
        instance.model = checkpoint['model']
        instance.config = checkpoint['config']
        instance.feature_names_ = checkpoint.get('feature_names_', None)
        instance._is_trained = True

        instance.logger.info(f"模型加载成功: {model_path}")
        return instance

    def get_feature_importance(self) -> pd.Series:
        """获取特征重要性"""
        if not self.is_trained or self.model is None:
            raise RuntimeError("模型尚未训练")
        # 使用保存的特征列名
        return pd.Series(
            self.model.feature_importances_,
            index=self.feature_names_
        )

    def analyze_feature_interactions(self, features: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """基于SHAP Interaction Values的高效特征交互分析

        返回:
            pd.DataFrame: 特征交互矩阵，值表示交互强度（绝对值均值）
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("模型尚未训练")

        if isinstance(features, pd.DataFrame):
            if len(features.columns) < 2:
                raise ValueError("特征交互分析需要至少两列特征")
        elif isinstance(features, np.ndarray):
            if features.shape[1] < 2:
                raise ValueError("特征交互分析需要至少两列特征")

        # 初始化 TreeExplainer
        explainer = shap.TreeExplainer(self.model)

        # 计算 SHAP 交互值
        shap_interaction_values = explainer.shap_interaction_values(features)

        # 取第一个输出的交互值（针对回归任务）
        interaction_matrix = np.mean(np.abs(shap_interaction_values), axis=0)

        # 转换为 DataFrame
        if isinstance(features, pd.DataFrame):
            interaction_df = pd.DataFrame(
                interaction_matrix,
                columns=features.columns,
                index=features.columns
            )
        else:
            interaction_df = pd.DataFrame(
                interaction_matrix,
                columns=[f"feature_{i}" for i in range(features.shape[1])],
                index=[f"feature_{i}" for i in range(features.shape[1])]
            )

        return interaction_df

    def get_model(self) -> Any:
        """返回具体的模型实例"""
        return self.model
