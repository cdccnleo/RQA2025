# src/models/base_model.py
from __future__ import annotations
import os
import joblib
import torch
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Iterator, Union
from sklearn.base import BaseEstimator
from torch.utils.data import DataLoader

from src.infrastructure.utils.logger import get_logger

logger = get_logger(__name__)  # 自动继承全局配置

# 类型变量用于子类返回类型
T = TypeVar("T", bound="BaseModel")


class ModelPersistence:
    @staticmethod
    def save_model(model, path, overwrite=False):
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(f"模型文件已存在: {path}")
        joblib.dump(model, path)

    @staticmethod
    def load_model(path):
        return joblib.load(path)


class BaseModel(ABC):
    """机器学习模型抽象基类，定义统一接口

    属性:
        model_name (str): 模型唯一标识
        config (Dict[str, Any]): 模型超参数配置
        model (Any): 具体模型实例
        is_trained (bool): 模型训练状态标记
    """

    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        """初始化基础模型
        Args:
            model_name: 模型唯一标识
            config: 模型超参数字典
        """
        self.model_name = model_name
        self.config = config or {}
        self.config['model_name'] = model_name
        self.logger = logger
        self._is_trained = False
        self.feature_names_ = None  # 新增特征列顺序属性
        self.model = None  # 明确初始化 model 属性

    @abstractmethod
    def train(self,
              features: pd.DataFrame,
              target: pd.Series,
              **kwargs) -> BaseModel:
        """训练模型抽象方法
        Args:
            features: 特征数据
            target: 目标变量
        Returns:
            训练后的模型实例
        """
        pass

    def _validate_feature_order(self, features):
        """校验输入特征列顺序或名称集合是否与训练时一致"""
        if self.feature_names_ is not None:
            expected_num_features = len(self.feature_names_)
            # 检查是否配置了特征顺序校验
            strict_feature_order = getattr(self, 'strict_feature_order', True)

            if isinstance(features, pd.DataFrame):
                if strict_feature_order:
                    # 严格校验特征列顺序
                    if list(features.columns) != self.feature_names_:
                        raise ValueError("特征列顺序与训练时不一致！")
                else:
                    # 校验特征名称集合
                    if set(features.columns) != set(self.feature_names_):
                        raise ValueError("特征名称集合与训练时不一致！")
            elif isinstance(features, np.ndarray):
                # 校验特征维度
                if features.shape[1] != expected_num_features:
                    raise ValueError(
                        f"输入特征维度不匹配：预期 {expected_num_features} 个特征，实际 {features.shape[1]} 个特征")
            else:
                raise TypeError(f"不支持的输入数据类型: {type(features)}")

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """执行预测抽象方法

        参数:
            features (pd.DataFrame): 输入特征数据

        返回:
            np.ndarray: 预测结果数组

        异常:
            RuntimeError: 模型未训练时调用预测抛出
        """
        self._validate_feature_order(features)  # 校验特征列顺序
        # 具体预测逻辑由子类实现
        raise NotImplementedError("子类必须实现 predict 方法")

    def save(self, dir_path: Union[str, Path], model_name: Optional[str] = None, overwrite: bool = False) -> Path:
        model_name = model_name or self.model_name
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        if isinstance(self.model, BaseEstimator):
            file_ext = ".pkl"
        elif isinstance(self.model, torch.nn.Module):
            file_ext = ".pt"
        else:
            raise NotImplementedError("不支持的模型保存类型")

        save_path = dir_path / f"{model_name}{file_ext}"

        if save_path.exists() and not overwrite:
            raise FileExistsError(f"模型文件已存在: {save_path}")

        if isinstance(self.model, BaseEstimator):
            joblib.dump(self.model, save_path)
        elif isinstance(self.model, torch.nn.Module):
            # 显式存储 is_trained 和 feature_names_
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
                'is_trained': self.is_trained,  # 显式保存训练状态
                'feature_names_': self.feature_names_
            }, save_path)
        else:
            raise NotImplementedError("不支持的模型保存类型")

        self.logger.info(f"模型保存成功: {save_path}")
        return save_path

    @classmethod
    def load(cls: Type[T], dir_path: Union[str, Path], model_name: str) -> T:
        dir_path = Path(dir_path)
        possible_exts = ['.pkl', '.pt']
        model_path = None

        for ext in possible_exts:
            candidate = dir_path / f"{model_name}{ext}"
            if candidate.exists():
                model_path = candidate
                break

        if not model_path:
            raise FileNotFoundError(f"模型文件不存在: {dir_path}/{model_name}")

        if model_path.suffix not in possible_exts:
            raise ValueError(f"不支持的模型文件扩展名: {model_path.suffix}。支持的扩展名: {possible_exts}")

        try:
            if model_path.suffix == ".pkl":
                return cls._load_sklearn(model_path)
            elif model_path.suffix == ".pt":
                return cls._load_pytorch(model_path)
        except Exception as e:
            raise ValueError(f"无效的模型文件: {model_path}") from e

        raise ValueError("未知的模型文件格式")

    def build_model(self):
        """子类必须实现模型构建逻辑"""
        pass

    def _save_sklearn(self, path: Path):
        """保存Scikit-learn风格模型"""
        joblib.dump(self.model, path)

    def _save_pytorch(self, path: Path):
        """保存PyTorch模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, path)

    @classmethod
    def _load_sklearn(cls, path: Path) -> BaseModel:
        """加载Scikit-learn模型"""
        model = joblib.load(path)
        instance = cls(model_name=path.stem)
        instance.model = model
        instance._is_trained = True
        return instance

    @classmethod
    def _load_pytorch(cls, path: Path) -> BaseModel:
        """加载PyTorch模型"""
        checkpoint = torch.load(path)
        config = checkpoint['config']
        model_name = config.get('model_name', 'default_name')

        # 创建实例但不调用 __init__
        instance = cls.__new__(cls)

        # 手动设置必要属性
        instance.model_name = model_name
        instance.config = config
        instance.logger = get_logger(__name__)
        instance._is_trained = checkpoint.get('is_trained', False)  # 显式恢复训练状态
        instance.feature_names_ = checkpoint.get('feature_names_', None)

        # 构建模型
        instance.build_model()

        # 加载模型状态
        instance.model.load_state_dict(checkpoint['model_state_dict'])

        return instance

    @property
    def is_trained(self) -> bool:
        """模型是否已完成训练"""
        return self._is_trained

    @is_trained.setter
    def is_trained(self, value: bool):
        self._is_trained = value

    def validate(self,
                 features: pd.DataFrame,
                 target: pd.Series,
                 metrics: Dict[str, callable]) -> Dict[str, float]:
        """
        通用模型验证方法
        :param features: 验证特征数据
        :param target: 验证目标数据
        :param metrics: 评估指标字典 {名称: 函数}
        :return: 评估结果字典
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练")

        predictions = self.predict(features)
        return {
            name: metric(target, predictions)
            for name, metric in metrics.items()
        }

    def get_feature_importance(self) -> pd.Series:
        """获取特征重要性
        Returns:
            特征重要性Series，索引为特征名
        Raises:
            NotImplementedError: 当模型不支持特征重要性时
        """
        if hasattr(self.model, "feature_importances_"):
            return pd.Series(
                self.model.feature_importances_,
                index=self.model.feature_names_in_
            ).sort_values(ascending=False)
        raise NotImplementedError("该模型不支持特征重要性获取")


class TorchModelMixin(ABC):
    """PyTorch模型混合类"""

    @abstractmethod
    def get_model(self) -> torch.nn.Module:
        """返回具体的 PyTorch 模型实例"""
        pass

    def configure_optimizer(self) -> torch.optim.Optimizer:
        """优化器配置（需子类实现）"""
        model = self.get_model()
        if model is None:
            raise NotImplementedError("子类必须实现 get_model 方法")
        return torch.optim.Adam(model.parameters(), lr=1e-3)

    def configure_loss(self) -> torch.nn.Module:
        """损失函数配置（需子类实现）"""
        model = self.get_model()
        if model is None:
            raise NotImplementedError("子类必须实现 get_model 方法")
        # 默认使用均方误差损失，子类可根据需要重写此方法
        return torch.nn.MSELoss()

    def train_epoch(self, train_loader: Union[DataLoader, Iterator], optimizer: torch.optim.Optimizer,
                    loss_fn: torch.nn.Module, device: str = "auto") -> float:
        """单epoch训练逻辑"""
        model = self.get_model()
        if model is None:
            raise NotImplementedError("子类必须实现 get_model 方法")

        # 确保模型在正确设备上
        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.train()

        total_loss = 0
        batch_count = 0

        for batch in train_loader:
            inputs, targets = batch
            inputs = inputs.to(device)  # 确保输入在正确设备上
            targets = targets.to(device)  # 确保目标在正确设备上

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        if batch_count == 0:
            raise ValueError("训练数据加载器中没有批次")

        return total_loss / batch_count
