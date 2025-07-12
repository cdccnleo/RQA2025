# src/features/processors/feature_standardizer.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from sklearn.exceptions import NotFittedError
from src.infrastructure.utils.logger import get_logger

logger = get_logger(__name__)  # 自动继承全局配置


class FeatureStandardizer:
    """特征标准化模块，专注于特征的标准化处理

    属性:
        scaler (StandardScaler): 标准化器实例
        scaler_path (Path): 标准化器存储路径
    """

    def __init__(self, model_path: Path):
        """初始化标准化器"""
        self.model_path = Path(model_path) if isinstance(model_path, str) else model_path
        self.scaler = StandardScaler()
        self.scaler_path = self.model_path / "feature_scaler.pkl"
        self.logger = logger
        self.is_fitted = False  # 添加拟合状态标记

    def fit_transform(self, features: pd.DataFrame, is_training: bool = True, metadata=None) -> pd.DataFrame:
        """拟合并转换特征"""
        # 空数据直接返回（保持类型一致性）
        if features.empty:
            raise ValueError("输入数据为空，请确保数据不为空")

        # 过滤掉非数值列
        numeric_features = features.select_dtypes(include=[np.number])

        if numeric_features.empty:
            raise ValueError("输入数据不包含数值列")

        if is_training:
            self.scaler.fit(numeric_features)
            joblib.dump(self.scaler, self.scaler_path)
            self.is_fitted = True  # 标记为已拟合

            # 验证文件是否保存成功
            if not self.scaler_path.exists():
                raise RuntimeError(f"特征标准化器保存失败，路径: {self.scaler_path}")

            if metadata:
                metadata.scaler_path = self.scaler_path
        else:
            try:
                self.scaler = joblib.load(self.scaler_path)
            except FileNotFoundError:
                self.logger.warning("标准化器文件未找到，返回原始数据")
                return features
            except Exception as e:
                logger.error(f"特征标准化失败: {str(e)}")
                raise

        scaled_array = self.scaler.transform(numeric_features)
        standardized_features = pd.DataFrame(scaled_array, columns=numeric_features.columns,
                                             index=numeric_features.index)

        return standardized_features

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """应用标准化转换"""
        numeric_features = features.select_dtypes(include=[np.number])
        if numeric_features.empty:
            raise ValueError("输入数据不包含数值列")

        if not self.is_fitted:  # 使用状态标记检查是否拟合
            raise RuntimeError("标准化器尚未拟合")

        try:
            standardized_data = self.scaler.transform(numeric_features)
        except NotFittedError as e:
            raise RuntimeError("标准化器尚未拟合") from e  # 修改异常信息

        return pd.DataFrame(
            standardized_data,
            columns=numeric_features.columns,
            index=numeric_features.index
        )

    def load_scaler(self, scaler_path: Path) -> None:
        """加载预训练的标准化器"""
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        else:
            raise FileNotFoundError(f"标准化器文件不存在: {scaler_path}")

    def partial_fit(self, features: pd.DataFrame):
        """增量更新标准化器"""
        self.scaler.partial_fit(features.select_dtypes(include=[np.number]))
