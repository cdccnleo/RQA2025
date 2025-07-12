# src/features/processors/feature_selector.py
from typing import Optional, List

import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import RFECV, SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
from sklearn.utils.validation import check_is_fitted
from src.infrastructure.utils.logger import get_logger

logger = get_logger(__name__)  # 自动继承全局配置


class FeatureSelector:
    """特征选择模块，专注于特征的选择和优化，支持RFECV和SelectKBest

    属性:
        selector (RFECV | SelectKBest): 特征选择器实例
        selected_features (List[str]): 选择后的特征列
        selector_path (Path): 选择器存储路径
        preserve_features (List[str]): 必须保留的特征列
        is_fitted (bool): 选择器是否已经拟合的标记
    """

    def __init__(
            self,
            selector_type: str = "rfecv",
            model_path: Optional[Path] = None,
            n_features: int = 15,
            min_features_to_select: int = 3,
            cv: int = 5,
            preserve_features: List[str] = None  # 添加需要保留的特征
    ):
        """初始化特征选择器"""
        self.selector_type = selector_type
        self.selector: Optional[BaseEstimator] = None
        self.model_path = model_path
        self.n_features = n_features
        if min_features_to_select <= 0:
            raise ValueError("min_features_to_select必须为正整数")
        self.min_features_to_select = max(3, min_features_to_select)  # 确保最小值
        self.cv = cv
        self.selected_features = []
        self.preserve_features = preserve_features or []  # 默认为空列表
        self.logger = logger
        self._init_selector()  # 初始化选择器
        self.is_fitted = False  # 添加拟合状态标记

    def _init_selector(self) -> None:
        """根据selector_type初始化特征选择器实例"""
        if self.selector_type == "rfecv":
            self.selector = RFECV(
                estimator=RandomForestRegressor(n_estimators=50),
                min_features_to_select=self.min_features_to_select,
                cv=self.cv
            )
        elif self.selector_type == "kbest":
            if self.n_features <= 0:
                raise ValueError("k must be positive")
            self.selector = SelectKBest(
                score_func=f_regression,
                k=self.n_features  # 使用类属性中的值
            )
        else:
            raise ValueError(f"无效的选择器类型: {self.selector_type}")

    def fit(self, features: pd.DataFrame, target: pd.Series, is_training=True):
        """拟合特征选择器，确保保留指定特征"""
        # 将特征数据转换为 pandas.DataFrame
        if isinstance(features, np.ndarray):
            features = pd.DataFrame(features)

        # 检查目标变量是否为空或未提供
        if is_training:
            if target is None:
                raise ValueError("目标变量不能为空")  # 明确抛出异常

            # 将目标变量转换为 pandas.Series
            if isinstance(target, np.ndarray):
                target = pd.Series(target, index=features.index)
            elif not isinstance(target, pd.Series):
                raise TypeError("目标变量必须是 pandas.Series 或 numpy.ndarray")

            if target.empty:
                raise ValueError("目标变量为空")

        if features.empty:
            self.logger.warning("有效特征列表为空")
            # 显式初始化选择器以避免后续错误
            self._init_selector()
            # 清空选中的特征列表
            self.selected_features = []
            self.is_fitted = False  # 标记为未拟合
            return

        # 继续执行拟合逻辑
        if self.selector is None:
            self._init_selector()

        try:
            self.selector.fit(features, target)  # 直接拟合当前实例
            # 更新选中的特征列
            if self.selector_type == "kbest":
                mask = self.selector.get_support()
                self.selected_features = features.columns[mask].tolist()
            elif self.selector_type == "rfecv":
                self.selected_features = self.selector.get_feature_names_out().tolist()

            # 合并必须保留的特征
            self.selected_features = list(set(self.selected_features) | set(self.preserve_features))

            self._save_selector()
            self.is_fitted = True  # 标记为已拟合
        except Exception as e:
            self.logger.error(f"特征选择器拟合失败: {str(e)}")
            raise

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """应用特征选择器"""
        # 空数据直接返回
        if features.empty:
            self.logger.warning("输入特征数据为空")
            return pd.DataFrame()

        # 未拟合选择器处理
        if not self.selector or not self.is_fitted:
            self.logger.warning("特征选择器尚未拟合")
            return features.copy()  # 返回原始数据而非空DataFrame

        try:
            check_is_fitted(self.selector)
        except NotFittedError as e:
            self.logger.warning(f"特征选择器尚未拟合: {str(e)}，返回原始数据")
            return features.copy()  # 返回原始数据
        except Exception as e:
            self.logger.error(f"特征选择失败: {str(e)}")
            return features.copy()  # 返回原始数据

        # 新增校验：特征列名列表是否为空
        if len(self.selected_features) == 0:
            self.logger.warning("特征选择失败：有效特征列表为空")
            return pd.DataFrame()  # 返回空 DataFrame

        # 验证输入特征维度与训练时一致
        if features.shape[1] != self.selector.n_features_in_:
            raise ValueError("输入特征数量不一致")

        if self.selector_type == "rfecv":
            try:
                selected = self.selector.transform(features)
                return pd.DataFrame(selected, columns=self.selected_features, index=features.index)
            except Exception as e:
                self.logger.warning(f"特征选择失败: {str(e)}")
                return features.copy()  # 返回原始数据
        else:
            try:
                return features[self.selected_features]
            except Exception as e:
                self.logger.warning(f"特征选择失败: {str(e)}")
                return features.copy()  # 返回原始数据

    def _load_selector(self) -> None:
        """从文件加载选择器"""
        if self.model_path is None:
            raise RuntimeError("模型路径未设置，无法加载选择器")

        full_path = self.model_path / "feature_selector.pkl"
        try:
            if not full_path.exists():
                self.logger.warning(
                    "特征选择器文件未找到 | 路径: %s",
                    str(full_path.resolve())  # 显示完整路径
                )
                raise FileNotFoundError

            self.selector = joblib.load(full_path)
            self.selected_features = list(joblib.load(self.model_path / "selected_features.pkl"))
        except FileNotFoundError as e:
            self.logger.warning("特征选择器文件未找到")  # 增强日志记录
            raise  # 不再封装为RuntimeError，直接抛出原始异常
        except Exception as e:
            raise RuntimeError("加载特征选择器失败") from e

    def _save_selector(self) -> None:
        """保存选择器到文件"""
        if self.selector and self.model_path:
            self.model_path.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.selector, self.model_path / "feature_selector.pkl")
            joblib.dump(self.selected_features, self.model_path / "selected_features.pkl")

    def _get_selected_features(self, features: pd.DataFrame) -> List[str]:
        """根据选择器类型提取被选中的特征列名"""
        if not hasattr(self.selector, "support_"):
            raise NotFittedError("特征选择器尚未拟合")

        try:
            if self.selector_type == "rfecv":
                # 对于RFECV，使用get_feature_names_out()
                selected = self.selector.get_feature_names_out(features.columns)
            elif self.selector_type == "kbest":
                # 对于SelectKBest，通过布尔掩码获取特征列名
                mask = self.selector.get_support()
                selected = features.columns[mask].tolist()
            else:
                raise ValueError(f"不支持的选择器类型: {self.selector_type}")
        except AttributeError as e:
            raise RuntimeError(f"无法提取选中特征: {str(e)}") from e

        # 合并必须保留的特征
        selected = list(set(selected) | set(self.preserve_features))
        return selected

    def update_selector_params(self, **params):
        """动态更新选择器参数，允许在未拟合状态下更新"""
        if self.selector is None:
            self._init_selector()  # 确保选择器已初始化

        # 动态更新参数并重新初始化选择器
        if self.selector_type == "rfecv":
            # 提取允许更新的参数
            estimator = params.get("estimator", self.selector.estimator if self.selector else None)
            self.cv = params.get("cv", self.cv)
            self.min_features_to_select = max(3, params.get(
                "min_features_to_select",
                params.get("min_features", self.min_features_to_select)
            ))

            # 重新初始化 RFECV
            self.selector = RFECV(
                estimator=estimator,
                min_features_to_select=self.min_features_to_select,
                cv=self.cv
            )
        elif self.selector_type == "kbest":
            k = params.get("k")
            if k is not None:
                if k <= 0:
                    raise ValueError("k必须为正整数")
                self.selector.k = k
                self.n_features = k
