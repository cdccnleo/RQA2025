# src/features/processors/feature_selector.py
from typing import Optional, List, Callable, Any

import numpy as np
import pandas as pd
import joblib
# 使用统一的sklearn导入工具
import logging
from ..utils.sklearn_imports import (
    BaseEstimator, NotFittedError, RFECV, SelectKBest, f_regression,
    RandomForestRegressor, check_is_fitted
)
from pathlib import Path
from ..core.config_integration import get_config_integration_manager, ConfigScope

logger = logging.getLogger(__name__)


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
            preserve_features: List[str] = None,  # 添加需要保留的特征
            strategy: Optional[str] = None,  # 兼容旧接口
            custom_strategy: Optional[Callable] = None,  # 兼容旧接口
            threshold: float = 0.9  # 兼容旧接口
    ):
        """初始化特征选择器"""
        # 配置集成
        self.config_manager = get_config_integration_manager()
        selector_config = self.config_manager.get_config(ConfigScope.PROCESSING)
        if selector_config:
            selector_type = selector_config.get('selector_type', selector_type)
            n_features = selector_config.get('n_features', n_features)
            min_features_to_select = selector_config.get(
                'min_features_to_select', min_features_to_select)
            cv = selector_config.get('cv', cv)
            threshold = selector_config.get('threshold', threshold)
        # 兼容旧接口参数
        if strategy:
            selector_type = strategy
        if custom_strategy:
            self.custom_strategy = custom_strategy
        else:
            self.custom_strategy = None
        self.threshold = threshold

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
        # 注册配置变更监听器
        self.config_manager.register_config_watcher(ConfigScope.PROCESSING, self._on_config_change)

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
        elif self.selector_type == "variance":
            # 方差选择器
            self.selector = None
        elif self.selector_type == "correlation":
            # 相关性选择器
            self.selector = None
        elif self.selector_type == "importance":
            # 重要性选择器
            self.selector = None
        else:
            raise ValueError(f"无效的选择器类型: {self.selector_type}")

    def select(self, features: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        """选择特征（兼容旧接口）"""
        if self.selector_type in ["variance", "correlation", "importance"]:
            return self._select_by_strategy(features, target)
        else:
            # 先拟合再转换
            if target is not None:
                self.fit(features, target)
            return self.transform(features)

    def select_features(self, features: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        选择特征（新接口）

        Args:
            features: 特征数据
            target: 目标变量

        Returns:
            选择后的特征数据
        """
        return self.select(features, target)

    def _select_by_strategy(self, features: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        """根据策略选择特征"""
        if features.empty:
            return pd.DataFrame()

        if self.selector_type == "variance":
            # 方差选择
            variances = features.var()
            threshold = variances.quantile(0.1)  # 保留方差最大的90%
            selected_cols = variances[variances >= threshold].index.tolist()
            return features[selected_cols]

        elif self.selector_type == "correlation":
            # 相关性选择
            if target is None:
                return features

            correlations = features.corrwith(target).abs()
            selected_cols = correlations[correlations >= self.threshold].index.tolist()
            return features[selected_cols]

        elif self.selector_type == "importance":
            # 重要性选择（使用随机森林）
            if target is None:
                return features

            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(n_estimators=10, random_state=42)
            rf.fit(features, target)
            importances = rf.feature_importances_
            selected_cols = features.columns[importances > np.mean(importances)].tolist()
            return features[selected_cols]

        else:
            return features

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

            # 设置特征数量（用于占位符类）
            if hasattr(self.selector, '_n_features'):
                self.selector._n_features = features.shape[1]

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

        # 验证输入特征维度与训练时一致（允许额外列，自动对齐）
        if features.shape[1] != self.selector.n_features_in_:
            original_names = getattr(self, "original_feature_names", features.columns.tolist())
            missing_columns = [col for col in original_names if col not in features.columns]
            if missing_columns:
                raise ValueError(f"输入特征缺少列: {missing_columns}")
            features = features[original_names]

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
                    f"特征选择器文件未找到 | 路径: {str(full_path.resolve())}"  # 显示完整路径
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
            if self.selector_type == "kbest":
                mask = self.selector.get_support()
                return features.columns[mask].tolist()
            elif self.selector_type == "rfecv":
                return self.selector.get_feature_names_out().tolist()
            else:
                return []
        except Exception as e:
            self.logger.error(f"获取选中特征失败: {str(e)}")
            return []

    def update_selector_params(self, **params):
        """更新选择器参数"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.logger.warning(f"未知参数: {key}")

        # 重新初始化选择器
        self._init_selector()
        self.is_fitted = False

    def _on_config_change(self, scope: ConfigScope, key: str, old_value: Any, new_value: Any):

        if scope == ConfigScope.PROCESSING:
            if key == "selector_type":
                self.selector_type = new_value
                self._init_selector()
            elif key == "n_features":
                self.n_features = new_value
                self._init_selector()
            elif key == "min_features_to_select":
                self.min_features_to_select = max(3, new_value)
                self._init_selector()
            elif key == "cv":
                self.cv = new_value
                self._init_selector()
            elif key == "threshold":
                self.threshold = new_value
