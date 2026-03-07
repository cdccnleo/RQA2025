# src / features / intelligent / auto_feature_selector.py
"""
自动特征选择器模块
实现智能化的特征选择功能，包括多种选择策略和自动优化
"""

import logging
from typing import Optional, List, Tuple, Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression,
    SelectFromModel, RFECV
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

from ..core.config_integration import get_config_integration_manager, ConfigScope

logger = logging.getLogger(__name__)


class AutoFeatureSelector:

    """自动特征选择器

    支持多种特征选择策略：
    - 统计方法：方差、相关性、互信息
    - 模型方法：基于重要性的选择
    - 包装方法：递归特征消除
    - 自动优化：多策略组合
    """

    def __init__(


        self,
        strategy: str = "auto",
        task_type: str = "classification",
        max_features: Optional[int] = None,
        min_features: int = 3,
        cv_folds: int = 5,
        random_state: int = 42,
        config_manager=None
    ):
        """
        初始化自动特征选择器

        Args:
            strategy: 选择策略 ('auto', 'statistical', 'model_based', 'wrapper', 'ensemble')
            task_type: 任务类型 ('classification', 'regression')
            max_features: 最大特征数量
            min_features: 最小特征数量
            cv_folds: 交叉验证折数
            random_state: 随机种子
            config_manager: 配置管理器
        """
        self.strategy = strategy
        self.task_type = task_type
        self.max_features = max_features
        self.min_features = min_features
        self.cv_folds = cv_folds
        self.random_state = random_state

        # 配置管理集成
        self.config_manager = config_manager or get_config_integration_manager()
        self.config_manager.register_config_watcher(ConfigScope.PROCESSING, self._on_config_change)

        # 初始化组件
        self.scaler = StandardScaler()
        self.selected_features = []
        self.feature_importance = {}
        self.selection_history = []
        self.is_fitted = False

        logger.info(f"初始化自动特征选择器: strategy={strategy}, task_type={task_type}")

    def _on_config_change(self, scope: ConfigScope, key: str, value: Any) -> None:
        """配置变更处理"""
        if scope == ConfigScope.PROCESSING:
            if key == "max_features":
                self.max_features = value
                logger.info(f"更新最大特征数量: {value}")
            elif key == "min_features":
                self.min_features = value
                logger.info(f"更新最小特征数量: {value}")

    def select_features(


        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_features: Optional[int] = None
    ) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
        """
        自动选择特征

        Args:
            X: 特征数据
            y: 目标变量
            target_features: 目标特征数量

        Returns:
            Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
            (选择的特征数据, 特征名称列表, 选择信息)
        """
        if X.empty or y.empty:
            raise ValueError("输入数据不能为空")

        # 确定目标特征数量
        if target_features is None:
            target_features = self._determine_optimal_feature_count(X, y)

        # 根据策略选择特征
        if self.strategy == "auto":
            return self._auto_select(X, y, target_features)
        elif self.strategy == "statistical":
            return self._statistical_select(X, y, target_features)
        elif self.strategy == "model_based":
            return self._model_based_select(X, y, target_features)
        elif self.strategy == "wrapper":
            return self._wrapper_select(X, y, target_features)
        elif self.strategy == "ensemble":
            return self._ensemble_select(X, y, target_features)
        else:
            raise ValueError(f"不支持的选择策略: {self.strategy}")

    def _determine_optimal_feature_count(self, X: pd.DataFrame, y: pd.Series) -> int:
        """确定最优特征数量"""
        n_features = X.shape[1]
        n_samples = X.shape[0]

        # 基于样本数量的启发式规则
        if n_samples < 100:
            optimal = min(10, n_features // 2)
        elif n_samples < 1000:
            optimal = min(20, n_features // 3)
        else:
            optimal = min(50, n_features // 4)

        # 应用约束
        optimal = max(self.min_features, min(optimal, n_features))
        if self.max_features:
            optimal = min(optimal, self.max_features)

        logger.info(f"确定最优特征数量: {optimal} (总特征数: {n_features})")
        return optimal

    def _auto_select(


        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_features: int
    ) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
        """自动选择策略 - 组合多种方法"""
        results = {}

        # 1. 统计方法
        try:
            X_stat, features_stat, info_stat = self._statistical_select(X, y, target_features)
            results['statistical'] = {
                'features': features_stat,
                'score': self._evaluate_selection(X_stat, y)
            }
        except Exception as e:
            logger.warning(f"统计方法失败: {e}")

        # 2. 模型方法
        try:
            X_model, features_model, info_model = self._model_based_select(X, y, target_features)
            results['model_based'] = {
                'features': features_model,
                'score': self._evaluate_selection(X_model, y)
            }
        except Exception as e:
            logger.warning(f"模型方法失败: {e}")

        # 3. 包装方法
        try:
            X_wrapper, features_wrapper, info_wrapper = self._wrapper_select(X, y, target_features)
            results['wrapper'] = {
                'features': features_wrapper,
                'score': self._evaluate_selection(X_wrapper, y)
            }
        except Exception as e:
            logger.warning(f"包装方法失败: {e}")

        # 选择最佳结果
        best_method = max(results.keys(), key=lambda k: results[k]['score'])
        best_features = results[best_method]['features']

        logger.info(f"自动选择最佳方法: {best_method}, 分数: {results[best_method]['score']}")

        return X[best_features], best_features, {
            'method': best_method,
            'all_results': results,
            'target_features': target_features
        }

    def _statistical_select(


        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_features: int
    ) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
        """统计方法特征选择"""
        # 1. 方差选择
        variance_selector = SelectKBest(
            score_func=lambda X, y: np.var(X, axis=0),
            k=target_features
        )
        variance_selector.fit(X, y)
        variance_features = X.columns[variance_selector.get_support()].tolist()

        # 2. 相关性选择
        if self.task_type == "classification":
            corr_selector = SelectKBest(
                score_func=f_classif,
                k=target_features
            )
        else:
            corr_selector = SelectKBest(
                score_func=f_regression,
                k=target_features
            )
        corr_selector.fit(X, y)
        corr_features = X.columns[corr_selector.get_support()].tolist()

        # 3. 互信息选择
        if self.task_type == "classification":
            mi_selector = SelectKBest(
                score_func=mutual_info_classif,
                k=target_features
            )
        else:
            mi_selector = SelectKBest(
                score_func=mutual_info_regression,
                k=target_features
            )
        mi_selector.fit(X, y)
        mi_features = X.columns[mi_selector.get_support()].tolist()

        # 选择最佳统计方法
        methods = {
            'variance': variance_features,
            'correlation': corr_features,
            'mutual_info': mi_features
        }

        best_method = 'correlation'  # 默认使用相关性
        best_features = methods[best_method]

        return X[best_features], best_features, {
            'method': 'statistical',
            'sub_method': best_method,
            'all_methods': methods
        }

    def _model_based_select(


        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_features: int
    ) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
        """基于模型的特征选择"""
        models = {}

        # 1. 随机森林
        if self.task_type == "classification":
            rf_model = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state
            )
        else:
            rf_model = RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state
            )

        rf_selector = SelectFromModel(
            estimator=rf_model,
            max_features=target_features
        )
        rf_selector.fit(X, y)
        rf_features = X.columns[rf_selector.get_support()].tolist()
        models['random_forest'] = rf_features

        # 2. Lasso / Logistic回归
        if self.task_type == "classification":
            lasso_model = LogisticRegression(
                penalty='l1',
                solver='liblinear',
                random_state=self.random_state
            )
        else:
            lasso_model = Lasso(alpha=0.01, random_state=self.random_state)

        lasso_selector = SelectFromModel(
            estimator=lasso_model,
            max_features=target_features
        )
        lasso_selector.fit(X, y)
        lasso_features = X.columns[lasso_selector.get_support()].tolist()
        models['lasso'] = lasso_features

        # 3. 决策树
        if self.task_type == "classification":
            dt_model = DecisionTreeClassifier(random_state=self.random_state)
        else:
            dt_model = DecisionTreeRegressor(random_state=self.random_state)

        dt_selector = SelectFromModel(
            estimator=dt_model,
            max_features=target_features
        )
        dt_selector.fit(X, y)
        dt_features = X.columns[dt_selector.get_support()].tolist()
        models['decision_tree'] = dt_features

        # 选择最佳模型方法
        best_method = 'random_forest'  # 默认使用随机森林
        best_features = models[best_method]

        return X[best_features], best_features, {
            'method': 'model_based',
            'sub_method': best_method,
            'all_models': models
        }

    def _wrapper_select(


        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_features: int
    ) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
        """包装方法特征选择"""
        # 使用RFECV进行递归特征消除
        if self.task_type == "classification":
            estimator = RandomForestClassifier(
                n_estimators=50,
                random_state=self.random_state
            )
        else:
            estimator = RandomForestRegressor(
                n_estimators=50,
                random_state=self.random_state
            )

        rfe_selector = RFECV(
            estimator=estimator,
            min_features_to_select=self.min_features,
            cv=self.cv_folds,
            step=1
        )

        rfe_selector.fit(X, y)
        rfe_features = X.columns[rfe_selector.get_support()].tolist()

        return X[rfe_features], rfe_features, {
            'method': 'wrapper',
            'sub_method': 'rfecv',
            'n_features_selected': len(rfe_features)
        }

    def _ensemble_select(


        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_features: int
    ) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
        """集成方法特征选择"""
        # 获取多种方法的结果
        methods_results = {}

        # 统计方法
        try:
            _, stat_features, _ = self._statistical_select(X, y, target_features)
            methods_results['statistical'] = stat_features
        except Exception as e:
            logger.warning(f"集成方法中统计方法失败: {e}")

        # 模型方法
        try:
            _, model_features, _ = self._model_based_select(X, y, target_features)
            methods_results['model_based'] = model_features
        except Exception as e:
            logger.warning(f"集成方法中模型方法失败: {e}")

        # 包装方法
        try:
            _, wrapper_features, _ = self._wrapper_select(X, y, target_features)
            methods_results['wrapper'] = wrapper_features
        except Exception as e:
            logger.warning(f"集成方法中包装方法失败: {e}")

        # 投票选择特征
        feature_votes = {}
        for method, features in methods_results.items():
            for feature in features:
                if feature not in feature_votes:
                    feature_votes[feature] = 0
                feature_votes[feature] += 1

        # 选择得票最多的特征
        sorted_features = sorted(
            feature_votes.items(),
            key=lambda x: x[1],
            reverse=True
        )

        selected_features = [f[0] for f in sorted_features[:target_features]]

        return X[selected_features], selected_features, {
            'method': 'ensemble',
            'feature_votes': feature_votes,
            'methods_used': list(methods_results.keys())
        }

    def _evaluate_selection(self, X_selected: pd.DataFrame, y: pd.Series) -> float:
        """评估特征选择结果"""
        if X_selected.empty:
            return 0.0

        # 使用简单的交叉验证评估
        if self.task_type == "classification":
            estimator = RandomForestClassifier(
                n_estimators=50,
                random_state=self.random_state
            )
            scoring = 'accuracy'
        else:
            estimator = RandomForestRegressor(
                n_estimators=50,
                random_state=self.random_state
            )
            scoring = 'r2'

        try:
            scores = cross_val_score(
                estimator, X_selected, y,
                cv=min(3, len(y) // 10),  # 避免过少的折数
                scoring=scoring
            )
            return scores.mean()
        except Exception as e:
            logger.warning(f"评估特征选择失败: {e}")
            return 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'AutoFeatureSelector':
        """拟合特征选择器"""
        self.selected_features = []
        self.feature_importance = {}

        # 执行特征选择
        X_selected, features, info = self.select_features(X, y)

        self.selected_features = features
        self.selection_info = info
        self.is_fitted = True

        logger.info(f"特征选择器拟合完成，选择了 {len(features)} 个特征")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """转换数据"""
        if not self.is_fitted:
            raise ValueError("特征选择器尚未拟合")

        if not self.selected_features:
            return pd.DataFrame()

        return X[self.selected_features]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """拟合并转换"""
        self.fit(X, y)
        return self.transform(X)

    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        return self.feature_importance

    def save(self, filepath: Union[str, Path]) -> None:
        """保存特征选择器"""
        if not self.is_fitted:
            raise ValueError("特征选择器尚未拟合")

        data = {
            'selected_features': self.selected_features,
            'feature_importance': self.feature_importance,
            'selection_info': self.selection_info,
            'strategy': self.strategy,
            'task_type': self.task_type
        }

        joblib.dump(data, filepath)
        logger.info(f"特征选择器已保存到: {filepath}")

    def load(self, filepath: Union[str, Path]) -> None:
        """加载特征选择器"""
        data = joblib.load(filepath)

        self.selected_features = data['selected_features']
        self.feature_importance = data['feature_importance']
        self.selection_info = data['selection_info']
        self.strategy = data['strategy']
        self.task_type = data['task_type']
        self.is_fitted = True

        logger.info(f"特征选择器已从 {filepath} 加载")
