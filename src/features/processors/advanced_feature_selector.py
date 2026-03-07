"""
高级特征选择器
实现基于机器学习的特征选择、特征重要性评估和性能优化
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import joblib
from pathlib import Path

# 机器学习相关导入
try:
    from sklearn.feature_selection import (
        SelectKBest, RFECV, SelectFromModel,
        mutual_info_classif, mutual_info_regression,
        f_classif, f_regression
    )
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import Lasso, LogisticRegression
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
except ImportError as e:
    logging.warning(f"sklearn导入失败: {e}")

try:
    from boruta import BorutaPy
except ImportError:
    BorutaPy = None
    logging.warning("BorutaPy未安装，Boruta特征选择不可用")

logger = logging.getLogger(__name__)


class SelectionMethod(Enum):

    """特征选择方法枚举"""
    K_BEST = "k_best"
    RFECV = "rfecv"
    SELECT_FROM_MODEL = "select_from_model"
    MUTUAL_INFO = "mutual_info"
    PERMUTATION = "permutation"
    BORUTA = "boruta"
    CORRELATION = "correlation"
    VARIANCE = "variance"
    PCA = "pca"


class TaskType(Enum):

    """任务类型枚举"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


@dataclass
class FeatureImportance:

    """特征重要性信息"""
    feature_name: str
    importance_score: float
    importance_rank: int
    selection_method: str
    p_value: Optional[float] = None
    std_score: Optional[float] = None


@dataclass
class SelectionResult:

    """特征选择结果"""
    selected_features: List[str]
    feature_importances: List[FeatureImportance]
    selection_method: str
    selection_time: float
    performance_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedFeatureSelector:

    """高级特征选择器"""

    def __init__(


        self,
        task_type: TaskType = TaskType.REGRESSION,
        random_state: int = 42,
        n_jobs: int = -1,
        cache_dir: Optional[str] = None
    ):
        """
        初始化高级特征选择器

        Args:
            task_type: 任务类型
            random_state: 随机种子
            n_jobs: 并行作业数
            cache_dir: 缓存目录
        """
        self.task_type = task_type
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # 初始化选择器
        self.selectors = {}
        self.results_cache = {}

        # 性能监控
        self.performance_history = []

        logger.info(f"高级特征选择器初始化完成，任务类型: {task_type.value}")

    def select_features(


        self,
        X: pd.DataFrame,
        y: pd.Series,
        methods: List[Union[str, SelectionMethod]] = None,
        max_features: Optional[int] = None,
        min_features: int = 3,
        cv_folds: int = 5,
        scoring: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict[str, SelectionResult]:
        """
        执行特征选择

        Args:
            X: 特征数据
            y: 目标变量
            methods: 选择方法列表
            max_features: 最大特征数
            min_features: 最小特征数
            cv_folds: 交叉验证折数
            scoring: 评分方法
            use_cache: 是否使用缓存

        Returns:
            特征选择结果字典
        """
        if methods is None:
            methods = [SelectionMethod.K_BEST, SelectionMethod.RFECV, SelectionMethod.MUTUAL_INFO]

        # 转换方法名称
        methods = [self._normalize_method(method) for method in methods]

        # 验证输入
        self._validate_input(X, y)

        # 检查缓存
        if use_cache and self.cache_dir:
            cached_result = self._load_from_cache(X, y, methods)
            if cached_result:
                logger.info("使用缓存的特征选择结果")
                return cached_result

        start_time = time.time()
        results = {}

        # 并行执行特征选择
        with ThreadPoolExecutor(max_workers=self.n_jobs if self.n_jobs > 0 else None) as executor:
            future_to_method = {}

            for method in methods:
                future = executor.submit(
                    self._select_with_method,
                    X, y, method, max_features, min_features, cv_folds, scoring
                )
                future_to_method[future] = method

            # 收集结果
            for future in as_completed(future_to_method):
                method = future_to_method[future]
                try:
                    result = future.result()
                    results[method.value] = result
                    logger.info(f"方法 {method.value} 完成，选择了 {len(result.selected_features)} 个特征")
                except Exception as e:
                    logger.error(f"方法 {method.value} 执行失败: {e}")

        # 保存缓存
        if use_cache and self.cache_dir:
            self._save_to_cache(X, y, methods, results)

        execution_time = time.time() - start_time
        self._record_performance(methods, execution_time, results)

        return results

    def _select_with_method(


        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: SelectionMethod,
        max_features: Optional[int],
        min_features: int,
        cv_folds: int,
        scoring: Optional[str]
    ) -> SelectionResult:
        """使用指定方法进行特征选择"""
        start_time = time.time()

        if method == SelectionMethod.K_BEST:
            result = self._select_k_best(X, y, max_features)
        elif method == SelectionMethod.RFECV:
            result = self._select_rfecv(X, y, min_features, cv_folds, scoring)
        elif method == SelectionMethod.SELECT_FROM_MODEL:
            result = self._select_from_model(X, y, max_features)
        elif method == SelectionMethod.MUTUAL_INFO:
            result = self._select_mutual_info(X, y, max_features)
        elif method == SelectionMethod.PERMUTATION:
            result = self._select_permutation(X, y, max_features)
        elif method == SelectionMethod.BORUTA:
            result = self._select_boruta(X, y)
        elif method == SelectionMethod.CORRELATION:
            result = self._select_correlation(X, y, max_features)
        elif method == SelectionMethod.VARIANCE:
            result = self._select_variance(X, y, max_features)
        elif method == SelectionMethod.PCA:
            result = self._select_pca(X, y, max_features)
        else:
            raise ValueError(f"不支持的选择方法: {method}")

        result.selection_time = time.time() - start_time
        result.selection_method = method.value

        return result

    def _select_k_best(self, X: pd.DataFrame, y: pd.Series, max_features: Optional[int]) -> SelectionResult:
        """K最佳特征选择"""
        if max_features is None:
            max_features = min(20, X.shape[1])

        # 选择评分函数
        if self.task_type == TaskType.CLASSIFICATION:
            score_func = f_classif
        else:
            score_func = f_regression

        selector = SelectKBest(score_func=score_func, k=max_features)
        selector.fit(X, y)

        # 获取选择的特征
        selected_indices = selector.get_support(indices=True)
        selected_features = [X.columns[i] for i in selected_indices]

        # 计算重要性
        scores = selector.scores_
        p_values = selector.pvalues_

        importances = []
        for i, feature in enumerate(X.columns):
            if i in selected_indices:
                importances.append(FeatureImportance(
                    feature_name=feature,
                    importance_score=scores[i],
                    importance_rank=np.argsort(scores)[::-1].tolist().index(i) + 1,
                    selection_method="k_best",
                    p_value=p_values[i] if p_values is not None else None
                ))

        return SelectionResult(
            selected_features=selected_features,
            feature_importances=importances,
            selection_method="k_best",
            selection_time=0.0,
            metadata={"scores": scores, "p_values": p_values}
        )

    def _select_rfecv(


        self,
        X: pd.DataFrame,
        y: pd.Series,
        min_features: int,
        cv_folds: int,
        scoring: Optional[str]
    ) -> SelectionResult:
        """递归特征消除交叉验证"""
        start_time = time.time()

        # 对于大数据集，使用更激进的轻量配置
        if X.shape[0] > 300 or X.shape[1] > 25:
            # 大幅减少树的数量和交叉验证折数
            n_estimators = 25
            cv_folds = 2
            # 完全禁用并行，避免死锁
            n_jobs = 1
            # 使用更简单的模型
            use_linear_model = True
        elif X.shape[0] > 500 or X.shape[1] > 30:
            # 中等优化
            n_estimators = 50
            cv_folds = min(cv_folds, 3)
            n_jobs = 1
            use_linear_model = False
        else:
            # 小数据集使用标准配置
            n_estimators = 100
            n_jobs = 1  # 避免嵌套并行
            use_linear_model = False

        # 选择基础模型
        if use_linear_model:
            # 使用线性模型，计算更快
            if self.task_type == TaskType.CLASSIFICATION:
                estimator = LogisticRegression(random_state=self.random_state, max_iter=1000)
            else:
                estimator = Lasso(random_state=self.random_state, max_iter=1000)
        else:
            if self.task_type == TaskType.CLASSIFICATION:
                estimator = RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=self.random_state,
                    n_jobs=1,  # 避免嵌套并行
                    max_depth=10,  # 限制树的深度
                    min_samples_split=10,  # 增加分裂所需的最小样本数
                    min_samples_leaf=5  # 增加叶节点最小样本数
                )
            else:
                estimator = RandomForestRegressor(
                    n_estimators=n_estimators,
                    random_state=self.random_state,
                    n_jobs=1,  # 避免嵌套并行
                    max_depth=10,  # 限制树的深度
                    min_samples_split=10,  # 增加分裂所需的最小样本数
                    min_samples_leaf=5  # 增加叶节点最小样本数
                )

        selector = RFECV(
            estimator=estimator,
            min_features_to_select=min_features,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=n_jobs,
            step=1  # 每次移除一个特征
        )

        selector.fit(X, y)

        # 获取选择的特征
        selected_features = list(selector.get_feature_names_out())

        # 计算重要性
        importances = []
        for i, feature in enumerate(X.columns):
            if feature in selected_features:
                importances.append(FeatureImportance(
                    feature_name=feature,
                    importance_score=selector.ranking_[i],
                    importance_rank=selector.ranking_[i],
                    selection_method="rfecv"
                ))

        selection_time = time.time() - start_time

        return SelectionResult(
            selected_features=selected_features,
            feature_importances=importances,
            selection_method="rfecv",
            selection_time=selection_time,
            metadata={"ranking": selector.ranking_, "cv_scores": selector.cv_results_}
        )

    def _select_from_model(self, X: pd.DataFrame, y: pd.Series, max_features: Optional[int]) -> SelectionResult:
        """基于模型的特征选择"""
        # 选择模型
        if self.task_type == TaskType.CLASSIFICATION:
            estimator = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        else:
            estimator = RandomForestRegressor(n_estimators=100, random_state=self.random_state)

        selector = SelectFromModel(
            estimator=estimator,
            max_features=max_features,
            prefit=False
        )

        selector.fit(X, y)

        # 获取选择的特征
        selected_indices = selector.get_support(indices=True)
        selected_features = [X.columns[i] for i in selected_indices]

        # 计算重要性
        importances = []
        if hasattr(estimator, 'feature_importances_'):
            feature_importances = estimator.feature_importances_
            for i, feature in enumerate(X.columns):
                if i in selected_indices:
                    importances.append(FeatureImportance(
                        feature_name=feature,
                        importance_score=feature_importances[i],
                        importance_rank=np.argsort(feature_importances)[::-1].tolist().index(i) + 1,
                        selection_method="select_from_model"
                    ))

        return SelectionResult(
            selected_features=selected_features,
            feature_importances=importances,
            selection_method="select_from_model",
            selection_time=0.0
        )

    def _select_mutual_info(self, X: pd.DataFrame, y: pd.Series, max_features: Optional[int]) -> SelectionResult:
        """基于互信息的特征选择"""
        if max_features is None:
            max_features = min(20, X.shape[1])

        # 选择互信息函数
        if self.task_type == TaskType.CLASSIFICATION:
            mi_func = mutual_info_classif
        else:
            mi_func = mutual_info_regression

        # 计算互信息
        mi_scores = mi_func(X, y, random_state=self.random_state)

        # 选择top特征
        top_indices = np.argsort(mi_scores)[-max_features:]
        selected_features = [X.columns[i] for i in top_indices]

        # 计算重要性
        importances = []
        for i, feature in enumerate(X.columns):
            if feature in selected_features:
                importances.append(FeatureImportance(
                    feature_name=feature,
                    importance_score=mi_scores[i],
                    importance_rank=np.argsort(mi_scores)[::-1].tolist().index(i) + 1,
                    selection_method="mutual_info"
                ))

        return SelectionResult(
            selected_features=selected_features,
            feature_importances=importances,
            selection_method="mutual_info",
            selection_time=0.0,
            metadata={"mi_scores": mi_scores}
        )

    def _select_permutation(self, X: pd.DataFrame, y: pd.Series, max_features: Optional[int]) -> SelectionResult:
        """基于排列重要性的特征选择"""
        if max_features is None:
            max_features = min(20, X.shape[1])

        # 训练基础模型
        if self.task_type == TaskType.CLASSIFICATION:
            model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)

        model.fit(X, y)

        # 计算排列重要性
        result = permutation_importance(
            model, X, y,
            n_repeats=10,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )

        # 选择top特征
        top_indices = np.argsort(result.importances_mean)[-max_features:]
        selected_features = [X.columns[i] for i in top_indices]

        # 计算重要性
        importances = []
        for i, feature in enumerate(X.columns):
            if feature in selected_features:
                importances.append(FeatureImportance(
                    feature_name=feature,
                    importance_score=result.importances_mean[i],
                    importance_rank=np.argsort(result.importances_mean)[::-1].tolist().index(i) + 1,
                    selection_method="permutation",
                    std_score=result.importances_std[i]
                ))

        return SelectionResult(
            selected_features=selected_features,
            feature_importances=importances,
            selection_method="permutation",
            selection_time=0.0,
            metadata={"importances_mean": result.importances_mean,
                      "importances_std": result.importances_std}
        )

    def _select_boruta(self, X: pd.DataFrame, y: pd.Series) -> SelectionResult:
        """Boruta特征选择"""
        if BorutaPy is None:
            raise ImportError("BorutaPy未安装，无法使用Boruta特征选择")

        # 选择基础模型
        if self.task_type == TaskType.CLASSIFICATION:
            base_model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        else:
            base_model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)

        # 创建Boruta选择器
        selector = BorutaPy(
            estimator=base_model,
            n_estimators='auto',
            max_iter=100,
            perc=90,
            random_state=self.random_state,
            verbose=0
        )

        selector.fit(X.values, y.values)

        # 获取选择的特征
        selected_indices = np.where(selector.support_)[0]
        selected_features = [X.columns[i] for i in selected_indices]

        # 计算重要性
        importances = []
        for i, feature in enumerate(X.columns):
            if i in selected_indices:
                importances.append(FeatureImportance(
                    feature_name=feature,
                    importance_score=selector.ranking_[i],
                    importance_rank=selector.ranking_[i],
                    selection_method="boruta"
                ))

        return SelectionResult(
            selected_features=selected_features,
            feature_importances=importances,
            selection_method="boruta",
            selection_time=0.0,
            metadata={"ranking": selector.ranking_, "support": selector.support_}
        )

    def _select_correlation(self, X: pd.DataFrame, y: pd.Series, max_features: Optional[int]) -> SelectionResult:
        """基于相关性的特征选择"""
        if max_features is None:
            max_features = min(20, X.shape[1])

        # 计算与目标变量的相关性
        correlations = []
        for col in X.columns:
            if self.task_type == TaskType.CLASSIFICATION:
                # 对于分类任务，使用点双列相关系数
                correlation = np.corrcoef(X[col], y)[0, 1]
            else:
                # 对于回归任务，使用皮尔逊相关系数
                correlation = X[col].corr(y)

            if pd.isna(correlation):
                correlation = 0.0

            correlations.append(abs(correlation))

        # 选择top特征
        top_indices = np.argsort(correlations)[-max_features:]
        selected_features = [X.columns[i] for i in top_indices]

        # 计算重要性
        importances = []
        for i, feature in enumerate(X.columns):
            if feature in selected_features:
                importances.append(FeatureImportance(
                    feature_name=feature,
                    importance_score=correlations[i],
                    importance_rank=np.argsort(correlations)[::-1].tolist().index(i) + 1,
                    selection_method="correlation"
                ))

        return SelectionResult(
            selected_features=selected_features,
            feature_importances=importances,
            selection_method="correlation",
            selection_time=0.0,
            metadata={"correlations": correlations}
        )

    def _select_variance(self, X: pd.DataFrame, y: pd.Series, max_features: Optional[int]) -> SelectionResult:
        """基于方差的特征选择"""
        if max_features is None:
            max_features = min(20, X.shape[1])

        # 计算方差
        variances = X.var()

        # 选择top特征
        top_indices = np.argsort(variances)[-max_features:]
        selected_features = [X.columns[i] for i in top_indices]

        # 计算重要性
        importances = []
        for i, feature in enumerate(X.columns):
            if feature in selected_features:
                importances.append(FeatureImportance(
                    feature_name=feature,
                    importance_score=variances[i],
                    importance_rank=np.argsort(variances)[::-1].tolist().index(i) + 1,
                    selection_method="variance"
                ))

        return SelectionResult(
            selected_features=selected_features,
            feature_importances=importances,
            selection_method="variance",
            selection_time=0.0,
            metadata={"variances": variances}
        )

    def _select_pca(self, X: pd.DataFrame, y: pd.Series, max_features: Optional[int]) -> SelectionResult:
        """基于PCA的特征选择"""
        if max_features is None:
            max_features = min(20, X.shape[1])

        # 标准化数据
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 应用PCA
        pca = PCA(n_components=max_features, random_state=self.random_state)
        pca.fit(X_scaled)

        # 选择解释方差比例最高的特征
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)

        # 选择解释95 % 方差所需的最少组件数
        n_components = np.argmax(cumulative_variance >= 0.95) + 1
        n_components = min(n_components, max_features)

        # 重新拟合PCA
        pca = PCA(n_components=n_components, random_state=self.random_state)
        pca.fit(X_scaled)

        # 获取特征重要性（基于载荷）
        loadings = pca.components_
        feature_importance = np.sum(np.abs(loadings), axis=0)

        # 选择top特征
        top_indices = np.argsort(feature_importance)[-max_features:]
        selected_features = [X.columns[i] for i in top_indices]

        # 计算重要性
        importances = []
        for i, feature in enumerate(X.columns):
            if feature in selected_features:
                importances.append(FeatureImportance(
                    feature_name=feature,
                    importance_score=feature_importance[i],
                    importance_rank=np.argsort(feature_importance)[::-1].tolist().index(i) + 1,
                    selection_method="pca"
                ))

        return SelectionResult(
            selected_features=selected_features,
            feature_importances=importances,
            selection_method="pca",
            selection_time=0.0,
            metadata={
                "explained_variance_ratio": pca.explained_variance_ratio_,
                "n_components": n_components,
                "loadings": loadings
            }
        )

    def _calculate_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """训练轻量模型获取特征重要性"""
        if self.task_type == TaskType.CLASSIFICATION:
            model = RandomForestClassifier(
                n_estimators=64,
                random_state=self.random_state,
                max_depth=8,
                n_jobs=1
            )
        else:
            model = RandomForestRegressor(
                n_estimators=64,
                random_state=self.random_state,
                max_depth=8,
                n_jobs=1
            )

        model.fit(X, y)
        importance_scores = getattr(model, "feature_importances_", np.zeros(X.shape[1]))
        return {
            feature: float(max(0.0, score))
            for feature, score in zip(X.columns, importance_scores)
        }

    def _calculate_permutation_importance(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        n_repeats: int = 5
    ) -> Dict[str, float]:
        """基于排列的重要性估计"""
        result = permutation_importance(
            model,
            X,
            y,
            n_repeats=n_repeats,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        return {
            feature: float(max(0.0, score))
            for feature, score in zip(X.columns, result.importances_mean)
        }

    def _record_performance(
        self,
        methods: List[SelectionMethod],
        execution_time: float,
        results: Dict[str, SelectionResult]
    ) -> None:
        """记录一次特征选择执行的性能信息"""
        record = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'execution_time': execution_time,
            'methods_used': [method.value for method in methods],
            'selected_feature_counts': {
                method: len(result.selected_features)
                for method, result in results.items()
            }
        }
        self.performance_history.append(record)

    def _normalize_method(self, method: Union[str, SelectionMethod]) -> SelectionMethod:
        """标准化方法名称"""
        if isinstance(method, str):
            try:
                return SelectionMethod(method.lower())
            except ValueError:
                raise ValueError(f"不支持的选择方法: {method}")
        return method

    def _validate_input(self, X: pd.DataFrame, y: pd.Series):
        """验证输入数据"""
        if X.empty:
            raise ValueError("特征数据为空")

        if y.empty:
            raise ValueError("目标变量为空")

        if len(X) != len(y):
            raise ValueError("特征数据和目标变量长度不匹配")

        if X.isnull().any().any():
            logger.warning("特征数据包含缺失值，建议先进行数据清洗")

    def _load_from_cache(self, X: pd.DataFrame, y: pd.Series, methods: List[SelectionMethod]) -> Optional[Dict[str, SelectionResult]]:
        """从缓存加载结果"""
        if not self.cache_dir:
            return None

        cache_key = self._generate_cache_key(X, y, methods)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if cache_file.exists():
            try:
                return joblib.load(cache_file)
            except Exception as e:
                logger.warning(f"加载缓存失败: {e}")

        return None

    def _save_to_cache(self, X: pd.DataFrame, y: pd.Series, methods: List[SelectionMethod], results: Dict[str, SelectionResult]):
        """保存结果到缓存"""
        if not self.cache_dir:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = self._generate_cache_key(X, y, methods)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            joblib.dump(results, cache_file)
            logger.info(f"特征选择结果已缓存到: {cache_file}")
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")

    def _generate_cache_key(self, X: pd.DataFrame, y: pd.Series, methods: List[SelectionMethod]) -> str:
        """生成缓存键"""
        import hashlib

        # 创建数据摘要
        data_hash = hashlib.md5()
        data_hash.update(X.to_string().encode())
        data_hash.update(y.to_string().encode())

        # 创建方法摘要
        methods_str = "_".join([method.value for method in methods])

        return f"feature_selection_{data_hash.hexdigest()}_{methods_str}"

    def get_feature_importance_summary(self, results: Dict[str, SelectionResult]) -> pd.DataFrame:
        """获取特征重要性汇总"""
        summary_data = []

        for method, result in results.items():
            for importance in result.feature_importances:
                summary_data.append({
                    'feature_name': importance.feature_name,
                    'importance_score': importance.importance_score,
                    'importance_rank': importance.importance_rank,
                    'selection_method': method,
                    'p_value': importance.p_value,
                    'std_score': importance.std_score
                })

        df = pd.DataFrame(summary_data)

        if not df.empty:
            # 计算综合排名
            df['avg_rank'] = df.groupby('feature_name')['importance_rank'].transform('mean')
            df['avg_score'] = df.groupby('feature_name')['importance_score'].transform('mean')
            df = df.sort_values(['avg_rank', 'avg_score'])

        return df

    def plot_feature_importance(self, results: Dict[str, SelectionResult], top_n: int = 20) -> None:
        """绘制特征重要性图"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("matplotlib或seaborn未安装，无法绘制图表")
            return

        summary_df = self.get_feature_importance_summary(results)

        if summary_df.empty:
            logger.warning("没有特征重要性数据可绘制")
            return

        # 获取top特征
        top_features = summary_df.head(top_n)

        # 创建图表
        plt.figure(figsize=(12, 8))

        # 绘制平均重要性得分
        plt.subplot(2, 1, 1)
        sns.barplot(data=top_features, x='avg_score', y='feature_name')
        plt.title(f'Top {top_n} Features - Average Importance Score')
        plt.xlabel('Average Importance Score')

        # 绘制平均排名
        plt.subplot(2, 1, 2)
        sns.barplot(data=top_features, x='avg_rank', y='feature_name')
        plt.title(f'Top {top_n} Features - Average Rank')
        plt.xlabel('Average Rank')

        plt.tight_layout()
        plt.show()

    def evaluate_selection_performance(


        self,
        X: pd.DataFrame,
        y: pd.Series,
        results: Dict[str, SelectionResult],
        cv_folds: int = 5
    ) -> Dict[str, float]:
        """评估特征选择性能"""
        performance_scores = {}

        for method, result in results.items():
            if not result.selected_features:
                performance_scores[method] = 0.0
                continue

            # 选择特征
            X_selected = X[result.selected_features]

            # 训练模型
            if self.task_type == TaskType.CLASSIFICATION:
                model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
                scoring = 'accuracy'
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
                scoring = 'neg_mean_squared_error'

            # 交叉验证
            scores = cross_val_score(
                model, X_selected, y,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=self.n_jobs
            )

            # 计算平均分数
            if scoring == 'neg_mean_squared_error':
                performance_scores[method] = -scores.mean()
            else:
                performance_scores[method] = scores.mean()

        return performance_scores
