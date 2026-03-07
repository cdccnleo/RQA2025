"""
特征选择算法测试
测试特征重要性评估、特征选择策略、降维算法等
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


class FeatureSelectionAlgorithms:
    """特征选择算法实现"""

    @staticmethod
    def calculate_feature_importance(X: pd.DataFrame, y: pd.Series,
                                   method: str = 'random_forest') -> pd.Series:
        """计算特征重要性"""
        if method == 'random_forest':
            # 使用随机森林计算特征重要性
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            importance = rf.feature_importances_

        elif method == 'correlation':
            # 使用皮尔逊相关系数
            correlations = X.corrwith(y).abs()
            importance = correlations.values

        elif method == 'variance':
            # 使用方差
            variances = X.var()
            max_var = variances.max()
            importance = variances.values / max_var if max_var > 0 else np.ones(len(variances))

        else:
            # 默认平均重要性
            importance = np.ones(X.shape[1]) / X.shape[1]

        return pd.Series(importance, index=X.columns, name='importance')

    @staticmethod
    def select_features_by_importance(X: pd.DataFrame, importance: pd.Series,
                                    threshold: float = 0.01) -> list:
        """基于重要性选择特征"""
        selected_features = importance[importance >= threshold].index.tolist()
        return selected_features

    @staticmethod
    def select_features_by_count(X: pd.DataFrame, importance: pd.Series,
                               top_k: int = 10) -> list:
        """选择前K个最重要的特征"""
        sorted_importance = importance.sort_values(ascending=False)
        selected_features = sorted_importance.head(min(top_k, len(sorted_importance))).index.tolist()
        return selected_features

    @staticmethod
    def calculate_feature_correlation_matrix(X: pd.DataFrame) -> pd.DataFrame:
        """计算特征相关性矩阵"""
        return X.corr()

    @staticmethod
    def remove_highly_correlated_features(X: pd.DataFrame, threshold: float = 0.95) -> list:
        """移除高度相关的特征"""
        corr_matrix = X.corr().abs()

        # 获取上三角矩阵
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # 找到高度相关的特征对
        to_drop = []
        for column in upper.columns:
            if column not in to_drop:
                correlated_features = upper[column][upper[column] > threshold].index.tolist()
                if correlated_features:
                    # 保留第一个，移除其他的
                    to_drop.extend(correlated_features)

        # 返回保留的特征
        remaining_features = [col for col in X.columns if col not in to_drop]
        return remaining_features

    @staticmethod
    def calculate_feature_stability(X: pd.DataFrame, y: pd.Series,
                                  n_bootstraps: int = 10) -> pd.Series:
        """计算特征稳定性（bootstrap方法）"""
        np.random.seed(42)
        n_samples = len(X)

        importance_scores = []

        for _ in range(n_bootstraps):
            # Bootstrap采样
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X.iloc[indices]
            y_boot = y.iloc[indices]

            # 计算重要性
            importance = FeatureSelectionAlgorithms.calculate_feature_importance(X_boot, y_boot)
            importance_scores.append(importance)

        # 计算稳定性（标准差的倒数）
        importance_df = pd.DataFrame(importance_scores)
        stability = 1 / (importance_df.std() + 1e-8)  # 避免除零

        return pd.Series(stability.mean(), index=X.columns, name='stability')

    @staticmethod
    def recursive_feature_elimination(X: pd.DataFrame, y: pd.Series,
                                    n_features_to_select: int = 5) -> list:
        """递归特征消除"""
        remaining_features = list(X.columns)
        n_features = len(remaining_features)

        while n_features > n_features_to_select:
            # 计算当前特征的重要性
            importance = FeatureSelectionAlgorithms.calculate_feature_importance(
                X[remaining_features], y
            )

            # 移除最不重要的特征
            least_important = importance.idxmin()
            remaining_features.remove(least_important)
            n_features = len(remaining_features)

        return remaining_features

    @staticmethod
    def forward_feature_selection(X: pd.DataFrame, y: pd.Series,
                                n_features_to_select: int = 5) -> list:
        """前向特征选择"""
        selected_features = []
        remaining_features = list(X.columns)

        for _ in range(min(n_features_to_select, len(X.columns))):
            best_score = -np.inf
            best_feature = None

            # 尝试添加每个剩余特征
            for feature in remaining_features:
                candidate_features = selected_features + [feature]
                X_candidate = X[candidate_features]

                # 使用交叉验证分数作为评估标准
                scores = []
                for _ in range(3):  # 简化的交叉验证
                    # 随机分割数据
                    n_samples = len(X_candidate)
                    indices = np.random.permutation(n_samples)
                    train_size = int(0.8 * n_samples)

                    X_train = X_candidate.iloc[indices[:train_size]]
                    X_test = X_candidate.iloc[indices[train_size:]]
                    y_train = y.iloc[indices[:train_size]]
                    y_test = y.iloc[indices[train_size:]]

                    # 训练模型
                    rf = RandomForestClassifier(n_estimators=50, random_state=42)
                    rf.fit(X_train, y_train)

                    # 计算分数
                    score = rf.score(X_test, y_test)
                    scores.append(score)

                avg_score = np.mean(scores)

                if avg_score > best_score:
                    best_score = avg_score
                    best_feature = feature

            if best_feature:
                selected_features.append(best_feature)
                remaining_features.remove(best_feature)
            else:
                break

        return selected_features


class TestFeatureSelectionAlgorithms:
    """特征选择算法测试"""

    def setup_method(self):
        """测试前准备"""
        self.fs = FeatureSelectionAlgorithms()

        # 生成测试数据
        np.random.seed(42)
        n_samples, n_features = 1000, 20

        # 生成分类数据集
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=10,
            n_redundant=5,
            n_clusters_per_class=1,
            random_state=42
        )

        self.X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
        self.y = pd.Series(y, name='target')

        # 添加一些相关特征
        self.X['correlated_1'] = self.X['feature_0'] + np.random.normal(0, 0.1, n_samples)
        self.X['correlated_2'] = self.X['feature_1'] * 1.1 + np.random.normal(0, 0.1, n_samples)

    def test_feature_importance_calculation(self):
        """测试特征重要性计算"""
        importance = self.fs.calculate_feature_importance(self.X, self.y, method='random_forest')

        # 验证基本属性
        assert isinstance(importance, pd.Series)
        assert len(importance) == self.X.shape[1]
        assert importance.index.equals(self.X.columns)
        assert importance.name == 'importance'

        # 验证重要性值范围
        assert (importance >= 0).all()
        assert (importance <= 1).all()
        assert abs(importance.sum() - 1.0) < 1e-10  # 随机森林重要性应该归一化

    def test_different_importance_methods(self):
        """测试不同的重要性计算方法"""
        methods = ['random_forest', 'correlation', 'variance']

        for method in methods:
            importance = self.fs.calculate_feature_importance(self.X, self.y, method=method)

            assert isinstance(importance, pd.Series)
            assert len(importance) == self.X.shape[1]
            assert (importance >= 0).all()

            if method == 'random_forest':
                assert abs(importance.sum() - 1.0) < 1e-10
            elif method == 'variance':
                assert importance.max() <= 1.0  # 方差方法应该归一化

    def test_feature_selection_by_importance(self):
        """测试基于重要性选择特征"""
        importance = self.fs.calculate_feature_importance(self.X, self.y)

        # 测试阈值选择
        selected_threshold = self.fs.select_features_by_importance(self.X, importance, threshold=0.03)
        assert isinstance(selected_threshold, list)
        assert len(selected_threshold) <= len(self.X.columns)

        # 验证所有选中特征的重要性都超过阈值
        if selected_threshold:
            min_importance = importance[selected_threshold].min()
            assert min_importance >= 0.03

    def test_feature_selection_by_count(self):
        """测试基于数量选择特征"""
        importance = self.fs.calculate_feature_importance(self.X, self.y)

        # 测试前K个特征选择
        top_k = 5
        selected_top = self.fs.select_features_by_count(self.X, importance, top_k=top_k)

        assert isinstance(selected_top, list)
        assert len(selected_top) == min(top_k, len(self.X.columns))

        # 验证选择的特征是重要性最高的
        if len(selected_top) > 1:
            selected_importance = importance[selected_top]
            unselected_importance = importance[~importance.index.isin(selected_top)]

            assert selected_importance.min() >= unselected_importance.max()

    def test_correlation_matrix_calculation(self):
        """测试相关性矩阵计算"""
        corr_matrix = self.fs.calculate_feature_correlation_matrix(self.X)

        # 验证基本属性
        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape == (self.X.shape[1], self.X.shape[1])
        assert corr_matrix.index.equals(self.X.columns)
        assert corr_matrix.columns.equals(self.X.columns)

        # 验证对角线为1
        assert np.allclose(corr_matrix.values.diagonal(), 1.0)

        # 验证对称性
        assert np.allclose(corr_matrix.values, corr_matrix.values.T)

        # 验证相关性值范围
        assert (corr_matrix >= -1).all().all()
        assert (corr_matrix <= 1).all().all()

    def test_highly_correlated_feature_removal(self):
        """测试高度相关特征移除"""
        # 确保我们有高度相关的特征
        remaining_features = self.fs.remove_highly_correlated_features(self.X, threshold=0.9)

        assert isinstance(remaining_features, list)
        assert len(remaining_features) <= self.X.shape[1]

        # 验证剩余特征的相关性都低于阈值
        if len(remaining_features) > 1:
            remaining_corr = self.X[remaining_features].corr().abs()
            upper_triangle = remaining_corr.where(
                np.triu(np.ones(remaining_corr.shape), k=1).astype(bool)
            )

            max_corr = upper_triangle.max().max()
            assert max_corr < 0.9

    def test_feature_stability_calculation(self):
        """测试特征稳定性计算"""
        stability = self.fs.calculate_feature_stability(self.X, self.y, n_bootstraps=5)

        # 验证基本属性
        assert isinstance(stability, pd.Series)
        assert len(stability) == self.X.shape[1]
        assert stability.index.equals(self.X.columns)
        assert stability.name == 'stability'

        # 稳定性应该是正值
        assert (stability > 0).all()

    def test_recursive_feature_elimination(self):
        """测试递归特征消除"""
        n_features_to_select = 5
        selected_features = self.fs.recursive_feature_elimination(
            self.X, self.y, n_features_to_select=n_features_to_select
        )

        assert isinstance(selected_features, list)
        assert len(selected_features) == n_features_to_select
        assert all(feature in self.X.columns for feature in selected_features)

    def test_forward_feature_selection(self):
        """测试前向特征选择"""
        n_features_to_select = 3
        selected_features = self.fs.forward_feature_selection(
            self.X, self.y, n_features_to_select=n_features_to_select
        )

        assert isinstance(selected_features, list)
        assert len(selected_features) == n_features_to_select
        assert all(feature in self.X.columns for feature in selected_features)

        # 验证选择的特征各不相同
        assert len(set(selected_features)) == len(selected_features)

    def test_feature_selection_performance(self):
        """测试特征选择算法性能"""
        import time

        # 测试不同算法的性能
        algorithms = [
            ('importance_calculation', lambda: self.fs.calculate_feature_importance(self.X, self.y)),
            ('correlation_matrix', lambda: self.fs.calculate_feature_correlation_matrix(self.X)),
            ('correlation_removal', lambda: self.fs.remove_highly_correlated_features(self.X)),
        ]

        for algo_name, algo_func in algorithms:
            start_time = time.time()
            result = algo_func()
            elapsed = time.time() - start_time

            # 性能应该在合理范围内（< 5秒）
            assert elapsed < 5.0, f"{algo_name}算法执行时间过长: {elapsed:.3f}s"
            assert result is not None, f"{algo_name}算法返回结果为空"

    def test_feature_selection_robustness(self):
        """测试特征选择算法鲁棒性"""
        # 测试空数据
        empty_X = pd.DataFrame()
        empty_y = pd.Series([], dtype=float)

        # 应该优雅处理空数据
        try:
            importance = self.fs.calculate_feature_importance(empty_X, empty_y)
            assert len(importance) == 0
        except Exception:
            # 某些算法可能不支持空数据，这是可以接受的
            pass

        # 测试单特征数据
        single_X = self.X.iloc[:, :1]
        importance_single = self.fs.calculate_feature_importance(single_X, self.y)

        assert len(importance_single) == 1
        assert importance_single.iloc[0] >= 0

    def test_feature_selection_consistency(self):
        """测试特征选择算法一致性"""
        # 多次运行应该得到一致的结果
        np.random.seed(42)

        results = []
        for _ in range(3):
            importance = self.fs.calculate_feature_importance(self.X, self.y, method='random_forest')
            results.append(importance)

        # 计算结果的一致性
        result_df = pd.DataFrame(results)
        std_dev = result_df.std()

        # 标准差应该相对较小（由于设置了随机种子）
        mean_std = std_dev.mean()
        assert mean_std < 0.05, f"算法结果不一致，平均标准差: {mean_std:.4f}"

    def test_comprehensive_feature_selection_pipeline(self):
        """测试综合特征选择流水线"""
        # 1. 计算特征重要性
        importance = self.fs.calculate_feature_importance(self.X, self.y)

        # 2. 基于重要性选择特征
        important_features = self.fs.select_features_by_count(self.X, importance, top_k=10)

        # 3. 移除高度相关特征
        X_important = self.X[important_features]
        uncorrelated_features = self.fs.remove_highly_correlated_features(X_important, threshold=0.8)

        # 4. 最终选择的特征
        final_features = uncorrelated_features

        # 验证流水线结果
        assert isinstance(final_features, list)
        assert len(final_features) > 0
        assert len(final_features) <= 10  # 不超过初始选择的特征数
        assert all(feature in self.X.columns for feature in final_features)

        # 验证最终特征的相关性
        if len(final_features) > 1:
            final_corr = self.X[final_features].corr().abs()
            upper_triangle = final_corr.where(
                np.triu(np.ones(final_corr.shape), k=1).astype(bool)
            )
            max_corr = upper_triangle.max().max()
            assert max_corr < 0.8, f"最终特征仍有高度相关性: {max_corr:.3f}"

    def test_feature_selection_with_different_data_types(self):
        """测试不同数据类型的特征选择"""
        # 创建混合数据类型的DataFrame
        mixed_data = pd.DataFrame({
            'numeric_1': np.random.normal(0, 1, 100),
            'numeric_2': np.random.normal(5, 2, 100),
            'categorical_1': np.random.choice(['A', 'B', 'C'], 100),
            'categorical_2': np.random.choice([1, 2, 3], 100),
        })

        # 将分类变量转换为数值
        mixed_data_encoded = pd.get_dummies(mixed_data, columns=['categorical_1', 'categorical_2'])

        # 创建目标变量
        y_mixed = pd.Series(np.random.choice([0, 1], 100))

        # 测试特征选择
        importance_mixed = self.fs.calculate_feature_importance(mixed_data_encoded, y_mixed)

        assert isinstance(importance_mixed, pd.Series)
        assert len(importance_mixed) == mixed_data_encoded.shape[1]
        assert (importance_mixed >= 0).all()

    def test_feature_selection_edge_cases(self):
        """测试特征选择边界情况"""
        # 测试只有一个特征的情况
        single_feature_X = self.X.iloc[:, :1]
        importance_single = self.fs.calculate_feature_importance(single_feature_X, self.y)

        assert len(importance_single) == 1
        assert importance_single.iloc[0] >= 0

        # 测试选择超过可用特征数量的情况
        too_many_features = self.fs.select_features_by_count(
            self.X, importance_single, top_k=10
        )

        assert len(too_many_features) == 1  # 只能返回1个特征

        # 测试零重要性阈值
        all_features = self.fs.select_features_by_importance(
            single_feature_X, importance_single, threshold=0.0
        )

        assert len(all_features) == 1

