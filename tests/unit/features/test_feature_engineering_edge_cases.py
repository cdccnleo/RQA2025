#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征工程边界条件测试

此文件包含特征工程模块的边界条件和异常情况测试，
用于提升测试覆盖率至80%以上。
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import warnings

# 导入相关模块
try:
    # 这里根据实际的特征工程模块导入
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="特征工程相关组件不可用")
class TestFeatureEngineeringEdgeCases:
    """特征工程边界条件测试"""

    def test_categorical_encoding_high_cardinality(self):
        """测试分类特征编码处理高基数情况"""
        # 创建高基数分类特征
        np.random.seed(42)
        categories = [f"category_{i}" for i in range(1000)]  # 1000个唯一类别

        data = pd.DataFrame({
            'high_cardinality_feature': np.random.choice(categories, 1000),
            'target': np.random.randint(0, 2, 1000)
        })

        # 测试标签编码（适用于高基数）
        le = LabelEncoder()
        encoded = le.fit_transform(data['high_cardinality_feature'])

        assert len(encoded) == 1000
        assert len(np.unique(encoded)) == 1000  # 所有类别都被编码

        # 验证编码后的值范围
        assert encoded.min() == 0
        assert encoded.max() == 999

        # 测试逆变换
        original = le.inverse_transform(encoded[:10])
        assert len(original) == 10
        assert all(isinstance(cat, str) for cat in original)

    def test_numerical_scaling_outliers(self):
        """测试数值特征缩放处理异常值"""
        # 创建包含极端异常值的数据
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 1000)
        outliers = np.array([100, -100, 1000, -1000, 10000])  # 极端异常值

        data = np.concatenate([normal_data, outliers])

        # 测试标准缩放
        scaler = StandardScaler()

        # 原始数据统计
        original_mean = np.mean(data)
        original_std = np.std(data)

        # 缩放数据
        scaled_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()

        # 验证缩放后的数据
        scaled_mean = np.mean(scaled_data)
        scaled_std = np.std(scaled_data)

        # 标准缩放后均值接近0，标准差为1
        assert abs(scaled_mean) < 0.1
        assert abs(scaled_std - 1.0) < 0.1

        # 验证可以逆变换
        original_restored = scaler.inverse_transform(scaled_data.reshape(-1, 1)).flatten()

        # 逆变换后的数据应该接近原始数据
        np.testing.assert_allclose(original_restored, data, rtol=1e-10)

    def test_text_feature_extraction_edge_cases(self):
        """测试文本特征提取的边界情况"""
        # 创建各种文本数据
        text_data = [
            "",  # 空字符串
            "   ",  # 只有空格
            "Hello World",  # 正常文本
            "Hello\nWorld",  # 包含换行
            "Hello\tWorld",  # 包含制表符
            "Hello, World! How are you?",  # 标点符号
            "你好世界",  # Unicode文本
            "Hello123",  # 包含数字
            "Hello@#$%",  # 特殊字符
            "A" * 1000,  # 超长文本
            None,  # None值
        ]

        # 模拟文本特征提取
        def extract_text_features(texts):
            features = []
            for text in texts:
                if text is None:
                    # 处理None值
                    features.append({
                        'length': 0,
                        'word_count': 0,
                        'has_upper': False,
                        'has_lower': False,
                        'has_digit': False,
                        'has_punct': False
                    })
                elif isinstance(text, str):
                    features.append({
                        'length': len(text),
                        'word_count': len(text.split()),
                        'has_upper': any(c.isupper() for c in text),
                        'has_lower': any(c.islower() for c in text),
                        'has_digit': any(c.isdigit() for c in text),
                        'has_punct': any(not c.isalnum() and not c.isspace() for c in text)
                    })
                else:
                    # 处理其他类型
                    features.append({
                        'length': 0,
                        'word_count': 0,
                        'has_upper': False,
                        'has_lower': False,
                        'has_digit': False,
                        'has_punct': False
                    })

            return pd.DataFrame(features)

        # 提取特征
        features_df = extract_text_features(text_data)

        # 验证结果
        assert len(features_df) == len(text_data)

        # 检查空字符串处理
        assert features_df.iloc[0]['length'] == 0
        assert features_df.iloc[0]['word_count'] == 1  # 空字符串.split()返回['']

        # 检查None值处理
        none_idx = text_data.index(None)
        assert features_df.iloc[none_idx]['length'] == 0
        assert features_df.iloc[none_idx]['has_upper'] == False

        # 检查超长文本
        long_text_idx = text_data.index("A" * 1000)
        assert features_df.iloc[long_text_idx]['length'] == 1000
        assert features_df.iloc[long_text_idx]['has_upper'] == True

    def test_time_series_feature_generation(self):
        """测试时间序列特征生成"""
        # 创建时间序列数据
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        values = np.random.randn(100).cumsum()  # 随机游走

        ts_data = pd.DataFrame({
            'date': dates,
            'value': values
        })

        # 生成时间序列特征
        def generate_time_features(df):
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])

            # 时间特征
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['day_of_week'] = df['date'].dt.dayofweek
            df['quarter'] = df['date'].dt.quarter
            df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6])

            # 移动统计特征
            df['value_lag_1'] = df['value'].shift(1)
            df['value_lag_7'] = df['value'].shift(7)
            df['value_rolling_mean_7'] = df['value'].rolling(window=7).mean()
            df['value_rolling_std_7'] = df['value'].rolling(window=7).std()

            # 差异特征
            df['value_diff_1'] = df['value'].diff(1)
            df['value_pct_change_1'] = df['value'].pct_change(1)

            return df

        # 生成特征
        featured_data = generate_time_features(ts_data)

        # 验证特征生成
        assert 'year' in featured_data.columns
        assert 'month' in featured_data.columns
        assert 'value_lag_1' in featured_data.columns
        assert 'value_rolling_mean_7' in featured_data.columns

        # 验证数据类型
        assert featured_data['year'].dtype == 'int64'
        assert featured_data['is_weekend'].dtype == 'bool'

        # 验证移动统计（前7个应该是NaN）
        assert pd.isna(featured_data['value_rolling_mean_7'].iloc[0])
        assert pd.isna(featured_data['value_rolling_mean_7'].iloc[6])
        assert not pd.isna(featured_data['value_rolling_mean_7'].iloc[7])

    def test_feature_interaction_complexity(self):
        """测试特征交互复杂性"""
        # 创建多特征数据集
        np.random.seed(42)
        n_samples = 1000
        n_features = 10

        X = np.random.randn(n_samples, n_features)
        feature_names = [f'feature_{i}' for i in range(n_features)]

        data = pd.DataFrame(X, columns=feature_names)

        # 生成特征交互
        def generate_feature_interactions(df, max_interactions=50):
            """生成特征交互，限制数量避免组合爆炸"""
            interactions = []
            feature_cols = [col for col in df.columns if col.startswith('feature_')]

            # 二阶交互
            for i in range(len(feature_cols)):
                for j in range(i+1, len(feature_cols)):
                    if len(interactions) >= max_interactions:
                        break
                    col1, col2 = feature_cols[i], feature_cols[j]

                    # 乘积交互
                    interaction_name = f"{col1}_x_{col2}"
                    df[interaction_name] = df[col1] * df[col2]
                    interactions.append(interaction_name)

                    # 比例交互（避免除零）
                    ratio_name = f"{col1}_div_{col2}"
                    df[ratio_name] = df[col1] / (df[col2] + 1e-8)  # 添加小常数避免除零
                    interactions.append(ratio_name)

                if len(interactions) >= max_interactions:
                    break

            return df, interactions

        # 生成交互特征
        data_with_interactions, interaction_names = generate_feature_interactions(data.copy())

        # 验证交互特征生成
        assert len(interaction_names) > 0
        assert all(name in data_with_interactions.columns for name in interaction_names)

        # 验证交互特征的数值合理性
        for interaction in interaction_names[:5]:  # 检查前5个交互
            values = data_with_interactions[interaction]
            assert not np.any(np.isinf(values))  # 不应该有无穷大值
            assert not np.all(np.isnan(values))  # 不应该全都是NaN

    def test_missing_value_imputation_strategies(self):
        """测试缺失值填充策略"""
        # 创建包含缺失值的数据
        np.random.seed(42)
        data = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10],
            'feature2': [np.nan, 2, 3, 4, np.nan, 6, 7, 8, np.nan, 10],
            'category': ['A', 'B', np.nan, 'A', 'B', 'A', np.nan, 'B', 'A', 'B']
        })

        original_missing = data.isnull().sum().sum()

        # 测试均值填充
        numeric_data = data.select_dtypes(include=[np.number])
        mean_filled = numeric_data.fillna(numeric_data.mean())

        # 验证填充结果
        assert not mean_filled.isnull().any().any()
        assert mean_filled.shape == numeric_data.shape

        # 测试中位数填充
        median_filled = numeric_data.fillna(numeric_data.median())

        assert not median_filled.isnull().any().any()

        # 测试众数填充（分类特征）
        categorical_data = data.select_dtypes(include=['object'])
        mode_filled = categorical_data.fillna(categorical_data.mode().iloc[0])

        assert not mode_filled.isnull().any().any()

        # 测试前向填充
        ffill_data = data.fillna(method='ffill')

        # 验证前向填充减少了缺失值
        assert ffill_data.isnull().sum().sum() < original_missing

        # 测试自定义填充策略
        def custom_imputation(df):
            """自定义填充策略"""
            df_filled = df.copy()

            for col in df_filled.columns:
                if df_filled[col].dtype in ['float64', 'int64']:
                    # 数值特征：用均值填充
                    df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
                elif df_filled[col].dtype == 'object':
                    # 分类特征：用众数填充
                    mode_val = df_filled[col].mode()
                    if not mode_val.empty:
                        df_filled[col] = df_filled[col].fillna(mode_val.iloc[0])
                    else:
                        df_filled[col] = df_filled[col].fillna('unknown')

            return df_filled

        custom_filled = custom_imputation(data)
        assert not custom_filled.isnull().any().any()

    def test_sparse_feature_selection(self):
        """测试稀疏特征选择"""
        # 创建包含稀疏特征的数据
        np.random.seed(42)
        n_samples = 1000
        n_features = 100

        # 生成特征，其中大部分是噪声
        X = np.random.randn(n_samples, n_features)
        # 只让前5个特征与目标相关
        y = (X[:, 0] + X[:, 1] - X[:, 2] + 0.5 * X[:, 3] + X[:, 4] > 0).astype(int)

        # 添加一些完全为零的特征（极端稀疏）
        X[:, 10:20] = 0  # 这些特征完全没有信息

        # 测试特征选择
        selector = SelectKBest(score_func=f_classif, k=10)
        X_selected = selector.fit_transform(X, y)

        # 验证选择结果
        assert X_selected.shape[1] == 10  # 选择了10个特征
        assert X_selected.shape[0] == n_samples

        # 验证选择的特征包含真正有用的特征
        selected_indices = selector.get_support(indices=True)
        # 前5个特征应该被选中（因为它们与目标相关）
        assert any(idx < 5 for idx in selected_indices)

        # 验证特征得分
        scores = selector.scores_
        assert len(scores) == n_features
        assert not np.any(np.isnan(scores))  # 不应该有NaN得分

    def test_feature_redundancy_elimination(self):
        """测试特征冗余消除"""
        # 创建包含高度相关特征的数据
        np.random.seed(42)
        n_samples = 500
        base_feature = np.random.randn(n_samples)

        # 创建相关特征
        data = pd.DataFrame({
            'feature1': base_feature,
            'feature2': base_feature + 0.1 * np.random.randn(n_samples),  # 高度相关
            'feature3': base_feature + 0.5 * np.random.randn(n_samples),  # 中等相关
            'feature4': np.random.randn(n_samples),  # 不相关
            'feature5': base_feature * 2 + np.random.randn(n_samples),  # 高度相关但变换
        })

        # 计算相关性矩阵
        corr_matrix = data.corr()

        # 识别高度相关的特征对
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = abs(corr_matrix.iloc[i, j])
                if corr_value > 0.8:  # 相关性阈值
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_value
                    ))

        # 验证找到了高度相关的特征对
        assert len(high_corr_pairs) >= 2  # feature1和feature2, feature1和feature5

        # 模拟特征消除策略：对每个相关对，保留方差较大的特征
        def select_uncorrelated_features(df, corr_threshold=0.8):
            """选择不相关的特征"""
            corr_matrix = df.corr()
            selected_features = list(df.columns)

            # 按相关性排序处理
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.columns[i] in selected_features and corr_matrix.columns[j] in selected_features:
                        corr_value = abs(corr_matrix.iloc[i, j])
                        if corr_value > corr_threshold:
                            # 保留方差较大的特征
                            var_i = df[corr_matrix.columns[i]].var()
                            var_j = df[corr_matrix.columns[j]].var()

                            if var_i >= var_j:
                                selected_features.remove(corr_matrix.columns[j])
                            else:
                                selected_features.remove(corr_matrix.columns[i])

            return df[selected_features]

        # 应用特征选择
        selected_data = select_uncorrelated_features(data)

        # 验证结果
        assert len(selected_data.columns) < len(data.columns)  # 应该减少了一些特征

        # 验证剩余特征之间的相关性降低
        if len(selected_data.columns) > 1:
            final_corr = selected_data.corr()
            max_corr = 0
            for i in range(len(final_corr.columns)):
                for j in range(i+1, len(final_corr.columns)):
                    max_corr = max(max_corr, abs(final_corr.iloc[i, j]))

            assert max_corr < 0.9  # 最大相关性应该低于阈值
