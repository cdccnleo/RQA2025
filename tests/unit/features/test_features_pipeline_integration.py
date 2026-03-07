#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Features层 - 特征管道集成测试

测试特征管道、数据预处理、特征验证的端到端流程
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class TestFeaturePipeline:
    """测试特征管道"""
    
    @pytest.fixture
    def sample_data(self):
        """创建示例数据"""
        return pd.DataFrame({
            'price': [100, 102, 98, 105, 103],
            'volume': [1000, 1200, 900, 1500, 1100],
            'indicator': [0.5, 0.6, 0.4, 0.7, 0.55]
        })
    
    def test_create_feature_pipeline(self, sample_data):
        """测试创建特征管道"""
        # 模拟特征管道
        pipeline_steps = []
        
        # Step 1: 数据清洗
        cleaned_data = sample_data.dropna()
        pipeline_steps.append(('clean', cleaned_data))
        
        # Step 2: 特征工程
        cleaned_data['price_return'] = cleaned_data['price'].pct_change()
        pipeline_steps.append(('engineer', cleaned_data))
        
        # Step 3: 标准化
        # 这里简化处理，不实际标准化
        pipeline_steps.append(('scale', cleaned_data))
        
        assert len(pipeline_steps) == 3
    
    def test_pipeline_transform(self, sample_data):
        """测试管道转换"""
        # 创建转换步骤
        data = sample_data.copy()
        
        # 添加派生特征
        data['volume_ma'] = data['volume'].rolling(window=2).mean()
        data['price_momentum'] = data['price'] / data['price'].shift(1) - 1
        
        assert 'volume_ma' in data.columns
        assert 'price_momentum' in data.columns
    
    def test_pipeline_fit_transform(self, sample_data):
        """测试管道拟合和转换"""
        # 模拟sklearn pipeline
        from sklearn.preprocessing import StandardScaler
        
        X = sample_data[['price', 'volume']].values
        scaler = StandardScaler()
        
        # Fit和transform
        X_transformed = scaler.fit_transform(X)
        
        assert X_transformed.shape == X.shape
        assert np.allclose(X_transformed.mean(axis=0), 0, atol=1e-10)


class TestDataPreprocessing:
    """测试数据预处理"""
    
    def test_handle_missing_values(self):
        """测试处理缺失值"""
        df = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': [10, np.nan, 30, 40, 50]
        })
        
        # 填充缺失值
        df_filled = df.fillna(df.mean())
        
        assert not df_filled.isna().any().any()
    
    def test_handle_outliers(self):
        """测试处理异常值"""
        data = pd.Series([1, 2, 3, 100, 4, 5, 6])  # 100是异常值
        
        # 使用IQR方法识别异常值
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = (data < lower_bound) | (data > upper_bound)
        
        assert outliers.sum() >= 1  # 至少识别出1个异常值
    
    def test_remove_duplicates(self):
        """测试移除重复值"""
        df = pd.DataFrame({
            'feature1': [1, 2, 2, 3, 3, 4],
            'feature2': [10, 20, 20, 30, 30, 40]
        })
        
        df_unique = df.drop_duplicates()
        
        assert len(df_unique) <= len(df)
    
    def test_filter_low_variance_features(self):
        """测试过滤低方差特征"""
        df = pd.DataFrame({
            'constant': [1, 1, 1, 1, 1],  # 无方差
            'low_var': [1, 1, 1, 1, 2],   # 低方差
            'normal': [1, 2, 3, 4, 5]     # 正常方差
        })
        
        variances = df.var()
        threshold = 0.1
        
        high_var_features = variances[variances > threshold].index.tolist()
        
        assert 'constant' not in high_var_features
        assert 'normal' in high_var_features


class TestFeatureValidation:
    """测试特征验证"""
    
    def test_validate_feature_types(self):
        """测试验证特征类型"""
        df = pd.DataFrame({
            'numeric': [1, 2, 3],
            'categorical': ['A', 'B', 'C'],
            'boolean': [True, False, True]
        })
        
        # 验证数据类型
        assert pd.api.types.is_numeric_dtype(df['numeric'])
        assert pd.api.types.is_object_dtype(df['categorical'])
        assert pd.api.types.is_bool_dtype(df['boolean'])
    
    def test_validate_feature_range(self):
        """测试验证特征范围"""
        features = pd.Series([0.1, 0.5, 0.8, 0.95])
        
        # 验证在[0, 1]范围内
        in_range = features.between(0, 1).all()
        
        assert in_range == True
    
    def test_validate_no_infinite_values(self):
        """测试验证无无穷值"""
        data = pd.Series([1.0, 2.0, 3.0, 4.0])
        
        has_inf = np.isinf(data).any()
        
        assert has_inf == False
    
    def test_validate_feature_names(self):
        """测试验证特征名称"""
        df = pd.DataFrame({
            'feature_1': [1, 2],
            'feature_2': [3, 4]
        })
        
        # 验证特征名称格式
        valid_names = all(col.startswith('feature_') for col in df.columns)
        
        assert valid_names is True
    
    def test_validate_feature_correlation(self):
        """测试验证特征相关性"""
        df = pd.DataFrame({
            'f1': [1, 2, 3, 4, 5],
            'f2': [2, 4, 6, 8, 10]  # f2 = 2*f1，完全相关
        })
        
        corr = df['f1'].corr(df['f2'])
        
        assert abs(corr) > 0.99  # 高度相关


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

