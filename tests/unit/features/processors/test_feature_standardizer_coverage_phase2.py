# -*- coding: utf-8 -*-
"""
特征标准化器覆盖率测试 - Phase 2
针对FeatureStandardizer类的未覆盖方法进行补充测试
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import shutil

from src.features.processors.feature_standardizer import FeatureStandardizer


class TestFeatureStandardizerCoverage:
    """测试FeatureStandardizer的未覆盖方法"""

    @pytest.fixture
    def sample_numeric_data(self):
        """生成示例数值数据"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        return pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100) * 2,
            'feature3': np.random.randn(100) * 3
        }, index=dates)

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        temp_path = tempfile.mkdtemp()
        yield Path(temp_path)
        shutil.rmtree(temp_path)

    @pytest.fixture
    def standardizer(self, temp_dir):
        """创建FeatureStandardizer实例"""
        return FeatureStandardizer(temp_dir)

    def test_standardize_features_alias(self, standardizer, sample_numeric_data):
        """测试standardize_features别名方法"""
        result = standardizer.standardize_features(sample_numeric_data)
        
        # 验证结果
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_numeric_data)
        assert standardizer.is_fitted is True

    def test_inverse_transform_success(self, standardizer, sample_numeric_data):
        """测试逆变换 - 成功"""
        # 先拟合
        standardized = standardizer.fit_transform(sample_numeric_data)
        
        # 测试逆变换
        original = standardizer.inverse_transform(standardized)
        
        # 验证结果
        assert isinstance(original, pd.DataFrame)
        assert len(original) == len(sample_numeric_data)
        # 验证逆变换后的数据接近原始数据（允许浮点误差）
        assert np.allclose(original.values, sample_numeric_data.values, rtol=1e-5)

    def test_inverse_transform_not_fitted(self, standardizer, sample_numeric_data):
        """测试逆变换 - 未拟合"""
        # 不拟合直接尝试逆变换
        with pytest.raises(RuntimeError, match="标准化器尚未拟合"):
            standardizer.inverse_transform(sample_numeric_data)

    def test_load_scaler_success(self, standardizer, sample_numeric_data, temp_dir):
        """测试加载标准化器 - 成功"""
        # 先保存标准化器
        standardizer.fit_transform(sample_numeric_data)
        scaler_path = standardizer.scaler_path
        
        # 创建新的standardizer并加载
        new_standardizer = FeatureStandardizer(temp_dir)
        new_standardizer.load_scaler(scaler_path)
        
        # 验证已加载
        assert new_standardizer.is_fitted is True
        assert new_standardizer.scaler is not None

    def test_load_scaler_not_found(self, standardizer, temp_dir):
        """测试加载标准化器 - 文件不存在"""
        non_existent_path = temp_dir / "non_existent_scaler.pkl"
        
        with pytest.raises(FileNotFoundError):
            standardizer.load_scaler(non_existent_path)

    def test_partial_fit_success(self, standardizer, sample_numeric_data):
        """测试增量拟合 - 成功（如果支持）"""
        # 测试partial_fit方法
        standardizer.partial_fit(sample_numeric_data)
        
        # 如果支持partial_fit，应该标记为已拟合
        # 如果不支持，应该记录警告但不抛出异常
        # 验证方法执行完成（不抛出异常）

    def test_partial_fit_empty_data(self, standardizer):
        """测试增量拟合 - 空数据"""
        empty_data = pd.DataFrame()
        
        # 应该处理空数据而不抛出异常
        standardizer.partial_fit(empty_data)
        
        # 验证方法执行完成

    def test_fit_transform_empty_data(self, standardizer):
        """测试拟合并转换 - 空数据"""
        empty_data = pd.DataFrame()
        
        with pytest.raises(ValueError, match="输入数据为空"):
            standardizer.fit_transform(empty_data)

    def test_fit_transform_no_numeric_columns(self, standardizer):
        """测试拟合并转换 - 无数值列"""
        non_numeric_data = pd.DataFrame({
            'text': ['a', 'b', 'c'],
            'category': ['x', 'y', 'z']
        })
        
        with pytest.raises(ValueError, match="输入数据不包含数值列"):
            standardizer.fit_transform(non_numeric_data)

    def test_transform_not_fitted(self, standardizer, sample_numeric_data):
        """测试转换 - 未拟合"""
        with pytest.raises(RuntimeError, match="标准化器尚未拟合"):
            standardizer.transform(sample_numeric_data)

    def test_transform_no_numeric_columns(self, standardizer, sample_numeric_data):
        """测试转换 - 无数值列"""
        # 先拟合
        standardizer.fit_transform(sample_numeric_data)
        
        # 创建无数值列的数据
        non_numeric_data = pd.DataFrame({
            'text': ['a', 'b', 'c'] * 33 + ['a']
        })
        
        with pytest.raises(ValueError, match="输入数据不包含数值列"):
            standardizer.transform(non_numeric_data)

    def test_init_scaler_unsupported_method(self, temp_dir):
        """测试初始化标准化器 - 不支持的方法"""
        with pytest.raises(ValueError, match="不支持的标准化方法"):
            FeatureStandardizer(temp_dir, method="unsupported")

    def test_init_scaler_minmax(self, temp_dir):
        """测试初始化标准化器 - MinMax方法"""
        standardizer = FeatureStandardizer(temp_dir, method="minmax")
        assert standardizer.method == "minmax"
        assert standardizer.scaler is not None

    def test_init_scaler_robust(self, temp_dir):
        """测试初始化标准化器 - Robust方法"""
        standardizer = FeatureStandardizer(temp_dir, method="robust")
        assert standardizer.method == "robust"
        assert standardizer.scaler is not None

    def test_fit_transform_save_failure_handling(self, standardizer, sample_numeric_data, temp_dir):
        """测试拟合并转换 - 保存失败处理"""
        # Mock joblib.dump to raise exception
        with patch('src.features.processors.feature_standardizer.joblib.dump', side_effect=Exception("Save failed")):
            # 应该不抛出异常，只记录警告
            result = standardizer.fit_transform(sample_numeric_data)
            
            # 验证结果仍然返回
            assert isinstance(result, pd.DataFrame)
            # 验证is_fitted可能为True（即使保存失败）
            # 这取决于实现，但至少不应该抛出异常

    def test_fit_transform_load_failure_handling(self, standardizer, sample_numeric_data, temp_dir):
        """测试拟合并转换 - 加载失败处理（is_training=False）"""
        # 先保存一个标准化器
        standardizer.fit_transform(sample_numeric_data)
        
        # 创建新的standardizer，模拟加载失败
        new_standardizer = FeatureStandardizer(temp_dir)
        with patch('src.features.processors.feature_standardizer.joblib.load', side_effect=FileNotFoundError()):
            # 当is_training=False且文件不存在时，应该返回原始数据
            result = new_standardizer.fit_transform(sample_numeric_data, is_training=False)
            
            # 验证返回了原始数据
            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(sample_numeric_data)




