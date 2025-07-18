import pytest
import pandas as pd
import numpy as np

from src.features.processors.feature_standardizer import FeatureStandardizer

class TestFeatureStandardizer:
    """FeatureStandardizer标准化测试"""
    
    @pytest.fixture
    def empty_df(self):
        return pd.DataFrame({'empty': []})

    @pytest.fixture
    def no_numeric_df(self, sample_features):
        return sample_features[[]]

    @pytest.mark.parametrize("method,expected_range", [
        ("zscore", (-3, 3)),
        ("minmax", (0, 1)),
        ("robust", (-2, 2))
    ])
    def test_standardization_methods(self, method, expected_range, sample_features, tmp_model_path):
        """测试不同标准化方法"""
        standardizer = FeatureStandardizer(model_path=tmp_model_path)
        df = sample_features.select_dtypes(include=np.number)

        # 拟合并转换
        transformed = standardizer.fit_transform(df)

        # 验证数值范围
        for col in transformed.columns:
            assert transformed[col].min() >= expected_range[0]
            assert transformed[col].max() <= expected_range[1]

    def test_inverse_transform(self, sample_features, tmp_model_path):
        """测试逆变换精度"""
        standardizer = FeatureStandardizer(model_path=tmp_model_path)
        df = sample_features[['close', 'sentiment']]

        # 转换并逆转换
        transformed = standardizer.fit_transform(df)
        inverted = standardizer.transform(transformed)

        # 验证逆变换精度
        pd.testing.assert_frame_equal(
            df,
            inverted,
            check_exact=False,
            rtol=1e-5
        )

    def test_incremental_fit(self, sample_features, tmp_model_path):
        """测试增量更新"""
        standardizer = FeatureStandardizer(model_path=tmp_model_path)
        df1 = sample_features.iloc[:3]
        df2 = sample_features.iloc[3:]

        # 分批拟合
        standardizer.partial_fit(df1)
        standardizer.partial_fit(df2)

        # 验证完整数据转换
        full_transformed = standardizer.transform(sample_features)
        assert not full_transformed.isna().any().any()

    def test_error_handling_broken_features(self, broken_features, tmp_model_path):
        """测试异常输入处理 - 包含NaN"""
        standardizer = FeatureStandardizer(model_path=tmp_model_path)
        with pytest.raises(ValueError):
            standardizer.fit_transform(broken_features)

    def test_error_handling_empty_df(self, empty_df, tmp_model_path):
        """测试异常输入处理 - 空DataFrame"""
        standardizer = FeatureStandardizer(model_path=tmp_model_path)
        with pytest.raises(ValueError):
            standardizer.fit_transform(empty_df)

    def test_error_handling_no_numeric_df(self, no_numeric_df, tmp_model_path):
        """测试异常输入处理 - 无数字特征"""
        standardizer = FeatureStandardizer(model_path=tmp_model_path)
        with pytest.raises(ValueError):
            standardizer.fit_transform(no_numeric_df)

    def test_feature_consistency(self, sample_features, tmp_model_path):
        """测试特征一致性（输入输出列相同）"""
        standardizer = FeatureStandardizer(model_path=tmp_model_path)
        df = sample_features.select_dtypes(include=np.number)

        transformed = standardizer.fit_transform(df)
        assert set(transformed.columns) == set(df.columns)
