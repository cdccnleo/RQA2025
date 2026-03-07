"""
基础设施工具层MathUtils模块测试
"""

import pytest
import numpy as np
from src.infrastructure.utils.tools.math_utils import MathUtils


class TestMathUtils:
    """测试基础设施工具层MathUtils模块"""

    @pytest.fixture
    def math_utils(self):
        """创建MathUtils实例"""
        return MathUtils()

    def test_math_utils_initialization(self, math_utils):
        """测试MathUtils初始化"""
        assert math_utils is not None
        assert isinstance(math_utils, MathUtils)

    def test_mean_calculation(self, math_utils):
        """测试平均值计算"""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = math_utils.mean(values)
        assert result == 3.0

    def test_mean_empty_list(self, math_utils):
        """测试空列表的平均值"""
        result = math_utils.mean([])
        assert result == 0.0

    def test_median_calculation(self, math_utils):
        """测试中位数计算"""
        values = [1.0, 3.0, 5.0, 7.0, 9.0]
        result = math_utils.median(values)
        assert result == 5.0

    def test_median_even_count(self, math_utils):
        """测试偶数个元素的平均值"""
        values = [1.0, 2.0, 3.0, 4.0]
        result = math_utils.median(values)
        assert result == 2.5

    def test_median_empty_list(self, math_utils):
        """测试空列表的中位数"""
        result = math_utils.median([])
        assert result == 0.0

    def test_std_dev_calculation(self, math_utils):
        """测试标准差计算"""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = math_utils.std_dev(values)
        # 对于样本[1,2,3,4,5]，总体标准差约为1.414
        assert abs(result - 1.414) < 0.01

    def test_std_dev_empty_list(self, math_utils):
        """测试空列表的标准差"""
        result = math_utils.std_dev([])
        assert result == 0.0

    def test_variance_calculation(self, math_utils):
        """测试方差计算"""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = math_utils.variance(values)
        # 对于样本[1,2,3,4,5]，总体方差约为2.0
        assert abs(result - 2.0) < 0.01

    def test_correlation_calculation(self, math_utils):
        """测试相关性计算"""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        result = math_utils.correlation(x, y)
        # x和y应该是完全正相关
        assert abs(result - 1.0) < 0.01

    def test_correlation_empty_lists(self, math_utils):
        """测试空列表的相关性"""
        result = math_utils.correlation([], [])
        assert result == 0.0

    def test_correlation_different_lengths(self, math_utils):
        """测试不同长度列表的相关性"""
        x = [1.0, 2.0, 3.0]
        y = [1.0, 2.0]
        result = math_utils.correlation(x, y)
        assert result == 0.0

    def test_percentile_calculation(self, math_utils):
        """测试百分位数计算"""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        result = math_utils.percentile(values, 50)
        assert result == 5.5

    def test_percentile_empty_list(self, math_utils):
        """测试空列表的百分位数"""
        result = math_utils.percentile([], 50)
        assert result == 0.0

    def test_zscore_calculation(self, math_utils):
        """测试Z分数计算"""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = math_utils.zscore(values)
        assert len(result) == len(values)
        # 均值的z分数应该接近0
        mean_zscore = np.mean(result)
        assert abs(mean_zscore) < 0.01

    def test_zscore_empty_list(self, math_utils):
        """测试空列表的Z分数"""
        result = math_utils.zscore([])
        assert len(result) == 0