#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施工具层数据转换工具测试

测试目标：提升utils/tools/convert.py的真实覆盖率
实际导入和使用src.infrastructure.utils.tools.convert模块
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from decimal import Decimal


class TestDataConvertConstants:
    """测试数据转换常量"""
    
    def test_constants(self):
        """测试常量值"""
        from src.infrastructure.utils.tools.convert import DataConvertConstants
        
        assert DataConvertConstants.DECIMAL_PRECISION == 8
        assert DataConvertConstants.NORMAL_STOCK_MULTIPLIER == 1.1
        assert DataConvertConstants.ST_STOCK_MULTIPLIER == 1.05
        assert DataConvertConstants.PRICE_CALCULATION_BASE == 2
        assert DataConvertConstants.PRICE_MIN_CHANGE == 0.01
        assert DataConvertConstants.INITIAL_CUM_FACTOR == 1.0
        assert DataConvertConstants.PRICE_ROUNDING_DECIMALS == 2


class TestDataConverter:
    """测试数据转换器类"""
    
    def test_calculate_limit_prices_normal_stock(self):
        """测试计算普通股票涨跌停价格"""
        from src.infrastructure.utils.tools.convert import DataConverter
        
        prev_close = 10.0
        result = DataConverter.calculate_limit_prices(prev_close, is_st=False)
        
        assert "upper_limit" in result
        assert "lower_limit" in result
        assert result["upper_limit"] == pytest.approx(11.0, rel=1e-2)
        assert result["lower_limit"] == pytest.approx(9.0, rel=1e-2)
    
    def test_calculate_limit_prices_st_stock(self):
        """测试计算ST股票涨跌停价格"""
        from src.infrastructure.utils.tools.convert import DataConverter
        
        prev_close = 10.0
        result = DataConverter.calculate_limit_prices(prev_close, is_st=True)
        
        assert "upper_limit" in result
        assert "lower_limit" in result
        assert result["upper_limit"] == pytest.approx(10.5, rel=1e-2)
        assert result["lower_limit"] == pytest.approx(9.5, rel=1e-2)
    
    def test_calculate_limit_prices_invalid_type(self):
        """测试无效类型输入"""
        from src.infrastructure.utils.tools.convert import DataConverter
        
        with pytest.raises(ValueError, match="必须是数值类型"):
            DataConverter.calculate_limit_prices("invalid")
    
    def test_apply_adjustment_factor(self):
        """测试应用复权因子"""
        from src.infrastructure.utils.tools.convert import DataConverter
        
        # 创建测试数据
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        data = pd.DataFrame({
            'open': [10.0, 11.0, 12.0, 13.0, 14.0],
            'high': [10.5, 11.5, 12.5, 13.5, 14.5],
            'low': [9.5, 10.5, 11.5, 12.5, 13.5],
            'close': [10.2, 11.2, 12.2, 13.2, 14.2],
            'volume': [1000, 1100, 1200, 1300, 1400]
        }, index=dates)
        
        factors = {
            '2024-01-03': 1.1  # 在第三天应用复权因子
        }
        
        result = DataConverter.apply_adjustment_factor(data, factors, inplace=False)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(data)
    
    def test_apply_adjustment_factor_empty_factors(self):
        """测试空复权因子"""
        from src.infrastructure.utils.tools.convert import DataConverter
        
        dates = pd.date_range('2024-01-01', periods=3, freq='D')
        data = pd.DataFrame({
            'open': [10.0, 11.0, 12.0],
            'close': [10.2, 11.2, 12.2]
        }, index=dates)
        
        result = DataConverter.apply_adjustment_factor(data, {}, inplace=False)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(data)
    
    def test_parse_margin_data(self):
        """测试解析融资融券数据"""
        from src.infrastructure.utils.tools.convert import DataConverter
        
        raw_data = {
            "symbol": "000001",
            "name": "平安银行",
            "margin_balance": 1000000.0,
            "short_balance": 500000.0,
            "margin_buy": 200000.0,
            "short_sell": 100000.0,
            "repayment": 50000.0,
        }
        
        result = DataConverter.parse_margin_data(raw_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert "net_margin" in result.columns
    
    def test_parse_margin_data_missing_field(self):
        """测试缺少必要字段"""
        from src.infrastructure.utils.tools.convert import DataConverter
        
        raw_data = {
            "symbol": "000001",
            # 缺少其他必要字段
        }
        
        with pytest.raises(ValueError, match="缺少必要字段"):
            DataConverter.parse_margin_data(raw_data)
    
    def test_normalize_dragon_board(self):
        """测试标准化龙虎榜数据"""
        from src.infrastructure.utils.tools.convert import DataConverter
        
        raw_data = [
            {
                "branch_name": "营业部 A",
                "direction": "买入",
                "amount": "100万"
            },
            {
                "branch_name": "营业部 B",
                "direction": "卖出",
                "amount": "50万"
            }
        ]
        
        result = DataConverter.normalize_dragon_board(raw_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "is_buy" in result.columns
    
    def test_convert_frequency(self):
        """测试转换数据频率"""
        from src.infrastructure.utils.tools.convert import DataConverter
        
        # 创建分钟级数据
        dates = pd.date_range('2024-01-01 09:30', periods=60, freq='1min')
        data = pd.DataFrame({
            'open': np.random.rand(60) * 100,
            'high': np.random.rand(60) * 100,
            'low': np.random.rand(60) * 100,
            'close': np.random.rand(60) * 100,
            'volume': np.random.randint(1000, 10000, 60)
        }, index=dates)
        
        result = DataConverter.convert_frequency(data, '5min')
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) < len(data)
    
    def test_convert_frequency_invalid_index(self):
        """测试无效索引"""
        from src.infrastructure.utils.tools.convert import DataConverter
        
        data = pd.DataFrame({
            'open': [10.0, 11.0, 12.0],
            'close': [10.2, 11.2, 12.2]
        })
        
        with pytest.raises(ValueError, match="必须包含datetime索引"):
            DataConverter.convert_frequency(data, '5min')
    
    def test_convert_frequency_custom_rules(self):
        """测试自定义聚合规则"""
        from src.infrastructure.utils.tools.convert import DataConverter
        
        dates = pd.date_range('2024-01-01 09:30', periods=60, freq='1min')
        data = pd.DataFrame({
            'open': np.random.rand(60) * 100,
            'close': np.random.rand(60) * 100,
            'volume': np.random.randint(1000, 10000, 60)
        }, index=dates)
        
        agg_rules = {
            'open': 'first',
            'close': 'last',
            'volume': 'sum'
        }
        
        result = DataConverter.convert_frequency(data, '5min', agg_rules)

        assert isinstance(result, pd.DataFrame)
        assert len(result) < len(data)


class TestConvertFunctions:
    """测试convert模块中的独立函数"""

    def test_apply_adjustment_factors_vectorized_basic(self):
        """测试基本的复权因子应用"""
        import pandas as pd
        from src.infrastructure.utils.tools.convert import _apply_adjustment_factors_vectorized

        # 创建测试数据
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'open': [100] * 10,
            'high': [105] * 10,
            'low': [95] * 10,
            'close': [102] * 10,
            'volume': [1000] * 10
        }, index=dates)

        # 因子数据
        factor_dates = [dates[2], dates[5], dates[8]]
        factor_values = [1.1, 1.05, 0.95]

        # 应用因子
        result = _apply_adjustment_factors_vectorized(data.copy(), factor_dates, factor_values)

        # 验证结果
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10
        assert all(col in result.columns for col in ['open', 'high', 'low', 'close', 'volume'])

    def test_apply_adjustment_factors_vectorized_empty_factors(self):
        """测试空因子列表的情况"""
        import pandas as pd
        from src.infrastructure.utils.tools.convert import _apply_adjustment_factors_vectorized

        # 创建测试数据
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        data = pd.DataFrame({
            'open': [100] * 5,
            'high': [105] * 5,
            'low': [95] * 5,
            'close': [102] * 5,
            'volume': [1000] * 5
        }, index=dates)

        # 空因子列表
        result = _apply_adjustment_factors_vectorized(data.copy(), [], [])

        # 验证数据不变（允许数据类型变化）
        assert len(result) == len(data)
        assert list(result.columns) == list(data.columns)

    def test_apply_adjustment_factors_vectorized_single_factor(self):
        """测试单个因子的情况"""
        import pandas as pd
        from src.infrastructure.utils.tools.convert import _apply_adjustment_factors_vectorized

        # 创建测试数据
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        data = pd.DataFrame({
            'open': [100] * 5,
            'high': [105] * 5,
            'low': [95] * 5,
            'close': [102] * 5,
            'volume': [1000] * 5
        }, index=dates)

        # 单个因子
        factor_dates = [dates[2]]
        factor_values = [1.2]

        result = _apply_adjustment_factors_vectorized(data.copy(), factor_dates, factor_values)

        # 验证结果
        assert result.loc[dates[0], 'open'] == 100  # 第一个因子之前，价格不变
        assert result.loc[dates[2], 'open'] == 120  # 因子应用点
        assert result.loc[dates[4], 'open'] == 120  # 因子之后保持
        assert result.loc[dates[0], 'volume'] == 1000  # 成交量不变
        assert result.loc[dates[2], 'volume'] == 1000 / 1.2  # 成交量调整

    def test_apply_adjustment_factors_vectorized_multiple_factors(self):
        """测试多个因子的累积效果"""
        import pandas as pd
        from src.infrastructure.utils.tools.convert import _apply_adjustment_factors_vectorized

        # 创建测试数据
        dates = pd.date_range('2023-01-01', periods=6, freq='D')
        data = pd.DataFrame({
            'open': [100] * 6,
            'close': [102] * 6,
            'volume': [1000] * 6
        }, index=dates)

        # 多个因子
        factor_dates = [dates[1], dates[3], dates[5]]
        factor_values = [1.1, 1.05, 0.95]

        result = _apply_adjustment_factors_vectorized(data.copy(), factor_dates, factor_values)

        # 验证累积因子效果
        # dates[0]: 1.0 (无因子)
        # dates[1]: 1.1
        # dates[2]: 1.1
        # dates[3]: 1.1 * 1.05 = 1.155
        # dates[4]: 1.155
        # dates[5]: 1.155 * 0.95 = 1.09725

        assert abs(result.loc[dates[0], 'open'] - 100) < 0.001
        assert abs(result.loc[dates[1], 'open'] - 110) < 0.001
        assert abs(result.loc[dates[3], 'open'] - 115.5) < 0.001
        assert abs(result.loc[dates[5], 'open'] - 109.725) < 0.001

    def test_apply_adjustment_factors_vectorized_missing_columns(self):
        """测试缺少某些列的情况"""
        import pandas as pd
        from src.infrastructure.utils.tools.convert import _apply_adjustment_factors_vectorized

        # 创建不完整的测试数据
        dates = pd.date_range('2023-01-01', periods=3, freq='D')
        data = pd.DataFrame({
            'open': [100] * 3,
            'close': [102] * 3
            # 缺少volume列
        }, index=dates)

        factor_dates = [dates[1]]
        factor_values = [1.1]

        result = _apply_adjustment_factors_vectorized(data.copy(), factor_dates, factor_values)

        # 验证结果
        assert result.loc[dates[0], 'open'] == 100
        assert abs(result.loc[dates[1], 'open'] - 110) < 0.001
        assert 'volume' not in result.columns  # 没有volume列

    def test_apply_adjustment_factors_vectorized_factor_not_in_index(self):
        """测试因子日期不在数据索引中的情况"""
        import pandas as pd
        from src.infrastructure.utils.tools.convert import _apply_adjustment_factors_vectorized

        # 创建测试数据
        dates = pd.date_range('2023-01-01', periods=3, freq='D')
        data = pd.DataFrame({
            'open': [100] * 3,
            'volume': [1000] * 3
        }, index=dates)

        # 因子日期不在数据范围内
        factor_dates = [pd.Timestamp('2022-12-31')]
        factor_values = [1.1]

        result = _apply_adjustment_factors_vectorized(data.copy(), factor_dates, factor_values)

        # 验证数据不变（因为因子日期不在范围内）
        assert len(result) == len(data)
        assert list(result.columns) == list(data.columns)