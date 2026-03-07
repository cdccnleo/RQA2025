#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统集成测试

测试基础设施层各模块之间的集成功能，确保跨模块的数据流和业务流程正常工作。
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from src.infrastructure.utils.tools.convert import DataConverter
from src.infrastructure.utils.tools.math_utils import (
    calculate_returns, calculate_volatility, calculate_max_drawdown,
    calculate_sharpe_ratio, annualized_volatility
)


class TestDataProcessingIntegration:
    """数据处理集成测试"""

    def test_stock_data_processing_pipeline(self):
        """测试股票数据处理完整流程"""
        # 1. 准备原始股票数据
        raw_data = {
            "open": [100.0, 102.0, 98.0, 105.0],
            "high": [105.0, 103.0, 102.0, 108.0],
            "low": [99.0, 101.0, 97.0, 104.0],
            "close": [103.0, 101.0, 100.0, 107.0],
            "volume": [10000, 12000, 8000, 15000]
        }
        dates = pd.date_range('2024-01-01', periods=4)
        stock_data = pd.DataFrame(raw_data, index=dates)

        # 2. 应用复权因子调整
        factors = {
            dates[1]: 1.05,  # 5%复权
            dates[3]: 0.98   # -2%复权
        }

        adjusted_data = DataConverter.apply_adjustment_factor(stock_data, factors)

        # 验证调整后的数据结构
        assert isinstance(adjusted_data, pd.DataFrame)
        assert len(adjusted_data) == len(stock_data)
        assert list(adjusted_data.columns) == list(stock_data.columns)

        # 验证价格数据已被调整
        assert not adjusted_data.equals(stock_data)  # 数据应该有变化

    def test_quantitative_analysis_integration(self):
        """测试量化分析集成流程"""
        # 1. 生成模拟价格数据
        np.random.seed(42)  # 确保可重复性
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 100)))

        # 2. 计算收益率
        returns = calculate_returns(prices)

        # 3. 计算各项风险指标
        volatility = calculate_volatility(returns)
        annualized_vol = annualized_volatility(returns)
        max_dd = calculate_max_drawdown(pd.Series(prices))
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)

        # 验证计算结果的合理性
        assert isinstance(returns, list)
        assert len(returns) == len(prices)  # calculate_returns返回与输入相同长度的列表
        assert volatility > 0
        assert annualized_vol >= volatility  # 年化波动率应该大于等于普通波动率
        assert max_dd <= 0  # 最大回撤是负数
        assert isinstance(sharpe, float)

    def test_resource_and_data_integration(self):
        """测试资源管理和数据处理的集成"""
        try:
            from src.infrastructure.resource.core.system_resource_analyzer import SystemResourceAnalyzer
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine

            # 1. 创建资源分析器
            analyzer = SystemResourceAnalyzer()
            optimizer = ResourceOptimizationEngine()

            # 2. 收集系统资源信息
            with patch.object(analyzer, 'get_system_resources') as mock_get_resources:
                mock_resources = {
                    "cpu": {"usage_percent": 45.0, "count": 4},
                    "memory": {"usage_percent": 60.0, "available_gb": 8.0},
                    "disk": {"usage_percent": 70.0}
                }
                mock_get_resources.return_value = mock_resources

                # 3. 执行资源优化
                optimization_config = {
                    "optimization_priority": ["memory", "cpu"],
                    "memory_optimization": {"enabled": True, "gc_threshold": 75.0},
                    "cpu_optimization": {"enabled": True, "priority_threshold": 80.0}
                }

                result = optimizer.optimize_resources(optimization_config)

                # 验证集成结果
                assert isinstance(result, dict)
                assert "status" in result
                assert result["status"] == "success"

        except ImportError:
            pytest.skip("Resource integration modules not available")

    def test_error_handling_integration(self):
        """测试错误处理集成"""
        # 1. 测试数据转换错误处理
        invalid_data = pd.DataFrame({"invalid_column": [1, 2, 3]})

        # 应该能处理缺失的必需列
        result = DataConverter.apply_adjustment_factor(invalid_data, {})
        assert isinstance(result, pd.DataFrame)

        # 2. 测试数学计算错误处理
        empty_prices = []
        with pytest.raises(ValueError, match="价格序列不能为空"):
            calculate_returns(empty_prices)  # 空输入应该抛出异常

        # 3. 测试波动率计算的错误处理
        zero_returns = [0.0, 0.0, 0.0]
        vol = calculate_volatility(zero_returns)
        assert vol == 0.0  # 零波动率

    def test_performance_integration(self):
        """测试性能相关的集成"""
        # 1. 创建大数据集进行性能测试
        large_prices = pd.Series(np.random.normal(100, 10, 1000))

        # 2. 执行完整的量化分析流程
        import time
        start_time = time.time()

        returns = calculate_returns(large_prices.tolist())
        volatility = calculate_volatility(returns)
        max_dd = calculate_max_drawdown(large_prices)
        sharpe = calculate_sharpe_ratio(returns)

        end_time = time.time()
        processing_time = end_time - start_time

        # 验证性能要求（应该在合理时间内完成）
        assert processing_time < 1.0  # 1秒内完成
        assert all(isinstance(x, (int, float)) for x in [volatility, max_dd, sharpe])

    def test_data_validation_integration(self):
        """测试数据验证集成"""
        # 1. 测试无效数据处理
        invalid_stock_data = pd.DataFrame({
            "open": ["invalid", None, np.nan],
            "close": [100, "invalid", 105]
        })

        # 2. 应用数据转换（应该能处理无效数据）
        result = DataConverter.apply_adjustment_factor(invalid_stock_data, {})
        assert isinstance(result, pd.DataFrame)

        # 3. 测试边界情况
        single_price = [100.0]
        returns = calculate_returns(single_price)
        assert returns == [0.0]  # 单价格返回填充的零收益率

        # 4. 测试极值情况
        extreme_prices = [0.01, 1000000, 0.001]
        returns = calculate_returns(extreme_prices)
        assert len(returns) == 3  # 返回与输入相同长度的数组
        assert all(isinstance(r, float) for r in returns)
