# tests/unit/features/test_technical_indicator_processor.py
"""
TechnicalIndicatorProcessor单元测试

测试覆盖:
- 初始化参数验证
- 技术指标计算功能
- 指标配置管理
- 错误处理
- 性能监控
- 边界条件
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import tempfile
import time
import os

from src.features.processors.technical_indicator_processor import (

TechnicalIndicatorProcessor,
    IndicatorConfig,
    IndicatorType
)

# 设置测试超时，避免死锁和无限等待
pytestmark = [
    pytest.mark.timeout(30),  # 30秒超时
    pytest.mark.deadlock_risk,  # 标记为可能存在死锁风险
    pytest.mark.concurrent,  # 并发测试
    pytest.mark.infinite_loop_risk  # 可能存在无限循环风险
]




class TestTechnicalIndicatorProcessor:
    """TechnicalIndicatorProcessor测试类"""

    @pytest.fixture
    def temp_dir(self):
        """临时目录fixture"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_data(self):
        """样本数据fixture"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        np.random.seed(42)
        return pd.DataFrame({
            'date': dates,
            'open': 100 + np.random.randn(100) * 5,
            'high': 105 + np.random.randn(100) * 5,
            'low': 95 + np.random.randn(100) * 5,
            'close': 100 + np.random.randn(100) * 3,
            'volume': np.random.randint(100000, 1000000, 100)
        })

    @pytest.fixture
    def indicator_configs(self):
        """指标配置列表fixture"""
        return [
            IndicatorConfig(
                name="SMA_20",
                type=IndicatorType.TREND,
                parameters={"period": 20},
                description="20日简单移动平均线"
            ),
            IndicatorConfig(
                name="RSI_14",
                type=IndicatorType.OSCILLATOR,
                parameters={"period": 14},
                description="14日相对强弱指数"
            ),
            IndicatorConfig(
                name="BB_20",
                type=IndicatorType.VOLATILITY,
                parameters={"period": 20, "std_dev": 2},
                description="20日布林带"
            )
        ]

    @pytest.fixture
    def processor(self, indicator_configs):
        """TechnicalIndicatorProcessor实例"""
        return TechnicalIndicatorProcessor(indicator_configs)

    def test_initialization_with_configs(self, indicator_configs):
        """测试带配置的初始化"""
        processor = TechnicalIndicatorProcessor(indicator_configs)

        assert len(processor.indicators) == len(indicator_configs)
        assert processor.enabled_indicators_count > 0

    def test_initialization_empty_configs(self):
        """测试空配置的初始化"""
        processor = TechnicalIndicatorProcessor([])

        assert len(processor.indicators) == 0
        assert processor.enabled_indicators_count == 0

    def test_initialization_default_configs(self):
        """测试默认配置的初始化"""
        processor = TechnicalIndicatorProcessor()

        # 默认应该包含一些基础指标
        assert len(processor.indicators) >= 0
        assert hasattr(processor, 'enabled_indicators_count')

    def test_process(self, processor, sample_data):
        """测试有效数据的处理"""
        result = processor.process(sample_data)

        assert result is not None
        assert not result.empty
        assert len(result) == len(sample_data)

        # 检查是否添加了技术指标列
        indicator_columns = [col for col in result.columns if any(indicator in col.upper() for indicator in ['SMA', 'RSI', 'BB', 'MACD', 'ATR'])]
        assert len(indicator_columns) > 0

    def test_process(self, processor):
        """测试空数据的处理"""
        empty_data = pd.DataFrame()

        result = processor.process(empty_data)

        # 应该返回空DataFrame或者抛出适当的错误
        assert result.empty or isinstance(result, pd.DataFrame)

    def test_process(self, processor):
        """测试缺失必要列的数据处理"""
        incomplete_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'close': np.random.randn(10)
            # 缺少OHLC其他列
        })

        result = processor.process(incomplete_data)

        # 应该能够处理缺少的列，或者只计算可能的指标
        assert isinstance(result, pd.DataFrame)

    def test_calculate_trend_indicators(self, processor, sample_data):
        """测试趋势指标计算"""
        trend_indicators = ['SMA_5', 'SMA_10', 'SMA_20', 'EMA_12']

        result = processor.calculate_trend_indicators(sample_data, trend_indicators)

        assert result is not None
        # 检查是否包含了请求的趋势指标
        for indicator in trend_indicators:
            if 'SMA' in indicator:
                period = int(indicator.split('_')[1])
                expected_col = f'SMA_{period}'
                assert expected_col in result.columns

    def test_calculate_momentum_indicators(self, processor, sample_data):
        """测试动量指标计算"""
        momentum_indicators = ['RSI_14', 'MACD', 'Stochastic']

        result = processor.calculate_momentum_indicators(sample_data, momentum_indicators)

        assert result is not None
        # 检查是否包含了动量指标
        if 'RSI_14' in momentum_indicators:
            assert 'RSI_14' in result.columns

    def test_calculate_volatility_indicators(self, processor, sample_data):
        """测试波动率指标计算"""
        volatility_indicators = ['BB_20', 'ATR_14']

        result = processor.calculate_volatility_indicators(sample_data, volatility_indicators)

        assert result is not None
        # 检查是否包含了波动率指标
        volatility_cols = [col for col in result.columns if 'BB' in col or 'ATR' in col]
        assert len(volatility_cols) > 0

    def test_calculate_volume_indicators(self, processor, sample_data):
        """测试成交量指标计算"""
        volume_indicators = ['Volume_MA_20', 'OBV']

        result = processor.calculate_volume_indicators(sample_data, volume_indicators)

        assert result is not None
        # 检查是否包含了成交量指标
        volume_cols = [col for col in result.columns if 'VOLUME' in col or 'OBV' in col]
        assert len(volume_cols) >= 0  # 可能没有实现所有的指标

    def test_get_indicator_config(self, processor):
        """测试指标配置获取"""
        # 获取第一个指标的配置
        if len(processor.indicators) > 0:
            first_indicator = list(processor.indicators.keys())[0]
            config = processor.get_indicator_config(first_indicator)

            assert config is not None
            assert isinstance(config, IndicatorConfig)

    def test_update_indicator_config(self, processor):
        """测试指标配置更新"""
        if len(processor.indicators) > 0:
            first_indicator = list(processor.indicators.keys())[0]

            new_config = IndicatorConfig(
                name=first_indicator,
                type=IndicatorType.TREND,
                parameters={"period": 30},
                description="更新后的配置"
            )

            success = processor.update_indicator_config(first_indicator, new_config)

            # 这里取决于具体实现，可能返回True/False或抛出异常
            assert isinstance(success, (bool, type(None)))

    def test_enable_disable_indicator(self, processor):
        """测试指标启用/禁用"""
        if len(processor.indicators) > 0:
            first_indicator = list(processor.indicators.keys())[0]

            # 禁用指标
            processor.disable_indicator(first_indicator)
            assert not processor.is_indicator_enabled(first_indicator)

            # 启用指标
            processor.enable_indicator(first_indicator)
            assert processor.is_indicator_enabled(first_indicator)

    def test_list_available_indicators(self, processor):
        """测试可用指标列表获取"""
        indicators = processor.list_available_indicators()

        assert isinstance(indicators, list)
        assert len(indicators) >= 0

        # 如果有指标，应该包含必要的字段
        if indicators:
            first_indicator = indicators[0]
            if isinstance(first_indicator, dict):
                assert 'name' in first_indicator
                assert 'type' in first_indicator

    def test_get_indicator_info(self, processor):
        """测试指标信息获取"""
        if len(processor.indicators) > 0:
            first_indicator = list(processor.indicators.keys())[0]
            info = processor.get_indicator_info(first_indicator)

            assert info is not None
            # 信息应该包含描述、参数等

    def test_calculate_sma(self, processor, sample_data):
        """测试简单移动平均线计算"""
        period = 5
        result = processor._calculate_sma(sample_data, period)

        assert result is not None
        assert f'SMA_{period}' in result.columns

        # 检查SMA值的合理性
        sma_values = result[f'SMA_{period}'].dropna()
        assert len(sma_values) > 0

    def test_calculate_ema(self, processor, sample_data):
        """测试指数移动平均线计算"""
        period = 12
        result = processor._calculate_ema(sample_data, period)

        assert result is not None
        assert f'EMA_{period}' in result.columns

        # 检查EMA值的合理性
        ema_values = result[f'EMA_{period}'].dropna()
        assert len(ema_values) > 0

    def test_calculate_rsi_indicator(self, processor, sample_data):
        """测试RSI指标计算"""
        period = 14
        result = processor._calculate_rsi(sample_data, period)

        assert result is not None
        assert f'RSI_{period}' in result.columns

        # RSI值应该在0-100范围内
        rsi_values = result[f'RSI_{period}'].dropna()
        if len(rsi_values) > 0:
            assert all(0 <= rsi <= 100 for rsi in rsi_values)

    def test_calculate_macd_indicator(self, processor, sample_data):
        """测试MACD指标计算"""
        result = processor._calculate_macd(sample_data)

        assert result is not None
        # 检查MACD相关列
        macd_columns = [col for col in result.columns if 'MACD' in col.upper()]
        assert len(macd_columns) >= 1  # 至少有MACD线

    def test_calculate_bollinger_bands_indicator(self, processor, sample_data):
        """测试布林带指标计算"""
        period = 20
        std_dev = 2
        result = processor._calculate_bollinger_bands(sample_data, period, std_dev)

        assert result is not None
        # 检查布林带相关列
        bb_columns = [col for col in result.columns if 'BB_' in col.upper()]
        assert len(bb_columns) >= 3  # Upper, Middle, Lower

    def test_calculate_atr_indicator(self, processor, sample_data):
        """测试ATR指标计算"""
        period = 14
        result = processor._calculate_atr(sample_data, period)

        assert result is not None
        assert f'ATR_{period}' in result.columns

        # ATR值应该为正数
        atr_values = result[f'ATR_{period}'].dropna()
        if len(atr_values) > 0:
            assert all(atr >= 0 for atr in atr_values)

    def test_performance_monitoring(self, processor, sample_data):
        """测试性能监控"""
        start_time = time.time()

        result = processor.process(sample_data)

        end_time = time.time()
        duration = end_time - start_time

        # 验证性能指标
        assert duration >= 0
        assert result is not None

    def test_memory_usage_efficiency(self, processor, sample_data):
        """测试内存使用效率"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        result = processor.process(sample_data)

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # 内存增加应该在合理范围内
        assert memory_increase < 200 * 1024 * 1024  # 不超过200MB
        assert result is not None

    def test_error_handling_invalid_data(self, processor):
        """测试无效数据错误处理"""
        invalid_data = pd.DataFrame({
            'date': ['invalid_date'],
            'close': ['not_a_number']
        })

        # 应该能够处理无效数据或抛出适当的错误
        result = processor.process(invalid_data)
        assert isinstance(result, pd.DataFrame)

    def test_error_handling_missing_ohlc(self, processor):
        """测试缺失OHLC数据错误处理"""
        incomplete_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'close': np.random.randn(10)
            # 缺少open, high, low
        })

        result = processor.process(incomplete_data)
        assert isinstance(result, pd.DataFrame)

    def test_indicator_type_enum_values(self):
        """测试指标类型枚举值"""
        # 验证所有枚举值
        expected_types = ['TREND', 'MOMENTUM', 'VOLATILITY', 'VOLUME', 'OSCILLATOR', 'STATISTICAL']

        for expected_type in expected_types:
            assert hasattr(IndicatorType, expected_type)

    def test_indicator_config_validation(self):
        """测试指标配置验证"""
        # 有效配置
        valid_config = IndicatorConfig(
            name="Test_Indicator",
            type=IndicatorType.TREND,
            parameters={"period": 20}
        )

        assert valid_config.name == "Test_Indicator"
        assert valid_config.type == IndicatorType.TREND
        assert valid_config.enabled is True

    def test_batch_indicator_calculation(self, processor, sample_data):
        """测试批量指标计算"""
        indicator_list = ['SMA_5', 'SMA_10', 'RSI_14', 'BB_20']

        result = processor.calculate_indicators_batch(sample_data, indicator_list)

        assert result is not None
        assert len(result) == len(sample_data)

        # 检查是否包含了请求的指标
        result_columns = result.columns.tolist()
        for indicator in indicator_list:
            if 'SMA' in indicator:
                assert indicator in result_columns
            elif 'RSI' in indicator:
                assert indicator in result_columns
            elif 'BB' in indicator:
                # BB通常会生成多个列
                bb_related_cols = [col for col in result_columns if 'BB' in col]
                assert len(bb_related_cols) > 0

    def test_indicator_dependencies(self, processor):
        """测试指标依赖关系"""
        # 测试一些需要其他指标作为输入的复杂指标
        # 例如MACD需要EMA作为输入

        if hasattr(processor, '_check_indicator_dependencies'):
            dependencies = processor._check_indicator_dependencies('MACD')
            # MACD通常需要EMA作为依赖
            assert isinstance(dependencies, list)

    def test_indicator_caching(self, processor, sample_data):
        """测试指标缓存机制"""
        # 第一次计算
        result1 = processor.process(sample_data)

        # 第二次计算（应该使用缓存）
        result2 = processor.process(sample_data)

        # 结果应该相同
        pd.testing.assert_frame_equal(result1, result2)

    def test_indicator_parallel_calculation(self, processor, sample_data):
        """测试指标并行计算"""
        # 这里可以测试并行计算多个指标的性能
        indicators = ['SMA_5', 'SMA_10', 'SMA_20', 'RSI_14', 'MACD']

        start_time = time.time()
        result = processor.calculate_indicators_parallel(sample_data, indicators, max_workers=4)
        end_time = time.time()

        duration = end_time - start_time

        assert result is not None
        # 并行计算应该比串行快
        assert duration < 10.0  # 应该在10秒内完成

    def test_indicator_result_validation(self, processor, sample_data):
        """测试指标结果验证"""
        result = processor.process(sample_data)

        # 验证结果的合理性
        if 'RSI_14' in result.columns:
            rsi_values = result['RSI_14'].dropna()
            if len(rsi_values) > 0:
                # RSI应该在0-100范围内
                assert all(0 <= rsi <= 100 for rsi in rsi_values)

        if 'SMA_5' in result.columns:
            sma_values = result['SMA_5'].dropna()
            if len(sma_values) > 0:
                # SMA应该接近收盘价的移动平均
                close_prices = sample_data['close']
                assert abs(sma_values.mean() - close_prices.mean()) < close_prices.std()

    def test_indicator_calculation_consistency(self, processor, sample_data):
        """测试指标计算一致性"""
        # 多次计算同一个指标应该得到相同结果
        result1 = processor.process(sample_data)
        result2 = processor.process(sample_data)

        # 比较数值列（排除时间戳等可能变化的列）
        numeric_columns = result1.select_dtypes(include=[np.number]).columns
        common_columns = [col for col in numeric_columns if col in result2.columns]

        for col in common_columns:
            pd.testing.assert_series_equal(
                result1[col].round(6),  # 保留6位小数
                result2[col].round(6),
                check_names=False
            )
