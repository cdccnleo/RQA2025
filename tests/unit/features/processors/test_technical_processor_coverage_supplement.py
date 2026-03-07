#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
technical_processor补充测试覆盖
针对未覆盖的代码分支编写测试
"""

import pytest
import pandas as pd
import numpy as np
from src.features.processors.technical.technical_processor import (
    TechnicalProcessor,
    BaseTechnicalProcessor,
    SMAProcessor,
    EMAProcessor,
    RSIProcessor,
    MACDProcessor,
    BollingerBandsProcessor,
    ATRProcessor
)
from src.features.processors.base_processor import ProcessorConfig


class TestTechnicalProcessorCoverageSupplement:
    """technical_processor补充测试"""

    def test_ema_processor_column_not_exists(self):
        """测试EMAProcessor列不存在错误"""
        processor = EMAProcessor()
        data = pd.DataFrame({'other_col': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="列 close 不存在"):
            processor.calculate(data, {'period': 20, 'column': 'close'})

    def test_ema_processor_get_name(self):
        """测试EMAProcessor.get_name"""
        processor = EMAProcessor()
        assert processor.get_name() == "ema"

    def test_rsi_processor_column_not_exists(self):
        """测试RSIProcessor列不存在错误"""
        processor = RSIProcessor()
        data = pd.DataFrame({'other_col': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="列 close 不存在"):
            processor.calculate(data, {'period': 14, 'column': 'close'})

    def test_rsi_processor_get_name(self):
        """测试RSIProcessor.get_name"""
        processor = RSIProcessor()
        assert processor.get_name() == "rsi"

    def test_macd_processor_column_not_exists(self):
        """测试MACDProcessor列不存在错误"""
        processor = MACDProcessor()
        data = pd.DataFrame({'other_col': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="列 close 不存在"):
            processor.calculate(data, {'fast_period': 12, 'slow_period': 26, 'column': 'close'})

    def test_macd_processor_get_name(self):
        """测试MACDProcessor.get_name"""
        processor = MACDProcessor()
        assert processor.get_name() == "macd"

    def test_bollinger_bands_processor_column_not_exists(self):
        """测试BollingerBandsProcessor列不存在错误"""
        processor = BollingerBandsProcessor()
        data = pd.DataFrame({'other_col': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="列 close 不存在"):
            processor.calculate(data, {'period': 20, 'std_dev': 2.0, 'column': 'close'})

    def test_bollinger_bands_processor_get_name(self):
        """测试BollingerBandsProcessor.get_name"""
        processor = BollingerBandsProcessor()
        assert processor.get_name() == "bbands"

    def test_atr_processor_missing_columns(self):
        """测试ATRProcessor缺失必要列"""
        processor = ATRProcessor()
        data = pd.DataFrame({'high': [1, 2, 3]})  # 缺少low和close
        
        with pytest.raises(ValueError, match="缺失必要列"):
            processor.calculate(data, {'period': 14})

    def test_atr_processor_get_name(self):
        """测试ATRProcessor.get_name"""
        processor = ATRProcessor()
        assert processor.get_name() == "atr"

    def test_sma_processor_get_name(self):
        """测试SMAProcessor.get_name"""
        processor = SMAProcessor()
        assert processor.get_name() == "sma"

    def test_base_technical_processor_init(self):
        """测试BaseTechnicalProcessor初始化（抽象类，不能直接实例化）"""
        # BaseTechnicalProcessor是抽象类，不能直接实例化
        with pytest.raises(TypeError):
            BaseTechnicalProcessor()

    def test_technical_processor_calc_ma_with_defaults(self):
        """测试calc_ma使用默认参数"""
        processor = TechnicalProcessor()
        data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120]
        })
        
        result = processor.calc_ma(data=data)
        assert isinstance(result, pd.Series)
        assert len(result) == len(data)

    def test_technical_processor_calc_ma_with_custom_params(self):
        """测试calc_ma使用自定义参数"""
        processor = TechnicalProcessor()
        data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105]
        })
        
        result = processor.calc_ma(data=data, window=3, price_col='close')
        assert isinstance(result, pd.Series)

    def test_technical_processor_calculate_ma(self):
        """测试calculate_ma方法"""
        processor = TechnicalProcessor()
        data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120]
        })
        
        result = processor.calculate_ma(data=data)
        assert isinstance(result, pd.Series)

    def test_technical_processor_calculate_rsi_with_defaults(self):
        """测试calculate_rsi使用默认参数"""
        processor = TechnicalProcessor()
        data = pd.DataFrame({
            'close': [100, 101, 102, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83]
        })
        
        result = processor.calculate_rsi(data=data)
        assert isinstance(result, pd.Series)

    def test_technical_processor_calculate_macd_with_defaults(self):
        """测试calculate_macd使用默认参数"""
        processor = TechnicalProcessor()
        data = pd.DataFrame({
            'close': [100 + i*0.5 for i in range(50)]
        })
        
        result = processor.calculate_macd(data=data)
        assert isinstance(result, dict)
        assert 'macd' in result
        assert 'signal' in result
        assert 'histogram' in result

    def test_technical_processor_calculate_bollinger_bands_with_defaults(self):
        """测试calculate_bollinger_bands使用默认参数"""
        processor = TechnicalProcessor()
        data = pd.DataFrame({
            'close': [100 + i*0.5 for i in range(30)]
        })
        
        result = processor.calculate_bollinger_bands(data=data)
        assert isinstance(result, dict)
        assert 'upper' in result
        assert 'middle' in result
        assert 'lower' in result

    def test_technical_processor_validate_data_valid(self):
        """测试validate_data（有效数据）"""
        processor = TechnicalProcessor()
        data = pd.DataFrame({
            'close': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'volume': [1000, 1100, 1200]
        })
        
        assert processor.validate_data(data) is True

    def test_technical_processor_validate_data_empty(self):
        """测试validate_data（空数据）"""
        processor = TechnicalProcessor()
        data = pd.DataFrame()
        
        assert processor.validate_data(data) is False

    def test_technical_processor_validate_data_none(self):
        """测试validate_data（None数据）"""
        processor = TechnicalProcessor()
        
        # validate_data可能对None抛出异常，需要捕获
        try:
            result = processor.validate_data(None)
            assert result is False
        except (TypeError, AttributeError):
            # 如果抛出异常也是可以接受的
            pass

    def test_technical_processor_get_supported_indicators(self):
        """测试get_supported_indicators"""
        processor = TechnicalProcessor()
        indicators = processor.get_supported_indicators()
        
        assert isinstance(indicators, list)
        assert len(indicators) > 0
        assert 'sma' in indicators
        assert 'ema' in indicators
        assert 'rsi' in indicators

    def test_technical_processor_compute_feature_existing(self):
        """测试_compute_feature（特征存在）"""
        processor = TechnicalProcessor()
        data = pd.DataFrame({
            'close': [100, 101, 102],
            'sma': [100, 100.5, 101]
        })
        
        result = processor._compute_feature(data, 'sma', {})
        assert isinstance(result, pd.Series)

    def test_technical_processor_get_feature_metadata(self):
        """测试_get_feature_metadata"""
        processor = TechnicalProcessor()
        metadata = processor._get_feature_metadata('sma')
        
        assert isinstance(metadata, dict)
        assert 'name' in metadata
        assert 'type' in metadata

    def test_technical_processor_get_available_features(self):
        """测试_get_available_features"""
        processor = TechnicalProcessor()
        features = processor._get_available_features()
        
        assert isinstance(features, list)

    def test_technical_processor_calculate_indicator_atr(self):
        """测试calculate_indicator（ATR）"""
        processor = TechnicalProcessor()
        data = pd.DataFrame({
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [100, 101, 102, 103, 104]
        })
        
        result = processor.calculate_indicator(data, 'atr', {'period': 14})
        assert isinstance(result, pd.Series)

    def test_technical_processor_calculate_indicator_bollinger_bands(self):
        """测试calculate_indicator（布林带）"""
        processor = TechnicalProcessor()
        data = pd.DataFrame({
            'close': [100 + i*0.5 for i in range(30)]
        })
        
        result = processor.calculate_indicator(data, 'bbands', {'period': 20, 'std_dev': 2.0})
        assert isinstance(result, dict)
        assert 'upper' in result
        assert 'middle' in result
        assert 'lower' in result

    def test_technical_processor_calculate_multiple_indicators_with_atr(self):
        """测试calculate_multiple_indicators（包含ATR）"""
        processor = TechnicalProcessor()
        data = pd.DataFrame({
            'high': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120],
            'low': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'close': [100 + i for i in range(16)],
            'volume': [1000 + i*10 for i in range(16)]
        })
        
        indicators = ['sma', 'atr']
        result = processor.calculate_multiple_indicators(data, indicators, {'period': 5})
        assert isinstance(result, pd.DataFrame)
        assert 'sma' in result.columns

