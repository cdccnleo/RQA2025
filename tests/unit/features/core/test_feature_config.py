# -*- coding: utf-8 -*-
"""
特征配置测试
测试特征层的配置管理功能
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.features.core.config import (
    DefaultConfigs,
    FeatureConfig,
    FeatureProcessingConfig,
    FeatureRegistrationConfig,
    FeatureType,
    OrderBookConfig,
    OrderBookType,
    TechnicalParams,
    SentimentParams,
    TechnicalIndicatorType,
    SentimentType,
)


class TestFeatureConfig:
    """特征配置测试"""

    def test_feature_config_creation(self):
        """测试特征配置创建"""
        config = FeatureConfig(
            feature_types=[FeatureType.TECHNICAL],
            enable_feature_selection=True,
            technical_indicators=["sma", "rsi"]
        )

        assert config.feature_types == [FeatureType.TECHNICAL]
        assert config.enable_feature_selection is True
        assert config.technical_indicators == ["sma", "rsi"]
        assert isinstance(config.technical_params, TechnicalParams)

    def test_feature_config_defaults(self):
        """测试特征配置默认值"""
        config = FeatureConfig()

        assert config.feature_types == [FeatureType.TECHNICAL]
        assert config.enable_feature_selection is True
        assert config.enable_standardization is True
        assert config.technical_indicators == ["sma", "ema", "rsi", "macd", "bbands", "atr"]
        assert isinstance(config.technical_params, TechnicalParams)

    def test_feature_config_validation(self):
        """测试特征配置验证"""
        config = FeatureConfig(
            feature_types=[FeatureType.TECHNICAL],
            technical_indicators=["sma"],
            max_features=10,
            min_features=5,
            chunk_size=200,
        )
        assert config.validate() is True

    def test_feature_config_validation_failure(self, capsys):
        config = FeatureConfig(feature_types=[], technical_indicators=["sma"])
        assert config.validate() is False
        message = capsys.readouterr().out
        assert "至少需要指定一种特征类型" in message


class TestFeatureProcessingConfig:
    """特征处理配置测试"""

    def test_processing_config_creation(self):
        """测试处理配置创建"""
        config = FeatureProcessingConfig(
            batch_size=100,
            max_memory_usage=512.0,
            enable_memory_monitoring=False,
            retry_count=5
        )

        assert config.batch_size == 100
        assert config.max_memory_usage == 512.0
        assert config.enable_memory_monitoring is False
        assert config.retry_count == 5

    def test_processing_config_defaults(self):
        """测试处理配置默认值"""
        config = FeatureProcessingConfig()

        assert config.batch_size == 1000
        assert config.timeout == 30.0
        assert config.retry_count == 3
        assert config.enable_memory_monitoring is True
        assert config.continue_on_error is False

    def test_processing_config_validation(self):
        """测试处理配置验证"""
        config = FeatureProcessingConfig(batch_size=500, chunk_size=50, max_workers=2)
        assert config.to_dict()["batch_size"] == 500


class TestTechnicalParams:
    """技术参数测试"""

    def test_technical_params_creation(self):
        """测试技术参数创建"""
        config = FeatureConfig(
            technical_indicators=["sma"],
            technical_params=FeatureConfig().technical_params,
        )
        assert "sma" in config.technical_indicators

    def test_technical_params_defaults(self):
        """测试技术参数默认值"""
        params = TechnicalParams()

        assert params.rsi_period == 14
        assert params.macd_fast == 12
        assert params.macd_slow == 26
        assert params.macd_signal == 9
        assert params.bb_period == 20
        assert params.bb_std == 2.0
        assert params.atr_period == 14
        assert params.sma_periods == [5, 10, 20, 50]
        assert params.ema_periods == [12, 26]

    def test_technical_params_validation(self):
        """测试技术参数验证"""
        # 有效的参数
        params = TechnicalParams(rsi_period=21)
        assert params.rsi_period == 21

        # 边界值测试
        params_min = TechnicalParams(rsi_period=1)
        assert params_min.rsi_period == 1

        params_large = TechnicalParams(rsi_period=100)
        assert params_large.rsi_period == 100


class TestSentimentParams:
    """情感参数测试"""

    def test_sentiment_params_creation(self):
        """测试情感参数创建"""
        config = FeatureConfig(sentiment_params=FeatureConfig().sentiment_params)
        assert config.sentiment_params.news_lookback_days == 30

    def test_sentiment_params_defaults(self):
        """测试情感参数默认值"""
        params = SentimentParams()

        assert params.news_lookback_days == 30
        assert params.social_lookback_days == 7
        assert params.min_confidence == 0.6
        assert params.max_keywords == 100
        assert params.language == "zh - cn"

    def test_sentiment_params_validation(self):
        """测试情感参数验证"""
        # 有效的参数
        reg_config = FeatureRegistrationConfig(
            name="alpha",
            feature_type=FeatureType.TECHNICAL,
            params={"window": 5},
        )
        assert reg_config.name == "alpha"
        assert reg_config.params["window"] == 5


class TestOrderBookConfig:
    """订单簿配置测试"""

    def test_orderbook_config_creation(self):
        """测试订单簿配置创建"""
        config = OrderBookConfig(
            depth=15,
            update_frequency=0.5,
            enable_imbalance_analysis=False,
            enable_skew_analysis=True,
            imbalance_threshold=0.2,
            window_size=30
        )

        assert config.depth == 15
        assert config.update_frequency == 0.5
        assert config.enable_imbalance_analysis is False
        assert config.enable_skew_analysis is True
        assert config.imbalance_threshold == 0.2
        assert config.window_size == 30

    def test_orderbook_config_defaults(self):
        """测试订单簿配置默认值"""
        config = OrderBookConfig()

        assert config.depth == 10
        assert config.update_frequency == 1.0
        assert config.enable_imbalance_analysis is True
        assert config.enable_skew_analysis is True
        assert config.enable_spread_analysis is True
        assert config.imbalance_threshold == 0.1
        assert config.window_size == 20

    def test_orderbook_config_validation(self):
        """测试订单簿配置验证"""
        # 有效的深度
        config = OrderBookConfig(depth=20)
        assert config.depth == 20

        # 边界值测试
        config_min = OrderBookConfig(depth=1)
        assert config_min.depth == 1

        config_large = OrderBookConfig(depth=50)
        assert config_large.depth == 50

    def test_orderbook_config_to_dict_and_from_dict(self):
        config = OrderBookConfig(
            orderbook_type=OrderBookType.LEVEL3,
            depth=20,
            update_frequency=0.25,
            custom_config={"alpha": 1},
        )
        data = config.to_dict()
        assert data["orderbook_type"] == "level3"
        restored = OrderBookConfig.from_dict(data)
        assert restored.orderbook_type == OrderBookType.LEVEL3
        assert restored.depth == 20

    def test_orderbook_config_validate_failure(self, capsys):
        config = OrderBookConfig(depth=0, update_frequency=-1, max_workers=0, batch_size=0)
        assert config.validate() is False
        captured = capsys.readouterr().out
        assert "OrderBookConfig验证失败" in captured


class TestTechnicalIndicatorType:
    """技术指标类型枚举测试"""

    def test_indicator_type_values(self):
        """测试指标类型枚举值"""
        assert TechnicalIndicatorType.SMA.value == "sma"
        assert TechnicalIndicatorType.EMA.value == "ema"
        assert TechnicalIndicatorType.MACD.value == "macd"
        assert TechnicalIndicatorType.RSI.value == "rsi"
        assert TechnicalIndicatorType.ADX.value == "adx"
        assert TechnicalIndicatorType.STOCH.value == "stoch"

    def test_indicator_type_list(self):
        """测试指标类型列表"""
        all_types = list(TechnicalIndicatorType)
        assert len(all_types) == 17  # 根据实际枚举值调整
        assert TechnicalIndicatorType.RSI in all_types
        assert TechnicalIndicatorType.MACD in all_types


class TestSentimentType:
    """情感类型枚举测试"""

    def test_sentiment_type_values(self):
        """测试情感类型枚举值"""
        assert SentimentType.NEWS_SENTIMENT.value == "news_sentiment"
        assert SentimentType.SOCIAL_SENTIMENT.value == "social_sentiment"
        assert SentimentType.EARNINGS_SENTIMENT.value == "earnings_sentiment"
        assert SentimentType.ANALYST_SENTIMENT.value == "analyst_sentiment"

    def test_sentiment_type_list(self):
        """测试情感类型列表"""
        all_types = list(SentimentType)
        assert len(all_types) == 4
        assert SentimentType.NEWS_SENTIMENT in all_types
        assert SentimentType.SOCIAL_SENTIMENT in all_types


class TestOrderBookType:
    """订单簿类型枚举测试"""

    def test_orderbook_type_values(self):
        """测试订单簿类型枚举值"""
        assert OrderBookType.LEVEL1.value == "level1"
        assert OrderBookType.LEVEL2.value == "level2"
        assert OrderBookType.LEVEL3.value == "level3"

    def test_orderbook_type_list(self):
        """测试订单簿类型列表"""
        all_types = list(OrderBookType)
        assert len(all_types) >= 3
        assert OrderBookType.LEVEL2 in all_types


class TestDefaultConfigs:
    """默认配置测试"""

    def test_default_configs_static_methods(self):
        """测试默认配置静态方法"""
        # 检查静态方法存在
        assert hasattr(DefaultConfigs, 'get_basic_config')
        assert hasattr(DefaultConfigs, 'basic_technical')
        assert hasattr(DefaultConfigs, 'comprehensive_technical')

        # 测试静态方法调用
        basic_config = DefaultConfigs.get_basic_config()
        assert isinstance(basic_config, FeatureConfig)
        assert basic_config.feature_types == [FeatureType.TECHNICAL]

        tech_config = DefaultConfigs.basic_technical()
        assert isinstance(tech_config, FeatureConfig)
        assert "sma" in tech_config.technical_indicators
        assert "rsi" in tech_config.technical_indicators

        comp_config = DefaultConfigs.comprehensive_technical()
        assert isinstance(comp_config, FeatureConfig)
        assert len(comp_config.technical_indicators) > 5  # 应该包含更多指标

    def test_default_configs_values(self):
        """测试默认配置值"""
        basic_config = DefaultConfigs.get_basic_config()

        # 检查特征配置默认值
        assert basic_config.feature_types == [FeatureType.TECHNICAL]
        assert basic_config.enable_feature_selection is False
        assert basic_config.enable_standardization is True

        # 检查技术指标
        assert "sma" in basic_config.technical_indicators
        assert "rsi" in basic_config.technical_indicators
        assert "macd" in basic_config.technical_indicators


class TestConfigIntegration:
    """配置集成测试"""

    def test_config_serialization(self):
        """测试配置序列化"""
        config = FeatureConfig(
            feature_types=[FeatureType.TECHNICAL],
            enable_feature_selection=True,
            max_features=100
        )

        # 转换为字典（模拟序列化）
        config_dict = {
            'feature_types': [ft.value for ft in config.feature_types],
            'enable_feature_selection': config.enable_feature_selection,
            'max_features': config.max_features
        }

        assert config_dict['feature_types'] == ["technical"]
        assert config_dict['enable_feature_selection'] is True
        assert config_dict['max_features'] == 100

    def test_config_equality(self):
        """测试配置相等性"""
        config1 = FeatureConfig(feature_types=[FeatureType.TECHNICAL], max_features=50)
        config2 = FeatureConfig(feature_types=[FeatureType.TECHNICAL], max_features=50)
        config3 = FeatureConfig(feature_types=[FeatureType.SENTIMENT], max_features=100)

        # 相同配置应该相等（这里简单比较主要属性）
        assert config1.max_features == config2.max_features
        assert config1.feature_types == config2.feature_types

        # 不同配置应该不相等
        assert config1.max_features != config3.max_features
        assert config1.feature_types != config3.feature_types

    def test_config_copy(self):
        """测试配置复制"""
        original = FeatureConfig(
            feature_types=[FeatureType.TECHNICAL],
            max_features=50
        )

        # 复制配置
        copied = FeatureConfig(
            feature_types=original.feature_types.copy(),
            max_features=original.max_features
        )

        assert copied.max_features == original.max_features
        assert copied.feature_types == original.feature_types

        # 修改复制的配置不影响原配置
        copied.max_features = 100
        assert original.max_features == 50
        assert copied.max_features == 100


def test_feature_config_to_dict_and_json_roundtrip():
    config = FeatureConfig(
        feature_types=[FeatureType.TECHNICAL, FeatureType.SENTIMENT],
        enable_feature_selection=False,
        max_features=20,
        technical_params=TechnicalParams(sma_periods=[3, 5], macd_fast=10),
        sentiment_params=SentimentParams(language="en"),
    )
    data = config.to_dict()
    assert data["feature_types"] == ["technical", "sentiment"]
    assert data["technical_params"]["macd_fast"] == 10

    json_str = config.to_json()
    restored = FeatureConfig.from_json(json_str)
    assert restored.feature_types == [FeatureType.TECHNICAL, FeatureType.SENTIMENT]
    assert restored.technical_params.macd_fast == 10
    assert restored.sentiment_params.language == "en"


def test_feature_config_validate_errors(capsys):
    config = FeatureConfig(feature_types=[], technical_indicators=[])
    assert config.validate() is False
    captured = capsys.readouterr().out
    assert "至少需要指定一种特征类型" in captured

    config = FeatureConfig(feature_types=[FeatureType.TECHNICAL], technical_indicators=[])
    assert config.validate() is False

    config = FeatureConfig(feature_types=[FeatureType.TECHNICAL], technical_params=TechnicalParams(sma_periods=[0]))
    assert config.validate() is False


def test_feature_config_validate_success_with_all_types():
    config = FeatureConfig(
        feature_types=[FeatureType.TECHNICAL, FeatureType.SENTIMENT, FeatureType.ORDERBOOK],
        technical_indicators=["sma"],
        sentiment_types=["news_sentiment"],
    )
    assert config.validate() is True


def test_feature_processing_config_to_dict_and_from_dict():
    cfg = FeatureProcessingConfig(batch_size=200, enable_parallel_processing=False)
    data = cfg.to_dict()
    assert data["batch_size"] == 200
    assert data["enable_parallel_processing"] is False
    restored = FeatureProcessingConfig.from_dict(data)
    assert restored.batch_size == 200
    assert restored.enable_parallel_processing is False
