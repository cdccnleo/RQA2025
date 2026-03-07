#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High Frequency Optimizer测试覆盖
测试high_freq_optimizer.py
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List

try:
    from src.features.performance.high_freq_optimizer import (
        HighFreqConfig,
        HighFreqOptimizer,
        CppHighFreqOptimizer,
        NUMBA_AVAILABLE
    )
    HIGH_FREQ_OPTIMIZER_AVAILABLE = True
except ImportError:
    HIGH_FREQ_OPTIMIZER_AVAILABLE = False
    HighFreqConfig = None
    HighFreqOptimizer = None
    CppHighFreqOptimizer = None


@pytest.mark.skipif(not HIGH_FREQ_OPTIMIZER_AVAILABLE, reason="HighFreqOptimizer not available")
class TestHighFreqConfig:
    """HighFreqConfig测试"""

    def test_config_initialization_default(self):
        """测试默认配置初始化"""
        config = HighFreqConfig()
        assert config.batch_size == 100
        assert config.prealloc_memory == 256
        assert config.parallel_threshold == 1000
        assert config.use_simd is True
        assert config.max_retries == 3
        assert config.enable_fallback is False

    def test_config_initialization_custom(self):
        """测试自定义配置初始化"""
        config = HighFreqConfig(
            batch_size=200,
            prealloc_memory=512,
            parallel_threshold=2000,
            use_simd=False,
            max_retries=5,
            enable_fallback=True
        )
        assert config.batch_size == 200
        assert config.prealloc_memory == 512
        assert config.parallel_threshold == 2000
        assert config.use_simd is False
        assert config.max_retries == 5
        assert config.enable_fallback is True


@pytest.mark.skipif(not HIGH_FREQ_OPTIMIZER_AVAILABLE, reason="HighFreqOptimizer not available")
class TestHighFreqOptimizer:
    """HighFreqOptimizer测试"""

    @pytest.fixture
    def mock_feature_engine(self):
        """创建模拟的FeatureEngineer"""
        engine = Mock()
        engine.register_feature = Mock()
        return engine

    @pytest.fixture
    def mock_level2_analyzer(self):
        """创建模拟的Level2Analyzer"""
        analyzer = Mock()
        analyzer.calculate_all_features = Mock()
        return analyzer

    @pytest.fixture
    def optimizer(self, mock_feature_engine):
        """创建HighFreqOptimizer实例"""
        with patch('src.features.performance.high_freq_optimizer.Level2Analyzer') as mock_level2:
            mock_level2.return_value = Mock()
            return HighFreqOptimizer(mock_feature_engine)

    def test_optimizer_initialization_default_config(self, mock_feature_engine):
        """测试优化器默认配置初始化"""
        with patch('src.features.performance.high_freq_optimizer.Level2Analyzer'):
            optimizer = HighFreqOptimizer(mock_feature_engine)
            assert optimizer.engine == mock_feature_engine
            assert optimizer.config.batch_size == 100
            assert hasattr(optimizer, 'level2_analyzer')

    def test_optimizer_initialization_custom_config(self, mock_feature_engine):
        """测试优化器自定义配置初始化"""
        config = HighFreqConfig(batch_size=200, prealloc_memory=0)
        with patch('src.features.performance.high_freq_optimizer.Level2Analyzer'):
            optimizer = HighFreqOptimizer(mock_feature_engine, config)
            assert optimizer.config.batch_size == 200
            assert not hasattr(optimizer, 'feature_buffer')  # prealloc_memory=0时不预分配

    def test_optimizer_preallocate_memory(self, mock_feature_engine):
        """测试预分配内存"""
        config = HighFreqConfig(batch_size=50, prealloc_memory=256)
        with patch('src.features.performance.high_freq_optimizer.Level2Analyzer'):
            optimizer = HighFreqOptimizer(mock_feature_engine, config)
            assert hasattr(optimizer, 'feature_buffer')
            assert optimizer.feature_buffer.shape == (50, 10)
            assert optimizer.temp_buffer.shape == (50, 5)

    def test_optimizer_register_features(self, mock_feature_engine):
        """测试注册特征"""
        with patch('src.features.performance.high_freq_optimizer.Level2Analyzer'):
            optimizer = HighFreqOptimizer(mock_feature_engine)
            # 应该注册了3个特征
            assert mock_feature_engine.register_feature.call_count == 3

    def test_calculate_momentum_small_batch(self, optimizer):
        """测试计算动量（小批量）"""
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0])
        volumes = np.array([1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100])
        
        result = optimizer.calculate_hf_momentum(prices, volumes)
        
        assert len(result) == len(prices)
        assert result[0] == 0.0  # 前10个应该是0
        assert result[10] != 0.0  # 第11个应该有值

    def test_calculate_momentum_large_batch(self, optimizer):
        """测试计算动量（大批量，使用numba）"""
        # 创建超过parallel_threshold的数据
        optimizer.config.parallel_threshold = 10
        prices = np.random.rand(100) * 100 + 100
        volumes = np.random.rand(100) * 1000 + 1000
        
        result = optimizer.calculate_hf_momentum(prices, volumes)
        
        assert len(result) == len(prices)
        assert isinstance(result, np.ndarray)

    def test_calculate_momentum_empty(self, optimizer):
        """测试计算动量（空数组）"""
        prices = np.array([])
        volumes = np.array([])
        
        result = optimizer.calculate_hf_momentum(prices, volumes)
        
        assert len(result) == 0

    def test_calculate_order_flow_imbalance(self, optimizer):
        """测试计算订单流不平衡"""
        order_book = {
            "bid": np.array([[100.0, 1000], [99.9, 2000], [99.8, 3000]]),
            "ask": np.array([[100.1, 500], [100.2, 600], [100.3, 700]]),
            "trade": np.array([[100.0, 1, 100], [100.1, -1, 50]])  # [price, direction, volume]
        }
        
        result = optimizer.calculate_order_flow_imbalance(order_book)
        
        assert isinstance(result, (float, np.floating))
        # bid_vol > ask_vol，应该返回正值
        assert result > 0

    def test_calculate_order_flow_imbalance_no_trades(self, optimizer):
        """测试计算订单流不平衡（无交易）"""
        order_book = {
            "bid": np.array([[100.0, 1000], [99.9, 2000]]),
            "ask": np.array([[100.1, 500], [100.2, 600]])
        }
        
        result = optimizer.calculate_order_flow_imbalance(order_book)
        
        assert isinstance(result, (float, np.floating))

    def test_calculate_instant_volatility(self, optimizer):
        """测试计算瞬时波动率"""
        prices = np.array([100.0, 101.0, 102.0, 101.5, 103.0, 102.5, 104.0, 103.5, 105.0, 104.5,
                          106.0, 105.5, 107.0, 106.5, 108.0, 107.5, 109.0, 108.5, 110.0, 109.5, 111.0])
        
        result = optimizer.calculate_instant_volatility(prices)
        
        assert isinstance(result, (float, np.floating))
        assert result >= 0  # 波动率应该非负

    def test_calculate_instant_volatility_insufficient_data(self, optimizer):
        """测试计算瞬时波动率（数据不足）"""
        prices = np.array([100.0, 101.0])  # 少于20个数据点
        
        result = optimizer.calculate_instant_volatility(prices)
        
        assert result == 0.0

    def test_batch_calculate_features(self, optimizer, mock_level2_analyzer):
        """测试批量计算特征"""
        optimizer.level2_analyzer = mock_level2_analyzer
        
        data_batch = [
            {
                "price": np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0]),
                "volume": np.array([1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]),
                "bid": np.array([[100.0, 1000]]),
                "ask": np.array([[100.1, 500]])
            },
            {
                "price": np.array([110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0]),
                "volume": np.array([2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000]),
                "bid": np.array([[110.0, 2000]]),
                "ask": np.array([[110.1, 1000]])
            }
        ]
        
        results = optimizer.batch_calculate_features(data_batch)
        
        assert "HF_MOMENTUM" in results
        assert "ORDER_FLOW_IMBALANCE" in results
        assert "INSTANT_VOLATILITY" in results
        assert len(results["HF_MOMENTUM"]) == 2
        assert len(results["ORDER_FLOW_IMBALANCE"]) == 2
        assert len(results["INSTANT_VOLATILITY"]) == 2

    def test_batch_calculate_features_empty_batch(self, optimizer):
        """测试批量计算特征（空批次）"""
        results = optimizer.batch_calculate_features([])
        
        assert "HF_MOMENTUM" in results
        assert len(results["HF_MOMENTUM"]) == 0

    def test_batch_calculate_features_missing_price(self, optimizer, mock_level2_analyzer):
        """测试批量计算特征（缺少价格数据）"""
        optimizer.level2_analyzer = mock_level2_analyzer
        
        data_batch = [
            {
                "volume": np.array([1000, 1100]),
                "bid": np.array([[100.0, 1000]]),
                "ask": np.array([[100.1, 500]])
            }
        ]
        
        results = optimizer.batch_calculate_features(data_batch)
        
        assert results["HF_MOMENTUM"][0] == 0.0
        assert results["INSTANT_VOLATILITY"][0] == 0.0


@pytest.mark.skipif(not HIGH_FREQ_OPTIMIZER_AVAILABLE, reason="CppHighFreqOptimizer not available")
class TestCppHighFreqOptimizer:
    """CppHighFreqOptimizer测试"""

    @pytest.fixture
    def mock_feature_engine(self):
        """创建模拟的FeatureEngineer"""
        engine = Mock()
        engine.register_feature = Mock()
        return engine

    def test_cpp_optimizer_initialization(self, mock_feature_engine):
        """测试C++优化器初始化"""
        with patch('src.features.performance.high_freq_optimizer.Level2Analyzer'):
            optimizer = CppHighFreqOptimizer(mock_feature_engine)
            assert optimizer.cpp_optimizer is None  # 暂时禁用C++扩展
            assert isinstance(optimizer, HighFreqOptimizer)

    def test_cpp_optimizer_batch_calculate_fallback(self, mock_feature_engine):
        """测试C++优化器回退到Python实现"""
        with patch('src.features.performance.high_freq_optimizer.Level2Analyzer'):
            optimizer = CppHighFreqOptimizer(mock_feature_engine)
            
            data_batch = [
                {
                    "price": np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0]),
                    "volume": np.array([1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]),
                    "bid": np.array([[100.0, 1000]]),
                    "ask": np.array([[100.1, 500]])
                }
            ]
            
            results = optimizer.batch_calculate_features(data_batch)
            
            assert "HF_MOMENTUM" in results
            assert "ORDER_FLOW_IMBALANCE" in results
            assert "INSTANT_VOLATILITY" in results


