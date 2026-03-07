#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Financial Data Loader 单元测试

测试 FinancialDataLoader 的核心功能，包括：
- 初始化和配置
- 数据加载和验证
- 市场和数据类型支持
- 错误处理
"""

import pytest
import time
from unittest.mock import Mock, patch
from typing import Dict, Any

from src.data.loader.financial_loader import FinancialDataLoader
from src.data.core.base_loader import LoaderConfig


class TestFinancialDataLoader:
    """FinancialDataLoader 单元测试"""

    def test_initialization_without_config(self):
        """测试无配置初始化"""
        loader = FinancialDataLoader()

        assert loader.config is None
        assert loader._supported_markets == ('CN', 'US', 'HK', 'JP')
        assert loader._supported_data_types == ('stock', 'index', 'fund', 'bond')
        assert not loader.is_initialized

    def test_initialization_with_config(self):
        """测试带配置初始化"""
        config = LoaderConfig(name="test_financial_loader")
        loader = FinancialDataLoader(config)

        assert loader.config == config
        assert loader.is_initialized

    def test_supported_markets_property(self):
        """测试 supported_markets 属性"""
        loader = FinancialDataLoader()

        markets = loader.supported_markets
        assert isinstance(markets, list)
        assert set(markets) == {'CN', 'US', 'HK', 'JP'}

        # 确保返回的是副本，不会影响原始数据
        markets.append('TEST')
        assert 'TEST' not in loader.supported_markets

    def test_supported_data_types_property(self):
        """测试 supported_data_types 属性"""
        loader = FinancialDataLoader()

        data_types = loader.supported_data_types
        assert isinstance(data_types, list)
        assert set(data_types) == {'stock', 'index', 'fund', 'bond'}

        # 确保返回的是副本
        data_types.append('test')
        assert 'test' not in loader.supported_data_types

    def test_get_metadata(self):
        """测试获取元数据"""
        loader = FinancialDataLoader()
        metadata = loader.get_metadata()

        assert isinstance(metadata, dict)
        assert metadata['loader'] == 'FinancialDataLoader'
        assert 'initialized' in metadata
        assert 'supported_markets' in metadata
        assert 'supported_data_types' in metadata

    def test_load_data_success(self):
        """测试成功加载数据"""
        config = LoaderConfig(name="test_loader")
        loader = FinancialDataLoader(config)

        result = loader.load_data('000001', market='CN', data_type='stock')

        assert isinstance(result, dict)
        assert result['symbol'] == '000001'
        assert result['market'] == 'CN'
        assert result['data_type'] == 'stock'
        assert result['price'] == 100.0
        assert result['volume'] == 1_000_000
        assert result['source'] == 'FinancialDataLoader'
        assert result['status'] == 'success'
        assert 'timestamp' in result

    def test_load_data_with_kwargs(self):
        """测试带额外参数的数据加载"""
        config = LoaderConfig(name="test_loader")
        loader = FinancialDataLoader(config)

        result = loader.load_data('AAPL', market='US', data_type='stock',
                                custom_field='test_value', quantity=100)

        assert result['custom_field'] == 'test_value'
        assert result['quantity'] == 100

    def test_load_method_delegation(self):
        """测试 load 方法委托给 load_data"""
        config = LoaderConfig(name="test_loader")
        loader = FinancialDataLoader(config)

        with patch.object(loader, 'load_data') as mock_load_data:
            mock_load_data.return_value = {'test': 'data'}
            result = loader.load('000001', market='CN')

            mock_load_data.assert_called_once_with('000001', market='CN', data_type='stock')
            assert result == {'test': 'data'}

    def test_load_data_not_initialized(self):
        """测试未初始化时加载数据"""
        loader = FinancialDataLoader()

        with pytest.raises(RuntimeError, match="Loader not initialized"):
            loader.load_data('000001')

    def test_load_data_empty_symbol(self):
        """测试空符号"""
        config = LoaderConfig(name="test_loader")
        loader = FinancialDataLoader(config)

        with pytest.raises(ValueError, match="Symbol is required"):
            loader.load_data('')

    def test_load_data_unsupported_market(self):
        """测试不支持的市场"""
        config = LoaderConfig(name="test_loader")
        loader = FinancialDataLoader(config)

        with pytest.raises(ValueError, match="Unsupported market: INVALID"):
            loader.load_data('000001', market='INVALID')

    def test_load_data_unsupported_data_type(self):
        """测试不支持的数据类型"""
        config = LoaderConfig(name="test_loader")
        loader = FinancialDataLoader(config)

        with pytest.raises(ValueError, match="Unsupported data type: invalid"):
            loader.load_data('000001', data_type='invalid')

    def test_validate_data_valid(self):
        """测试有效数据验证"""
        loader = FinancialDataLoader()

        valid_data = {
            'symbol': '000001',
            'price': 100.0,
            'timestamp': time.time(),
            'extra_field': 'value'
        }

        assert loader.validate_data(valid_data) is True

    def test_validate_data_invalid_type(self):
        """测试无效数据类型验证"""
        loader = FinancialDataLoader()

        assert loader.validate_data("not_a_dict") is False
        assert loader.validate_data(123) is False
        assert loader.validate_data(None) is False

    def test_validate_data_missing_fields(self):
        """测试缺失必要字段的数据验证"""
        loader = FinancialDataLoader()

        # 缺少 symbol
        invalid_data1 = {'price': 100.0, 'timestamp': time.time()}
        assert loader.validate_data(invalid_data1) is False

        # 缺少 price
        invalid_data2 = {'symbol': '000001', 'timestamp': time.time()}
        assert loader.validate_data(invalid_data2) is False

        # 缺少 timestamp
        invalid_data3 = {'symbol': '000001', 'price': 100.0}
        assert loader.validate_data(invalid_data3) is False

    def test_validate_data_empty_dict(self):
        """测试空字典验证"""
        loader = FinancialDataLoader()

        assert loader.validate_data({}) is False

    def test_constants(self):
        """测试常量定义"""
        loader = FinancialDataLoader()

        assert FinancialDataLoader.DEFAULT_MARKETS == ('CN', 'US', 'HK', 'JP')
        assert FinancialDataLoader.DEFAULT_DATA_TYPES == ('stock', 'index', 'fund', 'bond')

    def test_different_market_data_types(self):
        """测试不同市场和数据类型的组合"""
        config = LoaderConfig(name="test_loader")
        loader = FinancialDataLoader(config)

        test_cases = [
            ('000001', 'CN', 'stock'),
            ('AAPL', 'US', 'stock'),
            ('000001', 'HK', 'index'),
            ('BND', 'US', 'bond'),
            ('159919', 'CN', 'fund'),
        ]

        for symbol, market, data_type in test_cases:
            result = loader.load_data(symbol, market=market, data_type=data_type)
            assert result['symbol'] == symbol
            assert result['market'] == market
            assert result['data_type'] == data_type
            assert result['status'] == 'success'

    def test_timestamp_accuracy(self):
        """测试时间戳准确性"""
        config = LoaderConfig(name="test_loader")
        loader = FinancialDataLoader(config)

        start_time = time.time()
        result = loader.load_data('000001')
        end_time = time.time()

        assert start_time <= result['timestamp'] <= end_time
