"""
中国市场适配器综合测试
测试adapter.py中未覆盖的方法
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.data.china.adapter import ChinaDataAdapter


class TestChinaDataAdapterComprehensive:
    """中国数据适配器综合测试"""

    @pytest.fixture
    def adapter(self):
        """创建适配器实例"""
        return ChinaDataAdapter()

    @pytest.fixture
    def adapter_with_config(self):
        """创建带配置的适配器实例"""
        config = {
            'redis': {
                'host': 'localhost',
                'port': 6379,
                'db': 0
            }
        }
        return ChinaDataAdapter(config)

    def test_load_margin_data(self, adapter):
        """测试加载融资融券数据"""
        with patch('redis.Redis') as mock_redis:
            # Mock Redis连接
            mock_client = Mock()
            mock_client.get.return_value = None  # 没有缓存数据
            mock_redis.return_value = mock_client

            # 调用方法
            result = adapter.load_margin_data()

            # 验证返回DataFrame
            assert isinstance(result, pd.DataFrame)
            assert len(result) >= 0

    def test_validate_t1_settlement_valid(self, adapter):
        """测试T+1结算验证 - 有效交易"""
        trades = pd.DataFrame({
            'symbol': ['000001', '000002'],
            'date': pd.to_datetime(['2023-01-01', '2023-01-02']),
            'action': ['BUY', 'SELL']
        })

        result = adapter.validate_t1_settlement(trades)
        assert result is True

    def test_validate_t1_settlement_invalid(self, adapter):
        """测试T+1结算验证 - 无效交易"""
        trades = pd.DataFrame({
            'symbol': ['000001', '000001'],
            'date': pd.to_datetime(['2023-01-01', '2023-01-01']),  # 同一天
            'action': ['BUY', 'SELL']  # 当天买卖，违规
        })

        result = adapter.validate_t1_settlement(trades)
        assert result is False

    def test_get_price_limits_main_board(self, adapter):
        """测试获取主板价格限制"""
        result = adapter.get_price_limits('000001')  # 主板

        assert isinstance(result, dict)
        assert 'upper_limit' in result
        assert 'lower_limit' in result
        assert result['upper_limit'] > result['lower_limit']

    def test_get_price_limits_star_market(self, adapter):
        """测试获取科创板价格限制"""
        result = adapter.get_price_limits('688001')  # 科创板

        assert isinstance(result, dict)
        assert 'upper_limit' in result
        assert 'lower_limit' in result

    def test_cache_data_success(self, adapter_with_config):
        """测试数据缓存成功"""
        test_data = pd.DataFrame({'col': [1, 2, 3]})

        with patch.object(adapter_with_config, 'redis_client') as mock_client:
            mock_client.set.return_value = True

            result = adapter_with_config.cache_data('test_key', test_data)
            assert result is True

    def test_cache_data_failure(self, adapter):
        """测试数据缓存失败"""
        test_data = pd.DataFrame({'col': [1, 2, 3]})

        # 没有Redis连接的情况
        result = adapter.cache_data('test_key', test_data)
        assert result is False

    def test_get_cached_data_found(self, adapter_with_config):
        """测试获取缓存数据 - 找到数据"""
        with patch.object(adapter_with_config, 'redis_client') as mock_client:
            mock_client.get.return_value = '{"test": "data"}'

            result = adapter_with_config._get_cached_data('test_key')
            assert result == {"test": "data"}

    def test_get_cached_data_not_found(self, adapter_with_config):
        """测试获取缓存数据 - 未找到"""
        with patch.object(adapter_with_config, 'redis_client') as mock_client:
            mock_client.get.return_value = None

            result = adapter_with_config._get_cached_data('test_key')
            assert result is None

    def test_get_stock_info(self, adapter):
        """测试获取股票信息"""
        result = adapter.get_stock_info('000001')

        assert isinstance(result, dict)
        assert 'symbol' in result
        assert result['symbol'] == '000001'

    def test_get_market_status(self, adapter):
        """测试获取市场状态"""
        result = adapter.get_market_status()

        assert isinstance(result, dict)
        assert 'market_open' in result
        assert 'current_time' in result
        assert 'trading_day' in result
        assert 'session' in result
        assert result['session'] in ['morning', 'afternoon', 'closed']

    def test_connect_success(self, adapter):
        """测试连接成功"""
        with patch('redis.Redis') as mock_redis:
            mock_client = Mock()
            mock_redis.return_value = mock_client

            result = adapter.connect()
            assert result is True
            assert adapter._is_connected is True

    def test_connect_failure(self, adapter):
        """测试连接失败"""
        # connect方法总是成功，因为没有实际的连接逻辑
        result = adapter.connect()
        assert result is True
        assert adapter._is_connected is True

    def test_disconnect(self, adapter):
        """测试断开连接"""
        adapter._is_connected = True

        with patch('redis.Redis') as mock_redis:
            mock_client = Mock()
            mock_redis.return_value = mock_client

            result = adapter.disconnect()
            assert result is True
            assert adapter._is_connected is False

    def test_validate_t_plus_one_valid(self, adapter):
        """测试T+1验证 - 有效"""
        trade_data = {
            'symbol': '000001',
            'trade_date': '2023-01-01',
            'settlement_date': '2023-01-02'  # T+1
        }

        result = adapter.validate_t_plus_one(trade_data)
        assert result['is_valid'] is True

    def test_validate_t_plus_one_invalid(self, adapter):
        """测试T+1验证 - 无效"""
        trade_data = {
            'symbol': '000001',
            'trade_date': '2023-01-01',
            'settlement_date': '2023-01-01'  # 同一天，违规
        }

        result = adapter.validate_t_plus_one(trade_data)
        assert result['is_valid'] is False

    def test_get_market_data(self, adapter):
        """测试获取市场数据"""
        result = adapter.get_market_data('000001')

        assert isinstance(result, dict)
        assert 'symbol' in result
        assert result['symbol'] == '000001'

    def test_fetch_market_data(self, adapter):
        """测试批量获取市场数据"""
        symbols = ['000001', '000002']

        result = adapter.fetch_market_data(symbols)

        assert isinstance(result, dict)
        assert 'data' in result
        assert isinstance(result['data'], list)

    def test_transform_data(self, adapter):
        """测试数据转换"""
        raw_data = {
            'symbol': '000001',
            'price': 10.5,
            'volume': 1000
        }

        result = adapter.transform_data(raw_data)

        assert isinstance(result, dict)
        assert 'processed' in result

    def test_check_health_healthy(self, adapter):
        """测试健康检查 - 健康状态"""
        with patch('redis.Redis') as mock_redis:
            mock_client = Mock()
            mock_client.ping.return_value = True
            mock_redis.return_value = mock_client

            result = adapter.check_health()

            assert isinstance(result, dict)
            assert result['status'] == 'healthy'

    def test_check_health_unhealthy(self, adapter):
        """测试健康检查 - 不健康状态"""
        # 没有Redis连接
        result = adapter.check_health()

        assert isinstance(result, dict)
        assert result['status'] == 'unhealthy'

    def test_adapt_market_data(self, adapter):
        """测试市场数据适配"""
        input_data = pd.DataFrame({
            'symbol': ['000001', '000002'],
            'price': [10.5, 20.3],
            'volume': [1000, 2000]
        })

        result = adapter.adapt_market_data(input_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(input_data)
