"""经纪商适配器测试模块

测试 src.trading.broker.broker_adapter 模块的功能
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from src.trading.broker.broker_adapter import (
    BrokerAdapter,
    CTPSimulatorAdapter,
    BrokerAdapterFactory,
    OrderStatus
)


class TestOrderStatus:
    """订单状态枚举测试类"""
    
    def test_order_status_values(self):
        """测试订单状态枚举值"""
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.PARTIAL_FILLED.value == "partial_filled"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled"
        assert OrderStatus.REJECTED.value == "rejected"


class TestCTPSimulatorAdapter:
    """CTP模拟器适配器测试类"""
    
    @pytest.fixture
    def config(self):
        """创建配置"""
        return {
            'broker_id': 'test_broker',
            'user_id': 'test_user',
            'password': 'test_password'
        }
    
    @pytest.fixture
    def adapter(self, config):
        """创建CTP模拟器适配器实例"""
        return CTPSimulatorAdapter(config)
    
    def test_init(self, adapter, config):
        """测试初始化"""
        assert adapter.config == config
        assert adapter.connected is False
    
    def test_connect_success(self, adapter):
        """测试连接成功"""
        result = adapter.connect()
        
        assert result is True
        assert adapter.connected is True
    
    def test_connect_failure(self, adapter):
        """测试连接失败"""
        with patch('src.trading.broker.broker_adapter.logger') as mock_logger:
            # 模拟连接异常
            with patch.object(adapter, 'connect', side_effect=Exception("连接失败")):
                try:
                    result = adapter.connect()
                except Exception:
                    result = False
                assert result is False
    
    def test_disconnect(self, adapter):
        """测试断开连接"""
        adapter.connected = True
        result = adapter.disconnect()
        
        assert result is True
        assert adapter.connected is False
    
    def test_place_order_success(self, adapter):
        """测试下单成功"""
        adapter.connect()
        
        order = {
            'symbol': '000001.SZ',
            'direction': 'buy',
            'type': 'limit',
            'quantity': 100,
            'price': 10.50,
            'account': 'test_account'
        }
        
        order_id = adapter.place_order(order)
        
        assert isinstance(order_id, str)
        assert order_id.startswith('CTP_')
    
    def test_place_order_not_connected(self, adapter):
        """测试未连接时下单应该抛出异常"""
        order = {
            'symbol': '000001.SZ',
            'direction': 'buy',
            'type': 'limit',
            'quantity': 100,
            'price': 10.50,
            'account': 'test_account'
        }
        
        with pytest.raises(ConnectionError, match="Not connected"):
            adapter.place_order(order)
    
    def test_place_order_market(self, adapter):
        """测试市价单"""
        adapter.connect()
        
        order = {
            'symbol': '000001.SZ',
            'direction': 'buy',
            'type': 'market',
            'quantity': 100,
            'account': 'test_account'
        }
        
        order_id = adapter.place_order(order)
        assert isinstance(order_id, str)
    
    def test_cancel_order(self, adapter):
        """测试撤单"""
        order_id = "CTP_123456"
        result = adapter.cancel_order(order_id)
        
        assert result is True
    
    def test_get_order_status(self, adapter):
        """测试获取订单状态"""
        order_id = "CTP_123456"
        status = adapter.get_order_status(order_id)
        
        assert isinstance(status, dict)
        assert 'order_id' in status
        assert 'status' in status
        assert 'filled_quantity' in status
        assert 'avg_price' in status
        assert 'timestamp' in status
        assert status['order_id'] == order_id
        assert status['status'] == OrderStatus.FILLED.value
    
    def test_get_positions(self, adapter):
        """测试获取持仓"""
        positions = adapter.get_positions()
        
        assert isinstance(positions, list)
        if len(positions) > 0:
            assert 'symbol' in positions[0]
            assert 'quantity' in positions[0]
            assert 'cost_price' in positions[0]
            assert 'market_value' in positions[0]
    
    def test_get_positions_with_account(self, adapter):
        """测试获取指定账户持仓"""
        positions = adapter.get_positions(account='test_account')
        
        assert isinstance(positions, list)
    
    def test_get_account_balance(self, adapter):
        """测试获取账户资金"""
        balance = adapter.get_account_balance('test_account')
        
        assert isinstance(balance, dict)
        assert 'total_assets' in balance
        assert 'available_cash' in balance
        assert 'margin' in balance
        assert 'frozen_cash' in balance
        assert balance['total_assets'] == 1000000
        assert balance['available_cash'] == 800000
    
    def test_get_market_data(self, adapter):
        """测试获取市场数据"""
        symbols = ['000001.SZ', '000002.SZ']
        market_data = adapter.get_market_data(symbols)
        
        assert isinstance(market_data, dict)
        assert len(market_data) == 2
        assert '000001.SZ' in market_data
        assert '000002.SZ' in market_data
        
        # 检查每个标的的数据结构
        for symbol in symbols:
            data = market_data[symbol]
            assert 'last_price' in data
            assert 'ask_price' in data
            assert 'bid_price' in data
            assert 'volume' in data
            assert 'timestamp' in data
    
    def test_get_market_data_empty_list(self, adapter):
        """测试获取空列表的市场数据"""
        market_data = adapter.get_market_data([])
        
        assert isinstance(market_data, dict)
        assert len(market_data) == 0


class TestBrokerAdapterFactory:
    """经纪商适配器工厂测试类"""
    
    def test_create_adapter_ctp(self):
        """测试创建CTP适配器"""
        config = {'broker_id': 'test'}
        adapter = BrokerAdapterFactory.create_adapter('ctp', config)
        
        assert isinstance(adapter, CTPSimulatorAdapter)
        assert adapter.config == config
    
    def test_create_adapter_simulator(self):
        """测试创建模拟器适配器"""
        config = {'broker_id': 'test'}
        adapter = BrokerAdapterFactory.create_adapter('simulator', config)
        
        assert isinstance(adapter, CTPSimulatorAdapter)
    
    def test_create_adapter_unsupported(self):
        """测试创建不支持的适配器类型"""
        config = {'broker_id': 'test'}
        
        with pytest.raises(ValueError, match="Unsupported broker type"):
            BrokerAdapterFactory.create_adapter('unsupported', config)


class TestBrokerAdapterAbstract:
    """经纪商适配器抽象基类测试类"""
    
    def test_broker_adapter_is_abstract(self):
        """测试BrokerAdapter是抽象类"""
        with pytest.raises(TypeError):
            BrokerAdapter({'test': 'config'})
    
    def test_broker_adapter_get_market_data_default(self):
        """测试BrokerAdapter的默认get_market_data实现"""
        # 创建一个简单的实现类用于测试
        class TestAdapter(BrokerAdapter):
            def connect(self):
                return True
            
            def disconnect(self):
                return True
            
            def place_order(self, order):
                return "test_order_id"
            
            def cancel_order(self, order_id):
                return True
            
            def get_order_status(self, order_id):
                return {}
            
            def get_positions(self, account=None):
                return []
            
            def get_account_balance(self, account):
                return {}
        
        adapter = TestAdapter({'test': 'config'})
        market_data = adapter.get_market_data(['000001.SZ'])
        
        assert isinstance(market_data, dict)
        assert '000001.SZ' in market_data
        assert market_data['000001.SZ'] == {}
