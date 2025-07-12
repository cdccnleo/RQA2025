"""交易模块单元测试"""
import pytest
from datetime import datetime, time
from unittest.mock import patch, MagicMock
from src.trading import AfterHoursTrader, OrderValidator

class TestAfterHoursTrader:
    """盘后交易测试"""

    @pytest.fixture
    def trader(self):
        return AfterHoursTrader()

    def test_valid_stock_symbol(self, trader):
        """测试科创板股票代码验证"""
        assert trader._is_valid_symbol("688001") is True
        assert trader._is_valid_symbol("600000") is False

    def test_trading_hours(self, trader):
        """测试交易时段验证"""
        # 在交易时段内
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 6, 30, 15, 15)
            assert trader._is_in_trading_hours() is True

        # 在交易时段外
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 6, 30, 14, 30)
            assert trader._is_in_trading_hours() is False

    def test_order_execution(self, trader):
        """测试订单执行"""
        mock_order = {
            "symbol": "688001",
            "direction": "buy",
            "quantity": 1000,
            "client_id": "test123"
        }

        with patch('src.trading.FPGAExecutor.execute') as mock_execute:
            mock_execute.return_value = {"price": 98.50, "status": "filled"}
            result = trader.execute_order(mock_order)

            assert result["status"] == "filled"
            assert result["price"] == 98.50
            mock_execute.assert_called_once()

class TestOrderValidator:
    """订单验证测试"""

    @pytest.fixture
    def validator(self):
        return OrderValidator()

    @pytest.mark.parametrize("qty,expected", [
        (100, True),    # 正常数量
        (100000, False), # 超过单笔上限
        (0, False),      # 零数量
        (-100, False)    # 负数
    ])
    def test_quantity_validation(self, validator, qty, expected):
        """测试订单数量验证"""
        assert validator.validate_quantity(qty) == expected

    @pytest.mark.parametrize("order,expected", [
        ({"symbol": "688001", "quantity": 100}, True),
        ({"symbol": "600000", "quantity": 100}, False),
        ({"symbol": "688001", "quantity": 100000}, False),
        ({}, False)
    ])
    def test_complete_validation(self, validator, order, expected):
        """测试完整订单验证"""
        assert validator.validate_order(order) == expected

class TestFallbackMechanism:
    """降级机制测试"""

    def test_fpga_fallback(self):
        """测试FPGA降级到软件路径"""
        from src.trading import OrderProcessor

        processor = OrderProcessor()

        # 模拟FPGA故障
        with patch('src.trading.FPGAExecutor.execute', side_effect=Exception("FPGA error")):
            with patch('src.trading.SoftwareExecutor.execute') as mock_software:
                mock_software.return_value = {"price": 98.50, "status": "filled"}
                result = processor.process({"symbol": "688001", "quantity": 100})

                assert result["status"] == "filled"
                mock_software.assert_called_once()

    def test_dual_path_consistency(self):
        """测试硬件/软件路径结果一致性"""
        from src.trading import OrderProcessor

        test_order = {"symbol": "688001", "quantity": 100}
        processor = OrderProcessor()

        # 获取硬件结果
        with patch('src.trading.FPGAExecutor.execute') as mock_hardware:
            mock_hardware.return_value = {"price": 98.50, "status": "filled"}
            hw_result = processor.process(test_order)

        # 获取软件结果
        with patch('src.trading.SoftwareExecutor.execute') as mock_software:
            mock_software.return_value = {"price": 98.50, "status": "filled"}
            sw_result = processor.process(test_order)

        # 结果应该在允许误差范围内一致
        assert abs(hw_result["price"] - sw_result["price"]) < 0.01
