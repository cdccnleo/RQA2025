#!/usr/bin/env python3
"""
RQA2025 测试框架示例

展示如何使用统一的测试框架和Mock对象
"""

import pytest
from unittest.mock import Mock
from decimal import Decimal


def test_mock_cache_operations(mock_cache):
    """测试缓存操作"""
    # 测试设置和获取
    mock_cache.set("test_key", "test_value")
    result = mock_cache.get("test_key")
    mock_cache.get.assert_called_with("test_key")

    # 测试删除
    mock_cache.delete("test_key")
    mock_cache.delete.assert_called_with("test_key")

    # 测试存在性检查
    mock_cache.exists("test_key")
    mock_cache.exists.assert_called_with("test_key")


def test_mock_trading_service(mock_trading_service):
    """测试交易服务"""
    order = {
        "symbol": "000001.SZ",
        "quantity": 100,
        "price": Decimal('10.0'),
        "direction": "buy"
    }

    # 测试提交订单
    result = mock_trading_service.submit_order(order)

    # 验证返回结果
    assert result["order_id"] == "mock_order_123"
    assert result["status"] == "submitted"

    # 验证调用
    mock_trading_service.submit_order.assert_called_once()


def test_mock_risk_service(mock_risk_service):
    """测试风控服务"""
    order = {"symbol": "000001.SZ", "quantity": 100}

    # 测试风控检查
    result = mock_risk_service.check_order_risk(order)

    assert result["approved"] == True
    assert "risk_score" in result
    # 注意：我们简化的Mock不包含warnings字段


def test_sample_data_usage(sample_orders, sample_portfolio, sample_signals):
    """测试使用示例数据"""
    # 测试订单数据
    assert len(sample_orders) == 5
    assert sample_orders[0]["symbol"] in ["000001.SZ", "000002.SZ", "600000.SH", "000858.SZ"]
    assert sample_orders[0]["quantity"] > 0
    assert sample_orders[0]["price"] > 0

    # 测试投资组合数据
    assert "cash" in sample_portfolio
    assert "total_value" in sample_portfolio
    # 注意：我们简化的投资组合数据不包含positions字段

    # 测试信号数据
    assert len(sample_signals) == 10
    assert sample_signals[0]["signal_type"] in ["BUY", "SELL", "HOLD"]
    assert 0 <= sample_signals[0]["strength"] <= 1


def test_combined_services(mock_trading_service, mock_risk_service, sample_orders):
    """测试服务组合使用"""
    order = sample_orders[0]

    # 先进行风控检查
    risk_result = mock_risk_service.check_order_risk(order)
    assert risk_result["approved"] == True

    # 然后提交订单
    trade_result = mock_trading_service.submit_order(order)
    assert trade_result["status"] == "submitted"

    # 验证调用顺序
    mock_risk_service.check_order_risk.assert_called_once()
    mock_trading_service.submit_order.assert_called_once()


def test_performance_monitoring():
    """测试性能监控"""
    import time

    start_time = time.time()
    time.sleep(0.01)  # 模拟一些操作
    execution_time = time.time() - start_time

    # 验证执行时间在合理范围内
    assert execution_time < 0.1


if __name__ == "__main__":
    # 可以直接运行测试
    pytest.main([__file__, "-v"])
