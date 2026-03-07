# -*- coding: utf-8 -*-
"""
交易层 - 交易执行器单元测试
测试覆盖率目标: 95%+
按照业务流程驱动架构设计测试交易执行器核心功能
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.trading.executor import TradingExecutor, OrderStatus, OrderType


class TestTradingExecutor:
    """测试交易执行器"""

    def setup_method(self, method):
        """设置测试环境"""
        config = {
            "max_orders_per_second": 100,
            "supported_exchanges": ["SH", "SZ"],
            "risk_limits": {"max_position": 10000}
        }
        self.executor = TradingExecutor(config)

        # 确保必要属性被设置
        self.executor.enable_monitoring = True
        self.executor.enable_caching = True

    def test_init(self):
        """测试初始化"""
        assert self.executor.config["max_orders_per_second"] == 100
        assert isinstance(self.executor._executors, dict)
        assert "_order_history" in self.executor.__dict__

    def test_init_default_config(self):
        """测试默认配置初始化"""
        executor = TradingExecutor()
        assert executor.config == {}
        assert isinstance(executor._executors, dict)

    def test_execute_order_market(self):
        """测试执行市价单"""
        order = {
            "order_id": "test_order_001",
            "symbol": "000001.SZ",
            "quantity": 100,
            "order_type": "market",
            "direction": "BUY",
            "exchange": "SZ"
        }

        result = self.executor.execute_order(order)

        # 验证返回结果结构
        assert isinstance(result, dict)
        assert "success" in result
        assert "order_id" in result
        if result["success"]:
            assert "executed_quantity" in result
            assert "executed_price" in result
            assert "timestamp" in result

    def test_execute_order_limit(self):
        """测试执行限价单"""
        import random
        from unittest.mock import patch
        
        order = {
            "order_id": "test_order_002",
            "symbol": "000002.SZ",
            "quantity": 200,
            "price": 15.50,
            "order_type": "limit",
            "side": "SELL",
            "exchange": "SH"
        }

        # Mock随机数确保执行成功
        with patch('random.random', return_value=0.5):  # 0.5 < 0.8，确保成功
            result = self.executor.execute_order(order)

        # 验证限价单的特殊处理
        assert isinstance(result, dict)
        assert result["order_id"] == "test_order_002"
        assert result["success"] is True
        assert "executed_quantity" in result
        assert "execution_strategy" in result

    def test_execute_order_stop(self):
        """测试执行止损单"""
        order = {
            "order_id": "test_order_003",
            "symbol": "000003.SZ",
            "quantity": 50,
            "stop_price": 12.00,
            "order_type": "stop",
            "direction": "BUY",
            "exchange": "SZ"
        }

        result = self.executor.execute_order(order)

        # 验证止损单的处理
        assert isinstance(result, dict)
        assert result["order_id"] == "test_order_003"

    def test_validate_order_valid(self):
        """测试订单验证（有效订单）"""
        valid_order = {
            "symbol": "000001.SZ",
            "quantity": 100,
            "order_type": "market",
            "side": "BUY"
        }

        result = self.executor._validate_order(valid_order)

        # 验证有效订单的验证结果
        assert result["valid"] is True

    def test_validate_order_invalid(self):
        """测试订单验证（无效订单）"""
        invalid_order = {
            "symbol": "",  # 无效的symbol
            "quantity": 0,  # 无效的数量
            "order_type": "INVALID",  # 无效的类型
            "side": "INVALID"  # 无效的方向
        }

        result = self.executor._validate_order(invalid_order)

        # 验证无效订单的验证结果
        assert result["valid"] is False
        assert "error" in result

    def test_select_execution_strategy(self):
        """测试执行策略选择"""
        # 测试不同类型的订单策略选择
        market_order = {
            "order_type": "market",
            "quantity": 1000
        }

        limit_order = {
            "order_type": "limit",
            "quantity": 100
        }

        stop_order = {
            "order_type": "stop",
            "quantity": 50
        }

        # 验证策略选择逻辑
        strategy1 = self.executor._select_execution_strategy(market_order)
        strategy2 = self.executor._select_execution_strategy(limit_order)
        strategy3 = self.executor._select_execution_strategy(stop_order)

        assert isinstance(strategy1, str)
        assert isinstance(strategy2, str)
        assert isinstance(strategy3, str)

    def test_health_check(self):
        """测试健康检查"""
        health_info = self.executor.health_check()

        # 验证健康检查返回的信息
        assert isinstance(health_info, dict)
        assert "status" in health_info
        assert "timestamp" in health_info
        assert "order_history_count" in health_info
        assert "available_strategies" in health_info

    def test_cancel_order(self):
        """测试取消订单"""
        # 取消订单
        cancel_result = self.executor.cancel_order("test_cancel_001")

        # 验证取消结果
        assert isinstance(cancel_result, dict)
        assert "order_id" in cancel_result
        assert "status" in cancel_result
        assert cancel_result["status"] == "cancelled"

    def test_get_order_status(self):
        """测试获取订单状态"""
        # 获取订单状态
        status_result = self.executor.get_order_status("test_status_001")

        # 验证状态查询结果
        assert isinstance(status_result, dict)
        assert "order_id" in status_result
        assert "status" in status_result
        assert status_result["status"] == "filled"

    def test_get_performance_metrics(self):
        """测试获取性能指标"""
        metrics = self.executor.get_performance_metrics()

        # 验证性能指标结构
        assert isinstance(metrics, dict)
        assert "timestamp" in metrics
        assert "total_orders" in metrics
        assert "success_rate" in metrics
        assert "orders_per_second" in metrics

    def test_get_execution_statistics(self):
        """测试获取执行统计"""
        stats = self.executor.get_execution_statistics()

        # 验证执行统计结构
        assert isinstance(stats, dict)
        assert "total_orders" in stats
        assert "successful_orders" in stats
        assert "success_rate" in stats
        assert "available_strategies" in stats

    def test_calculate_orders_per_second(self):
        """测试计算每秒订单数"""
        # 执行一些订单来产生统计数据
        for i in range(5):
            order = {
                "order_id": f"test_ops_{i}",
                "symbol": "000001.SZ",
                "quantity": 100,
                "order_type": "market",
                "direction": "BUY"
            }
            self.executor.execute_order(order)

        # 计算OPS
        ops = self.executor._calculate_orders_per_second()

        # 验证OPS计算结果
        assert isinstance(ops, float)
        assert ops >= 0

    def test_record_order_history(self):
        """测试记录订单历史"""
        order = {
            "order_id": "test_history_001",
            "symbol": "000001.SZ",
            "quantity": 100
        }

        result = {
            "success": True,
            "executed_quantity": 100,
            "executed_price": 15.0
        }

        # 记录订单历史
        self.executor.record_order_history(order, result)

        # 验证历史记录已被保存
        assert len(self.executor._order_history) >= 1
        history_entry = self.executor._order_history[-1]
        assert history_entry["order"] == order
        assert history_entry["result"] == result

    def test_load_config(self):
        """测试配置加载"""
        # 测试配置加载逻辑（使用临时配置文件路径）
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            f.write('{"max_orders_per_second": 200}')
            temp_path = f.name
        
        try:
            result = self.executor.load_config(temp_path)
            # 验证配置加载结果
            assert result is True
            assert isinstance(self.executor.config, dict)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_setup_default_executors(self):
        """测试默认执行器设置"""
        # 测试默认执行器设置逻辑
        self.executor.setup_default_executors()

        # 验证默认执行器已被设置
        assert isinstance(self.executor._executors, dict)
