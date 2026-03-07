# tests/unit/trading/test_execution_engine_core.py
"""
ExecutionEngine核心功能单元测试

测试覆盖:
- 交易执行引擎初始化和配置
- 订单创建和执行
- 执行状态跟踪和管理
- 市场订单执行
- 限价订单执行
- 执行监控和报告
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

try:
    from src.trading.execution.execution_engine import ExecutionEngine
except ImportError:
    from src.trading.execution_engine import (
        ExecutionEngine,
    ExecutionMode,
    ExecutionStatus,
    Order,
    OrderType,
    OrderSide,
    OrderStatus
)

# 设置测试超时，避免死锁和无限等待
pytestmark = [
    pytest.mark.timeout(30),  # 30秒超时
    pytest.mark.deadlock_risk,  # 标记为可能存在死锁风险
    pytest.mark.concurrent,  # 并发测试
    pytest.mark.infinite_loop_risk  # 可能存在无限循环风险
]


class TestExecutionEngineCore:
    """ExecutionEngine核心功能测试类"""

    @pytest.fixture
    def execution_engine(self):
        """ExecutionEngine实例"""
        config = {
            "max_concurrent_orders": 100,
            "execution_timeout": 300,
            "slippage_tolerance": 0.001,
            "market_impact_limit": 0.01,
            "monitoring_enabled": True,
            "performance_tracking": True
        }
        return ExecutionEngine(config)

    @pytest.fixture
    def sample_order(self):
        """样本订单"""
        return {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 1000,
            "price": 150.0,
            "order_type": "market",
            "time_in_force": "day"
        }

    def test_initialization(self, execution_engine):
        """测试执行引擎初始化"""
        assert execution_engine is not None
        assert execution_engine.max_concurrent_orders == 100
        assert execution_engine.execution_timeout == 300
        assert isinstance(execution_engine.executions, dict)

    def test_order_creation(self, execution_engine, sample_order):
        """测试订单创建"""
        order_id = execution_engine.create_order(sample_order)

        assert order_id is not None
        assert isinstance(order_id, str)
        assert len(order_id) > 0
        assert order_id in execution_engine.executions

    def test_market_order_execution(self, execution_engine, sample_order):
        """测试市价订单执行"""
        market_order = sample_order.copy()
        market_order["order_type"] = "market"

        order_id = execution_engine.create_order(market_order)

        with patch.object(execution_engine, '_execute_market_order', return_value={
            "status": "completed",
            "executed_quantity": 1000,
            "average_price": 150.0,
            "execution_time": datetime.now()
        }):
            execution_result = execution_engine.execute_order(order_id)

        assert execution_result is not None
        assert execution_result["status"] == "completed"
        assert "executed_quantity" in execution_result
        assert "average_price" in execution_result

    def test_limit_order_execution(self, execution_engine, sample_order):
        """测试限价订单执行"""
        limit_order = sample_order.copy()
        limit_order["order_type"] = "limit"
        limit_order["price"] = 149.50

        order_id = execution_engine.create_order(limit_order)

        with patch.object(execution_engine, '_execute_limit_order', return_value={
            "status": "completed",
            "executed_quantity": 1000,
            "average_price": 149.50,
            "execution_time": datetime.now()
        }):
            execution_result = execution_engine.execute_order(order_id)

        assert execution_result is not None
        assert execution_result["status"] == "completed"
        assert execution_result["average_price"] == 149.50

    def test_execution_status_tracking(self, execution_engine, sample_order):
        """测试执行状态跟踪"""
        order_id = execution_engine.create_order(sample_order)

        # 检查初始状态
        status = execution_engine.get_execution_status_dict(order_id)
        assert status["status"] == "pending"

        # 更新状态
        execution_engine.update_execution_status(order_id, "running")
        status = execution_engine.get_execution_status_dict(order_id)
        assert status["status"] == "running"

        # 完成执行
        execution_engine.update_execution_status(order_id, "completed")
        status = execution_engine.get_execution_status_dict(order_id)
        assert status["status"] == "completed"

    def test_execution_cancellation(self, execution_engine, sample_order):
        """测试执行取消"""
        order_id = execution_engine.create_order(sample_order)

        cancellation_result = execution_engine.cancel_execution_dict(order_id)

        assert cancellation_result is not None
        assert cancellation_result["cancelled"] is True
        assert cancellation_result["remaining_quantity"] == 1000

        status = execution_engine.get_execution_status_dict(order_id)
        assert status["status"] == "cancelled"

    def test_execution_modification(self, execution_engine, sample_order):
        """测试执行修改"""
        order_id = execution_engine.create_order(sample_order)

        modifications = {
            "quantity": 1500,
            "price": 152.0
        }

        modification_result = execution_engine.modify_execution(order_id, modifications)

        assert modification_result is not None
        assert modification_result["modified"] is True

        updated_execution = execution_engine.get_execution_details(order_id)
        assert updated_execution["quantity"] == 1500
        assert updated_execution["price"] == 152.0

    def test_execution_performance_monitoring(self, execution_engine, sample_order):
        """测试执行性能监控"""
        # 创建多个订单
        order_ids = []
        for i in range(5):
            order = sample_order.copy()
            order["quantity"] = 1000 + (i * 100)
            order_id = execution_engine.create_order(order)
            order_ids.append(order_id)

        # 执行订单
        for order_id in order_ids:
            execution_engine.execute_order(order_id)

        performance_metrics = execution_engine.get_execution_performance()

        assert performance_metrics is not None
        assert "total_orders" in performance_metrics
        assert "average_execution_time" in performance_metrics
        assert "execution_success_rate" in performance_metrics

    def test_execution_error_handling(self, execution_engine, sample_order):
        """测试执行错误处理"""
        order_id = execution_engine.create_order(sample_order)

        with patch.object(execution_engine, '_execute_market_order', side_effect=Exception("Connection failed")):
            execution_result = execution_engine.execute_order(order_id)

        assert execution_result is not None
        assert execution_result["status"] == "failed"
        assert "error_message" in execution_result
        assert "Connection failed" in execution_result["error_message"]

    def test_execution_statistics(self, execution_engine):
        """测试执行统计"""
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
        orders = []

        for i, symbol in enumerate(symbols):
            order = {
                "symbol": symbol,
                "side": "buy" if i % 2 == 0 else "sell",
                "quantity": 1000 * (i + 1),
                "price": 100.0 + (i * 10),
                "order_type": "market"
            }
            order_id = execution_engine.create_order(order)
            execution_engine.execute_order(order_id)
            orders.append((order_id, order))

        statistics = execution_engine.get_execution_statistics()

        assert statistics is not None
        assert "total_executions" in statistics
        assert "successful_executions" in statistics
        assert "symbol_performance" in statistics
        assert statistics["total_executions"] == len(symbols)

    def test_execution_reporting(self, execution_engine, sample_order, tmp_path):
        """测试执行报告生成"""
        for i in range(3):
            order = sample_order.copy()
            order["quantity"] = 1000 + (i * 200)
            order_id = execution_engine.create_order(order)
            execution_engine.execute_order(order_id)

        report_file = tmp_path / "execution_report.json"
        report_result = execution_engine.generate_execution_report(str(report_file))

        assert report_result is not None
        assert report_result["report_generated"] is True
        assert report_file.exists()

    def test_twap_execution(self, execution_engine, sample_order):
        """测试TWAP执行算法"""
        twap_order = sample_order.copy()
        twap_order["quantity"] = 10000
        twap_order["execution_mode"] = "twap"
        twap_order["duration_minutes"] = 60

        order_id = execution_engine.create_order(twap_order)

        with patch('time.sleep'):
            execution_result = execution_engine.execute_order(order_id)

        assert execution_result is not None
        assert execution_result["status"] == "completed"
        assert execution_result["execution_slices"] > 1

    def test_vwap_execution(self, execution_engine, sample_order):
        """测试VWAP执行算法"""
        vwap_order = sample_order.copy()
        vwap_order["execution_mode"] = "vwap"
        vwap_order["target_volume_percentage"] = 0.1

        order_id = execution_engine.create_order(vwap_order)

        with patch.object(execution_engine, '_get_volume_profile', return_value=[1000, 1200, 800, 1500]):
            execution_result = execution_engine.execute_order(order_id)

        assert execution_result is not None
        assert execution_result["status"] == "completed"
        assert "vwap_price" in execution_result

    def test_iceberg_execution(self, execution_engine, sample_order):
        """测试冰山订单执行"""
        iceberg_order = sample_order.copy()
        iceberg_order["quantity"] = 50000
        iceberg_order["execution_mode"] = "iceberg"
        iceberg_order["visible_quantity"] = 5000
        iceberg_order["peak_interval_minutes"] = 5

        order_id = execution_engine.create_order(iceberg_order)

        with patch('time.sleep'):
            execution_result = execution_engine.execute_order(order_id)

        assert execution_result is not None
        assert execution_result["status"] == "completed"
        assert len(execution_result["iceberg_slices"]) > 1

    def test_execution_queue_management(self, execution_engine):
        """测试执行队列管理"""
        order_ids = []
        for i in range(150):  # 超过max_concurrent_orders
            order = {
                "symbol": f"SYMBOL_{i}",
                "side": "buy",
                "quantity": 1000,
                "price": 100.0,
                "order_type": "market"
            }
            order_id = execution_engine.create_order(order)
            order_ids.append(order_id)

        queue_status = execution_engine.get_execution_queue_status()

        assert queue_status is not None
        assert "queued_orders" in queue_status
        assert "active_executions" in queue_status
        assert queue_status["active_executions"] <= execution_engine.max_concurrent_orders

    def test_execution_audit_trail(self, execution_engine, sample_order):
        """测试执行审计跟踪"""
        order_id = execution_engine.create_order(sample_order)
        execution_engine.execute_order(order_id)

        audit_trail = execution_engine.get_execution_audit_trail(order_id)

        assert audit_trail is not None
        assert isinstance(audit_trail, list)
        assert len(audit_trail) > 0

        first_entry = audit_trail[0]
        assert "timestamp" in first_entry
        assert "action" in first_entry
        assert "details" in first_entry

    def test_execution_compliance_checking(self, execution_engine, sample_order):
        """测试执行合规检查"""
        order_id = execution_engine.create_order(sample_order)

        compliance_result = execution_engine.check_execution_compliance(order_id)

        assert compliance_result is not None
        assert "compliance_status" in compliance_result
        assert "regulatory_checks" in compliance_result
        assert "risk_limits_check" in compliance_result

    def test_execution_cost_analysis(self, execution_engine, sample_order):
        """测试执行成本分析"""
        order_id = execution_engine.create_order(sample_order)
        execution_engine.execute_order(order_id)

        cost_analysis = execution_engine.analyze_execution_cost(order_id)

        assert cost_analysis is not None
        assert "total_cost" in cost_analysis
        assert "commission_cost" in cost_analysis
        assert "slippage_cost" in cost_analysis
        assert "market_impact_cost" in cost_analysis

    def test_execution_smart_routing(self, execution_engine, sample_order):
        """测试执行智能路由"""
        venues = {
            "venue_a": {"commission": 0.001, "liquidity": 0.8, "latency": 10},
            "venue_b": {"commission": 0.002, "liquidity": 0.9, "latency": 5},
            "venue_c": {"commission": 0.0015, "liquidity": 0.7, "latency": 8}
        }

        execution_engine.configure_smart_routing(venues)

        order_id = execution_engine.create_order(sample_order)
        routing_result = execution_engine.execute_with_smart_routing(order_id)

        assert routing_result is not None
        assert "selected_venue" in routing_result
        assert "routing_reason" in routing_result
        assert "expected_cost_savings" in routing_result

    def test_execution_resource_monitoring(self, execution_engine):
        """测试执行资源监控"""
        for i in range(10):
            order = {
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 1000,
                "price": 150.0,
                "order_type": "market"
            }
            order_id = execution_engine.create_order(order)
            execution_engine.execute_order(order_id)

        resource_usage = execution_engine.get_resource_usage()

        assert resource_usage is not None
        assert "cpu_usage" in resource_usage
        assert "memory_usage" in resource_usage
        assert "network_io" in resource_usage
        assert "active_connections" in resource_usage
