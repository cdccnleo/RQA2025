"""FPGA加速模块集成测试"""
import pytest
import time
from unittest.mock import patch, MagicMock
from src.fpga.fpga_manager import FPGAController, FPGATestHarness
from src.trading import OrderEngine
from src.risk_control import RiskEngine

class TestFPGAIntegration:
    """FPGA集成测试套件"""

    @pytest.fixture
    def fpga_controller(self):
        """FPGA控制器fixture"""
        return FPGAController()

    @pytest.fixture
    def order_engine(self, fpga_controller):
        """订单引擎fixture"""
        return OrderEngine(fpga_controller=fpga_controller)

    @pytest.fixture
    def risk_engine(self, fpga_controller):
        """风控引擎fixture"""
        return RiskEngine(fpga_controller=fpga_controller)

    def test_order_execution_path(self, fpga_controller, order_engine):
        """测试订单执行路径"""
        test_order = {
            "symbol": "688001",
            "price": 100.0,
            "quantity": 1000,
            "strategy": "momentum"
        }

        # 正常情况FPGA路径
        with patch.object(fpga_controller.monitor, 'is_available', return_value=True):
            with patch('src.fpga.fpga_accelerators.FPGAOrderOptimizer.optimize') as mock_fpga:
                mock_fpga.return_value = {"status": "optimized"}
                result = order_engine.execute_order(test_order)
                assert result["status"] == "optimized"
                mock_fpga.assert_called_once()

        # 异常情况软件路径
        with patch.object(fpga_controller.monitor, 'is_available', return_value=False):
            with patch('src.fpga.software_fallback.SoftwareOrderOptimizer.optimize') as mock_software:
                mock_software.return_value = {"status": "fallback"}
                result = order_engine.execute_order(test_order)
                assert result["status"] == "fallback"
                mock_software.assert_called_once()

    def test_risk_check_path(self, fpga_controller, risk_engine):
        """测试风控检查路径"""
        test_order = {
            "symbol": "688001",
            "price": 100.0,
            "quantity": 100000,
            "client_id": "VIP001"
        }

        # FPGA路径测试
        with patch.object(fpga_controller.monitor, 'is_available', return_value=True):
            with patch('src.fpga.fpga_accelerators.FPGARiskEngine.check') as mock_fpga:
                mock_fpga.return_value = {"approved": True}
                result = risk_engine.check_order(test_order)
                assert result["approved"] is True
                mock_fpga.assert_called_once()

        # 软件路径测试
        with patch.object(fpga_controller.monitor, 'is_available', return_value=False):
            with patch('src.fpga.software_fallback.SoftwareRiskEngine.check') as mock_software:
                mock_software.return_value = {"approved": False, "reason": "fallback"}
                result = risk_engine.check_order(test_order)
                assert result["approved"] is False
                mock_software.assert_called_once()

    def test_consistency_between_paths(self, fpga_controller):
        """测试双路径结果一致性"""
        # 情感分析一致性测试
        test_data = {"text": "公司发布重大利好公告", "features": ["sentiment"]}
        assert FPGATestHarness.run_consistency_test(
            fpga_controller, "sentiment_analysis", test_data)

        # 订单优化一致性测试
        test_order = {"symbol": "688001", "price": 100.0, "quantity": 1000}
        assert FPGATestHarness.run_consistency_test(
            fpga_controller, "order_optimization", test_order, tolerance=0.01)

        # 风控检查一致性测试
        risk_data = {"order": test_order, "market_data": {}}
        assert FPGATestHarness.run_consistency_test(
            fpga_controller, "risk_check", risk_data)

    def test_performance_comparison(self, fpga_controller):
        """性能对比测试"""
        # 情感分析性能测试
        test_data = {"text": "公司发布重大利好公告", "features": ["sentiment"]}
        perf = FPGATestHarness.run_performance_test(
            fpga_controller, "sentiment_analysis", test_data, iterations=1000)
        assert perf["speedup_ratio"] > 3  # FPGA加速比至少3倍

        # 订单优化性能测试
        test_order = {"symbol": "688001", "price": 100.0, "quantity": 1000}
        perf = FPGATestHarness.run_performance_test(
            fpga_controller, "order_optimization", test_order, iterations=1000)
        assert perf["speedup_ratio"] > 5  # 订单优化加速比更高

    def test_failure_recovery(self, fpga_controller, order_engine):
        """测试故障恢复流程"""
        test_order = {
            "symbol": "688001",
            "price": 100.0,
            "quantity": 1000,
            "strategy": "arbitrage"
        }

        # 模拟FPGA超时故障
        with patch('src.fpga.fpga_accelerators.FPGAOrderOptimizer.optimize',
                 side_effect=Exception("FPGA timeout")):
            with patch('src.fpga.software_fallback.SoftwareOrderOptimizer.optimize') as mock_software:
                mock_software.return_value = {"status": "fallback"}

                # 第一次调用触发故障
                result = order_engine.execute_order(test_order)
                assert result["status"] == "fallback"

                # 检查FPGA状态是否被标记为不可用
                assert fpga_controller.monitor.status == FPGAStatus.UNHEALTHY

                # 第二次调用应直接走软件路径
                result = order_engine.execute_order(test_order)
                assert result["status"] == "fallback"
                mock_software.call_count == 2

class TestFPGAPressure:
    """FPGA压力测试"""

    @pytest.fixture
    def test_orders(self):
        """生成测试订单数据"""
        return [
            {
                "symbol": f"688{i:03d}",
                "price": 100.0 + i,
                "quantity": 1000 + i*100,
                "strategy": "momentum"
            } for i in range(1000)
        ]

    def test_fpga_throughput(self, fpga_controller, test_orders):
        """测试FPGA路径吞吐量"""
        from src.fpga import fpga_accelerators

        # 模拟FPGA健康状态
        fpga_controller.monitor.status = FPGAStatus.HEALTHY

        start_time = time.time()
        for order in test_orders:
            fpga_controller.execute(
                "order_optimization",
                {"order": order, "market_data": {}}
            )
        elapsed = time.time() - start_time

        # 验证吞吐量 > 5000 ops/sec
        throughput = len(test_orders) / elapsed
        assert throughput > 5000, f"实际吞吐量: {throughput:.2f} ops/sec"

    def test_software_fallback_throughput(self, fpga_controller, test_orders):
        """测试软件降级路径吞吐量"""
        from src.fpga import software_fallback

        # 强制使用软件路径
        fpga_controller.monitor.status = FPGAStatus.UNHEALTHY

        start_time = time.time()
        for order in test_orders:
            fpga_controller.execute(
                "order_optimization",
                {"order": order, "market_data": {}}
            )
        elapsed = time.time() - start_time

        # 验证吞吐量 > 1000 ops/sec
        throughput = len(test_orders) / elapsed
        assert throughput > 1000, f"实际吞吐量: {throughput:.2f} ops/sec"
