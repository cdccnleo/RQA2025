"""
量化策略生命周期管理业务流程测试

测试策略从创建到部署再到监控调优的全生命周期管理流程。
验证策略管理的完整性和正确性。
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List


class MockStrategyManager:
    """模拟策略管理器，用于业务流程测试"""

    def __init__(self):
        self.strategies = {}
        self.backtest_results = {}
        self.deployment_status = {}

    def create_strategy(self, config: Dict[str, Any]) -> str:
        """创建策略"""
        strategy_id = config.get("strategy_id", f"strategy_{len(self.strategies)}")
        self.strategies[strategy_id] = {
            "id": strategy_id,
            "config": config,
            "status": "created",
            "created_at": datetime.now().isoformat(),
            "version": "1.0"
        }
        return strategy_id

    def get_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """获取策略信息"""
        return self.strategies.get(strategy_id, {})

    def validate_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """验证策略"""
        if strategy_id not in self.strategies:
            return {"valid": False, "error": "Strategy not found"}

        strategy = self.strategies[strategy_id]
        # 简单的验证逻辑
        config = strategy["config"]
        is_valid = (
            "strategy_id" in config and
            "parameters" in config and
            "risk_limits" in config
        )

        if is_valid:
            strategy["status"] = "validated"

        return {
            "valid": is_valid,
            "errors": [] if is_valid else ["Missing required configuration"],
            "warnings": []
        }

    def run_backtest(self, strategy_id: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """运行回测"""
        if strategy_id not in self.strategies:
            return {"success": False, "error": "Strategy not found"}

        # 模拟回测结果
        self.backtest_results[strategy_id] = {
            "total_return": 0.15,
            "sharpe_ratio": 1.8,
            "max_drawdown": 0.08,
            "win_rate": 0.65,
            "total_trades": 45,
            "completed_at": datetime.now().isoformat()
        }

        self.strategies[strategy_id]["status"] = "backtested"
        self.strategies[strategy_id]["backtest_results"] = self.backtest_results[strategy_id]

        return {"success": True, "results": self.backtest_results[strategy_id]}

    def deploy_strategy(self, strategy_id: str, environment: str) -> Dict[str, Any]:
        """部署策略"""
        if strategy_id not in self.strategies:
            return {"success": False, "error": "Strategy not found"}

        self.deployment_status[strategy_id] = {
            "environment": environment,
            "status": "deployed",
            "deployed_at": datetime.now().isoformat(),
            "deployment_id": f"deploy_{strategy_id}_{environment}"
        }

        self.strategies[strategy_id]["status"] = "deployed"
        self.strategies[strategy_id]["deployment"] = self.deployment_status[strategy_id]

        return {"success": True, "deployment": self.deployment_status[strategy_id]}

    def start_monitoring(self, strategy_id: str) -> Dict[str, Any]:
        """启动监控"""
        if strategy_id not in self.strategies:
            return {"success": False, "error": "Strategy not found"}

        self.strategies[strategy_id]["status"] = "monitoring"
        self.strategies[strategy_id]["monitoring_started"] = datetime.now().isoformat()

        return {"success": True, "monitoring": {"status": "active"}}

    def stop_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """停止策略"""
        if strategy_id not in self.strategies:
            return {"success": False, "error": "Strategy not found"}

        self.strategies[strategy_id]["status"] = "stopped"
        self.strategies[strategy_id]["stopped_at"] = datetime.now().isoformat()

        return {"success": True}


class TestStrategyLifecycleManagement:
    """量化策略生命周期管理测试"""

    @pytest.fixture
    def strategy_manager(self):
        """创建策略管理器实例"""
        return MockStrategyManager()

    @pytest.fixture
    def sample_strategy_config(self) -> Dict[str, Any]:
        """创建示例策略配置"""
        return {
            "strategy_id": "test_strategy_001",
            "name": "Test Moving Average Strategy",
            "type": "technical",
            "parameters": {
                "short_window": 5,
                "long_window": 20,
                "symbol": "AAPL",
                "initial_capital": 100000.0
            },
            "risk_limits": {
                "max_position_size": 0.1,
                "max_drawdown": 0.05,
                "max_daily_loss": 0.02
            }
        }

    @pytest.fixture
    def mock_market_data(self) -> Dict[str, Any]:
        """创建模拟市场数据"""
        base_date = datetime.now()
        return {
            "symbol": "AAPL",
            "data": [
                {
                    "timestamp": (base_date - timedelta(days=i)).isoformat(),
                    "open": 150.0 + i * 0.5,
                    "high": 152.0 + i * 0.5,
                    "low": 148.0 + i * 0.5,
                    "close": 151.0 + i * 0.5,
                    "volume": 1000000 + i * 10000
                }
                for i in range(30)  # 30天的数据
            ]
        }

    def test_strategy_creation_flow(self, strategy_manager, sample_strategy_config):
        """测试策略创建流程"""
        # 1. 创建策略
        strategy_id = strategy_manager.create_strategy(sample_strategy_config)

        # 2. 验证策略创建成功
        assert strategy_id == sample_strategy_config["strategy_id"]

        # 3. 获取策略信息
        strategy_info = strategy_manager.get_strategy(strategy_id)
        assert strategy_info is not None
        assert strategy_info["status"] == "created"
        assert strategy_info["config"] == sample_strategy_config

        # 4. 验证策略参数正确性
        assert strategy_info["config"]["parameters"]["short_window"] == 5
        assert strategy_info["config"]["parameters"]["long_window"] == 20

    def test_strategy_validation_flow(self, strategy_manager, sample_strategy_config):
        """测试策略验证流程"""
        # 1. 创建策略
        strategy_id = strategy_manager.create_strategy(sample_strategy_config)

        # 2. 执行策略验证
        validation_result = strategy_manager.validate_strategy(strategy_id)

        # 3. 验证策略验证结果
        assert validation_result["valid"] is True
        assert "errors" in validation_result
        assert len(validation_result["errors"]) == 0

        # 4. 验证策略状态更新
        strategy_info = strategy_manager.get_strategy(strategy_id)
        assert strategy_info["status"] == "validated"

    def test_strategy_backtest_flow(self, strategy_manager, sample_strategy_config, mock_market_data):
        """测试策略回测流程"""
        # 1. 创建策略
        strategy_id = strategy_manager.create_strategy(sample_strategy_config)

        # 2. 执行策略回测
        backtest_result = strategy_manager.run_backtest(strategy_id, mock_market_data)

        # 3. 验证回测结果
        assert backtest_result["success"] is True
        assert "results" in backtest_result
        results = backtest_result["results"]
        assert results["total_return"] == 0.15
        assert results["sharpe_ratio"] == 1.8
        assert results["max_drawdown"] == 0.08

        # 4. 验证策略状态更新
        strategy_info = strategy_manager.get_strategy(strategy_id)
        assert strategy_info["status"] == "backtested"
        assert "backtest_results" in strategy_info

    def test_strategy_deployment_flow(self, strategy_manager, sample_strategy_config):
        """测试策略部署流程"""
        # 1. 创建并验证策略
        strategy_id = strategy_manager.create_strategy(sample_strategy_config)
        strategy_manager.validate_strategy(strategy_id)

        # 2. 部署策略
        deployment_result = strategy_manager.deploy_strategy(strategy_id, "paper_trading")

        # 3. 验证部署结果
        assert deployment_result["success"] is True
        assert "deployment" in deployment_result

        # 4. 验证策略状态更新
        strategy_info = strategy_manager.get_strategy(strategy_id)
        assert strategy_info["status"] == "deployed"
        assert "deployment" in strategy_info

    def test_strategy_monitoring_flow(self, strategy_manager, sample_strategy_config):
        """测试策略监控流程"""
        # 1. 创建并部署策略
        strategy_id = strategy_manager.create_strategy(sample_strategy_config)
        strategy_manager.validate_strategy(strategy_id)
        strategy_manager.deploy_strategy(strategy_id, "paper_trading")

        # 2. 启动策略监控
        monitor_result = strategy_manager.start_monitoring(strategy_id)

        # 3. 验证监控启动
        assert monitor_result["success"] is True

        # 4. 验证策略状态
        strategy_info = strategy_manager.get_strategy(strategy_id)
        assert strategy_info["status"] == "monitoring"
        assert "monitoring_started" in strategy_info

    def test_strategy_stop_flow(self, strategy_manager, sample_strategy_config):
        """测试策略停止流程"""
        # 1. 创建并部署策略
        strategy_id = strategy_manager.create_strategy(sample_strategy_config)
        strategy_manager.validate_strategy(strategy_id)
        strategy_manager.deploy_strategy(strategy_id, "paper_trading")

        # 2. 停止策略
        stop_result = strategy_manager.stop_strategy(strategy_id)

        # 3. 验证停止结果
        assert stop_result["success"] is True

        # 4. 验证策略状态更新
        strategy_info = strategy_manager.get_strategy(strategy_id)
        assert strategy_info["status"] == "stopped"
        assert "stopped_at" in strategy_info

    def test_strategy_error_handling(self, strategy_manager):
        """测试策略错误处理"""
        # 1. 尝试获取不存在的策略
        result = strategy_manager.get_strategy("non_existent_strategy")
        assert result == {}, "应该返回空字典而不是抛出异常"

        # 2. 尝试验证不存在的策略
        validation_result = strategy_manager.validate_strategy("non_existent_strategy")
        assert validation_result["valid"] is False
        assert "error" in validation_result

        # 3. 尝试部署不存在的策略
        deployment_result = strategy_manager.deploy_strategy("non_existent_strategy", "paper_trading")
        assert deployment_result["success"] is False
        assert "error" in deployment_result

    def test_strategy_lifecycle_state_transitions(self, strategy_manager, sample_strategy_config):
        """测试策略生命周期状态转换"""
        strategy_id = strategy_manager.create_strategy(sample_strategy_config)

        # 验证初始状态
        info = strategy_manager.get_strategy(strategy_id)
        assert info["status"] == "created"

        # 状态转换链: created -> validated -> backtested -> deployed -> monitoring -> stopped
        states = ["created", "validated", "backtested", "deployed", "monitoring", "stopped"]

        for expected_state in states[1:]:
            # 执行相应操作
            if expected_state == "validated":
                strategy_manager.validate_strategy(strategy_id)
            elif expected_state == "backtested":
                strategy_manager.run_backtest(strategy_id, {})
            elif expected_state == "deployed":
                strategy_manager.deploy_strategy(strategy_id, "paper_trading")
            elif expected_state == "monitoring":
                strategy_manager.start_monitoring(strategy_id)
            elif expected_state == "stopped":
                strategy_manager.stop_strategy(strategy_id)

            # 验证状态转换
            info = strategy_manager.get_strategy(strategy_id)
            assert info["status"] == expected_state, f"状态转换失败: 期望{expected_state}, 实际{info['status']}"

    def test_strategy_concurrent_operations(self, strategy_manager, sample_strategy_config):
        """测试策略并发操作"""
        # 创建多个策略
        strategy_ids = []
        for i in range(3):
            config = sample_strategy_config.copy()
            config["strategy_id"] = f"concurrent_strategy_{i}"
            strategy_id = strategy_manager.create_strategy(config)
            strategy_ids.append(strategy_id)

        # 并发执行验证操作
        import concurrent.futures
        import threading

        results = []
        def validate_strategy(strategy_id):
            result = strategy_manager.validate_strategy(strategy_id)
            results.append(result)
            return result

        # 使用线程池执行并发操作
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(validate_strategy, sid) for sid in strategy_ids]
            concurrent.futures.wait(futures)

        # 验证所有操作都成功
        assert len(results) == 3, "应该有3个结果"
        for result in results:
            assert result["valid"] is True, f"并发验证失败: {result}"

        # 验证所有策略状态
        for strategy_id in strategy_ids:
            info = strategy_manager.get_strategy(strategy_id)
            assert info["status"] == "validated", f"策略{strategy_id}状态错误: {info['status']}"
