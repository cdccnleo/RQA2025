#!/usr/bin/env python3
"""
策略层核心组件深度测试覆盖率提升
目标：大幅提升策略层核心组件测试覆盖率，从0%提升至>50%
策略：系统性地测试核心策略组件，确保全面覆盖
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock


class TestStrategyCoreComponents:
    """策略层核心组件全面覆盖测试"""

    @pytest.fixture(autouse=True)
    def setup_core_test(self):
        """设置核心组件测试环境"""
        import sys
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        src_path = os.path.join(project_root, "src")

        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        yield

    def test_strategy_service_core(self):
        """测试策略服务核心功能"""
        try:
            from src.strategy.core.strategy_service import UnifiedStrategyService

            service = UnifiedStrategyService()
            assert service is not None

            # 测试策略创建
            strategy_config = {
                'name': 'test_strategy',
                'type': 'momentum',
                'parameters': {'window': 20}
            }

            strategy_id = service.create_strategy(strategy_config)
            assert strategy_id is not None

            # 测试策略获取
            retrieved = service.get_strategy(strategy_id)
            assert retrieved is not None

            # 测试策略更新
            updated_config = strategy_config.copy()
            updated_config['parameters']['window'] = 30
            success = service.update_strategy(strategy_id, updated_config)
            assert success is True

            # 测试策略删除
            delete_success = service.delete_strategy(strategy_id)
            assert delete_success is True

        except ImportError:
            pytest.skip("UnifiedStrategyService not available")

    def test_strategy_interfaces(self):
        """测试策略接口定义"""
        try:
            from src.strategy.interfaces.strategy_interfaces import StrategyConfig, StrategyResult

            # 测试策略配置
            config = StrategyConfig(
                name="test_strategy",
                type="momentum",
                parameters={"window": 20},
                enabled=True,
                description="Test strategy",
                version="1.0.0",
                author="test_author",
                tags=["test", "momentum"]
            )
            assert config.name == "test_strategy"
            assert config.type == "momentum"
            assert config.parameters["window"] == 20
            assert config.enabled is True

            # 测试策略结果
            result = StrategyResult(
                strategy_id="test_001",
                performance_metrics={"total_return": 0.15, "sharpe_ratio": 1.8},
                execution_status="completed"
            )
            assert result.strategy_id == "test_001"
            assert result.performance_metrics["total_return"] == 0.15
            assert result.execution_status == "completed"

        except ImportError:
            pytest.skip("Strategy interfaces not available")

    def test_strategy_lifecycle_manager(self):
        """测试策略生命周期管理"""
        try:
            from src.strategy.lifecycle.strategy_lifecycle_manager import StrategyLifecycleManager

            manager = StrategyLifecycleManager()
            assert manager is not None

            # 测试策略生命周期状态转换
            strategy_id = "test_strategy_001"

            # 初始化状态
            initial_state = manager.get_strategy_state(strategy_id)
            assert initial_state == "draft" or initial_state is not None

            # 转换为开发状态
            success = manager.transition_to_development(strategy_id)
            assert success is True

            # 转换为测试状态
            success = manager.transition_to_testing(strategy_id)
            assert success is True

            # 转换为生产状态
            success = manager.transition_to_production(strategy_id)
            assert success is True

            # 获取生命周期历史
            history = manager.get_lifecycle_history(strategy_id)
            assert isinstance(history, list)
            assert len(history) > 0

        except ImportError:
            pytest.skip("StrategyLifecycleManager not available")

    def test_strategy_persistence(self):
        """测试策略持久化功能"""
        try:
            from src.strategy.persistence.strategy_persistence import StrategyPersistence

            persistence = StrategyPersistence()
            assert persistence is not None

            # 测试策略保存
            strategy_data = {
                'id': 'test_strategy_001',
                'name': 'test_strategy',
                'config': {'type': 'momentum', 'parameters': {'window': 20}},
                'created_at': '2024-01-01T00:00:00Z'
            }

            success = persistence.save_strategy(strategy_data)
            assert success is True

            # 测试策略加载
            loaded = persistence.load_strategy('test_strategy_001')
            assert loaded is not None
            assert loaded['id'] == 'test_strategy_001'

            # 测试策略列表
            strategies = persistence.list_strategies()
            assert isinstance(strategies, list)
            assert len(strategies) > 0

            # 测试策略删除
            delete_success = persistence.delete_strategy('test_strategy_001')
            assert delete_success is True

        except ImportError:
            pytest.skip("StrategyPersistence not available")

    def test_ai_strategy_optimizer(self):
        """测试AI策略优化器"""
        try:
            from src.strategy.intelligence.ai_strategy_optimizer import AIStrategyOptimizer

            optimizer = AIStrategyOptimizer()
            assert optimizer is not None

            # 创建测试数据
            historical_data = pd.DataFrame({
                'close': np.random.normal(100, 5, 100),
                'volume': np.random.uniform(100000, 500000, 100)
            })

            # 测试策略优化
            base_strategy = {
                'name': 'ai_test_strategy',
                'type': 'trend_following',
                'parameters': {'fast_period': 12, 'slow_period': 26}
            }

            optimized = optimizer.optimize_strategy(base_strategy, historical_data)
            assert isinstance(optimized, dict)
            assert 'optimized_parameters' in optimized
            assert 'expected_improvement' in optimized

            # 测试多策略比较
            strategies = [base_strategy, {**base_strategy, 'name': 'strategy_2'}]
            comparison = optimizer.compare_strategies(strategies, historical_data)
            assert isinstance(comparison, dict)
            assert 'rankings' in comparison

        except ImportError:
            pytest.skip("AIStrategyOptimizer not available")

    def test_multi_strategy_optimizer(self):
        """测试多策略优化器"""
        try:
            from src.strategy.intelligence.multi_strategy_optimizer import MultiStrategyOptimizer

            optimizer = MultiStrategyOptimizer()
            assert optimizer is not None

            # 测试策略组合优化
            strategies = [
                {'name': 'momentum', 'weight': 0.4, 'expected_return': 0.12},
                {'name': 'mean_reversion', 'weight': 0.3, 'expected_return': 0.08},
                {'name': 'trend_following', 'weight': 0.3, 'expected_return': 0.10}
            ]

            # 协方差矩阵
            covariance_matrix = pd.DataFrame({
                'momentum': [0.04, 0.01, 0.015],
                'mean_reversion': [0.01, 0.025, 0.008],
                'trend_following': [0.015, 0.008, 0.036]
            })

            optimized_portfolio = optimizer.optimize_portfolio(strategies, covariance_matrix)
            assert isinstance(optimized_portfolio, dict)
            assert 'optimal_weights' in optimized_portfolio
            assert 'expected_return' in optimized_portfolio
            assert 'portfolio_volatility' in optimized_portfolio

            # 验证权重和为1
            weights = optimized_portfolio['optimal_weights']
            total_weight = sum(weights.values())
            assert abs(total_weight - 1.0) < 0.01

        except ImportError:
            pytest.skip("MultiStrategyOptimizer not available")

    def test_strategy_monitoring_service(self):
        """测试策略监控服务"""
        try:
            from src.strategy.monitoring.monitoring_service import StrategyMonitoringService

            monitoring = StrategyMonitoringService()
            assert monitoring is not None

            # 测试性能监控
            strategy_id = "test_strategy_001"
            performance_data = {
                'total_return': 0.15,
                'sharpe_ratio': 1.8,
                'max_drawdown': -0.05,
                'win_rate': 0.65
            }

            monitoring.record_performance(strategy_id, performance_data)

            # 测试监控指标获取
            metrics = monitoring.get_performance_metrics(strategy_id)
            assert isinstance(metrics, dict)
            assert 'total_return' in metrics

            # 测试告警检查
            alerts = monitoring.check_performance_alerts(strategy_id)
            assert isinstance(alerts, list)

            # 测试监控报告生成
            report = monitoring.generate_monitoring_report(strategy_id)
            assert isinstance(report, dict)
            assert 'performance_summary' in report

        except ImportError:
            pytest.skip("StrategyMonitoringService not available")

    def test_distributed_strategy_manager(self):
        """测试分布式策略管理器"""
        try:
            from src.strategy.distributed.distributed_strategy_manager import DistributedStrategyManager

            manager = DistributedStrategyManager()
            assert manager is not None

            # 测试分布式部署
            strategy_config = {
                'name': 'distributed_test',
                'type': 'momentum',
                'distributed_config': {
                    'num_workers': 4,
                    'coordinator_node': 'node_001'
                }
            }

            deployment_result = manager.deploy_distributed_strategy(strategy_config)
            assert isinstance(deployment_result, dict)
            assert 'deployment_id' in deployment_result
            assert 'worker_nodes' in deployment_result

            # 测试分布式执行
            execution_result = manager.execute_distributed_strategy(deployment_result['deployment_id'])
            assert isinstance(execution_result, dict)
            assert 'status' in execution_result

            # 测试负载均衡
            load_balance = manager.check_load_balance()
            assert isinstance(load_balance, dict)
            assert 'node_utilization' in load_balance

        except ImportError:
            pytest.skip("DistributedStrategyManager not available")

    def test_strategy_visualization_service(self):
        """测试策略可视化服务"""
        try:
            from src.strategy.visualization.backtest_visualizer import BacktestVisualizer

            visualizer = BacktestVisualizer()
            assert visualizer is not None

            # 创建测试数据
            backtest_results = {
                'returns': pd.Series(np.random.normal(0.001, 0.02, 100)),
                'metrics': {
                    'total_return': 0.25,
                    'sharpe_ratio': 1.8,
                    'max_drawdown': -0.08
                }
            }

            # 测试收益曲线图
            returns_plot = visualizer.plot_returns_curve(backtest_results['returns'])
            assert returns_plot is not None

            # 测试绩效指标图表
            metrics_plot = visualizer.plot_performance_metrics(backtest_results['metrics'])
            assert metrics_plot is not None

            # 测试回撤分析图
            drawdown_plot = visualizer.plot_drawdown_analysis(backtest_results['returns'])
            assert drawdown_plot is not None

            # 测试风险分析图
            risk_plot = visualizer.plot_risk_analysis(backtest_results)
            assert risk_plot is not None

        except ImportError:
            pytest.skip("Strategy visualization components not available")

    def test_strategy_factory(self):
        """测试策略工厂"""
        try:
            from src.strategy.strategies.factory import StrategyFactory

            factory = StrategyFactory()
            assert factory is not None

            # 测试策略创建
            strategy_config = {
                'type': 'momentum',
                'parameters': {'window': 20}
            }

            strategy = factory.create_strategy(strategy_config)
            assert strategy is not None
            assert hasattr(strategy, 'execute')

            # 测试策略注册
            factory.register_strategy_type('custom_strategy', lambda config: Mock())
            assert 'custom_strategy' in factory._strategy_types

            # 测试策略类型列表
            strategy_types = factory.get_available_strategy_types()
            assert isinstance(strategy_types, list)
            assert 'momentum' in strategy_types

        except ImportError:
            pytest.skip("StrategyFactory not available")
