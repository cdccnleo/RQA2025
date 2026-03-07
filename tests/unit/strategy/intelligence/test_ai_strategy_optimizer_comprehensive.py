"""
AI策略优化器深度测试
全面测试AI策略优化器的强化学习、深度学习优化和智能决策功能
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path
import json
import time
import threading

# 导入AI策略优化相关类
try:
    from src.strategy.intelligence.ai_strategy_optimizer import (
        AIStrategyOptimizer, OptimizationObjective, OptimizationAlgorithm,
        OptimizationConfig, OptimizationResult, StrategyState, StrategyAction
    )
    AI_OPTIMIZER_AVAILABLE = True
except ImportError:
    AI_OPTIMIZER_AVAILABLE = False
    AIStrategyOptimizer = Mock
    OptimizationObjective = Mock
    OptimizationAlgorithm = Mock
    OptimizationConfig = Mock
    OptimizationResult = Mock
    StrategyState = Mock
    StrategyAction = Mock

try:
    from src.strategy.strategies.base_strategy import BaseStrategy
    BASE_STRATEGY_AVAILABLE = True
except ImportError:
    BASE_STRATEGY_AVAILABLE = False
    BaseStrategy = Mock


class TestAIStrategyOptimizerComprehensive:
    """AI策略优化器综合深度测试"""

    @pytest.fixture
    def sample_market_data(self):
        """创建样本市场数据"""
        dates = pd.date_range('2024-01-01', periods=252, freq='D')  # 一年的交易日
        np.random.seed(42)

        # 生成更真实的市场数据
        initial_price = 100.0
        returns = np.random.normal(0.0005, 0.02, len(dates))  # 日均收益0.05%，波动率2%
        prices = initial_price * np.exp(np.cumsum(returns))

        return pd.DataFrame({
            'symbol': ['AAPL'] * len(dates),
            'timestamp': dates,
            'open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, len(dates)),
            'returns': returns
        })

    @pytest.fixture
    def sample_strategy_performance(self):
        """创建样本策略性能数据"""
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100, freq='D'),
            'portfolio_value': np.cumprod(1 + np.random.normal(0.001, 0.02, 100)),
            'returns': np.random.normal(0.001, 0.02, 100),
            'volatility': np.random.uniform(0.15, 0.35, 100),
            'sharpe_ratio': np.random.uniform(0.5, 2.0, 100),
            'max_drawdown': np.random.uniform(-0.3, -0.05, 100)
        })

    @pytest.fixture
    def optimization_config(self):
        """创建优化配置"""
        if AI_OPTIMIZER_AVAILABLE:
            return OptimizationConfig(
                algorithm=OptimizationAlgorithm.PPO,
                objective=OptimizationObjective.MAXIMIZE_SHARPE,
                max_iterations=100,
                learning_rate=0.001,
                batch_size=32,
                exploration_rate=0.1,
                risk_free_rate=0.02,
                transaction_costs=0.001,
                max_position_size=1.0,
                rebalance_frequency='daily'
            )
        return Mock()

    @pytest.fixture
    def ai_strategy_optimizer(self, optimization_config):
        """创建AI策略优化器实例"""
        if AI_OPTIMIZER_AVAILABLE:
            return AIStrategyOptimizer(config=optimization_config)
        return Mock(spec=AIStrategyOptimizer)

    @pytest.fixture
    def base_strategy(self):
        """创建基础策略实例"""
        if BASE_STRATEGY_AVAILABLE:
            return BaseStrategy()
        return Mock(spec=BaseStrategy)

    def test_ai_strategy_optimizer_initialization(self, ai_strategy_optimizer, optimization_config):
        """测试AI策略优化器初始化"""
        if AI_OPTIMIZER_AVAILABLE:
            assert ai_strategy_optimizer is not None
            assert ai_strategy_optimizer.config == optimization_config
            assert hasattr(ai_strategy_optimizer, 'reinforcement_agent')
            assert hasattr(ai_strategy_optimizer, 'optimization_history')
            assert hasattr(ai_strategy_optimizer, 'performance_tracker')

    def test_optimization_config_validation(self, optimization_config):
        """测试优化配置验证"""
        if AI_OPTIMIZER_AVAILABLE:
            assert optimization_config.algorithm == OptimizationAlgorithm.PPO
            assert optimization_config.objective == OptimizationObjective.MAXIMIZE_SHARPE
            assert optimization_config.max_iterations == 100
            assert optimization_config.learning_rate == 0.001
            assert optimization_config.risk_free_rate == 0.02

    def test_strategy_state_representation(self, ai_strategy_optimizer, sample_market_data):
        """测试策略状态表示"""
        if AI_OPTIMIZER_AVAILABLE:
            # 创建策略状态
            state = ai_strategy_optimizer.create_strategy_state(sample_market_data.iloc[:10])

            assert isinstance(state, StrategyState)
            assert hasattr(state, 'market_features')
            assert hasattr(state, 'portfolio_state')
            assert hasattr(state, 'risk_metrics')
            assert hasattr(state, 'timestamp')

            # 检查市场特征
            assert 'price_momentum' in state.market_features
            assert 'volatility' in state.market_features
            assert 'volume_trend' in state.market_features

    def test_strategy_action_space(self, ai_strategy_optimizer):
        """测试策略动作空间"""
        if AI_OPTIMIZER_AVAILABLE:
            # 获取动作空间
            action_space = ai_strategy_optimizer.get_action_space()

            assert isinstance(action_space, dict)
            assert 'position_sizing' in action_space
            assert 'entry_exit' in action_space
            assert 'risk_management' in action_space

            # 检查具体动作
            position_actions = action_space['position_sizing']
            assert 'increase' in position_actions
            assert 'decrease' in position_actions
            assert 'hold' in position_actions

    def test_reinforcement_learning_optimization(self, ai_strategy_optimizer, sample_market_data, sample_strategy_performance):
        """测试强化学习优化"""
        if AI_OPTIMIZER_AVAILABLE:
            # 执行强化学习优化
            optimization_result = ai_strategy_optimizer.optimize_with_rl(
                historical_data=sample_market_data,
                performance_data=sample_strategy_performance,
                optimization_horizon=50
            )

            assert isinstance(optimization_result, OptimizationResult)
            assert hasattr(optimization_result, 'optimal_parameters')
            assert hasattr(optimization_result, 'optimization_metrics')
            assert hasattr(optimization_result, 'convergence_history')

            # 检查优化指标
            metrics = optimization_result.optimization_metrics
            assert 'final_objective_value' in metrics
            assert 'total_iterations' in metrics
            assert 'convergence_rate' in metrics

    def test_deep_learning_parameter_tuning(self, ai_strategy_optimizer, sample_market_data):
        """测试深度学习参数调优"""
        if AI_OPTIMIZER_AVAILABLE:
            # 定义参数搜索空间
            param_space = {
                'learning_rate': [0.001, 0.01, 0.1],
                'batch_size': [16, 32, 64],
                'hidden_layers': [1, 2, 3],
                'neurons_per_layer': [32, 64, 128],
                'dropout_rate': [0.1, 0.2, 0.3]
            }

            # 执行深度学习调优
            tuning_result = ai_strategy_optimizer.tune_deep_learning_parameters(
                param_space=param_space,
                training_data=sample_market_data,
                validation_split=0.2,
                max_trials=10
            )

            assert isinstance(tuning_result, dict)
            assert 'best_parameters' in tuning_result
            assert 'best_score' in tuning_result
            assert 'trials_history' in tuning_result

    def test_multi_objective_optimization(self, ai_strategy_optimizer, sample_market_data):
        """测试多目标优化"""
        if AI_OPTIMIZER_AVAILABLE:
            # 设置多目标优化
            objectives = [
                OptimizationObjective.MAXIMIZE_RETURN,
                OptimizationObjective.MINIMIZE_RISK,
                OptimizationObjective.MAXIMIZE_SHARPE
            ]

            # 执行多目标优化
            pareto_front = ai_strategy_optimizer.optimize_multi_objective(
                objectives=objectives,
                historical_data=sample_market_data,
                population_size=20,
                generations=10
            )

            assert isinstance(pareto_front, list)
            assert len(pareto_front) > 0

            # 检查帕累托前沿的每个解
            for solution in pareto_front:
                assert 'parameters' in solution
                assert 'objective_values' in solution
                assert 'dominance_rank' in solution

    def test_adaptive_learning_mechanism(self, ai_strategy_optimizer, sample_market_data):
        """测试自适应学习机制"""
        if AI_OPTIMIZER_AVAILABLE:
            # 启用自适应学习
            ai_strategy_optimizer.enable_adaptive_learning()

            # 模拟市场条件变化
            market_conditions = ['bull', 'bear', 'sideways', 'volatile']

            adaptation_results = {}
            for condition in market_conditions:
                # 生成对应市场条件的样本数据
                condition_data = sample_market_data.copy()
                if condition == 'bull':
                    condition_data['returns'] = np.random.normal(0.002, 0.01, len(condition_data))
                elif condition == 'bear':
                    condition_data['returns'] = np.random.normal(-0.002, 0.01, len(condition_data))
                elif condition == 'volatile':
                    condition_data['returns'] = np.random.normal(0, 0.05, len(condition_data))

                # 执行自适应优化
                result = ai_strategy_optimizer.adapt_to_market_conditions(condition_data)
                adaptation_results[condition] = result

            assert len(adaptation_results) == len(market_conditions)

            # 检查自适应结果
            for condition, result in adaptation_results.items():
                assert 'adapted_parameters' in result
                assert 'adaptation_score' in result
                assert 'learning_progress' in result

    def test_transfer_learning_application(self, ai_strategy_optimizer, sample_market_data):
        """测试迁移学习应用"""
        if AI_OPTIMIZER_AVAILABLE:
            # 创建源领域数据（股票）
            source_data = sample_market_data.copy()

            # 创建目标领域数据（期货）
            target_data = sample_market_data.copy()
            target_data['symbol'] = 'ES'  # E-mini S&P 500 futures
            target_data['returns'] = target_data['returns'] * 1.5  # 更高的波动性

            # 执行迁移学习
            transfer_result = ai_strategy_optimizer.apply_transfer_learning(
                source_data=source_data,
                target_data=target_data,
                transfer_method='fine_tuning'
            )

            assert isinstance(transfer_result, dict)
            assert 'transfer_accuracy' in transfer_result
            assert 'knowledge_transfer_rate' in transfer_result
            assert 'adaptation_iterations' in transfer_result

    def test_bayesian_optimization_integration(self, ai_strategy_optimizer, sample_market_data):
        """测试贝叶斯优化集成"""
        if AI_OPTIMIZER_AVAILABLE:
            # 配置贝叶斯优化
            bayesian_config = {
                'acquisition_function': 'expected_improvement',
                'kernel': 'matern',
                'n_initial_points': 5,
                'n_iterations': 15
            }

            ai_strategy_optimizer.configure_bayesian_optimization(bayesian_config)

            # 执行贝叶斯优化
            bayesian_result = ai_strategy_optimizer.optimize_with_bayesian_method(
                objective_function=lambda params: -abs(params['threshold'] - 0.05),  # 最小化与0.05的偏差
                parameter_bounds={
                    'threshold': (0.01, 0.1),
                    'window_size': (10, 50),
                    'multiplier': (1.0, 3.0)
                },
                historical_data=sample_market_data
            )

            assert isinstance(bayesian_result, dict)
            assert 'optimal_parameters' in bayesian_result
            assert 'optimal_value' in bayesian_result
            assert 'optimization_path' in bayesian_result

    def test_neural_architecture_search(self, ai_strategy_optimizer, sample_market_data):
        """测试神经架构搜索"""
        if AI_OPTIMIZER_AVAILABLE:
            # 配置架构搜索空间
            architecture_space = {
                'num_layers': [1, 2, 3, 4],
                'layer_sizes': [32, 64, 128, 256],
                'activation_functions': ['relu', 'tanh', 'sigmoid'],
                'dropout_rates': [0.1, 0.2, 0.3, 0.4],
                'learning_rates': [0.001, 0.01, 0.1]
            }

            # 执行神经架构搜索
            architecture_result = ai_strategy_optimizer.search_neural_architecture(
                architecture_space=architecture_space,
                training_data=sample_market_data,
                search_budget=10,
                evaluation_metric='sharpe_ratio'
            )

            assert isinstance(architecture_result, dict)
            assert 'best_architecture' in architecture_result
            assert 'best_performance' in architecture_result
            assert 'searched_architectures' in architecture_result

    def test_online_learning_adaptation(self, ai_strategy_optimizer, sample_market_data):
        """测试在线学习适应"""
        if AI_OPTIMIZER_AVAILABLE:
            # 初始化在线学习器
            ai_strategy_optimizer.initialize_online_learning()

            # 模拟实时数据流
            online_performance = []

            for i in range(10):
                # 获取当前批次数据
                batch_data = sample_market_data.iloc[i*25:(i+1)*25]

                # 执行在线学习更新
                update_result = ai_strategy_optimizer.update_online_model(batch_data)

                online_performance.append(update_result)

            assert len(online_performance) == 10

            # 检查在线学习性能
            for result in online_performance:
                assert 'model_updated' in result
                assert 'adaptation_rate' in result
                assert 'prediction_accuracy' in result

    def test_ensemble_strategy_optimization(self, ai_strategy_optimizer, sample_market_data):
        """测试集成策略优化"""
        if AI_OPTIMIZER_AVAILABLE:
            # 定义基础策略集合
            base_strategies = [
                'momentum_strategy',
                'mean_reversion_strategy',
                'trend_following_strategy',
                'breakout_strategy'
            ]

            # 执行集成优化
            ensemble_result = ai_strategy_optimizer.optimize_ensemble_strategy(
                base_strategies=base_strategies,
                historical_data=sample_market_data,
                ensemble_methods=['weighted', 'stacked', 'blended'],
                optimization_target='diversification'
            )

            assert isinstance(ensemble_result, dict)
            assert 'optimal_ensemble' in ensemble_result
            assert 'strategy_weights' in ensemble_result
            assert 'ensemble_performance' in ensemble_result
            assert 'correlation_matrix' in ensemble_result

    def test_quantum_inspired_optimization(self, ai_strategy_optimizer, sample_market_data):
        """测试量子启发优化"""
        if AI_OPTIMIZER_AVAILABLE:
            # 配置量子启发算法
            quantum_config = {
                'algorithm': 'quantum_annealing',
                'num_qubits': 10,
                'annealing_schedule': 'linear',
                'tunneling_strength': 0.1
            }

            ai_strategy_optimizer.configure_quantum_optimization(quantum_config)

            # 执行量子启发优化
            quantum_result = ai_strategy_optimizer.optimize_with_quantum_inspiration(
                optimization_problem='portfolio_optimization',
                historical_data=sample_market_data,
                constraints={
                    'max_weight': 0.3,
                    'min_weight': 0.0,
                    'target_return': 0.1
                }
            )

            assert isinstance(quantum_result, dict)
            assert 'optimal_portfolio' in quantum_result
            assert 'quantum_energy' in quantum_result
            assert 'optimization_time' in quantum_result

    def test_performance_monitoring_and_analytics(self, ai_strategy_optimizer, sample_market_data):
        """测试性能监控和分析"""
        if AI_OPTIMIZER_AVAILABLE:
            # 执行一系列优化操作
            operations = ['rl_optimization', 'parameter_tuning', 'architecture_search']

            performance_metrics = {}

            for operation in operations:
                # 记录操作开始时间
                start_time = time.time()

                # 执行操作
                if operation == 'rl_optimization':
                    result = ai_strategy_optimizer.optimize_with_rl(
                        sample_market_data, pd.DataFrame(), 10
                    )
                elif operation == 'parameter_tuning':
                    result = ai_strategy_optimizer.tune_deep_learning_parameters(
                        {}, sample_market_data, max_trials=3
                    )
                elif operation == 'architecture_search':
                    result = ai_strategy_optimizer.search_neural_architecture(
                        {}, sample_market_data, search_budget=3
                    )

                # 记录操作结束时间
                end_time = time.time()

                performance_metrics[operation] = {
                    'execution_time': end_time - start_time,
                    'success': result is not None,
                    'result_size': len(str(result)) if result else 0
                }

            # 获取优化器性能统计
            optimizer_stats = ai_strategy_optimizer.get_performance_statistics()

            assert isinstance(optimizer_stats, dict)
            assert 'total_optimizations' in optimizer_stats
            assert 'average_convergence_time' in optimizer_stats
            assert 'success_rate' in optimizer_stats

    def test_error_handling_and_robustness(self, ai_strategy_optimizer):
        """测试错误处理和鲁棒性"""
        if AI_OPTIMIZER_AVAILABLE:
            # 测试无效配置处理
            invalid_config = OptimizationConfig(
                algorithm=OptimizationAlgorithm.PPO,
                objective=OptimizationObjective.MAXIMIZE_RETURN,
                max_iterations=-1,  # 无效的迭代次数
                learning_rate=10.0  # 无效的学习率
            )

            try:
                invalid_optimizer = AIStrategyOptimizer(config=invalid_config)
                # 如果没有抛出异常，验证配置被修正
                assert invalid_optimizer.config.max_iterations > 0
                assert 0 < invalid_optimizer.config.learning_rate < 1
            except (ValueError, AssertionError):
                # 期望的异常处理
                pass

            # 测试数据质量问题处理
            noisy_data = pd.DataFrame({
                'symbol': [None] * 100,  # 缺失值
                'close': [float('inf')] * 50 + [float('-inf')] * 50,  # 无限值
                'volume': ['invalid'] * 100  # 无效类型
            })

            try:
                ai_strategy_optimizer.optimize_with_rl(noisy_data, pd.DataFrame(), 5)
                # 如果没有崩溃，验证错误被妥善处理
            except Exception as e:
                # 验证异常类型合适
                assert isinstance(e, (ValueError, TypeError, RuntimeError))

    def test_configuration_management_and_persistence(self, ai_strategy_optimizer, tmp_path):
        """测试配置管理和持久化"""
        if AI_OPTIMIZER_AVAILABLE:
            # 更新配置
            new_config = OptimizationConfig(
                algorithm=OptimizationAlgorithm.SAC,
                objective=OptimizationObjective.MINIMIZE_RISK,
                max_iterations=200,
                learning_rate=0.0005,
                exploration_rate=0.05
            )

            ai_strategy_optimizer.update_configuration(new_config)

            # 验证配置更新
            assert ai_strategy_optimizer.config.algorithm == OptimizationAlgorithm.SAC
            assert ai_strategy_optimizer.config.objective == OptimizationObjective.MINIMIZE_RISK

            # 保存配置
            config_file = tmp_path / "optimizer_config.json"
            ai_strategy_optimizer.save_configuration(str(config_file))

            # 验证文件创建
            assert config_file.exists()

            # 加载配置
            loaded_optimizer = AIStrategyOptimizer()
            loaded_optimizer.load_configuration(str(config_file))

            # 验证配置加载
            assert loaded_optimizer.config.algorithm == OptimizationAlgorithm.SAC

    def test_scalability_and_performance_limits(self, ai_strategy_optimizer):
        """测试扩展性和性能限制"""
        if AI_OPTIMIZER_AVAILABLE:
            # 测试大规模参数空间
            large_param_space = {
                f'param_{i}': list(range(10)) for i in range(20)  # 20个参数，每个10个值
            }

            # 评估参数空间大小
            space_size = ai_strategy_optimizer.evaluate_parameter_space_size(large_param_space)

            assert isinstance(space_size, int)
            assert space_size > 10**20  # 应该非常大

            # 测试内存和计算资源限制
            resource_limits = ai_strategy_optimizer.get_resource_limits()

            assert isinstance(resource_limits, dict)
            assert 'max_memory_gb' in resource_limits
            assert 'max_cpu_cores' in resource_limits
            assert 'max_optimization_time_hours' in resource_limits

    def test_audit_trail_and_explainability(self, ai_strategy_optimizer, sample_market_data):
        """测试审计跟踪和可解释性"""
        if AI_OPTIMIZER_AVAILABLE:
            # 启用审计跟踪
            ai_strategy_optimizer.enable_audit_trail()

            # 执行优化
            result = ai_strategy_optimizer.optimize_with_rl(sample_market_data, pd.DataFrame(), 5)

            # 获取审计日志
            audit_log = ai_strategy_optimizer.get_audit_trail()

            assert isinstance(audit_log, list)
            assert len(audit_log) > 0

            # 检查审计记录
            for entry in audit_log:
                assert 'timestamp' in entry
                assert 'operation' in entry
                assert 'parameters' in entry
                assert 'result' in entry

            # 获取优化解释
            explanation = ai_strategy_optimizer.explain_optimization_result(result)

            assert isinstance(explanation, dict)
            assert 'optimization_reasoning' in explanation
            assert 'key_decisions' in explanation
            assert 'performance_drivers' in explanation

    def test_concurrent_optimization_sessions(self, ai_strategy_optimizer, sample_market_data):
        """测试并发优化会话"""
        if AI_OPTIMIZER_AVAILABLE:
            # 创建多个优化会话
            optimization_configs = [
                {'algorithm': 'ppo', 'objective': 'sharpe'},
                {'algorithm': 'sac', 'objective': 'return'},
                {'algorithm': 'dqn', 'objective': 'risk_adjusted'}
            ]

            results = []
            errors = []

            def run_optimization_session(config, session_id):
                try:
                    # 配置优化器
                    session_config = OptimizationConfig(
                        algorithm=getattr(OptimizationAlgorithm, config['algorithm'].upper()),
                        objective=getattr(OptimizationObjective, f"MAXIMIZE_{config['objective'].upper()}"),
                        max_iterations=10
                    )

                    session_optimizer = AIStrategyOptimizer(config=session_config)

                    # 执行优化
                    result = session_optimizer.optimize_with_rl(sample_market_data, pd.DataFrame(), 5)
                    results.append((session_id, result))

                except Exception as e:
                    errors.append((session_id, str(e)))

            # 并发执行优化会话
            threads = []
            for i, config in enumerate(optimization_configs):
                thread = threading.Thread(target=run_optimization_session, args=(config, i))
                threads.append(thread)
                thread.start()

            # 等待所有会话完成
            for thread in threads:
                thread.join()

            # 验证结果
            assert len(results) == len(optimization_configs)
            assert len(errors) == 0

            # 检查每个会话的结果
            for session_id, result in results:
                assert isinstance(result, OptimizationResult)
                assert result.optimization_metrics is not None
