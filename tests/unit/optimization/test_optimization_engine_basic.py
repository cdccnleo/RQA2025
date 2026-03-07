#!/usr/bin/env python3
"""
优化引擎基础测试用例

测试OptimizationEngine类的基本功能
"""

import sys
import importlib
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent
project_root_str = str(project_root)
src_path_str = str(project_root / "src")

if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

# 动态导入优化引擎模块
try:
    optimization_engine_module = importlib.import_module('optimization.core.optimization_engine')
    OptimizationEngine = getattr(optimization_engine_module, 'OptimizationEngine', None)
    OptimizationObjective = getattr(optimization_engine_module, 'OptimizationObjective', None)
    OptimizationConstraint = getattr(optimization_engine_module, 'OptimizationConstraint', None)
    OptimizationAlgorithm = getattr(optimization_engine_module, 'OptimizationAlgorithm', None)
    OptimizationResult = getattr(optimization_engine_module, 'OptimizationResult', None)
    
    if OptimizationEngine is None:
        pytest.skip("OptimizationEngine不可用", allow_module_level=True)
except ImportError:
    pytest.skip("优化引擎模块导入失败", allow_module_level=True)


class TestOptimizationEngineBasic:
    """优化引擎基础测试类"""

    @pytest.fixture
    def optimization_engine(self):
        """优化引擎实例"""
        return OptimizationEngine(name="test_engine")

    def test_initialization(self, optimization_engine):
        """测试初始化"""
        assert optimization_engine.name == "test_engine"
        assert isinstance(optimization_engine.objective_functions, dict)
        assert isinstance(optimization_engine.constraint_functions, dict)
        assert isinstance(optimization_engine.optimization_stats, dict)

        # 验证统计信息初始化
        assert optimization_engine.optimization_stats['total_runs'] == 0
        assert optimization_engine.optimization_stats['successful_runs'] == 0
        assert optimization_engine.optimization_stats['average_execution_time'] == 0.0

    def test_optimization_objective_enum(self):
        """测试优化目标枚举"""
        assert OptimizationObjective.MAXIMIZE_RETURN.value == "maximize_return"
        assert OptimizationObjective.MINIMIZE_RISK.value == "minimize_risk"
        assert OptimizationObjective.MAXIMIZE_SHARPE_RATIO.value == "maximize_sharpe_ratio"
        assert OptimizationObjective.MAXIMIZE_SORTINO_RATIO.value == "maximize_sortino_ratio"
        assert OptimizationObjective.MAXIMIZE_CALMAR_RATIO.value == "maximize_calmar_ratio"
        assert OptimizationObjective.MINIMIZE_DRAWDOWN.value == "minimize_drawdown"
        assert OptimizationObjective.MAXIMIZE_PROFIT_FACTOR.value == "maximize_profit_factor"

    def test_optimization_constraint_enum(self):
        """测试优化约束枚举"""
        assert OptimizationConstraint.NO_SHORT_SELLING.value == "no_short_selling"
        assert OptimizationConstraint.MAX_WEIGHT.value == "max_weight"
        assert OptimizationConstraint.MIN_WEIGHT.value == "min_weight"
        assert OptimizationConstraint.SECTOR_LIMITS.value == "sector_limits"
        assert OptimizationConstraint.TURNOVER_LIMIT.value == "turnover_limit"
        assert OptimizationConstraint.RISK_BUDGET.value == "risk_budget"

    def test_optimization_algorithm_enum(self):
        """测试优化算法枚举"""
        assert OptimizationAlgorithm.SLSQP.value == "SLSQP"
        assert OptimizationAlgorithm.COBYLA.value == "COBYLA"
        assert OptimizationAlgorithm.BFGS.value == "BFGS"
        # 注意：枚举中是LBFGSB，不是L_BFGS_B
        assert OptimizationAlgorithm.LBFGSB.value == "L - BFGS - B"
        assert OptimizationAlgorithm.TRUST_CONSTR.value == "trust - constr"
        # TNC可能不存在，检查实际存在的枚举
        assert OptimizationAlgorithm.DIFFERENTIAL_EVOLUTION.value == "differential_evolution"

    def test_register_objective_function(self, optimization_engine):
        """测试注册目标函数"""
        def custom_objective(weights, returns, cov_matrix):
            return -np.sum(weights * returns)  # 负号表示最大化

        optimization_engine.register_objective_function(
            OptimizationObjective.MAXIMIZE_RETURN,
            custom_objective
        )

        assert OptimizationObjective.MAXIMIZE_RETURN in optimization_engine.objective_functions
        assert optimization_engine.objective_functions[OptimizationObjective.MAXIMIZE_RETURN] == custom_objective

    def test_register_constraint_function(self, optimization_engine):
        """测试注册约束函数"""
        def custom_constraint(weights, **kwargs):
            return np.sum(weights) - 1.0  # 权重和为1

        optimization_engine.register_constraint_function(
            OptimizationConstraint.MAX_WEIGHT,
            custom_constraint
        )

        assert OptimizationConstraint.MAX_WEIGHT in optimization_engine.constraint_functions
        assert optimization_engine.constraint_functions[OptimizationConstraint.MAX_WEIGHT] == custom_constraint

    def test_portfolio_optimization_basic(self, optimization_engine):
        """测试基本投资组合优化"""
        # 创建模拟数据 - optimize_portfolio需要pd.DataFrame
        np.random.seed(42)
        n_assets = 4
        n_periods = 100
        # 生成历史收益数据（DataFrame格式）
        returns = pd.DataFrame(
            np.random.normal(0.001, 0.02, (n_periods, n_assets)),
            columns=[f'asset_{i}' for i in range(n_assets)]
        )

        # 执行优化
        result = optimization_engine.optimize_portfolio(
            returns=returns,
            objective=OptimizationObjective.MAXIMIZE_SHARPE_RATIO,
            constraints=[OptimizationConstraint.NO_SHORT_SELLING],
            algorithm=OptimizationAlgorithm.SLSQP
        )

        # 验证结果
        assert isinstance(result, OptimizationResult)
        assert hasattr(result, 'optimal_weights')
        assert hasattr(result, 'objective_value')
        assert hasattr(result, 'success')

        # 验证权重
        weights = result.optimal_weights
        assert len(weights) == n_assets
        assert np.all(weights >= 0)  # 无卖空约束
        assert np.isclose(np.sum(weights), 1.0, atol=1e-6)  # 权重和为1

    def test_portfolio_optimization_with_constraints(self, optimization_engine):
        """测试带约束的投资组合优化"""
        # 创建模拟数据 - optimize_portfolio需要pd.DataFrame
        np.random.seed(42)
        n_assets = 3
        n_periods = 100
        # 生成历史收益数据（DataFrame格式）
        returns = pd.DataFrame(
            np.random.normal(0.001, 0.02, (n_periods, n_assets)),
            columns=[f'asset_{i}' for i in range(n_assets)]
        )

        # 约束列表（只使用枚举类型）
        constraints = [
            OptimizationConstraint.NO_SHORT_SELLING
        ]

        # 使用bounds参数来设置最小权重约束，而不是通过约束函数
        bounds = [(0.1, 1.0), (0.0, 1.0), (0.0, 1.0)]  # 第一资产最小权重0.1

        result = optimization_engine.optimize_portfolio(
            returns=returns,
            objective=OptimizationObjective.MINIMIZE_RISK,
            constraints=constraints,
            algorithm=OptimizationAlgorithm.SLSQP,
            bounds=bounds
        )

        assert isinstance(result, OptimizationResult)
        assert hasattr(result, 'optimal_weights')

        weights = result.optimal_weights
        assert len(weights) == n_assets
        assert np.all(weights >= 0)  # 无卖空
        # 如果优化成功，第一资产应该满足最小权重约束
        if result.success:
            assert weights[0] >= 0.1 - 1e-6  # 允许小的数值误差
        assert np.isclose(np.sum(weights), 1.0, atol=1e-6)  # 权重和为1

    def test_strategy_optimization(self, optimization_engine):
        """测试策略优化"""
        # 定义策略参数范围
        param_ranges = {
            'fast_period': (5, 50),
            'slow_period': (10, 100),
            'stop_loss': (0.01, 0.1)
        }

        # 模拟策略函数（根据参数返回性能分数）
        def strategy_func(params, data):
            fast_period, slow_period, stop_loss = params['fast_period'], params['slow_period'], params['stop_loss']
            # 模拟性能计算
            if fast_period >= slow_period:
                return 0.0  # 无效参数
            return fast_period * 0.1 + slow_period * 0.05 + stop_loss * 10  # 返回性能分数

        # 创建历史数据
        historical_data = pd.DataFrame(
            np.random.normal(0.001, 0.02, (100, 3)),
            columns=['price', 'volume', 'indicator']
        )

        result = optimization_engine.optimize_strategy_parameters(
            strategy_func=strategy_func,
            parameter_bounds=param_ranges,
            historical_data=historical_data,
            objective=OptimizationObjective.MAXIMIZE_RETURN,
            algorithm=OptimizationAlgorithm.DIFFERENTIAL_EVOLUTION
        )

        assert isinstance(result, OptimizationResult)
        assert hasattr(result, 'optimal_weights')  # 使用optimal_weights字段存储参数
        if result.success:
            best_params = result.optimal_weights
            assert len(best_params) == 3

    def test_get_optimization_stats(self, optimization_engine):
        """测试获取优化统计信息"""
        stats = optimization_engine.get_optimization_stats()

        assert isinstance(stats, dict)
        assert 'total_runs' in stats
        assert 'successful_runs' in stats
        assert 'average_execution_time' in stats
        assert 'last_run_timestamp' in stats

        # 初始状态
        assert stats['total_runs'] == 0
        assert stats['successful_runs'] == 0

    def test_invalid_inputs_handling(self, optimization_engine):
        """测试无效输入处理"""
        # 空收益DataFrame
        empty_data = pd.DataFrame()
        result = optimization_engine.optimize_portfolio(
            returns=empty_data,
            objective=OptimizationObjective.MAXIMIZE_RETURN,
            constraints=[]
        )
        # 应该返回失败结果，而不是抛出异常
        assert isinstance(result, OptimizationResult)
        assert result.success is False

    def test_custom_objective_function(self, optimization_engine):
        """测试自定义目标函数"""
        def custom_sharpe_objective(weights, expected_returns, cov_matrix):
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -(portfolio_return / portfolio_risk) if portfolio_risk > 0 else 0  # 负夏普比率（最大化）

        optimization_engine.register_objective_function(
            OptimizationObjective.MAXIMIZE_SHARPE_RATIO,
            custom_sharpe_objective
        )

        # 创建测试数据 - optimize_portfolio需要pd.DataFrame
        n_periods = 100
        returns = pd.DataFrame(
            np.random.normal(0.001, 0.02, (n_periods, 3)),
            columns=['asset_0', 'asset_1', 'asset_2']
        )

        result = optimization_engine.optimize_portfolio(
            returns=returns,
            objective=OptimizationObjective.MAXIMIZE_SHARPE_RATIO,
            constraints=[OptimizationConstraint.NO_SHORT_SELLING],
            algorithm=OptimizationAlgorithm.SLSQP
        )

        assert isinstance(result, OptimizationResult)
        assert hasattr(result, 'optimal_weights')
        assert hasattr(result, 'objective_value')

    def test_constraint_validation(self, optimization_engine):
        """测试约束验证"""
        # 测试无卖空约束
        def no_short_selling_constraint(weights, **kwargs):
            return weights  # 返回权重，scipy会检查非负性

        optimization_engine.register_constraint_function(
            OptimizationConstraint.NO_SHORT_SELLING,
            no_short_selling_constraint
        )

        assert OptimizationConstraint.NO_SHORT_SELLING in optimization_engine.constraint_functions

    def test_optimization_with_bounds(self, optimization_engine):
        """测试带边界约束的优化"""
        # 创建测试数据 - optimize_portfolio需要pd.DataFrame
        n_periods = 100
        returns = pd.DataFrame(
            np.random.normal(0.001, 0.02, (n_periods, 3)),
            columns=['asset_0', 'asset_1', 'asset_2']
        )

        # 设置边界
        bounds = [(0.1, 0.5), (0.0, 0.4), (0.2, 0.6)]  # 每资产的权重边界

        result = optimization_engine.optimize_portfolio(
            returns=returns,
            objective=OptimizationObjective.MAXIMIZE_RETURN,
            constraints=[OptimizationConstraint.NO_SHORT_SELLING],
            bounds=bounds,
            algorithm=OptimizationAlgorithm.SLSQP
        )

        assert isinstance(result, OptimizationResult)
        if result.success:
            weights = result.optimal_weights
            # 验证边界约束
            assert 0.1 <= weights[0] <= 0.5
            assert 0.0 <= weights[1] <= 0.4
            assert 0.2 <= weights[2] <= 0.6
            assert np.isclose(np.sum(weights), 1.0, atol=1e-6)

    def test_performance_tracking(self, optimization_engine):
        """测试性能跟踪"""
        initial_stats = optimization_engine.get_optimization_stats()

        # 创建测试数据 - optimize_portfolio需要pd.DataFrame
        n_periods = 100
        returns = pd.DataFrame(
            np.random.normal(0.001, 0.02, (n_periods, 3)),
            columns=['asset_0', 'asset_1', 'asset_2']
        )

        optimization_engine.optimize_portfolio(
            returns=returns,
            objective=OptimizationObjective.MAXIMIZE_RETURN,
            constraints=[OptimizationConstraint.NO_SHORT_SELLING]
        )

        # 验证统计信息更新
        updated_stats = optimization_engine.get_optimization_stats()
        assert updated_stats['total_runs'] == initial_stats['total_runs'] + 1
        assert updated_stats['last_run_timestamp'] is not None

    def test_error_handling(self, optimization_engine):
        """测试错误处理"""
        # 测试无效的目标函数
        def invalid_objective(weights, expected_returns, cov_matrix):
            raise RuntimeError("Objective function error")

        optimization_engine.register_objective_function(
            OptimizationObjective.MAXIMIZE_RETURN,
            invalid_objective
        )

        # 创建测试数据 - optimize_portfolio需要pd.DataFrame
        n_periods = 100
        returns = pd.DataFrame(
            np.random.normal(0.001, 0.02, (n_periods, 2)),
            columns=['asset_0', 'asset_1']
        )

        # 应该不会抛出异常，而是返回失败结果
        result = optimization_engine.optimize_portfolio(
            returns=returns,
            objective=OptimizationObjective.MAXIMIZE_RETURN,
            constraints=[OptimizationConstraint.NO_SHORT_SELLING]
        )

        assert isinstance(result, OptimizationResult)
        assert hasattr(result, 'success')

    @pytest.mark.parametrize("objective", [
        OptimizationObjective.MAXIMIZE_RETURN,
        OptimizationObjective.MINIMIZE_RISK,
        OptimizationObjective.MAXIMIZE_SHARPE_RATIO,
    ])
    def test_different_objectives(self, optimization_engine, objective):
        """参数化测试不同目标函数"""
        # 创建测试数据 - optimize_portfolio需要pd.DataFrame
        n_periods = 100
        returns = pd.DataFrame(
            np.random.normal(0.001, 0.02, (n_periods, 3)),
            columns=['asset_0', 'asset_1', 'asset_2']
        )

        result = optimization_engine.optimize_portfolio(
            returns=returns,
            objective=objective,
            constraints=[OptimizationConstraint.NO_SHORT_SELLING],
            algorithm=OptimizationAlgorithm.SLSQP
        )

        assert isinstance(result, OptimizationResult)
        assert hasattr(result, 'optimal_weights')
        assert hasattr(result, 'objective_value')
        assert hasattr(result, 'success')

        if result.success:
            weights = result.optimal_weights
            assert len(weights) == len(returns.columns)
            assert np.all(weights >= 0)  # 默认无卖空
            assert np.isclose(np.sum(weights), 1.0, atol=1e-3)

    def test_empty_constraints_handling(self, optimization_engine):
        """测试空约束处理"""
        returns = np.array([0.1, 0.12, 0.08])
        cov_matrix = np.array([
            [0.04, 0.006, 0.012],
            [0.006, 0.09, 0.015],
            [0.012, 0.015, 0.16]
        ])

        result = optimization_engine.optimize_portfolio(
            returns=returns,
            cov_matrix=cov_matrix,
            objective=OptimizationObjective.MAXIMIZE_RETURN,
            constraints=[]  # 空约束列表
        )

        assert isinstance(result, OptimizationResult)
        assert hasattr(result, 'optimal_weights')

    def test_large_portfolio_optimization(self, optimization_engine):
        """测试大规模投资组合优化"""
        np.random.seed(42)
        n_assets = 20  # 20个资产

        # 生成随机数据
        returns_data = np.random.normal(0.1, 0.05, (100, n_assets))  # 100 periods
        returns = pd.DataFrame(returns_data, columns=[f'asset_{i}' for i in range(n_assets)])
        cov_matrix = np.random.rand(n_assets, n_assets)
        cov_matrix = (cov_matrix + cov_matrix.T) / 2
        np.fill_diagonal(cov_matrix, np.random.uniform(0.01, 0.1, n_assets))

        result = optimization_engine.optimize_portfolio(
            returns=returns,
            cov_matrix=cov_matrix,
            objective=OptimizationObjective.MAXIMIZE_SHARPE_RATIO,
            constraints=[],
            algorithm=OptimizationAlgorithm.SLSQP
        )

        assert isinstance(result, OptimizationResult)
        weights = result.optimal_weights

        assert len(weights) == n_assets
        assert np.all(weights >= 0)
        assert np.isclose(np.sum(weights), 1.0, atol=1e-3)
