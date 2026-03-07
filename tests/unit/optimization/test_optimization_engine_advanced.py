# -*- coding: utf-8 -*-
"""
优化层 - 优化引擎高级单元测试
测试覆盖率目标: 95%+
按照业务流程驱动架构设计测试优化引擎核心功能
"""

import sys
import importlib
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

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
    OptimizationResult = getattr(optimization_engine_module, 'OptimizationResult', None)
    OptimizationObjective = getattr(optimization_engine_module, 'OptimizationObjective', None)
    OptimizationConstraint = getattr(optimization_engine_module, 'OptimizationConstraint', None)
    OptimizationAlgorithm = getattr(optimization_engine_module, 'OptimizationAlgorithm', None)
    
    if OptimizationEngine is None:
        pytest.skip("OptimizationEngine不可用", allow_module_level=True)
except ImportError:
    pytest.skip("优化引擎模块导入失败", allow_module_level=True)


class TestOptimizationEngineCore:
    """测试优化引擎核心功能"""

    def setup_method(self, method):
        """设置测试环境"""
        self.engine = OptimizationEngine("test_engine")

    def test_optimization_engine_initialization(self):
        """测试优化引擎初始化"""
        assert self.engine.name == "test_engine"
        assert isinstance(self.engine.objectives, list)
        assert isinstance(self.engine.constraints, list)
        assert isinstance(self.engine.algorithms, list)
        assert self.engine.default_algorithm == OptimizationAlgorithm.SLSQP

    def test_objective_function_setup(self):
        """测试目标函数设置"""
        # 设置最大化夏普比率的目标
        self.engine.add_objective(OptimizationObjective.MAXIMIZE_SHARPE_RATIO)

        # 设置最小化风险的目标
        self.engine.add_objective(OptimizationObjective.MINIMIZE_RISK)

        assert len(self.engine.objectives) == 2
        assert OptimizationObjective.MAXIMIZE_SHARPE_RATIO in self.engine.objectives
        assert OptimizationObjective.MINIMIZE_RISK in self.engine.objectives

    def test_constraint_setup(self):
        """测试约束条件设置"""
        # 添加无空头约束
        self.engine.add_constraint(OptimizationConstraint.NO_SHORT_SELLING)

        # 添加最大权重约束
        self.engine.add_constraint(OptimizationConstraint.MAX_WEIGHT)

        assert len(self.engine.constraints) == 2
        assert OptimizationConstraint.NO_SHORT_SELLING in self.engine.constraints
        assert OptimizationConstraint.MAX_WEIGHT in self.engine.constraints

    def test_portfolio_optimization_basic(self):
        """测试基本投资组合优化"""
        # 模拟资产收益率数据
        np.random.seed(42)
        n_assets = 4
        n_periods = 100

        # 生成模拟收益率数据
        returns = np.random.normal(0.001, 0.02, (n_periods, n_assets))

        # 设置优化参数
        initial_weights = np.ones(n_assets) / n_assets
        bounds = [(0, 1) for _ in range(n_assets)]

        # 执行优化
        result = self.engine.optimize_portfolio(
            returns=returns,
            objective=OptimizationObjective.MAXIMIZE_SHARPE_RATIO,
            constraints=[OptimizationConstraint.NO_SHORT_SELLING],
            bounds=bounds,
            initial_weights=initial_weights
        )

        # 验证优化结果
        assert isinstance(result, OptimizationResult)
        assert result.success is True
        assert len(result.optimal_weights) == n_assets
        assert abs(sum(result.optimal_weights) - 1.0) < 1e-6  # 权重和为1
        assert all(0 <= w <= 1 for w in result.optimal_weights)  # 权重在边界内

    def test_mean_variance_optimization(self):
        """测试均值方差优化"""
        # 创建测试数据
        np.random.seed(42)
        n_assets = 3

        # 预期收益率
        expected_returns = np.array([0.12, 0.08, 0.15])

        # 协方差矩阵
        cov_matrix = np.array([
            [0.04, 0.02, 0.01],
            [0.02, 0.09, 0.03],
            [0.01, 0.03, 0.16]
        ])

        # 执行均值方差优化
        result = self.engine.mean_variance_optimize(
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            target_return=0.10
        )

        # 验证结果
        assert result.success is True
        assert len(result.optimal_weights) == n_assets
        assert abs(sum(result.optimal_weights) - 1.0) < 1e-6

        # 计算实际预期收益率
        actual_return = np.dot(result.optimal_weights, expected_returns)
        assert abs(actual_return - 0.10) < 0.01  # 接近目标收益率

    def test_risk_parity_optimization(self):
        """测试风险平价优化"""
        # 设置资产波动率
        volatilities = np.array([0.2, 0.25, 0.15, 0.3])

        # 相关性矩阵
        correlation_matrix = np.array([
            [1.0, 0.5, 0.3, 0.2],
            [0.5, 1.0, 0.4, 0.3],
            [0.3, 0.4, 1.0, 0.25],
            [0.2, 0.3, 0.25, 1.0]
        ])

        # 计算协方差矩阵
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix

        # 执行风险平价优化
        result = self.engine.risk_parity_optimize(cov_matrix)

        # 验证结果
        assert result.success is True
        assert len(result.optimal_weights) == len(volatilities)
        assert abs(sum(result.optimal_weights) - 1.0) < 1e-6

        # 计算风险贡献
        portfolio_volatility = np.sqrt(result.optimal_weights.T @ cov_matrix @ result.optimal_weights)
        marginal_contributions = cov_matrix @ result.optimal_weights / portfolio_volatility
        risk_contributions = result.optimal_weights * marginal_contributions

        # 验证风险平价（各资产风险贡献相近）
        risk_contribution_std = np.std(risk_contributions)
        assert risk_contribution_std < 0.01  # 风险贡献标准差很小

    def test_black_litterman_optimization(self):
        """测试Black-Litterman优化"""
        # 先验预期收益率
        prior_returns = np.array([0.08, 0.06, 0.10, 0.07])

        # 协方差矩阵
        cov_matrix = np.array([
            [0.04, 0.02, 0.01, 0.015],
            [0.02, 0.09, 0.03, 0.025],
            [0.01, 0.03, 0.16, 0.02],
            [0.015, 0.025, 0.02, 0.25]
        ])

        # 投资者观点
        views = np.array([0.12, 0.09])  # 对前两个资产的预期收益率
        view_confidences = np.array([0.7, 0.8])  # 观点信心

        # 执行Black-Litterman优化
        result = self.engine.black_litterman_optimize(
            prior_returns=prior_returns,
            cov_matrix=cov_matrix,
            views=views,
            view_confidences=view_confidences
        )

        # 验证结果
        assert result.success is True
        assert len(result.optimal_weights) == len(prior_returns)
        assert abs(sum(result.optimal_weights) - 1.0) < 1e-6

        # 后验收益率应该不同于先验收益率
        posterior_returns = result.convergence_info.get('posterior_returns', prior_returns)
        assert not np.allclose(posterior_returns, prior_returns)


class TestOptimizationConstraints:
    """测试优化约束条件"""

    def setup_method(self, method):
        """设置测试环境"""
        self.engine = OptimizationEngine()

    def test_no_short_selling_constraint(self):
        """测试无空头约束"""
        n_assets = 3

        # 设置约束
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 权重和为1
            {'type': 'ineq', 'fun': lambda x: x}             # 无空头（权重 >= 0）
        ]

        bounds = [(0, 1) for _ in range(n_assets)]  # 权重边界

        # 随机目标函数
        def objective(weights):
            return -np.sum(weights * np.random.random(n_assets))

        # 执行优化
        from scipy.optimize import minimize
        result = minimize(
            objective,
            np.ones(n_assets) / n_assets,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        # 验证约束满足
        assert result.success is True
        assert all(w >= 0 for w in result.x)  # 无空头
        assert abs(sum(result.x) - 1.0) < 1e-6  # 权重和为1

    def test_max_weight_constraint(self):
        """测试最大权重约束"""
        n_assets = 4
        max_weight = 0.3  # 最大30%权重

        # 设置约束
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 权重和为1
        ]

        bounds = [(0, max_weight) for _ in range(n_assets)]  # 最大权重约束

        # 随机目标函数
        def objective(weights):
            return np.sum(weights**2)  # 最小化权重方差

        # 执行优化
        from scipy.optimize import minimize
        result = minimize(
            objective,
            np.ones(n_assets) / n_assets,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        # 验证约束满足
        assert result.success is True
        assert all(w <= max_weight for w in result.x)  # 不超过最大权重
        assert abs(sum(result.x) - 1.0) < 1e-6  # 权重和为1

    def test_sector_limit_constraint(self):
        """测试行业限制约束"""
        # 模拟不同行业的资产
        sectors = ['tech', 'finance', 'consumer', 'healthcare']
        sector_weights = {
            'tech': 0.4,      # 科技股权重
            'finance': 0.3,   # 金融股权重
            'consumer': 0.2,  # 消费股权重
            'healthcare': 0.1  # 医疗股权重
        }

        # 为每个行业分配资产
        assets = []
        asset_sectors = []
        for sector, weight in sector_weights.items():
            n_assets_in_sector = 2  # 每个行业2个资产
            assets.extend([f"{sector}_asset_{i}" for i in range(n_assets_in_sector)])
            asset_sectors.extend([sector] * n_assets_in_sector)

        n_assets = len(assets)

        # 创建行业约束
        def sector_constraint(weights, sector, max_weight):
            """行业权重约束"""
            sector_indices = [i for i, s in enumerate(asset_sectors) if s == sector]
            sector_weight = sum(weights[i] for i in sector_indices)
            return max_weight - sector_weight

        # 设置约束
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 权重和为1
        ]

        # 添加行业约束
        for sector, max_weight in sector_weights.items():
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, s=sector, mw=max_weight: sector_constraint(x, s, mw)
            })

        bounds = [(0, 1) for _ in range(n_assets)]

        # 随机目标函数
        def objective(weights):
            return np.random.random()

        # 执行优化
        from scipy.optimize import minimize
        result = minimize(
            objective,
            np.ones(n_assets) / n_assets,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        # 验证行业约束满足
        assert result.success is True
        for sector, max_weight in sector_weights.items():
            sector_indices = [i for i, s in enumerate(asset_sectors) if s == sector]
            actual_sector_weight = sum(result.x[i] for i in sector_indices)
            assert actual_sector_weight <= max_weight


class TestOptimizationAlgorithms:
    """测试优化算法"""

    def setup_method(self, method):
        """设置测试环境"""
        self.engine = OptimizationEngine()

    def test_slsqp_algorithm(self):
        """测试SLSQP算法"""
        # 简单的二次规划问题
        def objective(x):
            return (x[0] - 1)**2 + (x[1] - 2)**2

        def constraint(x):
            return x[0] + x[1] - 1  # x[0] + x[1] = 1

        constraints = [{'type': 'eq', 'fun': constraint}]
        bounds = [(-10, 10), (-10, 10)]

        result = self.engine.optimize_with_algorithm(
            objective=objective,
            initial_guess=[0, 0],
            bounds=bounds,
            constraints=constraints,
            algorithm=OptimizationAlgorithm.SLSQP
        )

        # 验证结果
        assert result.success is True
        assert abs(result.x[0] + result.x[1] - 1) < 1e-6  # 约束满足
        assert abs(result.x[0] - 0.5) < 0.1  # 接近最优解
        assert abs(result.x[1] - 0.5) < 0.1

    def test_differential_evolution_algorithm(self):
        """测试差分进化算法"""
        # Rastrigin函数（多模态优化问题）
        def rastrigin(x):
            A = 10
            return A * len(x) + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

        bounds = [(-5.12, 5.12), (-5.12, 5.12)]

        result = self.engine.optimize_with_algorithm(
            objective=rastrigin,
            bounds=bounds,
            algorithm=OptimizationAlgorithm.DIFFERENTIAL_EVOLUTION
        )

        # 验证结果
        assert result.success is True
        # Rastrigin函数的最优值是0
        assert result.fun < 1.0  # 应该接近最优值

    def test_trust_region_algorithm(self):
        """测试信任域算法"""
        # Rosenbrock函数
        def rosenbrock(x):
            return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

        result = self.engine.optimize_with_algorithm(
            objective=rosenbrock,
            initial_guess=[0, 0],
            algorithm=OptimizationAlgorithm.TRUST_CONSTR
        )

        # 验证结果
        assert result.success is True
        # Rosenbrock函数在(1,1)处有最小值
        assert abs(result.x[0] - 1.0) < 0.1
        assert abs(result.x[1] - 1.0) < 0.1
        assert result.fun < 0.1  # 目标函数值很小


class TestOptimizationPerformance:
    """测试优化性能"""

    def setup_method(self, method):
        """设置测试环境"""
        self.engine = OptimizationEngine()

    def test_optimization_execution_time(self):
        """测试优化执行时间"""
        # 创建不同规模的优化问题
        problem_sizes = [5, 10, 20, 50]

        performance_results = []

        for n_assets in problem_sizes:
            # 生成随机数据
            np.random.seed(42)
            expected_returns = np.random.normal(0.1, 0.02, n_assets)
            cov_matrix = np.random.random((n_assets, n_assets))
            cov_matrix = cov_matrix @ cov_matrix.T  # 确保正定

            # 执行优化
            start_time = time.time()

            result = self.engine.mean_variance_optimize(
                expected_returns=expected_returns,
                cov_matrix=cov_matrix,
                target_return=0.08
            )

            end_time = time.time()

            execution_time = end_time - start_time

            performance_results.append({
                'problem_size': n_assets,
                'execution_time': execution_time,
                'success': result.success
            })

        # 验证性能
        for result in performance_results:
            assert result['execution_time'] > 0
            assert result['execution_time'] < 10  # 应该在10秒内完成
            assert result['success'] is True

        # 检查性能随问题规模的变化
        times = [r['execution_time'] for r in performance_results]
        sizes = [r['problem_size'] for r in performance_results]

        # 通常情况下，执行时间会随着问题规模增加
        if len(times) > 1:
            assert times[-1] >= times[0]  # 更大问题应该不更快

    def test_optimization_convergence_analysis(self):
        """测试优化收敛分析"""
        # 使用需要多次迭代的问题
        def difficult_objective(x):
            # Ackley函数 - 具有多个局部最优值
            a = 20
            b = 0.2
            c = 2 * np.pi
            d = len(x)

            sum1 = -a * np.exp(-b * np.sqrt(np.sum(x**2) / d))
            sum2 = -np.exp(np.sum(np.cos(c * x)) / d)

            return a + np.exp(1) + sum1 + sum2

        # 测试不同初始点
        initial_points = [
            [0, 0],
            [1, 1],
            [-1, -1],
            [2, 2],
            [-2, -2]
        ]

        convergence_results = []

        for initial_point in initial_points:
            result = self.engine.optimize_with_algorithm(
                objective=difficult_objective,
                initial_guess=initial_point,
                algorithm=OptimizationAlgorithm.BFGS
            )

            convergence_results.append({
                'initial_point': initial_point,
                'success': result.success,
                'final_value': result.fun,
                'nfev': result.nfev if hasattr(result, 'nfev') else None,
                'njev': result.njev if hasattr(result, 'njev') else None
            })

        # 验证收敛
        successful_runs = sum(1 for r in convergence_results if r['success'])
        assert successful_runs > 0  # 至少有一些成功收敛

        # 检查Ackley函数的最优值（应该是0）
        best_result = min(convergence_results, key=lambda x: x['final_value'] if x['success'] else float('in'))
        if best_result['success']:
            assert best_result['final_value'] < 1.0  # 应该接近最优值

    def test_optimization_memory_usage(self):
        """测试优化内存使用"""
        import psutil
        import os

        process = psutil.Process(os.getpid())

        # 记录初始内存
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 执行大规模优化
        n_assets = 100
        np.random.seed(42)

        expected_returns = np.random.normal(0.1, 0.02, n_assets)
        cov_matrix = np.random.random((n_assets, n_assets))
        cov_matrix = cov_matrix @ cov_matrix.T

        result = self.engine.mean_variance_optimize(
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            target_return=0.08
        )

        # 记录执行后内存
        final_memory = process.memory_info().rss / 1024 / 1024

        memory_increase = final_memory - initial_memory

        # 验证内存使用合理
        assert memory_increase >= 0
        assert memory_increase < 200  # 内存增加不应超过200MB
        assert result.success is True


class TestOptimizationResultAnalysis:
    """测试优化结果分析"""

    def setup_method(self, method):
        """设置测试环境"""
        self.engine = OptimizationEngine()

    def test_result_validation(self):
        """测试结果验证"""
        # 创建测试结果
        optimal_weights = np.array([0.3, 0.4, 0.3])
        result = OptimizationResult(
            success=True,
            optimal_weights=optimal_weights,
            objective_value=-0.15,  # 负的夏普比率（最大化）
            convergence_info={"iterations": 50, "function_evaluations": 150},
            execution_time=2.5,
            algorithm_used="SLSQP"
        )

        # 验证结果属性
        assert result.success is True
        assert len(result.optimal_weights) == 3
        assert abs(sum(result.optimal_weights) - 1.0) < 1e-10
        assert result.objective_value == -0.15
        assert result.execution_time == 2.5
        assert result.algorithm_used == "SLSQP"

        # 验证权重有效性
        assert all(0 <= w <= 1 for w in result.optimal_weights)

    def test_result_serialization(self):
        """测试结果序列化"""
        result = OptimizationResult(
            success=True,
            optimal_weights=np.array([0.25, 0.35, 0.4]),
            objective_value=0.12,
            convergence_info={"status": "converged", "iterations": 100},
            execution_time=1.8,
            algorithm_used="COBYLA"
        )

        # 序列化为字典
        result_dict = result.to_dict()

        # 验证序列化结果
        assert result_dict['success'] is True
        assert result_dict['objective_value'] == 0.12
        assert result_dict['execution_time'] == 1.8
        assert result_dict['algorithm_used'] == "COBYLA"
        assert isinstance(result_dict['optimal_weights'], list)
        assert len(result_dict['optimal_weights']) == 3
        assert abs(sum(result_dict['optimal_weights']) - 1.0) < 1e-10

    def test_result_comparison(self):
        """测试结果比较"""
        results = [
            OptimizationResult(True, np.array([0.4, 0.3, 0.3]), 0.15, {}, 1.2, "SLSQP"),
            OptimizationResult(True, np.array([0.3, 0.4, 0.3]), 0.18, {}, 1.5, "COBYLA"),
            OptimizationResult(True, np.array([0.2, 0.3, 0.5]), 0.12, {}, 0.8, "BFGS")
        ]

        # 比较目标函数值
        best_result = max(results, key=lambda x: x.objective_value)
        worst_result = min(results, key=lambda x: x.objective_value)

        assert best_result.objective_value == 0.18
        assert worst_result.objective_value == 0.12

        # 比较执行时间
        fastest_result = min(results, key=lambda x: x.execution_time)
        slowest_result = max(results, key=lambda x: x.execution_time)

        assert fastest_result.execution_time == 0.8
        assert slowest_result.execution_time == 1.5

    def test_result_statistics(self):
        """测试结果统计"""
        # 生成多个优化结果
        np.random.seed(42)
        results = []

        for i in range(10):
            weights = np.random.random(4)
            weights = weights / sum(weights)  # 归一化

            result = OptimizationResult(
                success=True,
                optimal_weights=weights,
                objective_value=np.random.normal(0.1, 0.02),
                convergence_info={"iterations": np.random.randint(20, 100)},
                execution_time=np.random.uniform(0.5, 3.0),
                algorithm_used="SLSQP"
            )
            results.append(result)

        # 计算统计信息
        objective_values = [r.objective_value for r in results]
        execution_times = [r.execution_time for r in results]

        avg_objective = np.mean(objective_values)
        std_objective = np.std(objective_values)
        avg_execution_time = np.mean(execution_times)
        std_execution_time = np.std(execution_times)

        # 验证统计结果
        assert avg_objective > 0
        assert std_objective >= 0
        assert avg_execution_time > 0
        assert std_execution_time >= 0

        # 检查结果的一致性
        assert std_objective < 0.1  # 目标函数值应该相对稳定
        assert avg_execution_time < 5  # 平均执行时间应该合理


class TestOptimizationWorkflowIntegration:
    """测试优化工作流集成"""

    def setup_method(self, method):
        """设置测试环境"""
        self.engine = OptimizationEngine()

    def test_complete_optimization_workflow(self):
        """测试完整优化工作流"""
        # 1. 数据准备
        np.random.seed(42)
        n_assets = 5
        n_periods = 252  # 一年的交易日

        # 生成模拟收益率数据
        returns = np.random.normal(0.0005, 0.02, (n_periods, n_assets))

        # 2. 设置优化目标和约束
        objectives = [OptimizationObjective.MAXIMIZE_SHARPE_RATIO]
        constraints = [
            OptimizationConstraint.NO_SHORT_SELLING,
            OptimizationConstraint.MAX_WEIGHT
        ]

        # 3. 执行优化
        initial_weights = np.ones(n_assets) / n_assets
        bounds = [(0, 0.3) for _ in range(n_assets)]  # 最大30%权重

        result = self.engine.optimize_portfolio(
            returns=returns,
            objective=OptimizationObjective.MAXIMIZE_SHARPE_RATIO,
            constraints=constraints,
            bounds=bounds,
            initial_weights=initial_weights
        )

        # 4. 结果分析
        assert result.success is True
        assert len(result.optimal_weights) == n_assets

        # 验证约束满足
        assert all(0 <= w <= 0.3 for w in result.optimal_weights)  # 权重边界
        assert abs(sum(result.optimal_weights) - 1.0) < 1e-6  # 权重和为1

        # 5. 性能评估
        portfolio_returns = returns @ result.optimal_weights
        sharpe_ratio = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252)

        assert sharpe_ratio > 0  # 夏普比率应该为正

    def test_optimization_parameter_sensitivity(self):
        """测试优化参数敏感性"""
        # 测试不同参数对结果的影响
        np.random.seed(42)
        n_assets = 4

        expected_returns = np.array([0.12, 0.08, 0.15, 0.10])
        cov_matrix = np.random.random((n_assets, n_assets))
        cov_matrix = cov_matrix @ cov_matrix.T

        # 测试不同的目标收益率
        target_returns = [0.08, 0.10, 0.12]

        results = []
        for target_return in target_returns:
            result = self.engine.mean_variance_optimize(
                expected_returns=expected_returns,
                cov_matrix=cov_matrix,
                target_return=target_return
            )

            if result.success:
                actual_return = np.dot(result.optimal_weights, expected_returns)
                portfolio_volatility = np.sqrt(result.optimal_weights.T @ cov_matrix @ result.optimal_weights)

                results.append({
                    'target_return': target_return,
                    'actual_return': actual_return,
                    'volatility': portfolio_volatility,
                    'weights': result.optimal_weights
                })

        # 验证参数敏感性
        assert len(results) == len(target_returns)

        # 检查实际收益率是否接近目标
        for result in results:
            assert abs(result['actual_return'] - result['target_return']) < 0.02

        # 检查波动率是否随预期收益率增加
        returns_and_vols = [(r['actual_return'], r['volatility']) for r in results]
        returns_and_vols.sort(key=lambda x: x[0])

        # 通常情况下，波动率会随着预期收益率的增加而增加
        assert returns_and_vols[-1][1] >= returns_and_vols[0][1]

    def test_optimization_algorithm_comparison(self):
        """测试优化算法比较"""
        # 使用同一个优化问题测试不同算法
        def objective(x):
            return (x[0] - 2)**2 + (x[1] - 3)**2 + (x[2] - 1)**2

        constraints = [
            {'type': 'ineq', 'fun': lambda x: x[0] + x[1] + x[2] - 1},  # 和 >= 1
            {'type': 'ineq', 'fun': lambda x: 2 - x[0] - x[1] - x[2]}   # 和 <= 2
        ]

        bounds = [(0, 2), (0, 2), (0, 2)]

        algorithms = [
            OptimizationAlgorithm.SLSQP,
            OptimizationAlgorithm.COBYLA,
            OptimizationAlgorithm.BFGS
        ]

        results = []
        for algorithm in algorithms:
            result = self.engine.optimize_with_algorithm(
                objective=objective,
                initial_guess=[0.5, 0.5, 0.5],
                bounds=bounds,
                constraints=constraints,
                algorithm=algorithm
            )

            if result.success:
                results.append({
                    'algorithm': algorithm.value,
                    'objective_value': result.fun,
                    'solution': result.x,
                    'execution_time': result.execution_time if hasattr(result, 'execution_time') else 0
                })

        # 验证算法比较
        assert len(results) > 0  # 至少有一个算法成功

        # 找到最优结果
        best_result = min(results, key=lambda x: x['objective_value'])

        # 验证最优解接近(2, 3, 1)
        expected_solution = np.array([2, 3, 1])
        solution_error = np.linalg.norm(best_result['solution'] - expected_solution)

        assert solution_error < 1.0  # 解应该相对准确
