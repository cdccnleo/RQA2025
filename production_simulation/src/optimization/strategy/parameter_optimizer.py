#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
"""
参数优化器实现
Parameter Optimizer Implementation

提供多种参数优化算法的实现，包括网格搜索、随机搜索、贝叶斯优化等。
"""

import itertools
import secrets
from typing import Dict, List, Any, Callable
from datetime import datetime
import logging
from ..interfaces.optimization_interfaces import (
    IParameterOptimizer, OptimizationAlgorithm, OptimizationResult
)

logger = logging.getLogger(__name__)


class ParameterOptimizer(IParameterOptimizer):

    """
    参数优化器
    Parameter Optimizer

    实现多种参数优化算法。
    """

    def __init__(self):
        """初始化参数优化器"""
        self.random_seed = 42
        np.random.seed(self.random_seed)

        logger.info("参数优化器初始化完成")

    async def optimize(self, strategy_id: str, parameter_ranges: Dict[str, List[Any]],
                       target_function: Callable, algorithm: OptimizationAlgorithm = OptimizationAlgorithm.GRID_SEARCH,
                       **kwargs) -> OptimizationResult:
        """
        执行参数优化

        Args:
            strategy_id: 策略ID
            parameter_ranges: 参数范围字典
            target_function: 目标函数
            algorithm: 优化算法
            **kwargs: 其他参数

        Returns:
            OptimizationResult: 优化结果
        """
        start_time = datetime.now()

        try:
            if algorithm == OptimizationAlgorithm.GRID_SEARCH:
                result = await self.grid_search(parameter_ranges, target_function)

            elif algorithm == OptimizationAlgorithm.RANDOM_SEARCH:
                n_iterations = kwargs.get('n_iterations', 100)
                result = await self.random_search(parameter_ranges, target_function, n_iterations)

            elif algorithm == OptimizationAlgorithm.BAYESIAN_OPTIMIZATION:
                n_iterations = kwargs.get('n_iterations', 50)
                result = await self.bayesian_optimization(parameter_ranges, target_function, n_iterations)

            elif algorithm == OptimizationAlgorithm.GENETIC_ALGORITHM:
                result = await self.genetic_algorithm(parameter_ranges, target_function, **kwargs)

            elif algorithm == OptimizationAlgorithm.PARTICLE_SWARM:
                result = await self.particle_swarm_optimization(parameter_ranges, target_function, **kwargs)

            else:
                raise ValueError(f"不支持的优化算法: {algorithm}")

            result.execution_time = (datetime.now() - start_time).total_seconds()
            result.status = "success"

            return result

        except Exception as e:
            logger.error(f"参数优化异常: {e}")
            return OptimizationResult(
                optimization_id="",
                strategy_id=strategy_id,
                best_parameters={},
                best_score=float('-inf'),
                all_results=[],
                convergence_history=[],
                execution_time=(datetime.now() - start_time).total_seconds(),
                status="failed",
                error_message=str(e),
                timestamp=datetime.now()
            )

    async def grid_search(self, parameter_ranges: Dict[str, List[Any]],
                          target_function: Callable) -> OptimizationResult:
        """
        网格搜索优化

        Args:
            parameter_ranges: 参数范围字典
            target_function: 目标函数

        Returns:
            OptimizationResult: 优化结果
        """
        logger.info("开始网格搜索优化")

        # 生成所有参数组合
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        all_combinations = list(itertools.product(*param_values))

        logger.info(f"参数组合总数: {len(all_combinations)}")

        # 评估所有参数组合
        all_results = []
        convergence_history = []

        for i, combination in enumerate(all_combinations):
            try:
                # 构建参数字典
                parameters = dict(zip(param_names, combination))

                # 评估参数组合
                score = await target_function(parameters)

                result = {
                    'parameters': parameters,
                    'score': score,
                    'iteration': i + 1
                }
                all_results.append(result)

                # 更新收敛历史
                if convergence_history:
                    convergence_history.append(max(convergence_history[-1], score))
                else:
                    convergence_history.append(score)

                logger.debug(f"评估参数组合 {i + 1}/{len(all_combinations)}: 得分={score}")

            except Exception as e:
                logger.error(f"参数组合评估失败: {combination}, 错误: {e}")
                continue

        # 找到最佳结果
        if all_results:
            best_result = max(all_results, key=lambda x: x['score'])
            best_parameters = best_result['parameters']
            best_score = best_result['score']
        else:
            best_parameters = {}
            best_score = float('-inf')

        logger.info(f"网格搜索完成，最佳得分: {best_score}")

        return OptimizationResult(
            optimization_id="",
            strategy_id="",
            best_parameters=best_parameters,
            best_score=best_score,
            all_results=all_results,
            convergence_history=convergence_history,
            execution_time=0.0,  # 将在上级函数中设置
            status="success",
            timestamp=datetime.now()
        )

    async def random_search(self, parameter_ranges: Dict[str, List[Any]],
                            target_function: Callable, n_iterations: int = 100) -> OptimizationResult:
        """
        随机搜索优化

        Args:
            parameter_ranges: 参数范围字典
            target_function: 目标函数
            n_iterations: 迭代次数

        Returns:
            OptimizationResult: 优化结果
        """
        logger.info(f"开始随机搜索优化，迭代次数: {n_iterations}")

        all_results = []
        convergence_history = []

        for i in range(n_iterations):
            try:
                # 随机选择参数组合
                parameters = {}
                for param_name, param_values in parameter_ranges.items():
                    parameters[param_name] = secrets.choice(param_values)

                # 评估参数组合
                score = await target_function(parameters)

                result = {
                    'parameters': parameters,
                    'score': score,
                    'iteration': i + 1
                }
                all_results.append(result)

                # 更新收敛历史
                if convergence_history:
                    convergence_history.append(max(convergence_history[-1], score))
                else:
                    convergence_history.append(score)

                logger.debug(f"随机搜索迭代 {i + 1}/{n_iterations}: 得分={score}")

            except Exception as e:
                logger.error(f"随机搜索迭代失败 {i + 1}: {e}")
                continue

        # 找到最佳结果
        if all_results:
            best_result = max(all_results, key=lambda x: x['score'])
            best_parameters = best_result['parameters']
            best_score = best_result['score']
        else:
            best_parameters = {}
            best_score = float('-inf')

        logger.info(f"随机搜索完成，最佳得分: {best_score}")

        return OptimizationResult(
            optimization_id="",
            strategy_id="",
            best_parameters=best_parameters,
            best_score=best_score,
            all_results=all_results,
            convergence_history=convergence_history,
            execution_time=0.0,
            status="success",
            timestamp=datetime.now()
        )

    async def bayesian_optimization(self, parameter_ranges: Dict[str, List[Any]],
                                    target_function: Callable, n_iterations: int = 50) -> OptimizationResult:
        """
        贝叶斯优化

        Args:
            parameter_ranges: 参数范围字典
            target_function: 目标函数
            n_iterations: 迭代次数

        Returns:
            OptimizationResult: 优化结果
        """
        logger.info(f"开始贝叶斯优化，迭代次数: {n_iterations}")

        # 这里简化实现贝叶斯优化
        # 实际应该使用如scikit - optimize或bayesian - optimization库

        all_results = []
        convergence_history = []

        # 初始随机采样
        n_initial = min(10, n_iterations // 5)
        for i in range(n_initial):
            parameters = {}
            for param_name, param_values in parameter_ranges.items():
                parameters[param_name] = secrets.choice(param_values)

            score = await target_function(parameters)

            result = {
                'parameters': parameters,
                'score': score,
                'iteration': i + 1
            }
            all_results.append(result)
            convergence_history.append(score)

        # 贝叶斯优化迭代
        for i in range(n_initial, n_iterations):
            try:
                # 简化的贝叶斯优化策略：选择表现最好的参数附近
                best_result = max(all_results, key=lambda x: x['score'])
                best_params = best_result['parameters']

                # 在最佳参数附近采样
                parameters = {}
                for param_name, param_values in parameter_ranges.items():
                    if secrets.random() < 0.8:  # 80 % 概率选择最佳参数附近
                        current_value = best_params[param_name]
                        # 找到当前值在参数列表中的位置
                        try:
                            current_index = param_values.index(current_value)
                            # 在附近选择
                            nearby_indices = [
                                max(0, current_index - 1),
                                current_index,
                                min(len(param_values) - 1, current_index + 1)
                            ]
                            chosen_index = secrets.choice(nearby_indices)
                            parameters[param_name] = param_values[chosen_index]
                        except (ValueError, IndexError):
                            parameters[param_name] = secrets.choice(param_values)
                    else:
                        # 随机选择
                        parameters[param_name] = secrets.choice(param_values)

                score = await target_function(parameters)

                result = {
                    'parameters': parameters,
                    'score': score,
                    'iteration': i + 1
                }
                all_results.append(result)

                # 更新收敛历史
                convergence_history.append(max(convergence_history[-1], score))

                logger.debug(f"贝叶斯优化迭代 {i + 1}/{n_iterations}: 得分={score}")

            except Exception as e:
                logger.error(f"贝叶斯优化迭代失败 {i + 1}: {e}")
                continue

        # 找到最佳结果
        if all_results:
            best_result = max(all_results, key=lambda x: x['score'])
            best_parameters = best_result['parameters']
            best_score = best_result['score']
        else:
            best_parameters = {}
            best_score = float('-inf')

        logger.info(f"贝叶斯优化完成，最佳得分: {best_score}")

        return OptimizationResult(
            optimization_id="",
            strategy_id="",
            best_parameters=best_parameters,
            best_score=best_score,
            all_results=all_results,
            convergence_history=convergence_history,
            execution_time=0.0,
            status="success",
            timestamp=datetime.now()
        )

    async def genetic_algorithm(self, parameter_ranges: Dict[str, List[Any]],
                                target_function: Callable, **kwargs) -> OptimizationResult:
        """
        遗传算法优化

        Args:
            parameter_ranges: 参数范围字典
            target_function: 目标函数
            **kwargs: 其他参数

        Returns:
            OptimizationResult: 优化结果
        """
        logger.info("开始遗传算法优化")

        # 遗传算法参数
        population_size = kwargs.get('population_size', 20)
        n_generations = kwargs.get('n_generations', 10)
        mutation_rate = kwargs.get('mutation_rate', 0.1)
        crossover_rate = kwargs.get('crossover_rate', 0.8)

        # 初始化种群
        population = self._initialize_population(parameter_ranges, population_size)

        all_results = []
        convergence_history = []

        for generation in range(n_generations):
            try:
                # 评估种群
                fitness_scores = []
                for individual in population:
                    score = await target_function(individual)
                    fitness_scores.append(score)

                    result = {
                        'parameters': individual,
                        'score': score,
                        'generation': generation + 1
                    }
                    all_results.append(result)

                # 更新收敛历史
                best_score = max(fitness_scores) if fitness_scores else float('-inf')
                convergence_history.append(best_score)

                # 选择、交叉、变异
                population = self._evolve_population(
                    population, fitness_scores, crossover_rate, mutation_rate,
                    parameter_ranges
                )

                logger.debug(f"遗传算法第 {generation + 1} 代完成，最佳得分: {best_score}")

            except Exception as e:
                logger.error(f"遗传算法第 {generation + 1} 代失败: {e}")
                continue

        # 找到最佳结果
        if all_results:
            best_result = max(all_results, key=lambda x: x['score'])
            best_parameters = best_result['parameters']
            best_score = best_result['score']
        else:
            best_parameters = {}
            best_score = float('-inf')

        logger.info(f"遗传算法完成，最佳得分: {best_score}")

        return OptimizationResult(
            optimization_id="",
            strategy_id="",
            best_parameters=best_parameters,
            best_score=best_score,
            all_results=all_results,
            convergence_history=convergence_history,
            execution_time=0.0,
            status="success",
            timestamp=datetime.now()
        )

    async def particle_swarm_optimization(self, parameter_ranges: Dict[str, List[Any]],
                                          target_function: Callable, **kwargs) -> OptimizationResult:
        """
        粒子群优化

        Args:
            parameter_ranges: 参数范围字典
            target_function: 目标函数
            **kwargs: 其他参数

        Returns:
            OptimizationResult: 优化结果
        """
        logger.info("开始粒子群优化")

        # 粒子群算法参数 - 简化的实现
        # n_particles = kwargs.get('n_particles', 20)
        n_iterations = kwargs.get('n_iterations', 10)
        # inertia_weight = kwargs.get('inertia_weight', 0.7)
        # cognitive_weight = kwargs.get('cognitive_weight', 1.4)
        # social_weight = kwargs.get('social_weight', 1.4)

        # 这里简化实现粒子群优化
        # 实际应该实现完整的PSO算法

        all_results = []
        convergence_history = []

        # 简化的实现：随机搜索
        for i in range(n_iterations):
            parameters = {}
            for param_name, param_values in parameter_ranges.items():
                parameters[param_name] = secrets.choice(param_values)

            score = await target_function(parameters)

            result = {
                'parameters': parameters,
                'score': score,
                'iteration': i + 1
            }
            all_results.append(result)
            convergence_history.append(score)

        # 找到最佳结果
        if all_results:
            best_result = max(all_results, key=lambda x: x['score'])
            best_parameters = best_result['parameters']
            best_score = best_result['score']
        else:
            best_parameters = {}
            best_score = float('-inf')

        logger.info(f"粒子群优化完成，最佳得分: {best_score}")

        return OptimizationResult(
            optimization_id="",
            strategy_id="",
            best_parameters=best_parameters,
            best_score=best_score,
            all_results=all_results,
            convergence_history=convergence_history,
            execution_time=0.0,
            status="success",
            timestamp=datetime.now()
        )

    def _initialize_population(self, parameter_ranges: Dict[str, List[Any]],


                               population_size: int) -> List[Dict[str, Any]]:
        """
        初始化种群

        Args:
            parameter_ranges: 参数范围字典
            population_size: 种群大小

        Returns:
            List[Dict[str, Any]]: 初始种群
        """
        population = []

        for _ in range(population_size):
            individual = {}
            for param_name, param_values in parameter_ranges.items():
                individual[param_name] = secrets.choice(param_values)
            population.append(individual)

        return population

    def _evolve_population(self, population: List[Dict[str, Any]],


                           fitness_scores: List[float], crossover_rate: float,
                           mutation_rate: float, parameter_ranges: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        种群进化

        Args:
            population: 当前种群
            fitness_scores: 适应度分数
            crossover_rate: 交叉率
            mutation_rate: 变异率
            parameter_ranges: 参数范围字典

        Returns:
            List[Dict[str, Any]]: 新种群
        """
        new_population = []

        # 精英保留
        elite_indices = sorted(range(len(fitness_scores)),
                               key=lambda i: fitness_scores[i], reverse=True)[:2]

        for idx in elite_indices:
            new_population.append(population[idx].copy())

        # 生成剩余个体
        while len(new_population) < len(population):
            # 选择父代
            parent1 = self._tournament_selection(population, fitness_scores)
            parent2 = self._tournament_selection(population, fitness_scores)

            # 交叉
            if secrets.random() < crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = parent1.copy()

            # 变异
            if secrets.random() < mutation_rate:
                child = self._mutate(child, parameter_ranges)

            new_population.append(child)

        return new_population

    def _tournament_selection(self, population: List[Dict[str, Any]],


                              fitness_scores: List[float]) -> Dict[str, Any]:
        """
        锦标赛选择

        Args:
            population: 种群
            fitness_scores: 适应度分数

        Returns:
            Dict[str, Any]: 选择的个体
        """
        tournament_size = 3
        tournament_indices = secrets.sample(range(len(population)), tournament_size)
        best_index = max(tournament_indices, key=lambda i: fitness_scores[i])
        return population[best_index].copy()

    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """
        交叉操作

        Args:
            parent1: 父代1
            parent2: 父代2

        Returns:
            Dict[str, Any]: 子代
        """
        child = {}
        for key in parent1.keys():
            child[key] = parent1[key] if secrets.random() < 0.5 else parent2[key]
        return child

    def _mutate(self, individual: Dict[str, Any],


                parameter_ranges: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        变异操作

        Args:
            individual: 个体
            parameter_ranges: 参数范围字典

        Returns:
            Dict[str, Any]: 变异后的个体
        """
        mutated = individual.copy()
        param_names = list(parameter_ranges.keys())

        # 随机选择一个参数进行变异
        param_to_mutate = secrets.choice(param_names)
        param_values = parameter_ranges[param_to_mutate]

        # 随机选择一个新值
        current_value = mutated[param_to_mutate]
        new_values = [v for v in param_values if v != current_value]

        if new_values:
            mutated[param_to_mutate] = secrets.choice(new_values)

        return mutated


# 导出类
__all__ = [
    'ParameterOptimizer'
]
