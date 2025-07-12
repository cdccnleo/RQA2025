#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
回测参数优化模块
实现多种参数优化算法
"""

import itertools
import numpy as np
from typing import Dict, List, Any, Tuple
from src.backtest.engine import BacktestEngine
from src.utils.logger import get_logger
from concurrent.futures import ProcessPoolExecutor

logger = get_logger(__name__)

class ParameterOptimizer:
    def __init__(self, engine: BacktestEngine):
        """
        初始化参数优化器
        :param engine: 回测引擎实例
        """
        self.engine = engine
        self.results = []

    def grid_search(self,
                   strategy: Any,
                   param_grid: Dict[str, List[Any]],
                   start: str,
                   end: str,
                   n_jobs: int = 1) -> List[Dict[str, Any]]:
        """
        网格搜索参数优化
        :param strategy: 策略类
        :param param_grid: 参数网格 {参数名: [值列表]}
        :param start: 回测开始日期
        :param end: 回测结束日期
        :param n_jobs: 并行任务数
        :return: 优化结果列表
        """
        logger.info("Starting grid search optimization")

        # 生成参数组合
        param_names = sorted(param_grid)
        param_values = [param_grid[name] for name in param_names]
        param_combinations = list(itertools.product(*param_values))

        # 并行执行回测
        if n_jobs > 1:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = []
                for params in param_combinations:
                    param_dict = dict(zip(param_names, params))
                    futures.append(
                        executor.submit(
                            self._run_backtest,
                            strategy,
                            param_dict,
                            start,
                            end
                        )
                    )

                # 收集结果
                self.results = [f.result() for f in futures]
        else:
            # 单进程模式
            self.results = []
            for params in param_combinations:
                param_dict = dict(zip(param_names, params))
                result = self._run_backtest(strategy, param_dict, start, end)
                self.results.append(result)

        # 按绩效排序结果
        self.results.sort(key=lambda x: x['performance']['sharpe'], reverse=True)

        logger.info(f"Grid search completed, tested {len(param_combinations)} combinations")
        return self.results

    def _run_backtest(self,
                     strategy: Any,
                     params: Dict[str, Any],
                     start: str,
                     end: str) -> Dict[str, Any]:
        """
        执行单次回测
        :param strategy: 策略类
        :param params: 策略参数
        :param start: 开始日期
        :param end: 结束日期
        :return: 回测结果
        """
        # 克隆引擎以避免状态污染
        engine = self.engine.__class__(self.engine.config)

        # 添加策略
        engine.add_strategy(strategy, params)

        # 运行回测
        engine.run_backtest(start, end)

        # 获取绩效
        performance = engine.get_performance()

        return {
            'params': params,
            'performance': performance
        }

    def genetic_optimize(self,
                        strategy: Any,
                        param_ranges: Dict[str, Tuple[Any, Any]],
                        start: str,
                        end: str,
                        population_size: int = 20,
                        generations: int = 10,
                        elite_size: int = 4,
                        mutation_rate: float = 0.1) -> List[Dict[str, Any]]:
        """
        遗传算法参数优化
        :param strategy: 策略类
        :param param_ranges: 参数范围 {参数名: (最小值, 最大值)}
        :param start: 回测开始日期
        :param end: 回测结束日期
        :param population_size: 种群大小
        :param generations: 迭代代数
        :param elite_size: 精英保留数量
        :param mutation_rate: 变异率
        :return: 优化结果列表
        """
        logger.info("Starting genetic algorithm optimization")

        # 初始化种群
        population = self._init_population(param_ranges, population_size)

        # 进化循环
        for gen in range(generations):
            # 评估种群
            ranked_pop = []
            for individual in population:
                params = dict(zip(param_ranges.keys(), individual))
                result = self._run_backtest(strategy, params, start, end)
                ranked_pop.append((result['performance']['sharpe'], individual))

            # 按绩效排序
            ranked_pop.sort(reverse=True)

            # 保留精英
            elite = [ind for (_, ind) in ranked_pop[:elite_size]]

            # 生成新一代
            children = []
            while len(children) < population_size - elite_size:
                # 选择
                parent1 = self._select(ranked_pop)
                parent2 = self._select(ranked_pop)

                # 交叉
                child = self._crossover(parent1, parent2)

                # 变异
                child = self._mutate(child, param_ranges, mutation_rate)

                children.append(child)

            # 新一代种群 = 精英 + 子代
            population = elite + children

            logger.info(f"Generation {gen+1} completed, best sharpe: {ranked_pop[0][0]:.2f}")

        # 最终评估
        self.results = []
        for individual in population:
            params = dict(zip(param_ranges.keys(), individual))
            result = self._run_backtest(strategy, params, start, end)
            self.results.append(result)

        # 按绩效排序
        self.results.sort(key=lambda x: x['performance']['sharpe'], reverse=True)

        logger.info("Genetic optimization completed")
        return self.results

    def _init_population(self,
                       param_ranges: Dict[str, Tuple[Any, Any]],
                       size: int) -> List[List[Any]]:
        """
        初始化种群
        :param param_ranges: 参数范围
        :param size: 种群大小
        :return: 初始种群
        """
        population = []
        param_names = sorted(param_ranges.keys())

        for _ in range(size):
            individual = []
            for name in param_names:
                min_val, max_val = param_ranges[name]
                if isinstance(min_val, int) and isinstance(max_val, int):
                    # 整数参数
                    val = np.random.randint(min_val, max_val+1)
                else:
                    # 浮点数参数
                    val = np.random.uniform(min_val, max_val)
                individual.append(val)

            population.append(individual)

        return population

    def _select(self, ranked_pop: List[Tuple[float, List[Any]]]) -> List[Any]:
        """
        轮盘赌选择
        :param ranked_pop: 已排序的种群
        :return: 选择的个体
        """
        # 线性排名选择
        ranks = np.array([i+1 for i in range(len(ranked_pop))])
        probs = ranks / ranks.sum()
        idx = np.random.choice(len(ranked_pop), p=probs)
        return ranked_pop[idx][1]

    def _crossover(self, parent1: List[Any], parent2: List[Any]) -> List[Any]:
        """
        单点交叉
        :param parent1: 父代1
        :param parent2: 父代2
        :return: 子代
        """
        crossover_point = np.random.randint(1, len(parent1))
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child

    def _mutate(self,
               individual: List[Any],
               param_ranges: Dict[str, Tuple[Any, Any]],
               rate: float) -> List[Any]:
        """
        变异操作
        :param individual: 个体
        :param param_ranges: 参数范围
        :param rate: 变异率
        :return: 变异后的个体
        """
        param_names = sorted(param_ranges.keys())
        for i in range(len(individual)):
            if np.random.random() < rate:
                # 执行变异
                name = param_names[i]
                min_val, max_val = param_ranges[name]
                if isinstance(min_val, int) and isinstance(max_val, int):
                    # 整数参数
                    individual[i] = np.random.randint(min_val, max_val+1)
                else:
                    # 浮点数参数
                    individual[i] = np.random.uniform(min_val, max_val)

        return individual

    def bayesian_optimize(self,
                        strategy: Any,
                        param_bounds: Dict[str, Tuple[float, float]],
                        start: str,
                        end: str,
                        n_iter: int = 30,
                        init_points: int = 5) -> List[Dict[str, Any]]:
        """
        贝叶斯优化
        :param strategy: 策略类
        :param param_bounds: 参数边界 {参数名: (最小值, 最大值)}
        :param start: 回测开始日期
        :param end: 回测结束日期
        :param n_iter: 迭代次数
        :param init_points: 初始随机采样点
        :return: 优化结果列表
        """
        try:
            from bayes_opt import BayesianOptimization
        except ImportError:
            logger.error("BayesianOptimization package not installed")
            return []

        logger.info("Starting Bayesian optimization")

        # 定义目标函数
        def objective(**params):
            result = self._run_backtest(strategy, params, start, end)
            return result['performance']['sharpe']

        # 创建优化器
        optimizer = BayesianOptimization(
            f=objective,
            pbounds=param_bounds,
            random_state=42
        )

        # 执行优化
        optimizer.maximize(
            init_points=init_points,
            n_iter=n_iter
        )

        # 整理结果
        self.results = []
        for i, res in enumerate(optimizer.res):
            self.results.append({
                'params': res['params'],
                'performance': {'sharpe': res['target']}
            })

        # 按绩效排序
        self.results.sort(key=lambda x: x['performance']['sharpe'], reverse=True)

        logger.info("Bayesian optimization completed")
        return self.results
