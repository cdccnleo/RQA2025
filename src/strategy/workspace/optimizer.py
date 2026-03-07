import logging
"""
策略优化器

from src.engine.logging.unified_logger import get_unified_logger
提供多种优化算法用于量化策略参数调优，包括：
- 遗传算法优化
- 贝叶斯优化
- 网格搜索
- 多目标优化
- 参数敏感性分析
"""

import numpy as np
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import secrets
from .visual_editor import VisualStrategyEditor

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):

    """优化方法"""
    GRID_SEARCH = "grid_search"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"
    RANDOM = "random"
    PARTICLE_SWARM = "particle_swarm"
    SIMULATED_ANNEALING = "simulated_annealing"
    MULTI_OBJECTIVE = "multi_objective"
    ENSEMBLE = "ensemble"


@dataclass
class OptimizationConfig:

    """优化配置"""
    method: OptimizationMethod
    max_iterations: int
    population_size: int = 50
    elite_size: int = 5
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    bounds: Dict = None
    # 粒子群优化参数
    particle_count: int = 30
    cognitive_weight: float = 2.0
    social_weight: float = 2.0
    inertia_weight: float = 0.7
    # 模拟退火参数
    initial_temperature: float = 100.0
    cooling_rate: float = 0.95
    # 多目标优化参数
    objectives: List[str] = None  # ["sharpe_ratio", "max_drawdown", "total_return"]
    weights: List[float] = None   # [0.4, 0.3, 0.3]
    # 集成优化参数
    ensemble_methods: List[OptimizationMethod] = None


@dataclass
class OptimizationResult:

    """优化结果"""
    best_params: Dict
    best_score: float
    all_results: List[Tuple[Dict, float]]
    convergence_history: List[float]


class StrategyOptimizer:

    """策略优化器"""

    def __init__(self):

        self.optimization_history = []

    def optimize(self, strategy: VisualStrategyEditor,
                 objective_function: Callable,
                 config: OptimizationConfig) -> OptimizationResult:
        """优化策略参数

        Args:
            strategy: 策略编辑器
            objective_function: 目标函数
            config: 优化配置

        Returns:
            OptimizationResult: 优化结果
        """
        try:
            if config.method == OptimizationMethod.GRID_SEARCH:
                return self._grid_search(strategy, objective_function, config)
            elif config.method == OptimizationMethod.BAYESIAN:
                return self._bayesian_optimization(strategy, objective_function, config)
            elif config.method == OptimizationMethod.GENETIC:
                return self._genetic_optimization(strategy, objective_function, config)
            elif config.method == OptimizationMethod.RANDOM:
                return self._random_search(strategy, objective_function, config)
            elif config.method == OptimizationMethod.PARTICLE_SWARM:
                return self._particle_swarm_optimization(strategy, objective_function, config)
            elif config.method == OptimizationMethod.SIMULATED_ANNEALING:
                return self._simulated_annealing(strategy, objective_function, config)
            elif config.method == OptimizationMethod.MULTI_OBJECTIVE:
                return self._multi_objective_optimization(strategy, objective_function, config)
            elif config.method == OptimizationMethod.ENSEMBLE:
                return self._ensemble_optimization(strategy, objective_function, config)
            else:
                raise ValueError(f"不支持的优化方法: {config.method}")

        except Exception as e:
            logger.error(f"策略优化失败: {e}")
            raise

    def _grid_search(self, strategy: VisualStrategyEditor,
                     objective_function: Callable,
                     config: OptimizationConfig) -> OptimizationResult:
        """网格搜索优化"""
        logger.info("开始网格搜索优化")

        # 获取参数空间
        param_space = self._get_parameter_space(strategy)
        if not param_space:
            raise ValueError("无法获取参数空间")

        best_params = None
        best_score = float('-inf')
        all_results = []

        # 生成参数组合
        param_combinations = self._generate_param_combinations(param_space)

        for i, params in enumerate(param_combinations):
            try:
                # 更新策略参数
                self._update_strategy_params(strategy, params)

                # 评估目标函数
                score = objective_function(strategy)

                all_results.append((params.copy(), score))

                if score > best_score:
                    best_score = score
                    best_params = params.copy()

                if i % 10 == 0:
                    logger.info(f"网格搜索进度: {i + 1}/{len(param_combinations)}")

            except Exception as e:
                logger.warning(f"参数评估失败: {e}")
                continue

        return OptimizationResult(
            best_params=best_params or {},
            best_score=best_score,
            all_results=all_results,
            convergence_history=[result[1] for result in all_results]
        )

    def _bayesian_optimization(self, strategy: VisualStrategyEditor,
                               objective_function: Callable,
                               config: OptimizationConfig) -> OptimizationResult:
        """贝叶斯优化"""
        logger.info("开始贝叶斯优化")

        # 简化的贝叶斯优化实现
        # 实际项目中可以使用scikit - optimize或optuna库

        param_space = self._get_parameter_space(strategy)
        if not param_space:
            raise ValueError("无法获取参数空间")

        best_params = None
        best_score = float('-inf')
        all_results = []
        convergence_history = []

        # 随机初始化
        for i in range(min(10, config.max_iterations)):
            params = self._random_sample_params(param_space)
            try:
                self._update_strategy_params(strategy, params)
                score = objective_function(strategy)

                all_results.append((params.copy(), score))
                convergence_history.append(score)

                if score > best_score:
                    best_score = score
                    best_params = params.copy()

            except Exception as e:
                logger.warning(f"贝叶斯优化评估失败: {e}")
                continue

        # 基于历史结果进行简单的高斯过程优化
        for i in range(10, config.max_iterations):
            # 这里应该实现真正的贝叶斯优化
            # 暂时使用随机搜索
            params = self._random_sample_params(param_space)
            try:
                self._update_strategy_params(strategy, params)
                score = objective_function(strategy)

                all_results.append((params.copy(), score))
                convergence_history.append(score)

                if score > best_score:
                    best_score = score
                    best_params = params.copy()

            except Exception as e:
                logger.warning(f"贝叶斯优化评估失败: {e}")
                continue

        return OptimizationResult(
            best_params=best_params or {},
            best_score=best_score,
            all_results=all_results,
            convergence_history=convergence_history
        )

    def _genetic_optimization(self, strategy: VisualStrategyEditor,

                              objective_function: Callable,
                              config: OptimizationConfig) -> OptimizationResult:
        """遗传算法优化"""
        logger.info("开始遗传算法优化")

        param_space = self._get_parameter_space(strategy)
        if not param_space:
            raise ValueError("无法获取参数空间")

        # 初始化种群
        population = self._initialize_population(param_space, config.population_size)
        best_params = None
        best_score = float('-inf')
        all_results = []
        convergence_history = []

        for generation in range(config.max_iterations):
            # 评估适应度
            fitness_scores = []
            for individual in population:
                try:
                    self._update_strategy_params(strategy, individual)
                    score = objective_function(strategy)
                    fitness_scores.append(score)

                    if score > best_score:
                        best_score = score
                        best_params = individual.copy()

                except Exception as e:
                    logger.warning(f"遗传算法评估失败: {e}")
                    fitness_scores.append(float('-inf'))

            all_results.extend(list(zip(population, fitness_scores)))
            convergence_history.append(max(fitness_scores))

            # 选择
            selected = self._selection(population, fitness_scores, config.elite_size)

            # 交叉
            offspring = self._crossover(selected, config.crossover_rate)

            # 变异
            offspring = self._mutation(offspring, param_space, config.mutation_rate)

            # 更新种群
            population = offspring

        if generation % 10 == 0:
            logger.info(f"遗传算法第{generation}代，最佳适应度: {best_score}")

        return OptimizationResult(
            best_params=best_params or {},
            best_score=best_score,
            all_results=all_results,
            convergence_history=convergence_history
        )

    def _random_search(self, strategy: VisualStrategyEditor,


                       objective_function: Callable,
                       config: OptimizationConfig) -> OptimizationResult:
        """随机搜索优化"""
        logger.info("开始随机搜索优化")

        param_space = self._get_parameter_space(strategy)
        if not param_space:
            raise ValueError("无法获取参数空间")

        best_params = None
        best_score = float('-inf')
        all_results = []
        convergence_history = []

        for i in range(config.max_iterations):
            params = self._random_sample_params(param_space)
            try:
                self._update_strategy_params(strategy, params)
                score = objective_function(strategy)

                all_results.append((params.copy(), score))
                convergence_history.append(score)

                if score > best_score:
                    best_score = score
                    best_params = params.copy()

            except Exception as e:
                logger.warning(f"随机搜索评估失败: {e}")
                continue

        return OptimizationResult(
            best_params=best_params or {},
            best_score=best_score,
            all_results=all_results,
            convergence_history=convergence_history
        )

    def _particle_swarm_optimization(self, strategy: VisualStrategyEditor,

                                     objective_function: Callable,
                                     config: OptimizationConfig) -> OptimizationResult:
        """粒子群优化"""
        logger.info("开始粒子群优化")

        param_space = self._get_parameter_space(strategy)
        if not param_space:
            raise ValueError("无法获取参数空间")

        # 初始化粒子群
        particles = []
        velocities = []
        personal_best = []
        personal_best_scores = []
        global_best = None
        global_best_score = float('-inf')

        param_names = list(param_space.keys())

        # 初始化粒子
        for _ in range(config.particle_count):
            particle = self._random_sample_params(param_space)
            particles.append(particle)
            personal_best.append(particle.copy())
            personal_best_scores.append(float('-inf'))

            # 初始化速度
            velocity = {}
            for name in param_names:
                param_range = param_space[name]["range"]
                velocity[name] = secrets.uniform(-(param_range[1] - param_range[0]) * 0.1,
                                                 (param_range[1] - param_range[0]) * 0.1)
            velocities.append(velocity)

        all_results = []
        convergence_history = []

        for iteration in range(config.max_iterations):
            # 评估所有粒子
            for i, particle in enumerate(particles):
                try:
                    self._update_strategy_params(strategy, particle)
                    score = objective_function(strategy)

                    all_results.append((particle.copy(), score))

                    # 更新个体最优
                    if score > personal_best_scores[i]:
                        personal_best_scores[i] = score
                        personal_best[i] = particle.copy()

                    # 更新全局最优
                    if score > global_best_score:
                        global_best_score = score
                        global_best = particle.copy()

                except Exception as e:
                    logger.warning(f"粒子群优化评估失败: {e}")
                    continue

            convergence_history.append(global_best_score)

            # 更新粒子位置和速度
        for i in range(config.particle_count):
            for param_name in param_names:
                # 更新速度
                cognitive_component = config.cognitive_weight * secrets.random() * \
                    (personal_best[i][param_name] - particles[i][param_name])
                social_component = config.social_weight * secrets.random() * \
                    (global_best[param_name] - particles[i][param_name])

                velocities[i][param_name] = config.inertia_weight * velocities[i][param_name] + \
                    cognitive_component + social_component

                # 更新位置
                particles[i][param_name] += velocities[i][param_name]

                # 边界处理
                param_range = param_space[param_name]["range"]
                particles[i][param_name] = max(param_range[0], min(
                    param_range[1], particles[i][param_name]))

        if iteration % 10 == 0:
            logger.info(f"粒子群优化第{iteration}代，最佳适应度: {global_best_score}")

        return OptimizationResult(
            best_params=global_best or {},
            best_score=global_best_score,
            all_results=all_results,
            convergence_history=convergence_history
        )

    def _simulated_annealing(self, strategy: VisualStrategyEditor,


                             objective_function: Callable,
                             config: OptimizationConfig) -> OptimizationResult:
        """模拟退火优化"""
        logger.info("开始模拟退火优化")

        param_space = self._get_parameter_space(strategy)
        if not param_space:
            raise ValueError("无法获取参数空间")

        # 初始化
        current_params = self._random_sample_params(param_space)
        current_score = float('-inf')

        try:
            self._update_strategy_params(strategy, current_params)
            current_score = objective_function(strategy)
        except Exception as e:
            logger.warning(f"模拟退火初始评估失败: {e}")

        best_params = current_params.copy()
        best_score = current_score

        temperature = config.initial_temperature
        all_results = []
        convergence_history = []

        for iteration in range(config.max_iterations):
            # 生成新解
            new_params = current_params.copy()
            for param_name in new_params:
                if secrets.random() < 0.3:  # 30 % 概率改变每个参数
                    param_range = param_space[param_name]["range"]
                    param_type = param_space[param_name]["type"]

                    if param_type == "int":
                        new_params[param_name] = secrets.randint(param_range[0], param_range[1])
                    else:
                        # 在当前值附近随机扰动
                        current_val = new_params[param_name]
                        perturbation = secrets.uniform(-0.1, 0.1) * \
                            (param_range[1] - param_range[0])
                        new_params[param_name] = current_val + perturbation
                        new_params[param_name] = max(param_range[0], min(
                            param_range[1], new_params[param_name]))

            # 评估新解
            try:
                self._update_strategy_params(strategy, new_params)
                new_score = objective_function(strategy)

                all_results.append((new_params.copy(), new_score))

                # 计算接受概率
                delta_e = new_score - current_score
                if delta_e > 0 or secrets.random() < np.exp(delta_e / temperature):
                    current_params = new_params.copy()
                    current_score = new_score

                    if new_score > best_score:
                        best_params = new_params.copy()
                        best_score = new_score

            except Exception as e:
                logger.warning(f"模拟退火评估失败: {e}")
                continue

            convergence_history.append(best_score)

            # 降温
            temperature *= config.cooling_rate

        if iteration % 10 == 0:
            logger.info(f"模拟退火第{iteration}代，温度: {temperature:.2f}，最佳适应度: {best_score}")

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            convergence_history=convergence_history
        )

    def _multi_objective_optimization(self, strategy: VisualStrategyEditor,


                                      objective_function: Callable,
                                      config: OptimizationConfig) -> OptimizationResult:
        """多目标优化"""
        logger.info("开始多目标优化")

        if not config.objectives or not config.weights:
            raise ValueError("多目标优化需要指定目标和权重")

        param_space = self._get_parameter_space(strategy)
        if not param_space:
            raise ValueError("无法获取参数空间")

        # 初始化种群
        population = self._initialize_population(param_space, config.population_size)
        best_params = None
        best_score = float('-inf')
        all_results = []
        convergence_history = []

        for generation in range(config.max_iterations):
            # 评估适应度
            fitness_scores = []
            for individual in population:
                try:
                    self._update_strategy_params(strategy, individual)

                    # 计算多个目标
                    objectives_scores = []
                    for obj_name in config.objectives:
                        if obj_name == "sharpe_ratio":
                            # 这里需要从策略模拟结果中获取夏普比率
                            obj_score = secrets.uniform(0.5, 2.0)  # 模拟值
                        elif obj_name == "max_drawdown":
                            obj_score = secrets.uniform(0.05, 0.3)  # 模拟值
                        elif obj_name == "total_return":
                            obj_score = secrets.uniform(0.1, 0.5)  # 模拟值
                        else:
                            obj_score = secrets.uniform(0, 1)

                        objectives_scores.append(obj_score)

                    # 加权组合
                    combined_score = sum(w * s for w, s in zip(config.weights, objectives_scores))
                    fitness_scores.append(combined_score)

                    if combined_score > best_score:
                        best_score = combined_score
                        best_params = individual.copy()

                except Exception as e:
                    logger.warning(f"多目标优化评估失败: {e}")
                    fitness_scores.append(float('-inf'))

            all_results.extend(list(zip(population, fitness_scores)))
            convergence_history.append(max(fitness_scores))

            # 选择、交叉、变异
            selected = self._selection(population, fitness_scores, config.elite_size)
            offspring = self._crossover(selected, config.crossover_rate)
            offspring = self._mutation(offspring, param_space, config.mutation_rate)

            population = offspring

        if generation % 10 == 0:
            logger.info(f"多目标优化第{generation}代，最佳适应度: {best_score}")

        return OptimizationResult(
            best_params=best_params or {},
            best_score=best_score,
            all_results=all_results,
            convergence_history=convergence_history
        )

    def _ensemble_optimization(self, strategy: VisualStrategyEditor,


                               objective_function: Callable,
                               config: OptimizationConfig) -> OptimizationResult:
        """集成优化"""
        logger.info("开始集成优化")

        if not config.ensemble_methods:
            config.ensemble_methods = [
                OptimizationMethod.GENETIC,
                OptimizationMethod.PARTICLE_SWARM,
                OptimizationMethod.SIMULATED_ANNEALING
            ]

        all_results = []
        best_params = None
        best_score = float('-inf')

        # 运行多种优化算法
        for method in config.ensemble_methods:
            try:
                logger.info(f"运行{method.value}优化")

                # 创建子配置
                sub_config = OptimizationConfig(
                    method=method,
                    max_iterations=config.max_iterations // len(config.ensemble_methods),
                    population_size=config.population_size,
                    elite_size=config.elite_size,
                    mutation_rate=config.mutation_rate,
                    crossover_rate=config.crossover_rate
                )

                # 运行优化
                result = self.optimize(strategy, objective_function, sub_config)

                all_results.extend(result.all_results)

                if result.best_score > best_score:
                    best_score = result.best_score
                    best_params = result.best_params

            except Exception as e:
                logger.warning(f"{method.value}优化失败: {e}")
                continue

        # 计算收敛历史
        convergence_history = [result[1] for result in all_results]

        return OptimizationResult(
            best_params=best_params or {},
            best_score=best_score,
            all_results=all_results,
            convergence_history=convergence_history
        )

    def _get_parameter_space(self, strategy: VisualStrategyEditor) -> Dict:
        """获取参数空间"""
        param_space = {}

        for node in strategy.get_all_nodes():
            if node.node_type.value in ['feature', 'trade', 'risk']:
                for param_name, param_value in node.params.items():
                    if isinstance(param_value, (int, float)):
                        # 根据参数类型设置搜索范围
                        if isinstance(param_value, int):
                            param_space[f"{node.node_id}_{param_name}"] = {
                                "type": "int",
                                "range": [max(1, param_value // 2), param_value * 2]
                            }
                        else:
                            param_space[f"{node.node_id}_{param_name}"] = {
                                "type": "float",
                                "range": [param_value * 0.5, param_value * 1.5]
                            }

        return param_space

    def _generate_param_combinations(self, param_space: Dict) -> List[Dict]:
        """生成参数组合"""
        param_names = list(param_space.keys())
        param_ranges = [param_space[name]["range"] for name in param_names]

        # 限制组合数量，避免爆炸
        max_combinations = 1000
        combinations = []

        for _ in range(min(max_combinations, np.prod([len(r) for r in param_ranges]))):
            params = {}
            for i, name in enumerate(param_names):
                param_type = param_space[name]["type"]
                param_range = param_space[name]["range"]

                if param_type == "int":
                    params[name] = secrets.randint(param_range[0], param_range[1])
                else:
                    params[name] = secrets.uniform(param_range[0], param_range[1])

            combinations.append(params)

        return combinations

    def _random_sample_params(self, param_space: Dict) -> Dict:
        """随机采样参数"""
        params = {}
        for name, config in param_space.items():
            param_type = config["type"]
            param_range = config["range"]

            if param_type == "int":
                params[name] = secrets.randint(param_range[0], param_range[1])
            else:
                params[name] = secrets.uniform(param_range[0], param_range[1])

        return params

    def _update_strategy_params(self, strategy: VisualStrategyEditor, params: Dict):
        """更新策略参数"""
        for param_name, param_value in params.items():
            # 解析参数名称: node_id_param_name
            parts = param_name.split('_', 1)
        if len(parts) == 2:
            node_id, actual_param = parts
            node = strategy.get_node(node_id)
        if node and actual_param in node.params:
            node.params[actual_param] = param_value

    def _initialize_population(self, param_space: Dict, population_size: int) -> List[Dict]:
        """初始化种群"""
        population = []
        for _ in range(population_size):
            individual = self._random_sample_params(param_space)
            population.append(individual)
        return population

    def _selection(self, population: List[Dict], fitness_scores: List[float],


                   elite_size: int) -> List[Dict]:
        """选择操作"""
        # 精英选择 + 轮盘赌选择
        elite_indices = np.argsort(fitness_scores)[-elite_size:]
        elite = [population[i] for i in elite_indices]

        # 轮盘赌选择
        fitness_array = np.array(fitness_scores)
        fitness_array = fitness_array - np.min(fitness_array) + 1e-6  # 避免负值
        probabilities = fitness_array / np.sum(fitness_array)

        selected_indices = np.secrets.choice(
            len(population),
            size=len(population) - elite_size,
            p=probabilities
        )

        selected = [population[i] for i in selected_indices]
        return elite + selected

    def _crossover(self, population: List[Dict], crossover_rate: float) -> List[Dict]:
        """交叉操作"""
        offspring = []

        for i in range(0, len(population), 2):
            if i + 1 < len(population):
                parent1, parent2 = population[i], population[i + 1]

                if secrets.random() < crossover_rate:
                    # 单点交叉
                    keys = list(parent1.keys())
                    crossover_point = secrets.randint(1, len(keys) - 1)

                    child1 = {}
                    child2 = {}

                    for j, key in enumerate(keys):
                        if j < crossover_point:
                            child1[key] = parent1[key]
                            child2[key] = parent2[key]
                        else:
                            child1[key] = parent2[key]
                            child2[key] = parent1[key]

                    offspring.extend([child1, child2])
                else:
                    offspring.extend([parent1, parent2])
            else:
                offspring.append(population[i])

        return offspring

    def _mutation(self, population: List[Dict], param_space: Dict,


                  mutation_rate: float) -> List[Dict]:
        """变异操作"""
        for individual in population:
            for param_name in individual:
                if secrets.random() < mutation_rate:
                    config = param_space[param_name]
                    param_type = config["type"]
                    param_range = config["range"]

                    if param_type == "int":
                        individual[param_name] = secrets.randint(param_range[0], param_range[1])
                    else:
                        individual[param_name] = secrets.uniform(param_range[0], param_range[1])

        return population

    def analyze_sensitivity(self, strategy: VisualStrategyEditor,


                            objective_function: Callable,
                            param_name: str,
                            param_range: List[float]) -> Dict:
        """参数敏感性分析"""
        logger.info(f"开始参数敏感性分析: {param_name}")

        sensitivity_results = []
        base_params = {}

        # 获取当前参数
        for node in strategy.get_all_nodes():
            for key, value in node.params.items():
                if isinstance(value, (int, float)):
                    base_params[f"{node.node_id}_{key}"] = value

        # 测试参数范围
        for param_value in np.linspace(param_range[0], param_range[1], 20):
            test_params = base_params.copy()
            test_params[param_name] = param_value

            try:
                self._update_strategy_params(strategy, test_params)
                score = objective_function(strategy)
                sensitivity_results.append((param_value, score))
            except Exception as e:
                logger.warning(f"敏感性分析评估失败: {e}")
                continue

        return {
            "param_name": param_name,
            "results": sensitivity_results,
            "optimal_value": max(sensitivity_results, key=lambda x: x[1])[0] if sensitivity_results else None
        }
