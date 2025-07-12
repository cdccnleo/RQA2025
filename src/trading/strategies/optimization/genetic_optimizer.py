import numpy as np
import random
from typing import Dict, List, Tuple
import logging
from dataclasses import dataclass
from enum import Enum, auto

logger = logging.getLogger(__name__)

class GeneType(Enum):
    """基因类型枚举"""
    FLOAT = auto()
    INT = auto()
    CATEGORICAL = auto()

@dataclass
class GeneSpec:
    """基因规格定义"""
    name: str
    gene_type: GeneType
    bounds: Tuple[float, float]  # 对于FLOAT/INT类型
    categories: List[str]  # 对于CATEGORICAL类型
    precision: int = 2  # 对于FLOAT类型的精度

class GeneticOptimizer:
    """遗传算法优化器"""

    def __init__(self,
                 gene_specs: List[GeneSpec],
                 population_size: int = 50,
                 elite_size: int = 5,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7):
        """
        初始化优化器

        Args:
            gene_specs: 基因规格列表
            population_size: 种群大小
            elite_size: 精英保留数量
            mutation_rate: 变异率
            crossover_rate: 交叉率
        """
        self.gene_specs = gene_specs
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.fitness_history = []

    def initialize_population(self):
        """初始化种群"""
        self.population = []
        for _ in range(self.population_size):
            chromosome = {}
            for spec in self.gene_specs:
                if spec.gene_type == GeneType.FLOAT:
                    value = round(random.uniform(*spec.bounds), spec.precision)
                elif spec.gene_type == GeneType.INT:
                    value = random.randint(*spec.bounds)
                else:  # CATEGORICAL
                    value = random.choice(spec.categories)
                chromosome[spec.name] = value
            self.population.append(chromosome)

    def evaluate_fitness(self, chromosome: Dict) -> float:
        """
        评估染色体适应度(需由子类实现)

        Args:
            chromosome: 染色体参数

        Returns:
            float: 适应度分数(越大越好)
        """
        raise NotImplementedError

    def rank_population(self):
        """对种群进行排序"""
        scored = [(self.evaluate_fitness(ind), ind) for ind in self.population]
        scored.sort(reverse=True, key=lambda x: x[0])
        self.population = [ind for (score, ind) in scored]
        self.fitness_history.append(scored[0][0])  # 记录最佳适应度

    def select_parents(self) -> List[Dict]:
        """选择父代(轮盘赌选择)"""
        # 计算适应度总和
        total_fitness = sum(self.evaluate_fitness(ind) for ind in self.population)

        # 计算选择概率
        probs = [self.evaluate_fitness(ind)/total_fitness for ind in self.population]

        # 轮盘赌选择
        parents = []
        for _ in range(self.population_size - self.elite_size):
            pick = random.uniform(0, 1)
            current = 0
            for i, ind in enumerate(self.population):
                current += probs[i]
                if current > pick:
                    parents.append(ind)
                    break
        return parents

    def crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """交叉操作(单点交叉)"""
        child1, child2 = {}, {}

        # 随机选择交叉点
        crossover_point = random.randint(1, len(self.gene_specs)-1)

        # 执行交叉
        for i, spec in enumerate(self.gene_specs):
            if i < crossover_point:
                child1[spec.name] = parent1[spec.name]
                child2[spec.name] = parent2[spec.name]
            else:
                child1[spec.name] = parent2[spec.name]
                child2[spec.name] = parent1[spec.name]

        return child1, child2

    def mutate(self, chromosome: Dict) -> Dict:
        """变异操作"""
        mutated = chromosome.copy()
        for spec in self.gene_specs:
            if random.random() < self.mutation_rate:
                if spec.gene_type == GeneType.FLOAT:
                    mutated[spec.name] = round(random.uniform(*spec.bounds), spec.precision)
                elif spec.gene_type == GeneType.INT:
                    mutated[spec.name] = random.randint(*spec.bounds)
                else:  # CATEGORICAL
                    mutated[spec.name] = random.choice(spec.categories)
        return mutated

    def evolve(self):
        """进化一代"""
        # 保留精英
        elites = self.population[:self.elite_size]

        # 选择父代
        parents = self.select_parents()

        # 生成下一代
        next_population = elites.copy()
        while len(next_population) < self.population_size:
            # 选择两个父代
            parent1, parent2 = random.sample(parents, 2)

            # 交叉
            if random.random() < self.crossover_rate:
                child1, child2 = self.crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # 变异
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)

            next_population.extend([child1, child2])

        # 保持种群大小
        self.population = next_population[:self.population_size]

    def optimize(self, generations: int = 100) -> Dict:
        """
        执行优化

        Args:
            generations: 进化代数

        Returns:
            Dict: 最佳参数组合
        """
        self.initialize_population()
        self.rank_population()

        for i in range(generations):
            self.evolve()
            self.rank_population()
            logger.info(f"Generation {i+1}, Best Fitness: {self.fitness_history[-1]}")

        return self.population[0]

class TradingStrategyOptimizer(GeneticOptimizer):
    """交易策略遗传算法优化器"""

    def __init__(self, strategy, **kwargs):
        """
        初始化策略优化器

        Args:
            strategy: 待优化的策略实例
        """
        # 定义基因规格
        gene_specs = [
            GeneSpec("lookback", GeneType.INT, (10, 100), []),
            GeneSpec("threshold", GeneType.FLOAT, (0.1, 0.9), [], 2),
            GeneSpec("position_size", GeneType.FLOAT, (0.1, 0.5), [], 2),
            GeneSpec("exit_mode", GeneType.CATEGORICAL, [], ["fixed", "trailing"])
        ]

        super().__init__(gene_specs, **kwargs)
        self.strategy = strategy

    def evaluate_fitness(self, chromosome: Dict) -> float:
        """评估策略适应度(使用夏普比率)"""
        # 更新策略参数
        self.strategy.set_params(**chromosome)

        # 执行回测
        results = self.strategy.backtest()

        # 使用夏普比率作为适应度
        return results["sharpe_ratio"]
