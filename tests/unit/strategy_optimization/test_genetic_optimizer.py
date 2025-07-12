import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from src.strategy_optimization.genetic_optimizer import (
    GeneticOptimizer,
    TradingStrategyOptimizer,
    GeneSpec,
    GeneType
)

@pytest.fixture
def sample_gene_specs():
    """创建测试用基因规格"""
    return [
        GeneSpec("param1", GeneType.FLOAT, (0.0, 1.0), [], 2),
        GeneSpec("param2", GeneType.INT, (1, 10), []),
        GeneSpec("param3", GeneType.CATEGORICAL, [], ["A", "B", "C"])
    ]

@pytest.fixture
def mock_strategy():
    """模拟策略类"""
    strategy = MagicMock()
    strategy.backtest.return_value = {"sharpe_ratio": np.random.uniform(0.5, 2.0)}
    return strategy

@pytest.fixture
def base_optimizer(sample_gene_specs):
    """基础优化器实例"""
    return GeneticOptimizer(
        gene_specs=sample_gene_specs,
        population_size=10,
        elite_size=2,
        mutation_rate=0.1,
        crossover_rate=0.7
    )

@pytest.fixture
def strategy_optimizer(mock_strategy):
    """策略优化器实例"""
    return TradingStrategyOptimizer(
        strategy=mock_strategy,
        population_size=10,
        elite_size=2,
        mutation_rate=0.1,
        crossover_rate=0.7
    )

def test_initialization(base_optimizer):
    """测试优化器初始化"""
    assert len(base_optimizer.gene_specs) == 3
    assert base_optimizer.population_size == 10
    assert base_optimizer.elite_size == 2
    assert base_optimizer.mutation_rate == 0.1
    assert base_optimizer.crossover_rate == 0.7

def test_population_initialization(base_optimizer):
    """测试种群初始化"""
    base_optimizer.initialize_population()
    assert len(base_optimizer.population) == 10

    # 检查基因值在合理范围内
    for individual in base_optimizer.population:
        assert 0.0 <= individual["param1"] <= 1.0
        assert 1 <= individual["param2"] <= 10
        assert individual["param3"] in ["A", "B", "C"]

def test_evaluate_fitness(strategy_optimizer):
    """测试适应度评估"""
    strategy_optimizer.initialize_population()
    individual = strategy_optimizer.population[0]

    # 模拟策略返回的夏普比率
    mock_sharpe = 1.5
    strategy_optimizer.strategy.backtest.return_value = {"sharpe_ratio": mock_sharpe}

    fitness = strategy_optimizer.evaluate_fitness(individual)
    assert fitness == mock_sharpe

    # 验证策略参数被正确设置
    strategy_optimizer.strategy.set_params.assert_called_with(**individual)

def test_rank_population(base_optimizer):
    """测试种群排序"""
    base_optimizer.initialize_population()

    # 模拟适应度评估
    base_optimizer.evaluate_fitness = lambda ind: ind["param1"]

    # 手动设置种群以控制测试
    test_population = [
        {"param1": 0.1, "param2": 1, "param3": "A"},
        {"param1": 0.5, "param2": 5, "param3": "B"},
        {"param1": 0.3, "param2": 3, "param3": "C"}
    ]
    base_optimizer.population = test_population

    base_optimizer.rank_population()

    # 验证排序结果
    assert base_optimizer.population[0]["param1"] == 0.5
    assert base_optimizer.population[1]["param1"] == 0.3
    assert base_optimizer.population[2]["param1"] == 0.1

    # 验证适应度历史记录
    assert base_optimizer.fitness_history[-1] == 0.5

def test_select_parents(base_optimizer):
    """测试父代选择"""
    base_optimizer.initialize_population()

    # 模拟适应度评估
    base_optimizer.evaluate_fitness = lambda ind: ind["param1"]

    # 手动设置种群以控制测试
    test_population = [
        {"param1": 0.9, "param2": 1, "param3": "A"},
        {"param1": 0.5, "param2": 5, "param3": "B"},
        {"param1": 0.1, "param2": 3, "param3": "C"}
    ]
    base_optimizer.population = test_population

    parents = base_optimizer.select_parents()

    # 验证选择结果
    assert len(parents) == 8  # population_size - elite_size
    # 高适应度个体应有更高概率被选中
    assert parents.count(test_population[0]) > parents.count(test_population[2])

def test_crossover(base_optimizer):
    """测试交叉操作"""
    base_optimizer.initialize_population()

    parent1 = {"param1": 0.1, "param2": 1, "param3": "A"}
    parent2 = {"param1": 0.9, "param2": 9, "param3": "C"}

    child1, child2 = base_optimizer.crossover(parent1, parent2)

    # 验证交叉点
    crossover_point = None
    if (child1["param1"] == parent1["param1"] and
        child1["param2"] == parent1["param2"] and
        child1["param3"] == parent2["param3"]):
        crossover_point = 2
    elif (child1["param1"] == parent1["param1"] and
          child1["param2"] == parent2["param2"] and
          child1["param3"] == parent2["param3"]):
        crossover_point = 1

    assert crossover_point is not None

    # 验证子代基因值
    for param in ["param1", "param2", "param3"]:
        assert child1[param] in [parent1[param], parent2[param]]
        assert child2[param] in [parent1[param], parent2[param]]

def test_mutate(base_optimizer):
    """测试变异操作"""
    base_optimizer.initialize_population()
    original = {"param1": 0.5, "param2": 5, "param3": "B"}

    # 设置高变异率以确保变异发生
    base_optimizer.mutation_rate = 1.0

    mutated = base_optimizer.mutate(original)

    # 验证变异结果
    assert mutated != original
    assert 0.0 <= mutated["param1"] <= 1.0
    assert 1 <= mutated["param2"] <= 10
    assert mutated["param3"] in ["A", "B", "C"]

def test_evolve(base_optimizer):
    """测试进化过程"""
    base_optimizer.initialize_population()

    # 模拟适应度评估
    base_optimizer.evaluate_fitness = lambda ind: ind["param1"]

    # 记录原始种群
    original_population = base_optimizer.population.copy()
    original_population.sort(key=lambda x: -x["param1"])

    # 执行进化
    base_optimizer.evolve()

    # 验证种群大小不变
    assert len(base_optimizer.population) == 10

    # 验证精英保留
    assert base_optimizer.population[0] == original_population[0]
    assert base_optimizer.population[1] == original_population[1]

    # 验证种群发生变化
    assert base_optimizer.population != original_population

def test_optimize(strategy_optimizer):
    """测试完整优化流程"""
    # 模拟适应度评估
    strategy_optimizer.strategy.backtest.return_value = {"sharpe_ratio": 1.5}

    best_params = strategy_optimizer.optimize(generations=5)

    # 验证返回最佳参数
    assert isinstance(best_params, dict)
    assert "lookback" in best_params
    assert "threshold" in best_params
    assert "position_size" in best_params
    assert "exit_mode" in best_params

    # 验证适应度历史记录
    assert len(strategy_optimizer.fitness_history) == 5 + 1  # 初始排序+5代
