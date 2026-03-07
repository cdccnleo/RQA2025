# 回测层（backtest）架构设计说明

## 1. 模块定位
backtest模块为RQA2025系统所有策略、模型、交易主流程提供高效、灵活的历史回测、绩效评估、参数优化、可视化等能力，是策略验证与风险控制的核心环节。

## 2. 主要子系统
- **回测引擎**：BacktestEngine、backtest_engine 支持事件驱动、单/多/优化模式回测，兼容多策略、参数优化、A股特有规则。
  - 如需硬件加速，FPGA相关模块应从`src/acceleration/fpga/`导入。
- **回测数据加载**：BacktestDataLoader 支持多源、多频率历史数据加载与预处理。
- **绩效分析与可视化**：PerformanceAnalyzer、Plotter 支持回测结果的绩效分析、风险指标、收益分布、可视化展示。
- **参数优化**：ParameterOptimizer、StrategyOptimizer 支持网格搜索、贝叶斯优化等多种参数优化算法，提升策略表现。
- **模型与策略评估**：ModelEvaluator 支持模型/策略的回测评估、指标计算、A股规则模拟。
- **回测工具与日志**：统一日志、日期、交易日历等工具，提升回测可追溯性和易用性。

## 2. 主要子系统（补充）

- **评估系统**：StrategyEvaluator、ModelEvaluator 支持多维度绩效与风险分析、归因分析、稳定性测试等。
- **工具模块**：BacktestUtils、StrategyValidationResult 提供策略验证、风险指标、数据验证、回测报告等辅助功能。
- **性能优化**：引擎与数据加载层均支持并行、缓存、内存监控，适配大数据量场景。
- **测试与质量保障**：pytest全覆盖，持续集成，主流程、边界、异常、性能等用例齐全。

## 3. 典型用法
### 回测主流程
```python
from src.backtest.backtest_engine import BacktestEngine
engine = BacktestEngine(config, strategy, data_provider)
results = engine.run()
```

### 参数优化
```python
from src.backtest.parameter_optimizer import ParameterOptimizer
optimizer = ParameterOptimizer(engine)
results = optimizer.grid_search(strategy, param_grid, start, end, n_jobs=4)
```

### 绩效分析与可视化
```python
from src.backtest.analyzer import PerformanceAnalyzer
from src.backtest.visualizer import Plotter
analyzer = PerformanceAnalyzer()
metrics = analyzer.analyze(returns)
plotter = Plotter()
plotter.plot_performance(results)
```

## 3. 典型用法（补充）

### 评估系统
```python
from src.backtest.evaluation import StrategyEvaluator
evaluator = StrategyEvaluator()
metrics = evaluator.evaluate_strategy(strategy_returns, benchmark_returns)
```

### 实时回测原型
```python
from src.backtest.engine import BacktestEngine
engine = BacktestEngine()
results = engine.run_realtime(strategy, live_data_stream)
```

## 4. 在主流程中的地位
- 为策略/模型/交易等提供历史验证和绩效评估，支撑策略落地和风险控制。
- 支持多策略、多参数、多市场、A股特有规则、并发优化，保障主流程的灵活性和高可用性。
- 接口抽象与注册机制，便于扩展新回测模式、适配新市场、Mock测试等。

## 4. 测试与质量保障（补充）

- 已实现高质量pytest单元测试，覆盖回测引擎、数据加载、评估、优化、可视化等所有核心功能。
- 测试用例见：tests/unit/backtest/ 目录下相关文件。 