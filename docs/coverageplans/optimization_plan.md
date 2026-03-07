# Optimization 模块补测计划

## 覆盖率结果概览
- 命令：pytest tests/unit/infrastructure/optimization --cov=src.infrastructure.optimization --cov-report=term-missing
- 主要文件覆盖率：
  - src/infrastructure/optimization/architecture_refactor.py：64%
  - src/infrastructure/optimization/performance_optimizer.py：85%
- 报告中出现 src/infrastructure/monitoring/* 为覆盖工具的附带统计，后续可通过 coverage omit 或更精确的 --cov 设置过滤。

## 补测重点
1. rchitecture_refactor.py
   - 补充 _execute_* 分支：真实执行模式 dry_run=False、导入修复失败路径、目录创建异常。
   - 覆盖 _display_plan_summary、un_full_refactor 全流程，并验证 efactor_actions 聚合列表。
   - 针对 _analyze_directory_compliance 新增 xpected_dirs 为空 / 目录缺失的边界测试。
2. performance_optimizer.py
   - 新增关于 optimize_memory_usage、optimize_io_operations、optimize_data_structures 的零值场景，验证 _calculate_improvement 的容错逻辑。
   - 针对 _optimize_* 内部方法（异步、连接池、对象池等）创建轻量 stub 测试，确认路径被触达。

## 建议测试文件
- 	ests/unit/infrastructure/optimization/test_architecture_refactor_core.py
- 	ests/unit/infrastructure/optimization/test_performance_optimizer_core.py

## 后续行动
1. 阶段一：完成 rchitecture_refactor 核心流程补测，目标覆盖率 ≥80%。
2. 阶段二：针对 performance_optimizer 编写零值 & 边界测试，保持/提高至 ≥85%。
3. 若需进一步提升，可检查其他子模块（如 optimization_engine、strategy_optimizer），在补测后再次生成覆盖率报告。

覆盖率详情：	est_logs/htmlcov_infra_optimization/index.html

