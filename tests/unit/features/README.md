# 特征层单元测试指引

本目录按 Phase 10.1 治理后的特征层架构组织单元测试，覆盖目标为 **通过率 100%**、**覆盖率 ≥ 80%**。测试执行统一使用 `pytest -n auto`，所有日志输出到 `test_logs/features`。

## 目录约定

- `core/`：特征引擎、配置、管理器等核心流程。
- `processors/`：通用处理器、技术指标处理器、标准化与质量评估。
- `indicators/`：技术指标计算器及批量指标引擎。
- `performance/` 与 `acceleration/`：并行、GPU、分布式加速策略。
- `store/`：特征存储、缓存、多级持久化适配。
- `monitoring/`：监控、告警、指标采集。
- `distributed/`：分布式处理与调度。
- `orderbook/`、`sentiment/`：订单簿与情感特征。
- `plugins/`、`utils/`、`fallback/`：插件、工具、降级策略。

各子目录可继续细分子模块（例如 `processors/technical/`），测试文件命名遵循 `test_*.py` 规范。

## 执行说明

```shell
pytest tests/unit/features -n auto --cov=src/features --cov-report=xml
```

若需生成阶段性覆盖报告，可附加 `--cov-report=term-missing` 方便定位缺口。

## 数据策略

- 默认使用轻量内存数据集，必要时通过参数化覆盖边界情况。
- 涉及数据层或外部资源时，优先使用 Fake/Mock，避免真实 I/O。
- GPU 与分布式场景以降级或模拟验证调度逻辑，不直接触发真实硬件。

## 夹具与工具

- 公共夹具定义在 `tests/unit/features/conftest.py`，提供：
  - 标准行情数据集（含 `open/high/low/close/volume`）。
  - 典型 `FeatureConfig` 组合。
  - 测试日志目录初始化。
- 若模块需要特定数据集，请在对应目录下新增本地夹具，并保持互斥命名。

## 提交流程

1. 新增或更新测试需覆盖主要分支与异常路径。
2. 本地运行最小化用例确保通过，再运行全量命令验证覆盖率。
3. 将执行日志保存到 `test_logs/features/<模块>/`，并在 MR/报告中引用。

通过以上约束，保证特征层单测体系与架构设计保持一致，并支持分阶段扩展覆盖。***

