# RQA2025 模块文档索引

## 概述
本索引包含所有架构模块的详细文档，便于开发者快速查找和理解模块功能。

## 📚 文档索引

**生成时间**: 2025-08-23 21:16:22

---

## 核心服务层

### [事件总线](core/event_bus.md)
- **模块标识**: `core.event_bus`
- **文档路径**: `docs/architecture/core/event_bus.md`
- **功能说明**: 事件总线的核心功能实现

### [依赖注入容器](core/container.md)
- **模块标识**: `core.container`
- **文档路径**: `docs/architecture/core/container.md`
- **功能说明**: 依赖注入容器的核心功能实现

### [业务流程编排器](core/business_process_orchestrator.md)
- **模块标识**: `core.business_process_orchestrator`
- **文档路径**: `docs/architecture/core/business_process_orchestrator.md`
- **功能说明**: 业务流程编排器的核心功能实现

### [架构层实现](core/architecture_layers.md)
- **模块标识**: `core.architecture_layers`
- **文档路径**: `docs/architecture/core/architecture_layers.md`
- **功能说明**: 架构层实现的核心功能实现

---

## 数据采集层

### [数据源适配器](data/adapters.md)
- **模块标识**: `data.adapters`
- **文档路径**: `docs/architecture/data/adapters.md`
- **功能说明**: 数据源适配器的核心功能实现

### [实时数据采集器](data/collector.md)
- **模块标识**: `data.collector`
- **文档路径**: `docs/architecture/data/collector.md`
- **功能说明**: 实时数据采集器的核心功能实现

### [数据验证器](data/validator.md)
- **模块标识**: `data.validator`
- **文档路径**: `docs/architecture/data/validator.md`
- **功能说明**: 数据验证器的核心功能实现

### [数据质量监控器](data/quality_monitor.md)
- **模块标识**: `data.quality_monitor`
- **文档路径**: `docs/architecture/data/quality_monitor.md`
- **功能说明**: 数据质量监控器的核心功能实现

---

## 特征处理层

### [智能特征工程](features/engineering.md)
- **模块标识**: `features.engineering`
- **文档路径**: `docs/architecture/features/engineering.md`
- **功能说明**: 智能特征工程的核心功能实现

### [分布式特征处理](features/distributed.md)
- **模块标识**: `features.distributed`
- **文档路径**: `docs/architecture/features/distributed.md`
- **功能说明**: 分布式特征处理的核心功能实现

### [硬件加速计算](features/acceleration.md)
- **模块标识**: `features.acceleration`
- **文档路径**: `docs/architecture/features/acceleration.md`
- **功能说明**: 硬件加速计算的核心功能实现

---

## 模型推理层

### [集成学习](ml/integration.md)
- **模块标识**: `ml.integration`
- **文档路径**: `docs/architecture/ml/integration.md`
- **功能说明**: 集成学习的核心功能实现

### [模型管理](ml/models.md)
- **模块标识**: `ml.models`
- **文档路径**: `docs/architecture/ml/models.md`
- **功能说明**: 模型管理的核心功能实现

### [推理引擎](ml/engine.md)
- **模块标识**: `ml.engine`
- **文档路径**: `docs/architecture/ml/engine.md`
- **功能说明**: 推理引擎的核心功能实现

---

## API网关层

### [API网关](gateway/api_gateway.md)
- **模块标识**: `gateway.api_gateway`
- **文档路径**: `docs/architecture/gateway/api_gateway.md`
- **功能说明**: API网关的核心功能实现

---

## 策略决策层

### [策略引擎](backtest/engine.md)
- **模块标识**: `backtest.engine`
- **文档路径**: `docs/architecture/backtest/engine.md`
- **功能说明**: 策略引擎的核心功能实现

### [策略分析器](backtest/analyzer.md)
- **模块标识**: `backtest.analyzer`
- **文档路径**: `docs/architecture/backtest/analyzer.md`
- **功能说明**: 策略分析器的核心功能实现

---

## 交易执行层

### [交易执行器](trading/executor.md)
- **模块标识**: `trading.executor`
- **文档路径**: `docs/architecture/trading/executor.md`
- **功能说明**: 交易执行器的核心功能实现

### [交易管理器](trading/manager.md)
- **模块标识**: `trading.manager`
- **文档路径**: `docs/architecture/trading/manager.md`
- **功能说明**: 交易管理器的核心功能实现

---

## 风控合规层

### [风险检查器](risk/checker.md)
- **模块标识**: `risk.checker`
- **文档路径**: `docs/architecture/risk/checker.md`
- **功能说明**: 风险检查器的核心功能实现

### [风险监控器](risk/monitor.md)
- **模块标识**: `risk.monitor`
- **文档路径**: `docs/architecture/risk/monitor.md`
- **功能说明**: 风险监控器的核心功能实现

---

## 监控反馈层

### [系统监控](engine/monitoring.md)
- **模块标识**: `engine.monitoring`
- **文档路径**: `docs/architecture/engine/monitoring.md`
- **功能说明**: 系统监控的核心功能实现

### [告警系统](engine/alerting.md)
- **模块标识**: `engine.alerting`
- **文档路径**: `docs/architecture/engine/alerting.md`
- **功能说明**: 告警系统的核心功能实现

---

## 📋 文档使用指南

### 文档阅读顺序
1. **总体架构文档**: 先了解整体架构设计
2. **核心服务层**: 理解系统核心组件
3. **基础设施层**: 掌握基础服务功能
4. **业务功能层**: 深入具体业务实现

### 文档更新机制
- 文档通过自动化工具生成
- 代码变更后自动更新文档
- 重要变更需要人工审核

### 文档反馈
- 发现问题请提交Issue
- 改进建议请提交PR
- 文档错误请及时修正

## 🔧 文档维护

### 维护工具
```bash
# 生成所有模块文档
python scripts/generate_module_docs.py --all

# 生成特定模块文档
python scripts/generate_module_docs.py --layer core --module event_bus

# 生成文档索引
python scripts/generate_module_docs.py --index
```

### 质量检查
```bash
# 检查文档完整性
python scripts/check_documentation.py

# 验证文档格式
python scripts/validate_docs.py
```

---

**索引版本**: 1.0
**维护人员**: 架构组
**更新频率**: 代码变更时自动更新
