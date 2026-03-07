# RQA2025 各层功能验证报告

## 📊 验证概览

**验证时间**: 2025-08-24T12:44:13.240692 至 2025-08-24T12:44:13.246416
**验证时长**: 0.0 秒
**总体状态**: ❌ 失败
**验证层数**: 11 层
**通过层数**: 0 层
**通过率**: 0.0%
**总得分**: 0 分
**平均得分**: 0.0 分

---

## 🏗️ 各层验证结果

### ❌ Core

**状态**: FAILED
**功能得分**: 0 分

**发现问题**:

- ⚠️ 核心服务层验证失败: No module named 'src.core'

### ❌ Infrastructure

**状态**: FAILED
**功能得分**: 0 分

**发现问题**:

- ⚠️ 基础设施层验证失败: cannot import name 'UnifiedConfigManager' from 'src.infrastructure' (unknown location)

### ❌ Data

**状态**: FAILED
**功能得分**: 0 分

**发现问题**:

- ⚠️ 数据采集层验证失败: No module named 'src.data'

### ❌ Gateway

**状态**: FAILED
**功能得分**: 0 分

**发现问题**:

- ⚠️ API网关层验证失败: No module named 'src.gateway'

### ❌ Features

**状态**: FAILED
**功能得分**: 0 分

**发现问题**:

- ⚠️ 特征处理层验证失败: cannot import name 'FeatureProcessor' from 'src.features' (C:\PythonProject\RQA2025\scripts\src\features\__init__.py)

### ❌ Ml

**状态**: FAILED
**功能得分**: 0 分

**发现问题**:

- ⚠️ 模型推理层验证失败: No module named 'src.ml'

### ❌ Backtest

**状态**: FAILED
**功能得分**: 0 分

**发现问题**:

- ⚠️ 策略决策层验证失败: No module named 'src.backtest'

### ❌ Risk

**状态**: FAILED
**功能得分**: 0 分

**发现问题**:

- ⚠️ 风控合规层验证失败: No module named 'src.risk'

### ❌ Trading

**状态**: FAILED
**功能得分**: 0 分

**发现问题**:

- ⚠️ 交易执行层验证失败: No module named 'src.trading'

### ❌ Engine

**状态**: FAILED
**功能得分**: 0 分

**发现问题**:

- ⚠️ 监控反馈层验证失败: No module named 'src.engine'

### ❌ Business Flow

**状态**: FAILED
**功能得分**: 0 分

**发现问题**:

- ⚠️ 业务流程验证失败: No module named 'src.core'

## ⚠️ 总体问题列表

- 核心服务层验证失败: No module named 'src.core'
- 基础设施层验证失败: cannot import name 'UnifiedConfigManager' from 'src.infrastructure' (unknown location)
- 数据采集层验证失败: No module named 'src.data'
- API网关层验证失败: No module named 'src.gateway'
- 特征处理层验证失败: cannot import name 'FeatureProcessor' from 'src.features' (C:\PythonProject\RQA2025\scripts\src\features\__init__.py)
- 模型推理层验证失败: No module named 'src.ml'
- 策略决策层验证失败: No module named 'src.backtest'
- 风控合规层验证失败: No module named 'src.risk'
- 交易执行层验证失败: No module named 'src.trading'
- 监控反馈层验证失败: No module named 'src.engine'
- 业务流程验证失败: No module named 'src.core'

---

**报告生成时间**: 2025-08-24T12:44:13.247447
**验证脚本**: scripts/layer_function_validation.py
**验证状态**: ❌ 失败
