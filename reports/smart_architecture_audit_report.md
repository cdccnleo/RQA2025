# 智能架构审计报告

## 📊 审计概览

**审计时间**: 2025-08-23T21:33:05.270704
**架构评分**: 99.2/100
**发现问题**: 69 个
**状态**: 架构质量优秀，完全符合设计要求

### 问题统计
| 问题类型 | 数量 | 严重程度 |
|---------|------|---------|
| 关键问题 | 0 | 🔴 高 |
| 警告问题 | 69 | 🟡 中 |
| 信息问题 | 0 | 🟢 低 |

---

## 🏗️ 层级架构审计结果

### CORE 层级 (核心服务层)
**文件数量**: 23 个
**子目录数量**: 2 个
**职责匹配度**: 44.5 个主要职责匹配
**跨层级概念**: 0 个

**子目录列表**:
- `integration`
- `optimizations`

### INFRASTRUCTURE 层级 (基础设施层)
**文件数量**: 344 个
**子目录数量**: 115 个
**职责匹配度**: 664.0 个主要职责匹配
**跨层级概念**: 0 个

**子目录列表**:
- `benchmark`
- `cache`
- `cloud_native`
- `config`
- `core`
- `deployment`
- `di`
- `disaster`
- `distributed`
- `edge_computing`
- `error`
- `extensions`
- `health`
- `interfaces`
- `logging`
- `mobile`
- `monitoring`
- `ops`
- `performance`
- `resource`
- `scheduler`
- `security`
- `services`
- `services.backup_20250823_212847`
- `testing`
- `trading`
- `utils`
- `utils.backup_20250823_212847`
- `versioning`
- `config`
- `core`
- `error`
- `event`
- `interfaces`
- `managers`
- `monitoring`
- `performance`
- `security`
- `services`
- `static`
- `storage`
- `strategies`
- `utils`
- `validation`
- `web`
- `async_processing`
- `cache`
- `cloud`
- `config`
- `database`
- `deployment`
- `di`
- `distributed`
- `error`
- `factories`
- `logging`
- `microservice`
- `monitoring`
- `performance`
- `resource_management`
- `security`
- `utils`
- `interfaces`
- `cache`
- `config`
- `core`
- `data`
- `event`
- `interfaces`
- `logs`
- `managers`
- `models`
- `performance`
- `security`
- `services`
- `static`
- `storage`
- `strategies`
- `utils`
- `validation`
- `web`
- `core`
- `core`
- `logger`
- `core`
- `interfaces`
- `monitoring_service`
- `core`
- `compliance`
- `dashboard`
- `email`
- `web`
- `alerting`
- `cache`
- `config`
- `core`
- `factories`
- `implementations`
- `interfaces`
- `models`
- `monitoring`
- `core`
- `monitoring_service`
- `templates`
- `docs`
- `cache`
- `database`
- `network`
- `notification`
- `security`
- `storage`
- `interfaces`
- `adapters`
- `helpers`
- `validators`

### DATA 层级 (数据采集层)
**文件数量**: 123 个
**子目录数量**: 44 个
**职责匹配度**: 154.5 个主要职责匹配
**跨层级概念**: 0 个

**子目录列表**:
- `adapters`
- `adapters.backup_20250823_212847`
- `alignment`
- `cache`
- `china`
- `collector`
- `compliance`
- `core`
- `decoders`
- `distributed`
- `edge`
- `export`
- `governance`
- `integration`
- `interfaces`
- `lake`
- `loader`
- `ml`
- `monitoring`
- `optimization`
- `parallel`
- `performance`
- `preload`
- `processing`
- `quality`
- `quality_monitor`
- `quantum`
- `realtime`
- `repair`
- `sources`
- `streaming`
- `sync`
- `transformers`
- `validation`
- `validator`
- `validators`
- `version_control`
- `miniqmt`
- `china`
- `crypto`
- `international`
- `macro`
- `news`
- `infrastructure`

**架构问题**:
- 🟡 文件包含受限概念 'order'，违反 数据采集层 的架构约束
- 🟡 文件包含受限概念 'order'，违反 数据采集层 的架构约束
- 🟡 文件包含受限概念 'order'，违反 数据采集层 的架构约束
- 🟡 文件包含受限概念 'trading'，违反 数据采集层 的架构约束
- 🟡 文件包含受限概念 'trading'，违反 数据采集层 的架构约束
- 🟡 文件包含受限概念 'trading'，违反 数据采集层 的架构约束
- 🟡 文件包含受限概念 'trading'，违反 数据采集层 的架构约束
- 🟡 文件包含受限概念 'trading'，违反 数据采集层 的架构约束
- 🟡 文件包含受限概念 'order'，违反 数据采集层 的架构约束
- 🟡 文件包含受限概念 'order'，违反 数据采集层 的架构约束
- 🟡 文件包含受限概念 'order'，违反 数据采集层 的架构约束
- 🟡 文件包含受限概念 'trading'，违反 数据采集层 的架构约束
- 🟡 文件包含受限概念 'trading'，违反 数据采集层 的架构约束
- 🟡 文件包含受限概念 'order'，违反 数据采集层 的架构约束
- 🟡 文件包含受限概念 'trading'，违反 数据采集层 的架构约束
- 🟡 文件包含受限概念 'order'，违反 数据采集层 的架构约束
- 🟡 文件包含受限概念 'execution'，违反 数据采集层 的架构约束
- 🟡 文件包含受限概念 'order'，违反 数据采集层 的架构约束
- 🟡 文件包含受限概念 'execution'，违反 数据采集层 的架构约束
- 🟡 文件包含受限概念 'order'，违反 数据采集层 的架构约束
- 🟡 文件包含受限概念 'trading'，违反 数据采集层 的架构约束
- 🟡 文件包含受限概念 'execution'，违反 数据采集层 的架构约束
- 🟡 文件包含受限概念 'execution'，违反 数据采集层 的架构约束
- 🟡 文件包含受限概念 'trading'，违反 数据采集层 的架构约束
- 🟡 文件包含受限概念 'trading'，违反 数据采集层 的架构约束
- 🟡 文件包含受限概念 'order'，违反 数据采集层 的架构约束

### GATEWAY 层级 (API网关层)
**文件数量**: 1 个
**子目录数量**: 0 个
**职责匹配度**: 6.5 个主要职责匹配
**跨层级概念**: 0 个

**架构问题**:
- 🟡 文件包含受限概念 'trading'，违反 API网关层 的架构约束
- 🟡 文件包含受限概念 'model'，违反 API网关层 的架构约束

### FEATURES 层级 (特征处理层)
**文件数量**: 90 个
**子目录数量**: 24 个
**职责匹配度**: 75.5 个主要职责匹配
**跨层级概念**: 0 个

**子目录列表**:
- `acceleration`
- `config`
- `core`
- `distributed`
- `engineering`
- `intelligent`
- `monitoring`
- `orderbook`
- `performance`
- `plugins`
- `processors`
- `sentiment`
- `technical`
- `types`
- `utils`
- `fpga`
- `gpu`
- `templates`
- `advanced`
- `deep_learning`
- `distributed`
- `gpu`
- `technical`
- `models`

**架构问题**:
- 🟡 文件包含受限概念 'order'，违反 特征处理层 的架构约束
- 🟡 文件包含受限概念 'order'，违反 特征处理层 的架构约束
- 🟡 文件包含受限概念 'order'，违反 特征处理层 的架构约束
- 🟡 文件包含受限概念 'order'，违反 特征处理层 的架构约束
- 🟡 文件包含受限概念 'order'，违反 特征处理层 的架构约束
- 🟡 文件包含受限概念 'trading'，违反 特征处理层 的架构约束
- 🟡 文件包含受限概念 'order'，违反 特征处理层 的架构约束
- 🟡 文件包含受限概念 'order'，违反 特征处理层 的架构约束
- 🟡 文件包含受限概念 'trading'，违反 特征处理层 的架构约束
- 🟡 文件包含受限概念 'order'，违反 特征处理层 的架构约束
- 🟡 文件包含受限概念 'execution'，违反 特征处理层 的架构约束
- 🟡 文件包含受限概念 'order'，违反 特征处理层 的架构约束
- 🟡 文件包含受限概念 'execution'，违反 特征处理层 的架构约束
- 🟡 文件包含受限概念 'order'，违反 特征处理层 的架构约束
- 🟡 文件包含受限概念 'execution'，违反 特征处理层 的架构约束
- 🟡 文件包含受限概念 'execution'，违反 特征处理层 的架构约束
- 🟡 文件包含受限概念 'order'，违反 特征处理层 的架构约束
- 🟡 文件包含受限概念 'order'，违反 特征处理层 的架构约束
- 🟡 文件包含受限概念 'execution'，违反 特征处理层 的架构约束
- 🟡 文件包含受限概念 'order'，违反 特征处理层 的架构约束
- 🟡 文件包含受限概念 'order'，违反 特征处理层 的架构约束
- 🟡 文件包含受限概念 'order'，违反 特征处理层 的架构约束
- 🟡 文件包含受限概念 'order'，违反 特征处理层 的架构约束
- 🟡 文件包含受限概念 'order'，违反 特征处理层 的架构约束

### ML 层级 (模型推理层)
**文件数量**: 28 个
**子目录数量**: 10 个
**职责匹配度**: 48.5 个主要职责匹配
**跨层级概念**: 0 个

**子目录列表**:
- `engine`
- `ensemble`
- `integration`
- `models`
- `models.backup_20250823_212847`
- `tuning`
- `inference`
- `evaluators`
- `optimizers`
- `utils`

**架构问题**:
- 🟡 文件包含受限概念 'order'，违反 模型推理层 的架构约束
- 🟡 文件包含受限概念 'execution'，违反 模型推理层 的架构约束

### BACKTEST 层级 (策略决策层)
**文件数量**: 22 个
**子目录数量**: 3 个
**职责匹配度**: 30.0 个主要职责匹配
**跨层级概念**: 0 个

**子目录列表**:
- `analysis`
- `evaluation`
- `utils`

### RISK 层级 (风控合规层)
**文件数量**: 10 个
**子目录数量**: 0 个
**职责匹配度**: 21.0 个主要职责匹配
**跨层级概念**: 0 个

### TRADING 层级 (交易执行层)
**文件数量**: 99 个
**子目录数量**: 17 个
**职责匹配度**: 107.0 个主要职责匹配
**跨层级概念**: 0 个

**子目录列表**:
- `advanced_analysis`
- `distributed`
- `execution`
- `ml_integration`
- `portfolio`
- `realtime`
- `risk`
- `settlement`
- `signal`
- `strategies`
- `strategy`
- `strategy_workspace`
- `universe`
- `china`
- `china`
- `optimization`
- `static`

**架构问题**:
- 🟡 文件包含受限概念 'backtest'，违反 交易执行层 的架构约束
- 🟡 文件包含受限概念 'backtest'，违反 交易执行层 的架构约束
- 🟡 文件包含受限概念 'backtest'，违反 交易执行层 的架构约束
- 🟡 文件包含受限概念 'backtest'，违反 交易执行层 的架构约束
- 🟡 文件包含受限概念 'backtest'，违反 交易执行层 的架构约束
- 🟡 文件包含受限概念 'backtest'，违反 交易执行层 的架构约束
- 🟡 文件包含受限概念 'backtest'，违反 交易执行层 的架构约束
- 🟡 文件包含受限概念 'backtest'，违反 交易执行层 的架构约束
- 🟡 文件包含受限概念 'backtest'，违反 交易执行层 的架构约束
- 🟡 文件包含受限概念 'backtest'，违反 交易执行层 的架构约束
- 🟡 文件包含受限概念 'backtest'，违反 交易执行层 的架构约束
- 🟡 文件包含受限概念 'simulation'，违反 交易执行层 的架构约束
- 🟡 文件包含受限概念 'backtest'，违反 交易执行层 的架构约束
- 🟡 文件包含受限概念 'simulation'，违反 交易执行层 的架构约束
- 🟡 文件包含受限概念 'simulation'，违反 交易执行层 的架构约束

### ENGINE 层级 (监控反馈层)
**文件数量**: 49 个
**子目录数量**: 16 个
**职责匹配度**: 108.0 个主要职责匹配
**跨层级概念**: 0 个

**子目录列表**:
- `config`
- `documentation`
- `inference`
- `level2`
- `logging`
- `monitoring`
- `monitoring.backup_20250823_212847`
- `optimization`
- `production`
- `testing`
- `web`
- `modules`
- `static`
- `templates`
- `css`
- `js`

## 🔍 架构问题详情

### 🟡 Wrong Layer Placement
**层级**: data
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'order'，违反 数据采集层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: data
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'order'，违反 数据采集层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: data
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'order'，违反 数据采集层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: data
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'trading'，违反 数据采集层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: data
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'trading'，违反 数据采集层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: data
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'trading'，违反 数据采集层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: data
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'trading'，违反 数据采集层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: data
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'trading'，违反 数据采集层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: data
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'order'，违反 数据采集层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: data
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'order'，违反 数据采集层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: data
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'order'，违反 数据采集层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: data
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'trading'，违反 数据采集层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: data
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'trading'，违反 数据采集层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: data
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'order'，违反 数据采集层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: data
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'trading'，违反 数据采集层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: data
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'order'，违反 数据采集层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: data
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'execution'，违反 数据采集层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: data
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'order'，违反 数据采集层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: data
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'execution'，违反 数据采集层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: data
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'order'，违反 数据采集层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: data
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'trading'，违反 数据采集层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: data
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'execution'，违反 数据采集层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: data
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'execution'，违反 数据采集层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: data
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'trading'，违反 数据采集层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: data
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'trading'，违反 数据采集层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: data
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'order'，违反 数据采集层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: gateway
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'trading'，违反 API网关层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: gateway
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'model'，违反 API网关层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: features
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'order'，违反 特征处理层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: features
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'order'，违反 特征处理层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: features
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'order'，违反 特征处理层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: features
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'order'，违反 特征处理层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: features
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'order'，违反 特征处理层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: features
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'trading'，违反 特征处理层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: features
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'order'，违反 特征处理层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: features
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'order'，违反 特征处理层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: features
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'trading'，违反 特征处理层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: features
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'order'，违反 特征处理层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: features
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'execution'，违反 特征处理层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: features
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'order'，违反 特征处理层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: features
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'execution'，违反 特征处理层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: features
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'order'，违反 特征处理层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: features
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'execution'，违反 特征处理层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: features
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'execution'，违反 特征处理层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: features
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'order'，违反 特征处理层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: features
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'order'，违反 特征处理层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: features
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'execution'，违反 特征处理层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: features
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'order'，违反 特征处理层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: features
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'order'，违反 特征处理层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: features
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'order'，违反 特征处理层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: features
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'order'，违反 特征处理层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: features
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'order'，违反 特征处理层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: ml
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'order'，违反 模型推理层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: ml
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'execution'，违反 模型推理层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: trading
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'backtest'，违反 交易执行层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: trading
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'backtest'，违反 交易执行层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: trading
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'backtest'，违反 交易执行层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: trading
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'backtest'，违反 交易执行层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: trading
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'backtest'，违反 交易执行层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: trading
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'backtest'，违反 交易执行层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: trading
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'backtest'，违反 交易执行层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: trading
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'backtest'，违反 交易执行层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: trading
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'backtest'，违反 交易执行层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: trading
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'backtest'，违反 交易执行层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: trading
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'backtest'，违反 交易执行层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: trading
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'simulation'，违反 交易执行层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: trading
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'backtest'，违反 交易执行层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: trading
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'simulation'，违反 交易执行层 的架构约束
**文件**: `unknown`

### 🟡 Wrong Layer Placement
**层级**: trading
**严重程度**: medium
**影响**: 架构职责混乱
**描述**: 文件包含受限概念 'simulation'，违反 交易执行层 的架构约束
**文件**: `unknown`

## 📈 质量指标

### 详细指标
- **架构完整性**: 100.0% - 层级完整程度
- **职责匹配度**: 159.6% - 文件职责符合度
- **接口符合性**: 66.7% - 标准接口使用率
- **依赖健康度**: 40.0% - 依赖关系健康度
- **综合评分**: 99.2% - 整体架构质量

### 评分标准
- **90-100%**: 架构质量优秀，完全符合设计要求
- **75-89%**: 架构质量良好，基本符合设计要求
- **60-74%**: 架构质量一般，存在一些需要改进的地方
- **0-59%**: 架构质量较差，需要重点改进

## 🔗 接口设计分析

### CORE 层级接口 (核心服务层)
- **接口文件**: 1 个
- **基础实现**: 1 个
- **标准接口**: 0 个

**接口问题**:
- ⚠️ core\base.py: 基础实现类不符合标准模式 Base{Name}Component
- ⚠️ core\layer_interfaces.py: 接口命名不符合标准规范 I{Name}Component

### INFRASTRUCTURE 层级接口 (基础设施层)
- **接口文件**: 10 个
- **基础实现**: 11 个
- **标准接口**: 8 个

**接口问题**:
- ⚠️ infrastructure\config\standard_interfaces.py: 接口命名不符合标准规范 I{Name}Component
- ⚠️ infrastructure\config\unified_interfaces.py: 接口命名不符合标准规范 I{Name}Component
- ⚠️ infrastructure\utils.backup_20250823_212847\base_database.py: 基础实现类不符合标准模式 Base{Name}Component
- ⚠️ infrastructure\utils.backup_20250823_212847\database.py: 基础实现类不符合标准模式 Base{Name}Component
- ⚠️ infrastructure\utils.backup_20250823_212847\unified_database.py: 基础实现类不符合标准模式 Base{Name}Component

### DATA 层级接口 (数据采集层)
- **接口文件**: 1 个
- **基础实现**: 1 个
- **标准接口**: 0 个

**接口问题**:
- ⚠️ data\interfaces.py: 接口命名不符合标准规范 I{Name}Component
- ⚠️ data\adapters.backup_20250823_212847\base.py: 基础实现类不符合标准模式 Base{Name}Component

### ML 层级接口 (模型推理层)
- **接口文件**: 0 个
- **基础实现**: 1 个
- **标准接口**: 0 个

**接口问题**:
- ⚠️ ml\tuning\optimizers\base.py: 基础实现类不符合标准模式 Base{Name}Component

## ⚡ 依赖关系分析

### CORE 层级依赖 (核心服务层)
- **内部导入**: 11 个
- **外部导入**: 213 个
- **跨层级导入**: 1 个

### INFRASTRUCTURE 层级依赖 (基础设施层)
- **内部导入**: 75 个
- **外部导入**: 2397 个
- **跨层级导入**: 40 个

### DATA 层级依赖 (数据采集层)
- **内部导入**: 70 个
- **外部导入**: 872 个
- **跨层级导入**: 97 个

**依赖问题**:
- ⚠️ data\api.py: 不合理的跨层级导入
- ⚠️ data\backup_recovery.py: 不合理的跨层级导入
- ⚠️ data\base_adapter.py: 不合理的跨层级导入
- ⚠️ data\data_manager.py: 不合理的跨层级导入
- ⚠️ data\data_manager.py: 不合理的跨层级导入
- ⚠️ data\data_manager.py: 不合理的跨层级导入
- ⚠️ data\data_manager.py: 不合理的跨层级导入
- ⚠️ data\enhanced_integration_manager.py: 不合理的跨层级导入
- ⚠️ data\market_data.py: 不合理的跨层级导入
- ⚠️ data\registry.py: 不合理的跨层级导入
- ⚠️ data\adapters\miniqmt\adapter.py: 不合理的跨层级导入
- ⚠️ data\adapters\miniqmt\adapter.py: 不合理的跨层级导入
- ⚠️ data\adapters\miniqmt\adapter.py: 不合理的跨层级导入
- ⚠️ data\adapters\miniqmt\adapter.py: 不合理的跨层级导入
- ⚠️ data\adapters\miniqmt\adapter.py: 不合理的跨层级导入
- ⚠️ data\adapters\miniqmt\miniqmt_data_adapter.py: 不合理的跨层级导入
- ⚠️ data\adapters\miniqmt\miniqmt_trade_adapter.py: 不合理的跨层级导入
- ⚠️ data\adapters.backup_20250823_212847\adapter_registry.py: 不合理的跨层级导入
- ⚠️ data\adapters.backup_20250823_212847\base_adapter.py: 不合理的跨层级导入
- ⚠️ data\adapters.backup_20250823_212847\generic_china_data_adapter.py: 不合理的跨层级导入
- ⚠️ data\adapters.backup_20250823_212847\china\adapter.py: 不合理的跨层级导入
- ⚠️ data\adapters.backup_20250823_212847\china\financial_adapter.py: 不合理的跨层级导入
- ⚠️ data\adapters.backup_20250823_212847\china\index_adapter.py: 不合理的跨层级导入
- ⚠️ data\adapters.backup_20250823_212847\china\news_adapter.py: 不合理的跨层级导入
- ⚠️ data\adapters.backup_20250823_212847\china\sentiment_adapter.py: 不合理的跨层级导入
- ⚠️ data\adapters.backup_20250823_212847\crypto\crypto_adapter.py: 不合理的跨层级导入
- ⚠️ data\adapters.backup_20250823_212847\crypto\crypto_adapter.py: 不合理的跨层级导入
- ⚠️ data\adapters.backup_20250823_212847\international\international_stock_adapter.py: 不合理的跨层级导入
- ⚠️ data\adapters.backup_20250823_212847\macro\macro_economic_adapter.py: 不合理的跨层级导入
- ⚠️ data\adapters.backup_20250823_212847\news\news_sentiment_adapter.py: 不合理的跨层级导入
- ⚠️ data\alignment\data_aligner.py: 不合理的跨层级导入
- ⚠️ data\alignment\data_aligner.py: 不合理的跨层级导入
- ⚠️ data\cache\cache_manager.py: 不合理的跨层级导入
- ⚠️ data\cache\cache_manager.py: 不合理的跨层级导入
- ⚠️ data\cache\disk_cache.py: 不合理的跨层级导入
- ⚠️ data\cache\disk_cache.py: 不合理的跨层级导入
- ⚠️ data\cache\enhanced_cache_manager.py: 不合理的跨层级导入
- ⚠️ data\cache\enhanced_cache_strategy.py: 不合理的跨层级导入
- ⚠️ data\cache\multi_level_cache.py: 不合理的跨层级导入
- ⚠️ data\cache\redis_cache_adapter.py: 不合理的跨层级导入
- ⚠️ data\china\adapter.py: 不合理的跨层级导入
- ⚠️ data\china\dragon_board_updater.py: 不合理的跨层级导入
- ⚠️ data\decoders\level2_decoder.py: 不合理的跨层级导入
- ⚠️ data\distributed\cluster_manager.py: 不合理的跨层级导入
- ⚠️ data\distributed\distributed_data_loader.py: 不合理的跨层级导入
- ⚠️ data\distributed\load_balancer.py: 不合理的跨层级导入
- ⚠️ data\distributed\sharding_manager.py: 不合理的跨层级导入
- ⚠️ data\export\data_exporter.py: 不合理的跨层级导入
- ⚠️ data\export\data_exporter.py: 不合理的跨层级导入
- ⚠️ data\governance\enterprise_governance.py: 不合理的跨层级导入
- ⚠️ data\integration\enhanced_data_integration.py: 不合理的跨层级导入
- ⚠️ data\lake\data_lake_manager.py: 不合理的跨层级导入
- ⚠️ data\lake\metadata_manager.py: 不合理的跨层级导入
- ⚠️ data\loader\bond_loader.py: 不合理的跨层级导入
- ⚠️ data\loader\bond_loader.py: 不合理的跨层级导入
- ⚠️ data\loader\commodity_loader.py: 不合理的跨层级导入
- ⚠️ data\loader\commodity_loader.py: 不合理的跨层级导入
- ⚠️ data\loader\crypto_loader.py: 不合理的跨层级导入
- ⚠️ data\loader\enhanced_data_loader.py: 不合理的跨层级导入
- ⚠️ data\loader\enhanced_data_loader.py: 不合理的跨层级导入
- ⚠️ data\loader\financial_loader.py: 不合理的跨层级导入
- ⚠️ data\loader\financial_loader.py: 不合理的跨层级导入
- ⚠️ data\loader\financial_loader.py: 不合理的跨层级导入
- ⚠️ data\loader\forex_loader.py: 不合理的跨层级导入
- ⚠️ data\loader\forex_loader.py: 不合理的跨层级导入
- ⚠️ data\loader\index_loader.py: 不合理的跨层级导入
- ⚠️ data\loader\index_loader.py: 不合理的跨层级导入
- ⚠️ data\loader\macro_loader.py: 不合理的跨层级导入
- ⚠️ data\loader\news_loader.py: 不合理的跨层级导入
- ⚠️ data\loader\options_loader.py: 不合理的跨层级导入
- ⚠️ data\loader\options_loader.py: 不合理的跨层级导入
- ⚠️ data\loader\parallel_loader.py: 不合理的跨层级导入
- ⚠️ data\loader\stock_loader.py: 不合理的跨层级导入
- ⚠️ data\loader\stock_loader.py: 不合理的跨层级导入
- ⚠️ data\monitoring\dashboard.py: 不合理的跨层级导入
- ⚠️ data\monitoring\performance_monitor.py: 不合理的跨层级导入
- ⚠️ data\monitoring\performance_monitor.py: 不合理的跨层级导入
- ⚠️ data\optimization\advanced_optimizer.py: 不合理的跨层级导入
- ⚠️ data\optimization\data_optimizer.py: 不合理的跨层级导入
- ⚠️ data\optimization\data_preloader.py: 不合理的跨层级导入
- ⚠️ data\optimization\performance_monitor.py: 不合理的跨层级导入
- ⚠️ data\optimization\performance_optimizer.py: 不合理的跨层级导入
- ⚠️ data\parallel\enhanced_parallel_loader.py: 不合理的跨层级导入
- ⚠️ data\parallel\parallel_loader.py: 不合理的跨层级导入
- ⚠️ data\preload\preloader.py: 不合理的跨层级导入
- ⚠️ data\processing\data_processor.py: 不合理的跨层级导入
- ⚠️ data\processing\unified_processor.py: 不合理的跨层级导入
- ⚠️ data\quality\advanced_quality_monitor.py: 不合理的跨层级导入
- ⚠️ data\quality\advanced_quality_monitor.py: 不合理的跨层级导入
- ⚠️ data\quality\enhanced_quality_monitor.py: 不合理的跨层级导入
- ⚠️ data\quality\enhanced_quality_monitor_v2.py: 不合理的跨层级导入
- ⚠️ data\repair\data_repairer.py: 不合理的跨层级导入
- ⚠️ data\sources\intelligent_source_manager.py: 不合理的跨层级导入
- ⚠️ data\sync\multi_market_sync.py: 不合理的跨层级导入
- ⚠️ data\version_control\test_version_manager.py: 不合理的跨层级导入
- ⚠️ data\version_control\version_manager.py: 不合理的跨层级导入
- ⚠️ data\version_control\version_manager.py: 不合理的跨层级导入

### GATEWAY 层级依赖 (API网关层)
- **内部导入**: 0 个
- **外部导入**: 17 个
- **跨层级导入**: 0 个

### FEATURES 层级依赖 (特征处理层)
- **内部导入**: 35 个
- **外部导入**: 699 个
- **跨层级导入**: 77 个

**依赖问题**:
- ⚠️ features\api.py: 不合理的跨层级导入
- ⚠️ features\config_integration.py: 不合理的跨层级导入
- ⚠️ features\config_integration.py: 不合理的跨层级导入
- ⚠️ features\feature_engineer.py: 不合理的跨层级导入
- ⚠️ features\feature_importance.py: 不合理的跨层级导入
- ⚠️ features\feature_manager.py: 不合理的跨层级导入
- ⚠️ features\feature_metadata.py: 不合理的跨层级导入
- ⚠️ features\feature_store.py: 不合理的跨层级导入
- ⚠️ features\minimal_feature_main_flow.py: 不合理的跨层级导入
- ⚠️ features\optimized_feature_manager.py: 不合理的跨层级导入
- ⚠️ features\parallel_feature_processor.py: 不合理的跨层级导入
- ⚠️ features\parallel_feature_processor.py: 不合理的跨层级导入
- ⚠️ features\quality_assessor.py: 不合理的跨层级导入
- ⚠️ features\sentiment_analyzer.py: 不合理的跨层级导入
- ⚠️ features\signal_generator.py: 不合理的跨层级导入
- ⚠️ features\signal_generator.py: 不合理的跨层级导入
- ⚠️ features\version_management.py: 不合理的跨层级导入
- ⚠️ features\acceleration\fpga\fpga_order_optimizer.py: 不合理的跨层级导入
- ⚠️ features\acceleration\fpga\fpga_risk_engine.py: 不合理的跨层级导入
- ⚠️ features\acceleration\fpga\fpga_sentiment_analyzer.py: 不合理的跨层级导入
- ⚠️ features\core\config.py: 不合理的跨层级导入
- ⚠️ features\core\engine.py: 不合理的跨层级导入
- ⚠️ features\core\engine.py: 不合理的跨层级导入
- ⚠️ features\core\engine.py: 不合理的跨层级导入
- ⚠️ features\core\engine.py: 不合理的跨层级导入
- ⚠️ features\core\engine.py: 不合理的跨层级导入
- ⚠️ features\core\engine.py: 不合理的跨层级导入
- ⚠️ features\core\manager.py: 不合理的跨层级导入
- ⚠️ features\core\manager.py: 不合理的跨层级导入
- ⚠️ features\distributed\distributed_processor.py: 不合理的跨层级导入
- ⚠️ features\distributed\task_scheduler.py: 不合理的跨层级导入
- ⚠️ features\distributed\worker_manager.py: 不合理的跨层级导入
- ⚠️ features\intelligent\auto_feature_selector.py: 不合理的跨层级导入
- ⚠️ features\intelligent\intelligent_enhancement_manager.py: 不合理的跨层级导入
- ⚠️ features\intelligent\ml_model_integration.py: 不合理的跨层级导入
- ⚠️ features\intelligent\smart_alert_system.py: 不合理的跨层级导入
- ⚠️ features\monitoring\alert_manager.py: 不合理的跨层级导入
- ⚠️ features\monitoring\benchmark_runner.py: 不合理的跨层级导入
- ⚠️ features\monitoring\benchmark_runner.py: 不合理的跨层级导入
- ⚠️ features\monitoring\features_monitor.py: 不合理的跨层级导入
- ⚠️ features\monitoring\metrics_collector.py: 不合理的跨层级导入
- ⚠️ features\monitoring\performance_analyzer.py: 不合理的跨层级导入
- ⚠️ features\performance\performance_optimizer.py: 不合理的跨层级导入
- ⚠️ features\performance\scalability_manager.py: 不合理的跨层级导入
- ⚠️ features\plugins\base_plugin.py: 不合理的跨层级导入
- ⚠️ features\plugins\base_plugin.py: 不合理的跨层级导入
- ⚠️ features\plugins\plugin_loader.py: 不合理的跨层级导入
- ⚠️ features\plugins\plugin_loader.py: 不合理的跨层级导入
- ⚠️ features\plugins\plugin_manager.py: 不合理的跨层级导入
- ⚠️ features\plugins\plugin_manager.py: 不合理的跨层级导入
- ⚠️ features\plugins\plugin_registry.py: 不合理的跨层级导入
- ⚠️ features\plugins\plugin_registry.py: 不合理的跨层级导入
- ⚠️ features\plugins\plugin_validator.py: 不合理的跨层级导入
- ⚠️ features\plugins\plugin_validator.py: 不合理的跨层级导入
- ⚠️ features\processors\base_processor.py: 不合理的跨层级导入
- ⚠️ features\processors\distributed_processor.py: 不合理的跨层级导入
- ⚠️ features\processors\distributed_processor.py: 不合理的跨层级导入
- ⚠️ features\processors\feature_correlation.py: 不合理的跨层级导入
- ⚠️ features\processors\feature_importance.py: 不合理的跨层级导入
- ⚠️ features\processors\feature_processor.py: 不合理的跨层级导入
- ⚠️ features\processors\feature_quality_assessor.py: 不合理的跨层级导入
- ⚠️ features\processors\feature_selector.py: 不合理的跨层级导入
- ⚠️ features\processors\feature_stability.py: 不合理的跨层级导入
- ⚠️ features\processors\feature_standardizer.py: 不合理的跨层级导入
- ⚠️ features\processors\general_processor.py: 不合理的跨层级导入
- ⚠️ features\processors\general_processor.py: 不合理的跨层级导入
- ⚠️ features\processors\advanced\advanced_feature_processor.py: 不合理的跨层级导入
- ⚠️ features\processors\distributed\distributed_feature_processor.py: 不合理的跨层级导入
- ⚠️ features\processors\gpu\gpu_technical_processor.py: 不合理的跨层级导入
- ⚠️ features\processors\gpu\multi_gpu_processor.py: 不合理的跨层级导入
- ⚠️ features\processors\technical\technical_processor.py: 不合理的跨层级导入
- ⚠️ features\processors\technical\technical_processor.py: 不合理的跨层级导入
- ⚠️ features\sentiment\sentiment_analyzer.py: 不合理的跨层级导入
- ⚠️ features\sentiment\sentiment_analyzer.py: 不合理的跨层级导入
- ⚠️ features\utils\feature_metadata.py: 不合理的跨层级导入
- ⚠️ features\utils\selector.py: 不合理的跨层级导入

### ML 层级依赖 (模型推理层)
- **内部导入**: 0 个
- **外部导入**: 265 个
- **跨层级导入**: 12 个

**依赖问题**:
- ⚠️ ml\models\ab_testing.py: 不合理的跨层级导入
- ⚠️ ml\models\api.py: 不合理的跨层级导入
- ⚠️ ml\models\api.py: 不合理的跨层级导入
- ⚠️ ml\models\api.py: 不合理的跨层级导入
- ⚠️ ml\models\api.py: 不合理的跨层级导入
- ⚠️ ml\models\deep_learning_models.py: 不合理的跨层级导入
- ⚠️ ml\models\deep_learning_models.py: 不合理的跨层级导入
- ⚠️ ml\models\inference\batch_inference_processor.py: 不合理的跨层级导入
- ⚠️ ml\models\inference\gpu_inference_engine.py: 不合理的跨层级导入
- ⚠️ ml\models\inference\inference_cache.py: 不合理的跨层级导入
- ⚠️ ml\models\inference\inference_manager.py: 不合理的跨层级导入
- ⚠️ ml\models\inference\model_loader.py: 不合理的跨层级导入

### BACKTEST 层级依赖 (策略决策层)
- **内部导入**: 1 个
- **外部导入**: 237 个
- **跨层级导入**: 6 个

**依赖问题**:
- ⚠️ backtest\analyzer.py: 不合理的跨层级导入
- ⚠️ backtest\parameter_optimizer.py: 不合理的跨层级导入
- ⚠️ backtest\visualizer.py: 不合理的跨层级导入
- ⚠️ backtest\utils\backtest_utils.py: 不合理的跨层级导入

### RISK 层级依赖 (风控合规层)
- **内部导入**: 0 个
- **外部导入**: 72 个
- **跨层级导入**: 0 个

### TRADING 层级依赖 (交易执行层)
- **内部导入**: 28 个
- **外部导入**: 570 个
- **跨层级导入**: 110 个

**依赖问题**:
- ⚠️ trading\backtester.py: 不合理的跨层级导入
- ⚠️ trading\trading_engine.py: 不合理的跨层级导入
- ⚠️ trading\trading_engine.py: 不合理的跨层级导入
- ⚠️ trading\trading_engine_with_distributed.py: 不合理的跨层级导入
- ⚠️ trading\trading_engine_with_distributed.py: 不合理的跨层级导入
- ⚠️ trading\trading_engine_with_distributed.py: 不合理的跨层级导入
- ⚠️ trading\advanced_analysis\clustering_engine.py: 不合理的跨层级导入
- ⚠️ trading\advanced_analysis\portfolio_optimizer.py: 不合理的跨层级导入
- ⚠️ trading\advanced_analysis\relationship_network.py: 不合理的跨层级导入
- ⚠️ trading\advanced_analysis\similarity_analyzer.py: 不合理的跨层级导入
- ⚠️ trading\distributed\distributed_trading_node.py: 不合理的跨层级导入
- ⚠️ trading\distributed\distributed_trading_node.py: 不合理的跨层级导入
- ⚠️ trading\distributed\distributed_trading_node.py: 不合理的跨层级导入
- ⚠️ trading\distributed\distributed_trading_node.py: 不合理的跨层级导入
- ⚠️ trading\distributed\intelligent_order_router.py: 不合理的跨层级导入
- ⚠️ trading\distributed\intelligent_order_router.py: 不合理的跨层级导入
- ⚠️ trading\distributed\intelligent_order_router.py: 不合理的跨层级导入
- ⚠️ trading\execution\execution_engine.py: 不合理的跨层级导入
- ⚠️ trading\execution\execution_engine.py: 不合理的跨层级导入
- ⚠️ trading\execution\order_router.py: 不合理的跨层级导入
- ⚠️ trading\execution\order_router.py: 不合理的跨层级导入
- ⚠️ trading\execution\reporting.py: 不合理的跨层级导入
- ⚠️ trading\ml_integration\auto_optimizer.py: 不合理的跨层级导入
- ⚠️ trading\ml_integration\hyperparameter_tuner.py: 不合理的跨层级导入
- ⚠️ trading\ml_integration\multi_objective_optimizer.py: 不合理的跨层级导入
- ⚠️ trading\ml_integration\optimization_engine.py: 不合理的跨层级导入
- ⚠️ trading\ml_integration\performance_predictor.py: 不合理的跨层级导入
- ⚠️ trading\ml_integration\recommendation_engine.py: 不合理的跨层级导入
- ⚠️ trading\ml_integration\similarity_analyzer.py: 不合理的跨层级导入
- ⚠️ trading\ml_integration\strategy_recommender.py: 不合理的跨层级导入
- ⚠️ trading\risk\risk_compliance_engine.py: 不合理的跨层级导入
- ⚠️ trading\risk\risk_compliance_engine.py: 不合理的跨层级导入
- ⚠️ trading\risk\risk_controller.py: 不合理的跨层级导入
- ⚠️ trading\risk\risk_controller.py: 不合理的跨层级导入
- ⚠️ trading\risk\china\circuit_breaker.py: 不合理的跨层级导入
- ⚠️ trading\risk\china\market_rule_checker.py: 不合理的跨层级导入
- ⚠️ trading\risk\china\position_limits.py: 不合理的跨层级导入
- ⚠️ trading\risk\china\star_market.py: 不合理的跨层级导入
- ⚠️ trading\strategies\base_strategy.py: 不合理的跨层级导入
- ⚠️ trading\strategies\enhanced.py: 不合理的跨层级导入
- ⚠️ trading\strategies\factory.py: 不合理的跨层级导入
- ⚠️ trading\strategies\reinforcement_learning.py: 不合理的跨层级导入
- ⚠️ trading\strategies\china\base_strategy.py: 不合理的跨层级导入
- ⚠️ trading\strategies\china\dragon_tiger.py: 不合理的跨层级导入
- ⚠️ trading\strategies\china\limit_up.py: 不合理的跨层级导入
- ⚠️ trading\strategies\china\margin.py: 不合理的跨层级导入
- ⚠️ trading\strategies\china\ml_strategy.py: 不合理的跨层级导入
- ⚠️ trading\strategies\china\st.py: 不合理的跨层级导入
- ⚠️ trading\strategies\optimization\performance_tuner.py: 不合理的跨层级导入
- ⚠️ trading\strategies\optimization\performance_tuner.py: 不合理的跨层级导入
- ⚠️ trading\strategies\optimization\performance_tuner.py: 不合理的跨层级导入
- ⚠️ trading\strategy_workspace\analyzer.py: 不合理的跨层级导入
- ⚠️ trading\strategy_workspace\optimizer.py: 不合理的跨层级导入
- ⚠️ trading\strategy_workspace\simulator.py: 不合理的跨层级导入
- ⚠️ trading\strategy_workspace\store.py: 不合理的跨层级导入
- ⚠️ trading\strategy_workspace\strategy_generator.py: 不合理的跨层级导入
- ⚠️ trading\strategy_workspace\web_interface.py: 不合理的跨层级导入

### ENGINE 层级依赖 (监控反馈层)
- **内部导入**: 34 个
- **外部导入**: 431 个
- **跨层级导入**: 34 个

**依赖问题**:
- ⚠️ engine\stress_test.py: 不合理的跨层级导入
- ⚠️ engine\stress_test.py: 不合理的跨层级导入
- ⚠️ engine\optimization\buffer_optimizer.py: 不合理的跨层级导入
- ⚠️ engine\optimization\buffer_optimizer.py: 不合理的跨层级导入
- ⚠️ engine\optimization\dispatcher_optimizer.py: 不合理的跨层级导入
- ⚠️ engine\optimization\dispatcher_optimizer.py: 不合理的跨层级导入
- ⚠️ engine\optimization\level2_optimizer.py: 不合理的跨层级导入
- ⚠️ engine\optimization\level2_optimizer.py: 不合理的跨层级导入
- ⚠️ engine\web\app_factory.py: 不合理的跨层级导入
- ⚠️ engine\web\app_factory.py: 不合理的跨层级导入
- ⚠️ engine\web\app_factory.py: 不合理的跨层级导入
- ⚠️ engine\web\app_factory.py: 不合理的跨层级导入
- ⚠️ engine\web\app_factory.py: 不合理的跨层级导入
- ⚠️ engine\web\app_factory.py: 不合理的跨层级导入
- ⚠️ engine\web\app_factory.py: 不合理的跨层级导入
- ⚠️ engine\web\data_api.py: 不合理的跨层级导入
- ⚠️ engine\web\data_api.py: 不合理的跨层级导入
- ⚠️ engine\web\data_api.py: 不合理的跨层级导入
- ⚠️ engine\web\data_api.py: 不合理的跨层级导入
- ⚠️ engine\web\unified_dashboard.py: 不合理的跨层级导入
- ⚠️ engine\web\unified_dashboard.py: 不合理的跨层级导入
- ⚠️ engine\web\unified_dashboard.py: 不合理的跨层级导入
- ⚠️ engine\web\unified_dashboard.py: 不合理的跨层级导入
- ⚠️ engine\web\websocket_api.py: 不合理的跨层级导入
- ⚠️ engine\web\websocket_api.py: 不合理的跨层级导入
- ⚠️ engine\web\websocket_api.py: 不合理的跨层级导入
- ⚠️ engine\web\websocket_api.py: 不合理的跨层级导入
- ⚠️ engine\web\websocket_api.py: 不合理的跨层级导入
- ⚠️ engine\web\modules\config_module.py: 不合理的跨层级导入
- ⚠️ engine\web\modules\fpga_module.py: 不合理的跨层级导入
- ⚠️ engine\web\modules\fpga_module.py: 不合理的跨层级导入
- ⚠️ engine\web\modules\resource_module.py: 不合理的跨层级导入
- ⚠️ engine\web\modules\resource_module.py: 不合理的跨层级导入
- ⚠️ engine\web\modules\resource_module.py: 不合理的跨层级导入

---

**审计工具**: scripts/smart_architecture_audit.py
**审计标准**: 基于架构设计文档 v5.0
**建议处理**: 按严重程度从高到低修复问题
