# 特征工程任务ID增强计划

## 目标
修改特征工程监控页面(feature-engineering-monitor.html)的创建特征提取任务功能，根据任务模式在任务ID中添加单只股票symbol或股票池表示，以便可以根据任务ID识别特征提取任务。

## 当前问题
- 当前任务ID格式：`feature_task_时间戳`
- 无法从任务ID识别是单只股票还是股票池任务
- 无法从任务ID识别具体的股票代码

## 预期改进

### 任务ID格式

#### 单只股票模式
- **格式**: `feature_task_single_{symbol}_{时间戳}`
- **示例**: `feature_task_single_002837_1771735627`

#### 股票池模式
- **格式**: `feature_task_pool_{时间戳}`
- **示例**: `feature_task_pool_1771735627`

## 检查内容

### 1. 前端代码检查
- **文件**: `web-static/feature-engineering-monitor.html`
- **检查项**:
  - 任务创建表单的股票选择逻辑
  - 任务ID生成逻辑
  - 任务模式识别（单只/股票池）

### 2. 后端API检查
- **文件**: `src/gateway/web/feature_engineering_routes.py`
- **检查项**:
  - 任务创建API端点
  - 任务ID生成逻辑
  - 任务参数解析

### 3. 数据存储检查
- **文件**: `src/gateway/web/feature_task_persistence.py`
- **检查项**:
  - 任务ID存储逻辑
  - 任务配置存储

## 实施步骤

### 步骤1: 检查当前任务ID生成逻辑
1. 查看前端创建任务的代码
2. 查看后端接收任务的代码
3. 确定任务ID生成位置

### 步骤2: 修改前端代码
1. 在任务创建时识别任务模式（单只/股票池）
2. 根据模式生成带标识的任务ID
3. 如果是单只股票，将symbol加入任务ID

### 步骤3: 修改后端代码（如需要）
1. 确保后端能正确接收带标识的任务ID
2. 确保任务存储和查询正常

### 步骤4: 验证
1. 创建单只股票任务，验证任务ID格式
2. 创建股票池任务，验证任务ID格式
3. 验证任务可以正常执行和查询

## 文件清单
- `web-static/feature-engineering-monitor.html`
- `src/gateway/web/feature_engineering_routes.py`
- `src/gateway/web/feature_task_persistence.py`
