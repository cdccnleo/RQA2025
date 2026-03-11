# 数据采集完成后后续业务处理检查计划

## 目标
全面检查今日数据采集完成后，各业务模块是否正确响应 DATA_COLLECTION_COMPLETED 事件并进行后续处理。

## 背景
- 数据源 `akshare_stock_a` 今日已完成数据采集
- 需要验证数据采集完成后，后续业务流程是否正常触发
- 基于业务流程驱动架构，各模块应订阅 DATA_COLLECTION_COMPLETED 事件

## 检查范围

### 1. 特征工程模块检查
**检查内容**:
- 是否正确订阅 DATA_COLLECTION_COMPLETED 事件
- 事件处理器是否被正确初始化
- 收到事件后是否创建特征提取任务
- 任务是否正确提交到统一调度器
- 特征提取任务是否正确执行
- 特征数据是否正确保存

**相关文件**:
- `src/features/core/event_listeners.py`
- `src/features/core/feature_task_service.py`
- `src/gateway/web/api.py` (事件监听器初始化)

### 2. 数据质量监控检查
**检查内容**:
- 是否正确订阅 DATA_COLLECTION_COMPLETED 事件
- 收到事件后是否执行数据质量检查
- 是否执行数据验证
- 是否生成质量报告
- 质量监控日志是否正常输出

**相关文件**:
- `src/data/quality/data_collection_event_handler.py`
- `src/data/quality/data_quality_monitor.py`

### 3. ML模块检查
**检查内容**:
- 是否正确订阅 DATA_COLLECTION_COMPLETED 事件
- 收到事件后是否更新特征工程
- 是否触发模型训练
- 模型训练条件判断是否正确（数据量>100）
- ML模块日志是否正常输出

**相关文件**:
- `src/ml/core/data_collection_event_handler.py`

### 4. WebSocket监控检查
**检查内容**:
- 是否正确订阅 DATA_COLLECTED 事件
- 是否通过 WebSocket 推送实时状态
- 前端监控页面是否正确显示状态更新
- WebSocket 连接是否正常

**相关文件**:
- `src/gateway/web/websocket_routes.py`
- `web-static/feature-engineering-monitor.html`

### 5. 事件发布检查
**检查内容**:
- DATA_COLLECTION_COMPLETED 事件是否正确发布
- 事件数据格式是否正确（source_id, task_id, result, source_config）
- 事件发布时机是否正确（任务完成后）

**相关文件**:
- `src/gateway/web/data_collection_scheduler_manager.py`
- `src/gateway/web/data_collectors.py`

## 实施步骤

### 步骤 1: 检查事件发布机制
1. 检查 DATA_COLLECTION_COMPLETED 事件发布代码
2. 验证事件数据格式是否包含所有必要字段
3. 确认事件发布时机（任务完成后）

### 步骤 2: 检查事件订阅机制
1. 检查各模块是否正确订阅事件
2. 验证事件处理器初始化
3. 确认事件总线连接正常

### 步骤 3: 检查特征工程模块
1. 验证 FeatureEventListeners 是否正确初始化
2. 检查事件处理日志
3. 确认特征提取任务是否创建
4. 验证任务是否正确提交到调度器

### 步骤 4: 检查数据质量监控
1. 验证数据质量事件处理器是否正确初始化
2. 检查质量检查日志
3. 确认质量报告生成

### 步骤 5: 检查ML模块
1. 验证ML事件处理器是否正确初始化
2. 检查特征工程更新日志
3. 确认模型训练触发逻辑

### 步骤 6: 检查WebSocket监控
1. 验证 WebSocket 连接
2. 检查实时状态推送
3. 确认前端监控页面显示

### 步骤 7: 端到端验证
1. 触发数据采集完成事件（手动或等待自动采集）
2. 观察完整流程：事件发布 → 各模块响应 → 后续处理
3. 验证整个链路是否正常

## 预期结果
- DATA_COLLECTION_COMPLETED 事件正确发布
- 特征工程模块正确响应并创建特征提取任务
- 数据质量监控正确执行质量检查
- ML模块正确更新特征工程并触发模型训练
- WebSocket 正确推送实时状态
- 前端监控页面正确显示状态更新

## 时间估计
- 步骤 1-2: 20分钟
- 步骤 3-5: 40分钟
- 步骤 6-7: 30分钟
- 总计: 约1.5小时
