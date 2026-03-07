# 计划：检查web-static目录监控页面所有涉及调度器的仪表盘是否已集成统一调度器

## 目标
检查web-static目录下的所有监控页面，确认它们是否正确集成了统一调度器（Unified Scheduler），并识别需要更新的页面。

## 背景
统一调度器已完成三个阶段的实现（立即、短期、中期行动阶段），具备以下核心能力：
- 38+种任务类型
- 数据库持久化
- 多通道告警
- 事件总线集成
- 性能优化（优先级队列、批量处理、任务缓存）
- Prometheus指标
- 安全加固（加密、RBAC、API密钥）

统一调度器的API端点位于 `/api/v1/scheduler/`，而旧的调度器可能使用各自的端点（如 `/ml/training/scheduler/`、`/features/engineering/scheduler/`）。

## 检查范围

### 1. 已识别涉及调度器的监控页面

| 页面文件 | 调度器类型 | 当前API端点 | 需要更新 |
|----------|------------|-------------|----------|
| model-training-monitor.html | 模型训练调度器 | /ml/training/scheduler/* | 待检查 |
| feature-engineering-monitor.html | 特征工程调度器 | /features/engineering/scheduler/* | 待检查 |
| data-sources-config.html | 数据采集调度器 | 待确认 | 待检查 |
| data-collection-monitor.html | 数据采集监控 | 待确认 | 待检查 |
| dashboard.html | 主仪表板 | 待确认 | 待检查 |

### 2. 检查步骤

#### 步骤1：分析每个页面的调度器集成情况
- [ ] 检查model-training-monitor.html
  - 查找所有scheduler相关的API调用
  - 确认当前使用的端点
  - 检查是否已适配统一调度器格式
  
- [ ] 检查feature-engineering-monitor.html
  - 查找所有scheduler相关的API调用
  - 确认当前使用的端点
  - 检查是否已适配统一调度器格式
  
- [ ] 检查data-sources-config.html
  - 查找scheduler相关代码
  - 确认数据采集调度器的集成方式
  
- [ ] 检查data-collection-monitor.html
  - 查找scheduler相关代码
  - 确认监控页面是否显示调度器状态
  
- [ ] 检查dashboard.html
  - 查找是否有统一调度器的总览
  - 确认是否显示调度器状态卡片

#### 步骤2：识别需要更新的内容

对于每个页面，检查以下内容：

1. **API端点更新**
   - 旧端点：`/ml/training/scheduler/status`
   - 新端点：`/api/v1/scheduler/status`

2. **数据格式适配**
   - 统一调度器使用 `queue_sizes` 字典
   - 需要适配新的响应格式

3. **功能完整性**
   - 是否支持启动/停止调度器
   - 是否显示任务统计
   - 是否显示工作进程状态

#### 步骤3：生成更新建议

对于需要更新的页面，提供：
- 具体的代码修改建议
- API端点映射表
- 数据格式转换逻辑

## 预期输出

1. **检查报告**：每个页面的调度器集成状态
2. **更新建议**：需要修改的页面列表和具体修改方案
3. **API映射表**：旧端点到新端点的映射

## 执行步骤

1. 读取并分析每个涉及调度器的监控页面
2. 记录当前使用的API端点和数据格式
3. 对比统一调度器的API规范
4. 生成检查报告和更新建议
