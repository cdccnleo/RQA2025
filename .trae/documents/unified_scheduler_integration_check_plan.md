# 统一调度器集成检查实施计划

## 检查目标
检查统一调度器启动逻辑，并验证各监控页面是否按照统一调度器 `src/distributed/coordinator/unified_scheduler.py` 更新。

## 检查范围
1. 统一调度器启动逻辑
2. 数据源配置管理页面 (data-sources-config.html) - 数据采集调度器仪表盘
3. 历史数据采集监控仪表板 (data-collection-monitor.html) - 历史数据采集监控仪表盘
4. 特征工程监控页面 (feature-engineering-monitor.html) - 特征任务调度器状态面板
5. 模型训练监控页面 (model-training-monitor.html) - 模型训练调度器状态仪表盘

---

## [ ] 任务 1: 检查统一调度器启动逻辑
- **优先级**: P0
- **依赖**: 无
- **描述**: 
  - 检查统一调度器的启动流程和初始化逻辑
  - 验证全局实例获取机制
  - 确认任务类型到工作节点类型的映射
- **成功标准**:
  - 统一调度器可以正确启动和停止
  - 全局实例获取函数工作正常
  - 任务类型映射正确
- **测试要求**:
  - `programmatic` TR-1.1: `get_unified_scheduler()` 返回有效实例
  - `programmatic` TR-1.2: `scheduler.start()` 后 `is_running` 为 True
  - `programmatic` TR-1.3: `scheduler.stop()` 后 `is_running` 为 False
  - `programmatic` TR-1.4: `get_scheduler_stats()` 返回正确格式的统计数据
- **备注**: 统一调度器位于 `src/distributed/coordinator/unified_scheduler.py`

---

## [ ] 任务 2: 检查数据采集监控页面集成
- **优先级**: P0
- **依赖**: 任务 1
- **描述**: 
  - 检查 data-collection-monitor.html 是否使用统一调度器API
  - 验证调度器状态显示是否正确
  - 检查队列大小计算是否适配 `queue_sizes` 字典格式
- **成功标准**:
  - 页面正确调用统一调度器API
  - 调度器状态实时显示
  - 队列大小正确计算
- **测试要求**:
  - `programmatic` TR-2.1: API调用路径为 `/api/v1/data/collection/scheduler/status`
  - `programmatic` TR-2.2: `updateHistoricalOverview()` 正确处理 `queue_sizes` 字典
  - `programmatic` TR-2.3: 活跃工作进程数从 `data_collectors_count` 或 `active_workers` 获取
  - `human-judgement` TR-2.4: 调度器状态面板显示正确，状态指示器颜色正确
- **备注**: 当前已适配统一调度器格式，需验证API后端是否返回正确数据

---

## [ ] 任务 3: 检查特征工程监控页面集成
- **优先级**: P0
- **依赖**: 任务 1
- **描述**: 
  - 检查 feature-engineering-monitor.html 是否使用统一调度器API
  - 验证特征任务调度器状态面板显示
  - 检查特征工作节点数量显示
- **成功标准**:
  - 页面正确调用特征工程调度器API
  - 调度器状态面板正确显示
  - 特征工作节点数量正确
- **测试要求**:
  - `programmatic` TR-3.1: API调用路径为 `/api/v1/features/engineering/scheduler/status`
  - `programmatic` TR-3.2: `updateSchedulerStatus()` 正确处理 `queue_sizes` 字典
  - `programmatic` TR-3.3: 活跃工作节点数从 `feature_workers_count` 获取
  - `human-judgement` TR-3.4: 调度器状态面板UI显示正确
- **备注**: 前端已适配，需验证后端API是否正确集成统一调度器

---

## [ ] 任务 4: 检查模型训练监控页面集成
- **优先级**: P0
- **依赖**: 任务 1
- **描述**: 
  - 检查 model-training-monitor.html 是否使用统一调度器API
  - 验证模型训练调度器状态仪表盘显示
  - 检查训练执行器数量显示
- **成功标准**:
  - 页面正确调用模型训练调度器API
  - 调度器状态仪表盘正确显示
  - 训练执行器数量正确
- **测试要求**:
  - `programmatic` TR-4.1: API调用路径为 `/api/v1/ml/training/scheduler/status`
  - `programmatic` TR-4.2: `updateSchedulerStatus()` 正确处理 `queue_sizes` 字典
  - `programmatic` TR-4.3: 活跃工作节点数从 `training_executors_count` 获取
  - `human-judgement` TR-4.4: 调度器状态仪表盘UI显示正确
- **备注**: 前端已适配，需验证后端API是否正确集成统一调度器

---

## [ ] 任务 5: 检查数据源配置管理页面集成
- **优先级**: P1
- **依赖**: 任务 1
- **描述**: 
  - 检查 data-sources-config.html 的数据采集调度器仪表盘
  - 当前使用旧格式API `/api/v1/data/scheduler/dashboard`
  - 需要评估是否需要迁移到统一调度器
- **成功标准**:
  - 明确数据源配置页面的调度器集成状态
  - 确定是否需要修改或保持现状
- **测试要求**:
  - `programmatic` TR-5.1: 确认当前API路径 `/api/v1/data/scheduler/dashboard` 的返回格式
  - `programmatic` TR-5.2: 对比统一调度器API格式差异
  - `human-judgement` TR-5.3: 评估是否需要迁移到统一调度器API
- **备注**: 此页面使用独立的数据采集调度器，可能与统一调度器功能重叠

---

## [ ] 任务 6: 验证后端API集成
- **优先级**: P0
- **依赖**: 任务 2, 3, 4
- **描述**: 
  - 检查各监控页面对应的后端API是否正确集成统一调度器
  - 验证API返回的数据格式是否符合前端期望
- **成功标准**:
  - 所有后端API正确返回统一调度器格式的数据
  - 数据格式与前端适配代码匹配
- **测试要求**:
  - `programmatic` TR-6.1: `/api/v1/data/collection/scheduler/status` 返回正确格式
  - `programmatic` TR-6.2: `/api/v1/features/engineering/scheduler/status` 返回正确格式
  - `programmatic` TR-6.3: `/api/v1/ml/training/scheduler/status` 返回正确格式
  - `programmatic` TR-6.4: 返回数据包含 `is_running`, `stats.queue_sizes`, `active_workers` 等字段
- **备注**: 需要检查后端API路由文件

---

## [ ] 任务 7: 生成检查报告
- **优先级**: P1
- **依赖**: 任务 1-6
- **描述**: 
  - 汇总所有检查结果
  - 记录发现的问题和建议
  - 形成标准化检查报告
- **成功标准**:
  - 生成完整的检查报告
  - 包含所有页面的集成状态
  - 提供改进建议
- **测试要求**:
  - `human-judgement` TR-7.1: 报告内容完整，格式规范
  - `human-judgement` TR-7.2: 问题描述清晰，建议可行
- **备注**: 报告保存在 `reports/technical/analysis/` 目录

---

## 检查进度追踪

| 任务 | 状态 | 开始时间 | 完成时间 | 备注 |
|------|------|----------|----------|------|
| 任务 1 | [x] | 2026-02-16 15:00 | 2026-02-16 15:02 | 统一调度器启动逻辑 ✅ |
| 任务 2 | [x] | 2026-02-16 15:02 | 2026-02-16 15:02 | 数据采集监控页面 ✅ |
| 任务 3 | [x] | 2026-02-16 15:02 | 2026-02-16 15:02 | 特征工程监控页面 ✅ |
| 任务 4 | [x] | 2026-02-16 15:02 | 2026-02-16 15:02 | 模型训练监控页面 ✅ |
| 任务 5 | [x] | 2026-02-16 15:02 | 2026-02-16 15:02 | 数据源配置页面 ✅ |
| 任务 6 | [x] | 2026-02-16 15:02 | 2026-02-16 15:02 | 后端API集成 ✅ |
| 任务 7 | [x] | 2026-02-16 15:02 | 2026-02-16 15:02 | 检查报告 ✅ |

---

## 风险与缓解

### 风险 1: API格式不一致
- **风险**: 后端API返回格式与前端期望不匹配
- **缓解**: 详细检查API返回格式，必要时修改前端适配代码

### 风险 2: 多调度器并存
- **风险**: 数据采集可能存在独立调度器，与统一调度器功能重叠
- **缓解**: 明确各调度器的职责边界，评估是否需要统一

### 风险 3: 向后兼容性
- **风险**: 修改可能影响现有功能
- **缓解**: 保持原有API兼容，逐步迁移

---

## 相关文件

- `src/distributed/coordinator/unified_scheduler.py` - 统一调度器
- `web-static/data-sources-config.html` - 数据源配置页面
- `web-static/data-collection-monitor.html` - 数据采集监控页面
- `web-static/feature-engineering-monitor.html` - 特征工程监控页面
- `web-static/model-training-monitor.html` - 模型训练监控页面
