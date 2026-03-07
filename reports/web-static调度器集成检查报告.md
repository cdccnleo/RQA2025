# Web-Static 目录监控页面调度器集成检查报告

**检查日期**: 2026-03-06  
**报告版本**: v1.0  
**检查范围**: web-static目录下所有涉及调度器的监控页面  
**统一调度器版本**: v2.0

---

## 执行摘要

本次检查针对web-static目录下的5个监控页面进行了调度器集成情况分析，识别出不同页面使用的调度器API端点差异，并提供迁移建议。

### 检查结果概览

| 页面文件 | 当前状态 | 使用端点 | 需要更新 | 优先级 |
|----------|----------|----------|----------|--------|
| model-training-monitor.html | ⚠️ 使用旧端点 | /ml/training/scheduler/* | ✅ 是 | 高 |
| feature-engineering-monitor.html | ⚠️ 使用旧端点 | /features/engineering/scheduler/* | ✅ 是 | 高 |
| data-sources-config.html | ✅ 已集成 | /api/v1/data/scheduler/* | ❌ 否 | - |
| data-collection-monitor.html | ✅ 已集成 | /api/v1/data/scheduler/* | ❌ 否 | - |
| dashboard.html | ⚠️ 无调度器显示 | - | ⚠️ 建议添加 | 中 |

---

## 详细检查结果

### 1. model-training-monitor.html

**文件路径**: `web-static/model-training-monitor.html`

#### 当前集成状态
- **调度器类型**: 模型训练调度器
- **集成状态**: ⚠️ 使用旧版独立调度器端点
- **数据格式适配**: 已部分适配统一调度器格式（注释显示"适配统一调度器格式"）

#### 使用的API端点

| 功能 | 当前端点 | 统一调度器端点 | 状态 |
|------|----------|----------------|------|
| 获取状态 | `/ml/training/scheduler/status` | `/api/v1/scheduler/status` | ⚠️ 需更新 |
| 启动调度器 | `/ml/training/scheduler/start` | `/api/v1/scheduler/start` | ⚠️ 需更新 |
| 停止调度器 | `/ml/training/scheduler/stop` | `/api/v1/scheduler/stop` | ⚠️ 需更新 |

#### 代码片段分析
```javascript
// 当前实现（需要更新）
const response = await fetch(getApiBaseUrl('/ml/training/scheduler/status'));

// 启动/停止使用独立端点
const response = await fetch(getApiBaseUrl('/ml/training/scheduler/start'), {...});
const response = await fetch(getApiBaseUrl('/ml/training/scheduler/stop'), {...});
```

#### 数据格式适配
页面代码中已有适配统一调度器格式的注释：
```javascript
// 更新统计数据（适配统一调度器格式）
// 统一调度器使用 queue_sizes 字典，需要计算总数
const queueSizes = stats.queue_sizes || {};
const totalQueueSize = Object.values(queueSizes).reduce((sum, size) => sum + size, 0);
```

#### 更新建议
1. 将API端点从 `/ml/training/scheduler/*` 更新为 `/api/v1/scheduler/*`
2. 添加任务类型过滤参数（如 `?task_type=model_training`）以显示模型训练相关任务
3. 更新启动/停止调度器的API调用

---

### 2. feature-engineering-monitor.html

**文件路径**: `web-static/feature-engineering-monitor.html`

#### 当前集成状态
- **调度器类型**: 特征工程调度器
- **集成状态**: ⚠️ 使用旧版独立调度器端点
- **数据格式适配**: 已部分适配统一调度器格式

#### 使用的API端点

| 功能 | 当前端点 | 统一调度器端点 | 状态 |
|------|----------|----------------|------|
| 获取状态 | `/features/engineering/scheduler/status` | `/api/v1/scheduler/status` | ⚠️ 需更新 |
| 启动调度器 | `/features/engineering/scheduler/start` | `/api/v1/scheduler/start` | ⚠️ 需更新 |
| 停止调度器 | `/features/engineering/scheduler/stop` | `/api/v1/scheduler/stop` | ⚠️ 需更新 |

#### 代码片段分析
```javascript
// 当前实现（需要更新）
const response = await fetch(getApiBaseUrl('/features/engineering/scheduler/status'));

// 启动/停止使用独立端点
const response = await fetch(getApiBaseUrl('/features/engineering/scheduler/start'), {...});
const response = await fetch(getApiBaseUrl('/features/engineering/scheduler/stop'), {...});
```

#### 更新建议
1. 将API端点从 `/features/engineering/scheduler/*` 更新为 `/api/v1/scheduler/*`
2. 添加任务类型过滤参数（如 `?task_type=feature_extraction`）以显示特征工程相关任务
3. 更新启动/停止调度器的API调用

---

### 3. data-sources-config.html

**文件路径**: `web-static/data-sources-config.html`

#### 当前集成状态
- **调度器类型**: 数据采集调度器
- **集成状态**: ✅ 已集成统一调度器
- **使用端点**: `/api/v1/data/scheduler/*`

#### 使用的API端点

| 功能 | 当前端点 | 状态 |
|------|----------|------|
| 仪表板数据 | `/api/v1/data/scheduler/dashboard` | ✅ 已使用统一调度器 |
| 自动采集状态 | `/api/v1/data/scheduler/auto-collection/status` | ✅ 已使用统一调度器 |
| 启动自动采集 | `/api/v1/data/scheduler/auto-collection/start` | ✅ 已使用统一调度器 |
| 停止自动采集 | `/api/v1/data/scheduler/auto-collection/stop` | ✅ 已使用统一调度器 |
| 调度器控制 | `/api/v1/data/scheduler/control` | ✅ 已使用统一调度器 |

#### 代码片段分析
```javascript
// 已使用统一调度器端点
const response = await fetch('/api/v1/data/scheduler/dashboard');
const response = await fetch('/api/v1/data/scheduler/auto-collection/status');
const response = await fetch('/api/v1/data/scheduler/auto-collection/start', {...});
const response = await fetch('/api/v1/data/scheduler/auto-collection/stop', {...});
```

#### 评估结论
该页面已正确集成统一调度器，无需更新。

---

### 4. data-collection-monitor.html

**文件路径**: `web-static/data-collection-monitor.html`

#### 当前集成状态
- **调度器类型**: 数据采集监控
- **集成状态**: ✅ 已集成统一调度器
- **使用端点**: `/api/v1/data/scheduler/*`

#### 使用的API端点

| 功能 | 当前端点 | 状态 |
|------|----------|------|
| 仪表板数据 | `/api/v1/data/scheduler/dashboard` | ✅ 已使用统一调度器 |
| 自动采集状态 | `/api/v1/data/scheduler/auto-collection/status` | ✅ 已使用统一调度器 |
| 启动自动采集 | `/api/v1/data/scheduler/auto-collection/start` | ✅ 已使用统一调度器 |
| 停止自动采集 | `/api/v1/data/scheduler/auto-collection/stop` | ✅ 已使用统一调度器 |
| 运行中任务 | `/api/v1/data/scheduler/tasks/running` | ✅ 已使用统一调度器 |
| 已完成任务 | `/api/v1/data/scheduler/tasks/completed` | ✅ 已使用统一调度器 |
| 任务详情 | `/api/v1/data/scheduler/tasks/{taskId}` | ✅ 已使用统一调度器 |
| 暂停任务 | `/api/v1/data/scheduler/tasks/{taskId}/pause` | ✅ 已使用统一调度器 |
| 恢复任务 | `/api/v1/data/scheduler/tasks/{taskId}/resume` | ✅ 已使用统一调度器 |
| 取消任务 | `/api/v1/data/scheduler/tasks/{taskId}/cancel` | ✅ 已使用统一调度器 |
| 重试任务 | `/api/v1/data/scheduler/tasks/{taskId}/retry` | ✅ 已使用统一调度器 |
| 更新优先级 | `/api/v1/data/scheduler/tasks/{taskId}/priority` | ✅ 已使用统一调度器 |

#### 功能完整性
该页面实现了完整的任务管理功能：
- ✅ 调度器状态显示
- ✅ 任务列表展示（运行中、已完成）
- ✅ 任务操作（暂停、恢复、取消、重试）
- ✅ 优先级调整
- ✅ 自动采集控制

#### 评估结论
该页面已完整集成统一调度器，功能完善，无需更新。

---

### 5. dashboard.html

**文件路径**: `web-static/dashboard.html`

#### 当前集成状态
- **调度器类型**: 主仪表板
- **集成状态**: ⚠️ 未显示调度器信息
- **使用端点**: 无

#### 评估结论
主仪表板页面当前未集成调度器状态显示。建议添加统一调度器的总览卡片，显示：
- 调度器运行状态
- 当前任务统计（运行中、待处理、已完成）
- 工作进程状态
- 快捷链接到详细监控页面

---

## API端点映射表

### 旧端点 → 新端点映射

| 功能 | 旧端点（ML训练） | 旧端点（特征工程） | 新端点（统一调度器） |
|------|------------------|-------------------|---------------------|
| 获取状态 | `/ml/training/scheduler/status` | `/features/engineering/scheduler/status` | `/api/v1/scheduler/status` |
| 启动调度器 | `/ml/training/scheduler/start` | `/features/engineering/scheduler/start` | `/api/v1/scheduler/start` |
| 停止调度器 | `/ml/training/scheduler/stop` | `/features/engineering/scheduler/stop` | `/api/v1/scheduler/stop` |

### 统一调度器完整API列表

| 功能 | 端点 | 方法 |
|------|------|------|
| 获取仪表板 | `/api/v1/scheduler/dashboard` | GET |
| 获取状态 | `/api/v1/scheduler/status` | GET |
| 获取指标 | `/api/v1/scheduler/metrics` | GET |
| 启动调度器 | `/api/v1/scheduler/start` | POST |
| 停止调度器 | `/api/v1/scheduler/stop` | POST |
| 获取配置 | `/api/v1/scheduler/config` | GET |
| 更新配置 | `/api/v1/scheduler/config` | POST |
| 提交任务 | `/api/v1/scheduler/tasks` | POST |
| 获取运行中任务 | `/api/v1/scheduler/tasks/running` | GET |
| 获取已完成任务 | `/api/v1/scheduler/tasks/completed` | GET |
| 获取任务详情 | `/api/v1/scheduler/tasks/{taskId}` | GET |
| 暂停任务 | `/api/v1/scheduler/tasks/{taskId}/pause` | POST |
| 恢复任务 | `/api/v1/scheduler/tasks/{taskId}/resume` | POST |
| 取消任务 | `/api/v1/scheduler/tasks/{taskId}/cancel` | POST |
| 重试任务 | `/api/v1/scheduler/tasks/{taskId}/retry` | POST |

---

## 更新建议

### 高优先级（立即执行）

#### 1. model-training-monitor.html
```javascript
// 修改前
const response = await fetch(getApiBaseUrl('/ml/training/scheduler/status'));

// 修改后
const response = await fetch(getApiBaseUrl('/api/v1/scheduler/status'));

// 可选：添加任务类型过滤
const response = await fetch(getApiBaseUrl('/api/v1/scheduler/status?task_type=model_training'));
```

#### 2. feature-engineering-monitor.html
```javascript
// 修改前
const response = await fetch(getApiBaseUrl('/features/engineering/scheduler/status'));

// 修改后
const response = await fetch(getApiBaseUrl('/api/v1/scheduler/status'));

// 可选：添加任务类型过滤
const response = await fetch(getApiBaseUrl('/api/v1/scheduler/status?task_type=feature_extraction'));
```

### 中优先级（建议执行）

#### 3. dashboard.html
建议添加统一调度器总览卡片：
```html
<!-- 调度器状态卡片 -->
<div class="dashboard-card" onclick="window.location.href='/data-collection-monitor.html'">
    <div class="card-header">
        <h3><i class="fas fa-tasks"></i> 统一调度器</h3>
        <span id="scheduler-status-indicator" class="status-badge">加载中...</span>
    </div>
    <div class="card-body">
        <div class="metric-grid">
            <div class="metric">
                <span id="scheduler-running-tasks" class="metric-value">--</span>
                <span class="metric-label">运行中任务</span>
            </div>
            <div class="metric">
                <span id="scheduler-pending-tasks" class="metric-value">--</span>
                <span class="metric-label">待处理任务</span>
            </div>
        </div>
    </div>
</div>
```

```javascript
// 加载调度器状态
async function loadSchedulerStatus() {
    try {
        const response = await fetch('/api/v1/scheduler/dashboard');
        const data = await response.json();
        
        // 更新UI
        document.getElementById('scheduler-running-tasks').textContent = data.tasks.running;
        document.getElementById('scheduler-pending-tasks').textContent = data.tasks.pending;
        
        const isRunning = data.scheduler.is_running;
        const indicator = document.getElementById('scheduler-status-indicator');
        indicator.textContent = isRunning ? '运行中' : '已停止';
        indicator.className = `status-badge ${isRunning ? 'success' : 'error'}`;
    } catch (error) {
        console.error('加载调度器状态失败:', error);
    }
}
```

---

## 总结

### 检查结果统计

| 状态 | 页面数量 | 说明 |
|------|----------|------|
| ✅ 已集成 | 2 | data-sources-config.html, data-collection-monitor.html |
| ⚠️ 需更新 | 2 | model-training-monitor.html, feature-engineering-monitor.html |
| ⚠️ 建议添加 | 1 | dashboard.html |

### 推荐行动

1. **立即执行**：更新 model-training-monitor.html 和 feature-engineering-monitor.html 的API端点
2. **短期执行**：在 dashboard.html 添加统一调度器总览卡片
3. **验证测试**：更新后在各页面验证调度器功能正常

### 迁移收益

- **统一维护**：减少多个独立调度器的维护成本
- **功能增强**：获得统一调度器的全部企业级功能
- **一致性**：所有监控页面使用相同的调度器后端
- **可扩展性**：易于添加新的任务类型和监控维度

---

**报告结束**
