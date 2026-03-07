# 前端Dashboard集成ML管道层实施计划

## 文档信息

| 属性 | 值 |
|------|-----|
| 文档编号 | PLAN-FRONTEND-IMPL-001 |
| 版本 | 1.0.0 |
| 创建日期 | 2026-02-26 |
| 计划状态 | 待审批 |

---

## 1. 当前进度

### 1.1 已完成工作

**后端API开发 (100%完成)**
- ✅ Pipeline API路由 (`src/api/routes/pipeline.py`)
- ✅ Monitoring API路由 (`src/api/routes/monitoring.py`)
- ✅ Alert API路由 (`src/api/routes/alerts.py`)
- ✅ API入口文件 (`src/api/__init__.py`)
- ✅ 34个REST API端点
- ✅ 4个WebSocket端点

**前端基础架构 (40%完成)**
- ✅ TypeScript类型定义
  - `src/web/types/pipeline.ts`
  - `src/web/types/monitoring.ts`
  - `src/web/types/alert.ts`
- ✅ API服务层
  - `src/web/services/api.ts`
  - `src/web/services/pipelineApi.ts`
  - `src/web/services/monitoringApi.ts`
  - `src/web/services/alertApi.ts`
- ✅ React Hooks (部分)
  - `src/web/hooks/usePipeline.ts`

### 1.2 剩余工作量

**前端开发 (60%剩余)**
- ⏳ React Hooks (2个)
- ⏳ UI组件 (15个)
- ⏳ 页面 (4个)
- ⏳ Dashboard集成

---

## 2. 实施计划

### 阶段1: 完成Hooks开发 (2小时)

| 任务ID | 任务名称 | 文件路径 | 说明 |
|--------|----------|----------|------|
| 1.1 | useMonitoring Hook | `src/web/hooks/useMonitoring.ts` | 监控数据获取和WebSocket |
| 1.2 | useAlerts Hook | `src/web/hooks/useAlerts.ts` | 告警管理和操作 |
| 1.3 | useWebSocket Hook | `src/web/hooks/useWebSocket.ts` | 通用WebSocket连接管理 |

### 阶段2: 开发UI组件 (6小时)

#### 2.1 Dashboard组件 (1小时)

| 任务ID | 组件名称 | 文件路径 | 功能说明 |
|--------|----------|----------|----------|
| 2.1.1 | PipelineStatusCard | `src/web/components/dashboard/PipelineStatusCard.tsx` | 管道状态概览卡片 |
| 2.1.2 | ModelMetricsCard | `src/web/components/dashboard/ModelMetricsCard.tsx` | 模型指标概览卡片 |
| 2.1.3 | AlertSummaryCard | `src/web/components/dashboard/AlertSummaryCard.tsx` | 告警摘要卡片 |

#### 2.2 管道监控组件 (2小时)

| 任务ID | 组件名称 | 文件路径 | 功能说明 |
|--------|----------|----------|----------|
| 2.2.1 | PipelineList | `src/web/components/pipeline/PipelineList.tsx` | 管道列表展示 |
| 2.2.2 | PipelineStage | `src/web/components/pipeline/PipelineStage.tsx` | 管道阶段可视化 |
| 2.2.3 | ExecutionTimeline | `src/web/components/pipeline/ExecutionTimeline.tsx` | 执行时间线 |
| 2.2.4 | LogViewer | `src/web/components/pipeline/LogViewer.tsx` | 日志查看器 |
| 2.2.5 | ExecutePipelineButton | `src/web/components/pipeline/ExecutePipelineButton.tsx` | 执行按钮 |

#### 2.3 性能监控组件 (2小时)

| 任务ID | 组件名称 | 文件路径 | 功能说明 |
|--------|----------|----------|----------|
| 2.3.1 | MetricsChart | `src/web/components/monitoring/MetricsChart.tsx` | 指标趋势图表 |
| 2.3.2 | RealtimeMetrics | `src/web/components/monitoring/RealtimeMetrics.tsx` | 实时指标展示 |
| 2.3.3 | DriftReportCard | `src/web/components/monitoring/DriftReportCard.tsx` | 漂移检测报告 |
| 2.3.4 | TechnicalMetricsPanel | `src/web/components/monitoring/TechnicalMetricsPanel.tsx` | 技术指标面板 |
| 2.3.5 | BusinessMetricsPanel | `src/web/components/monitoring/BusinessMetricsPanel.tsx` | 业务指标面板 |

#### 2.4 告警中心组件 (1小时)

| 任务ID | 组件名称 | 文件路径 | 功能说明 |
|--------|----------|----------|----------|
| 2.4.1 | AlertList | `src/web/components/alerts/AlertList.tsx` | 告警列表 |
| 2.4.2 | AlertItem | `src/web/components/alerts/AlertItem.tsx` | 告警项 |
| 2.4.3 | NotificationBadge | `src/web/components/alerts/NotificationBadge.tsx` | 通知徽章 |
| 2.4.4 | RollbackPanel | `src/web/components/alerts/RollbackPanel.tsx` | 回滚操作面板 |

### 阶段3: 开发页面 (4小时)

| 任务ID | 页面名称 | 文件路径 | 功能说明 |
|--------|----------|----------|----------|
| 3.1 | 管道列表页 | `src/web/pages/pipeline/index.tsx` | 展示所有管道 |
| 3.2 | 管道详情页 | `src/web/pages/pipeline/[id].tsx` | 管道执行详情 |
| 3.3 | 监控总览页 | `src/web/pages/monitoring/index.tsx` | 模型性能监控 |
| 3.4 | 告警中心页 | `src/web/pages/alerts/index.tsx` | 告警管理和操作 |

### 阶段4: Dashboard集成 (2小时)

| 任务ID | 任务名称 | 文件路径 | 功能说明 |
|--------|----------|----------|----------|
| 4.1 | 更新主页Dashboard | `src/web/pages/index.tsx` | 集成管道监控卡片 |
| 4.2 | 添加导航菜单 | `src/web/components/layout/` | 添加监控和告警入口 |
| 4.3 | 全局状态管理 | `src/web/store/` | 使用Zustand管理状态 |

### 阶段5: 测试和优化 (2小时)

| 任务ID | 任务名称 | 说明 |
|--------|----------|------|
| 5.1 | API连通性测试 | 验证前后端API连接 |
| 5.2 | WebSocket测试 | 验证实时推送功能 |
| 5.3 | 性能优化 | 组件懒加载、数据缓存 |
| 5.4 | 错误处理 | 完善错误边界和提示 |

---

## 3. 文件清单

### 需要创建的文件列表

```
src/web/
├── hooks/
│   ├── useMonitoring.ts          ⏳ 阶段1
│   ├── useAlerts.ts              ⏳ 阶段1
│   └── useWebSocket.ts           ⏳ 阶段1
├── components/
│   ├── dashboard/
│   │   ├── PipelineStatusCard.tsx    ⏳ 阶段2.1
│   │   ├── ModelMetricsCard.tsx      ⏳ 阶段2.1
│   │   └── AlertSummaryCard.tsx      ⏳ 阶段2.1
│   ├── pipeline/
│   │   ├── PipelineList.tsx          ⏳ 阶段2.2
│   │   ├── PipelineStage.tsx         ⏳ 阶段2.2
│   │   ├── ExecutionTimeline.tsx     ⏳ 阶段2.2
│   │   ├── LogViewer.tsx             ⏳ 阶段2.2
│   │   └── ExecutePipelineButton.tsx ⏳ 阶段2.2
│   ├── monitoring/
│   │   ├── MetricsChart.tsx          ⏳ 阶段2.3
│   │   ├── RealtimeMetrics.tsx       ⏳ 阶段2.3
│   │   ├── DriftReportCard.tsx       ⏳ 阶段2.3
│   │   ├── TechnicalMetricsPanel.tsx ⏳ 阶段2.3
│   │   └── BusinessMetricsPanel.tsx  ⏳ 阶段2.3
│   └── alerts/
│       ├── AlertList.tsx             ⏳ 阶段2.4
│       ├── AlertItem.tsx             ⏳ 阶段2.4
│       ├── NotificationBadge.tsx     ⏳ 阶段2.4
│       └── RollbackPanel.tsx         ⏳ 阶段2.4
├── pages/
│   ├── pipeline/
│   │   ├── index.tsx             ⏳ 阶段3
│   │   └── [id].tsx              ⏳ 阶段3
│   ├── monitoring/
│   │   └── index.tsx             ⏳ 阶段3
│   └── alerts/
│       └── index.tsx             ⏳ 阶段3
└── store/
    └── index.ts                  ⏳ 阶段4
```

---

## 4. 技术依赖

### 需要安装的前端依赖

```bash
# 图表库
npm install recharts

# 状态管理
npm install zustand

# 日期处理
npm install date-fns

# HTTP客户端
npm install axios

# 图标库（如未安装）
npm install lucide-react
```

---

## 5. 验收标准

### 功能验收

| 功能模块 | 验收标准 | 优先级 |
|----------|----------|--------|
| 管道监控 | 能查看管道列表、执行状态、进度、日志 | P0 |
| 性能监控 | 能查看实时指标、历史趋势、漂移检测 | P0 |
| 告警中心 | 能查看告警列表、确认/解决告警 | P0 |
| Dashboard集成 | 主页显示管道和告警摘要 | P1 |
| WebSocket实时推送 | 指标和状态实时更新 | P1 |

### 性能标准

| 指标 | 目标值 |
|------|--------|
| 页面加载时间 | < 3秒 |
| API响应时间 | < 500ms |
| 图表渲染时间 | < 1秒 |
| WebSocket延迟 | < 100ms |

---

## 6. 风险评估

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| WebSocket连接不稳定 | 中 | 高 | 实现自动重连和降级轮询 |
| 图表大数据量渲染慢 | 中 | 中 | 数据采样和虚拟滚动 |
| API跨域问题 | 低 | 高 | 配置CORS代理 |

---

## 7. 实施建议

### 建议1: 分阶段交付
1. **第一阶段** (4小时): 完成Hooks + Dashboard卡片
2. **第二阶段** (4小时): 完成管道监控页面
3. **第三阶段** (4小时): 完成性能监控和告警页面

### 建议2: 优先核心功能
- 优先实现管道状态展示和执行功能
- 优先实现关键指标展示（准确率、夏普比率）
- 优先实现告警列表和确认功能

### 建议3: 使用Mock数据
- 在API开发完成前，使用Mock数据进行前端开发
- 定义好数据结构，便于后续联调

---

## 8. 相关文档

| 文档 | 路径 |
|------|------|
| 前端集成计划 | `.trae/documents/前端Dashboard集成ML管道层计划.md` |
| 管道层架构设计 | `docs/architecture/pipeline_architecture_design.md` |
| 后端API路由 | `src/api/routes/` |

---

## 9. 版本历史

| 版本 | 日期 | 作者 | 变更说明 |
|------|------|------|----------|
| 1.0.0 | 2026-02-26 | 前端开发组 | 初始版本，详细实施计划 |

---

*文档结束*
