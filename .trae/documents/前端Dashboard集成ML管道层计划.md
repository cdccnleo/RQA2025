# 前端Dashboard集成ML自动化训练管道层计划

## 文档信息

| 属性 | 值 |
|------|-----|
| 文档编号 | PLAN-FRONTEND-PIPELINE-001 |
| 版本 | 1.0.0 |
| 创建日期 | 2026-02-26 |
| 计划状态 | 待审批 |

---

## 1. 概述

### 1.1 目标

将ML自动化训练管道层集成到量化交易系统前端Dashboard，实现：
- 管道执行状态可视化监控
- 实时模型性能指标展示
- 告警和漂移检测通知
- 手动触发管道执行
- 回滚操作界面

### 1.2 当前状态

**前端现状：**
- 技术栈：Next.js + React + TypeScript + Tailwind CSS
- 当前页面：主页Dashboard（`src/web/pages/index.tsx`）
- 功能：系统状态展示、健康检查、服务列表、快速操作

**后端现状：**
- ML自动化训练管道层已实现（`src/pipeline/`）
- 8阶段管道：数据准备→特征工程→模型训练→模型评估→模型验证→金丝雀部署→全面部署→监控
- 监控子系统：性能监控、漂移检测、告警管理、自动回滚

### 1.3 集成范围

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         集成范围示意图                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   前端 Dashboard                    后端 API                    管道层       │
│  ┌──────────────────┐            ┌──────────────────┐      ┌──────────────┐ │
│  │                  │            │                  │      │              │ │
│  │ 管道监控面板     │◄──────────►│ Pipeline API     │◄────►│ Pipeline     │ │
│  │ - 执行状态       │   HTTP     │ - 状态查询       │      │ Controller   │ │
│  │ - 进度跟踪       │            │ - 执行控制       │      │              │ │
│  │ - 阶段详情       │            │ - 日志获取       │      └──────────────┘ │
│  │                  │            │                  │            │          │
│  ├──────────────────┤            ├──────────────────┤            ▼          │
│  │                  │            │                  │      ┌──────────────┐ │
│  │ 模型性能面板     │◄──────────►│ Monitoring API   │◄────►│ Performance  │ │
│  │ - 准确率         │            │ - 指标查询       │      │ Monitor      │ │
│  │ - 夏普比率       │            │ - 实时推送       │      └──────────────┘ │
│  │ - 回撤曲线       │            │                  │            │          │
│  │                  │            │                  │            ▼          │
│  ├──────────────────┤            ├──────────────────┤      ┌──────────────┐ │
│  │                  │            │                  │      │ Drift        │ │
│  │ 告警中心         │◄──────────►│ Alert API        │◄────►│ Detector     │ │
│  │ - 告警列表       │            │ - 告警查询       │      └──────────────┘ │
│  │ - 漂移检测       │            │ - 确认/解决      │            │          │
│  │ - 通知推送       │            │                  │            ▼          │
│  │                  │            │                  │      ┌──────────────┐ │
│  ├──────────────────┤            ├──────────────────┤      │ Rollback     │ │
│  │                  │            │                  │      │ Manager      │ │
│  │ 操作控制台       │◄──────────►│ Control API      │◄────►│              │ │
│  │ - 启动管道       │            │ - 启动/停止      │      └──────────────┘ │
│  │ - 手动回滚       │            │ - 回滚操作       │                       │
│  │ - 配置管理       │            │ - 配置更新       │                       │
│  │                  │            │                  │                       │
│  └──────────────────┘            └──────────────────┘                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Dashboard + Pipeline 集成架构                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        前端展示层 (Next.js)                          │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │   │
│  │  │  系统概览    │ │  管道监控    │ │  模型性能    │ │  告警中心    │ │   │
│  │  │  Dashboard   │ │  Pipeline    │ │  Model Perf  │ │   Alerts     │ │   │
│  │  │              │ │   Monitor    │ │              │ │              │ │   │
│  │  │ • 状态卡片   │ │ • 执行列表   │ │ • 指标图表   │ │ • 告警列表   │ │   │
│  │  │ • 服务状态   │ │ • 进度追踪   │ │ • 趋势分析   │ │ • 漂移检测   │ │   │
│  │  │ • 快速操作   │ │ • 日志查看   │ │ • 对比分析   │ │ • 通知管理   │ │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ │   │
│  │                              │                                       │   │
│  │                    ┌─────────┴─────────┐                             │   │
│  │                    ▼                   ▼                             │   │
│  │           ┌──────────────┐    ┌──────────────┐                      │   │
│  │           │  WebSocket   │    │   REST API   │                      │   │
│  │           │  实时推送    │    │   请求响应   │                      │   │
│  │           └──────────────┘    └──────────────┘                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        API网关层 (FastAPI)                           │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │   │
│  │  │ Pipeline     │ │ Monitoring   │ │   Alert      │ │   Control    │ │   │
│  │  │   Router     │ │   Router     │ │   Router     │ │   Router     │ │   │
│  │  │              │ │              │ │              │ │              │ │   │
│  │  │ /api/v1/     │ │ /api/v1/     │ │ /api/v1/     │ │ /api/v1/     │ │   │
│  │  │ pipeline/*   │ │ monitoring/* │ │ alerts/*     │ │ control/*    │ │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      ML管道层 (Python)                               │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │   │
│  │  │   Pipeline   │ │ Performance  │ │    Drift     │ │  Rollback    │ │   │
│  │  │  Controller  │ │   Monitor    │ │  Detector    │ │   Manager    │ │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 前端组件架构

```
src/web/
├── pages/
│   ├── index.tsx                 # 主页Dashboard（现有）
│   ├── pipeline/
│   │   ├── index.tsx             # 管道列表页
│   │   ├── [id].tsx              # 管道详情页
│   │   └── create.tsx            # 创建管道页
│   ├── monitoring/
│   │   ├── index.tsx             # 监控总览页
│   │   ├── metrics.tsx           # 性能指标页
│   │   └── drift.tsx             # 漂移检测页
│   └── alerts/
│       └── index.tsx             # 告警中心页
├── components/
│   ├── dashboard/
│   │   ├── SystemStatusCard.tsx  # 系统状态卡片（现有扩展）
│   │   ├── PipelineStatusCard.tsx# 管道状态卡片（新增）
│   │   ├── ModelMetricsCard.tsx  # 模型指标卡片（新增）
│   │   └── AlertSummaryCard.tsx  # 告警摘要卡片（新增）
│   ├── pipeline/
│   │   ├── PipelineList.tsx      # 管道列表组件
│   │   ├── PipelineStage.tsx     # 管道阶段组件
│   │   ├── ExecutionTimeline.tsx # 执行时间线
│   │   └── LogViewer.tsx         # 日志查看器
│   ├── monitoring/
│   │   ├── MetricsChart.tsx      # 指标图表
│   │   ├── RealtimeMetrics.tsx   # 实时指标
│   │   └── DriftReport.tsx       # 漂移报告
│   └── alerts/
│       ├── AlertList.tsx         # 告警列表
│       ├── AlertDetail.tsx       # 告警详情
│       └── NotificationBadge.tsx # 通知徽章
├── hooks/
│   ├── usePipeline.ts            # 管道相关Hooks
│   ├── useMonitoring.ts          # 监控相关Hooks
│   ├── useAlerts.ts              # 告警相关Hooks
│   └── useWebSocket.ts           # WebSocket连接Hook
├── services/
│   ├── pipelineApi.ts            # 管道API服务
│   ├── monitoringApi.ts          # 监控API服务
│   ├── alertApi.ts               # 告警API服务
│   └── websocket.ts              # WebSocket服务
└── types/
    ├── pipeline.ts               # 管道类型定义
    ├── monitoring.ts             # 监控类型定义
    └── alert.ts                  # 告警类型定义
```

---

## 3. 功能模块设计

### 3.1 模块1：管道监控面板

**功能描述：**
展示ML自动化训练管道的执行状态和进度

**UI组件：**
```typescript
// 管道状态卡片组件
interface PipelineStatusCardProps {
  pipelineId: string;
  pipelineName: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'rolling_back';
  currentStage: string;
  progress: number; // 0-100
  startTime: Date;
  estimatedEndTime?: Date;
  stages: PipelineStageInfo[];
}

// 管道阶段信息
interface PipelineStageInfo {
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped';
  duration?: number; // 秒
  startTime?: Date;
  endTime?: Date;
  output?: Record<string, any>;
}
```

**API接口：**
```typescript
// GET /api/v1/pipeline/status
interface PipelineStatusResponse {
  pipelines: PipelineStatus[];
  total: number;
  running: number;
  completed: number;
  failed: number;
}

// GET /api/v1/pipeline/{id}/details
interface PipelineDetailsResponse {
  pipeline: PipelineDetails;
  stages: StageDetails[];
  logs: LogEntry[];
  metrics: PipelineMetrics;
}
```

**交互功能：**
- 实时刷新管道状态（5秒轮询 + WebSocket推送）
- 点击查看管道详情
- 手动触发管道执行
- 取消正在执行的管道
- 查看阶段日志

### 3.2 模块2：模型性能面板

**功能描述：**
展示模型性能指标的实时监控和历史趋势

**UI组件：**
```typescript
// 模型性能卡片
interface ModelMetricsCardProps {
  modelId: string;
  modelName: string;
  metrics: {
    technical: TechnicalMetrics;
    business: BusinessMetrics;
    resource: ResourceMetrics;
  };
  history: MetricsHistoryPoint[];
  alerts: ActiveAlert[];
}

// 技术指标
interface TechnicalMetrics {
  accuracy: number;
  f1Score: number;
  rocAuc: number;
  precision: number;
  recall: number;
}

// 业务指标
interface BusinessMetrics {
  totalReturn: number;
  annualizedReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
}

// 资源指标
interface ResourceMetrics {
  avgLatencyMs: number;
  p95LatencyMs: number;
  p99LatencyMs: number;
  errorRate: number;
  throughputRps: number;
}
```

**图表展示：**
- 准确率趋势图（折线图）
- 收益率曲线（面积图）
- 延迟分布（柱状图）
- 指标仪表盘（仪表盘图）

**API接口：**
```typescript
// GET /api/v1/monitoring/metrics/{model_id}
interface MetricsResponse {
  current: MetricsSnapshot;
  history: MetricsHistoryPoint[];
  statistics: MetricsStatistics;
}

// GET /api/v1/monitoring/metrics/{model_id}/history
interface MetricsHistoryResponse {
  data: MetricsHistoryPoint[];
  aggregation: '1m' | '5m' | '1h' | '1d';
}
```

### 3.3 模块3：告警中心

**功能描述：**
集中展示和管理所有告警信息

**UI组件：**
```typescript
// 告警列表
interface AlertListProps {
  alerts: Alert[];
  onAcknowledge: (alertId: string) => void;
  onResolve: (alertId: string) => void;
  onSuppress: (alertId: string, duration: number) => void;
}

// 告警项
interface Alert {
  id: string;
  title: string;
  message: string;
  severity: 'debug' | 'info' | 'warning' | 'error' | 'critical';
  status: 'active' | 'acknowledged' | 'resolved' | 'suppressed';
  source: string;
  metricName?: string;
  metricValue?: number;
  threshold?: number;
  timestamp: Date;
  acknowledgedBy?: string;
  acknowledgedAt?: Date;
}

// 漂移报告
interface DriftReportProps {
  reports: DriftReport[];
  onTriggerRetraining: () => void;
}

interface DriftReport {
  timestamp: Date;
  driftType: 'data_drift' | 'concept_drift' | 'feature_drift';
  severity: 'none' | 'low' | 'medium' | 'high';
  driftScore: number;
  affectedFeatures: string[];
  recommendations: string[];
}
```

**API接口：**
```typescript
// GET /api/v1/alerts
interface AlertsResponse {
  alerts: Alert[];
  total: number;
  bySeverity: Record<Severity, number>;
  byStatus: Record<Status, number>;
}

// POST /api/v1/alerts/{id}/acknowledge
interface AcknowledgeAlertRequest {
  acknowledgedBy: string;
}

// GET /api/v1/monitoring/drift
interface DriftResponse {
  reports: DriftReport[];
  summary: DriftSummary;
  shouldTriggerRetraining: boolean;
}
```

### 3.4 模块4：操作控制台

**功能描述：**
提供手动操作管道的界面

**功能列表：**
1. **启动管道**
   - 选择管道配置
   - 设置参数（股票代码、日期范围等）
   - 触发执行

2. **回滚操作**
   - 查看回滚建议
   - 确认回滚
   - 查看回滚历史

3. **配置管理**
   - 查看管道配置
   - 编辑告警阈值
   - 保存配置变更

**API接口：**
```typescript
// POST /api/v1/pipeline/execute
interface ExecutePipelineRequest {
  configId: string;
  context: {
    symbols: string[];
    startDate: string;
    endDate: string;
    [key: string]: any;
  };
}

// POST /api/v1/control/rollback
interface RollbackRequest {
  modelId: string;
  force?: boolean;
  targetVersion?: string;
}

// POST /api/v1/control/config/update
interface UpdateConfigRequest {
  section: string;
  key: string;
  value: any;
}
```

---

## 4. API设计

### 4.1 REST API规范

#### 4.1.1 管道管理API

```yaml
# 管道状态查询
GET /api/v1/pipeline/status
Response:
  200:
    body:
      pipelines:
        - pipeline_id: "pipe_001"
          name: "quant_trading_ml_pipeline"
          status: "running"
          current_stage: "model_training"
          progress: 45
          start_time: "2026-02-26T10:00:00Z"
          stages:
            - name: "data_preparation"
              status: "completed"
              duration: 300

# 管道详情
GET /api/v1/pipeline/{pipeline_id}/details
Response:
  200:
    body:
      pipeline:
        pipeline_id: "pipe_001"
        name: "quant_trading_ml_pipeline"
        status: "running"
      stages:
        - name: "data_preparation"
          status: "completed"
          output:
            data_size: 10000
            features_count: 50
      logs:
        - timestamp: "2026-02-26T10:05:00Z"
          level: "info"
          message: "Data preparation completed"

# 执行管道
POST /api/v1/pipeline/execute
Request:
  body:
    config_id: "default_config"
    context:
      symbols: ["AAPL", "GOOGL"]
      start_date: "2024-01-01"
      end_date: "2024-12-31"
Response:
  202:
    body:
      pipeline_id: "pipe_002"
      status: "pending"
      message: "Pipeline execution started"

# 取消管道
POST /api/v1/pipeline/{pipeline_id}/cancel
Response:
  200:
    body:
      success: true
      message: "Pipeline cancelled"
```

#### 4.1.2 监控API

```yaml
# 获取指标
GET /api/v1/monitoring/metrics/{model_id}
Response:
  200:
    body:
      current:
        timestamp: "2026-02-26T10:00:00Z"
        metrics:
          accuracy:
            value: 0.85
            threshold: 0.70
            status: "normal"
          sharpe_ratio:
            value: 1.2
            threshold: 1.0
            status: "normal"
      history:
        - timestamp: "2026-02-26T09:00:00Z"
          accuracy: 0.84
        - timestamp: "2026-02-26T10:00:00Z"
          accuracy: 0.85

# 获取漂移检测
GET /api/v1/monitoring/drift
Response:
  200:
    body:
      reports:
        - timestamp: "2026-02-26T10:00:00Z"
          drift_type: "data_drift"
          severity: "low"
          drift_score: 0.15
          affected_features: ["feature_1", "feature_2"]
      summary:
        total_detections: 10
        has_high_severity: false
      should_trigger_retraining: false
```

#### 4.1.3 告警API

```yaml
# 获取告警列表
GET /api/v1/alerts
Query:
  severity: "critical,error"
  status: "active"
  page: 1
  page_size: 20
Response:
  200:
    body:
      alerts:
        - id: "alert_001"
          title: "准确率下降"
          message: "模型准确率下降超过10%"
          severity: "critical"
          status: "active"
          metric_name: "accuracy"
          metric_value: 0.60
          threshold: 0.70
          timestamp: "2026-02-26T10:00:00Z"
      total: 5
      by_severity:
        critical: 2
        error: 3

# 确认告警
POST /api/v1/alerts/{alert_id}/acknowledge
Request:
  body:
    acknowledged_by: "admin"
Response:
  200:
    body:
      success: true

# 解决告警
POST /api/v1/alerts/{alert_id}/resolve
Response:
  200:
    body:
      success: true
```

### 4.2 WebSocket实时推送

```typescript
// WebSocket连接
ws://localhost:8000/ws/monitoring

// 订阅消息
interface SubscribeMessage {
  type: 'subscribe';
  channels: string[];
}

// 推送消息类型
interface WebSocketMessage {
  type: 'metrics' | 'alert' | 'pipeline_status' | 'drift_detected';
  timestamp: string;
  data: any;
}

// 示例：指标更新推送
{
  "type": "metrics",
  "timestamp": "2026-02-26T10:00:00Z",
  "data": {
    "model_id": "model_001",
    "metrics": {
      "accuracy": { "value": 0.85, "status": "normal" },
      "sharpe_ratio": { "value": 1.2, "status": "normal" }
    }
  }
}

// 示例：告警推送
{
  "type": "alert",
  "timestamp": "2026-02-26T10:00:00Z",
  "data": {
    "alert_id": "alert_001",
    "title": "准确率下降",
    "severity": "critical",
    "message": "模型准确率下降超过10%"
  }
}

// 示例：管道状态推送
{
  "type": "pipeline_status",
  "timestamp": "2026-02-26T10:00:00Z",
  "data": {
    "pipeline_id": "pipe_001",
    "status": "running",
    "current_stage": "model_training",
    "progress": 45
  }
}
```

---

## 5. 数据模型

### 5.1 TypeScript类型定义

```typescript
// types/pipeline.ts

export interface Pipeline {
  id: string;
  name: string;
  version: string;
  status: PipelineStatus;
  currentStage?: string;
  progress: number;
  startTime?: Date;
  endTime?: Date;
  duration?: number;
  stages: PipelineStage[];
  context?: Record<string, any>;
  error?: string;
}

export type PipelineStatus = 
  | 'pending' 
  | 'running' 
  | 'completed' 
  | 'failed' 
  | 'rolling_back' 
  | 'rolled_back' 
  | 'cancelled';

export interface PipelineStage {
  name: string;
  status: StageStatus;
  startTime?: Date;
  endTime?: Date;
  duration?: number;
  output?: Record<string, any>;
  error?: string;
  logs?: LogEntry[];
}

export type StageStatus = 
  | 'pending' 
  | 'running' 
  | 'completed' 
  | 'failed' 
  | 'skipped';

export interface LogEntry {
  timestamp: Date;
  level: 'debug' | 'info' | 'warning' | 'error';
  message: string;
}

// types/monitoring.ts

export interface MetricsSnapshot {
  timestamp: Date;
  modelId: string;
  metrics: Record<string, MetricValue>;
}

export interface MetricValue {
  value: number;
  threshold?: number;
  status: 'normal' | 'warning' | 'critical';
  unit?: string;
}

export interface MetricsHistoryPoint {
  timestamp: Date;
  values: Record<string, number>;
}

export interface DriftReport {
  timestamp: Date;
  driftType: 'data_drift' | 'concept_drift' | 'feature_drift';
  severity: 'none' | 'low' | 'medium' | 'high';
  driftScore: number;
  affectedFeatures: string[];
  statistics?: Record<string, any>;
  recommendations: string[];
}

// types/alert.ts

export interface Alert {
  id: string;
  title: string;
  message: string;
  severity: AlertSeverity;
  status: AlertStatus;
  source: string;
  metricName?: string;
  metricValue?: number;
  threshold?: number;
  timestamp: Date;
  acknowledgedBy?: string;
  acknowledgedAt?: Date;
  resolvedAt?: Date;
  metadata?: Record<string, any>;
}

export type AlertSeverity = 'debug' | 'info' | 'warning' | 'error' | 'critical';
export type AlertStatus = 'active' | 'acknowledged' | 'resolved' | 'suppressed';
```

---

## 6. 实施计划

### 6.1 阶段划分

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         实施阶段划分                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  阶段1: 基础架构搭建 (Week 1)                                               │
│  ├── 任务1.1: 创建API路由和控制器                                           │
│  ├── 任务1.2: 实现WebSocket服务                                             │
│  ├── 任务1.3: 创建前端基础组件和类型定义                                    │
│  └── 交付物: 基础API和前端框架                                              │
│                                                                             │
│  阶段2: 管道监控功能 (Week 2)                                               │
│  ├── 任务2.1: 实现管道状态查询API                                           │
│  ├── 任务2.2: 开发管道列表和详情页面                                        │
│  ├── 任务2.3: 实现管道执行控制功能                                          │
│  └── 交付物: 管道监控面板                                                   │
│                                                                             │
│  阶段3: 性能监控功能 (Week 3)                                               │
│  ├── 任务3.1: 实现指标收集和查询API                                         │
│  ├── 任务3.2: 开发指标图表组件                                              │
│  ├── 任务3.3: 实现实时指标推送                                              │
│  └── 交付物: 模型性能面板                                                   │
│                                                                             │
│  阶段4: 告警中心功能 (Week 4)                                               │
│  ├── 任务4.1: 实现告警管理API                                               │
│  ├── 任务4.2: 开发告警列表和详情组件                                        │
│  ├── 任务4.3: 实现漂移检测展示                                              │
│  └── 交付物: 告警中心                                                       │
│                                                                             │
│  阶段5: 集成测试和优化 (Week 5)                                             │
│  ├── 任务5.1: 端到端集成测试                                                │
│  ├── 任务5.2: 性能优化                                                      │
│  ├── 任务5.3: 用户体验优化                                                  │
│  └── 交付物: 完整集成系统                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 详细任务清单

#### 阶段1：基础架构搭建

| 任务ID | 任务名称 | 描述 | 预估工时 | 依赖 |
|--------|----------|------|----------|------|
| 1.1 | 创建Pipeline API路由 | 实现管道相关的REST API路由 | 4h | - |
| 1.2 | 创建Monitoring API路由 | 实现监控相关的REST API路由 | 4h | - |
| 1.3 | 创建Alert API路由 | 实现告警相关的REST API路由 | 4h | - |
| 1.4 | 实现WebSocket服务 | 实现实时数据推送服务 | 8h | - |
| 1.5 | 创建前端类型定义 | 创建TypeScript类型定义文件 | 4h | - |
| 1.6 | 创建API服务层 | 创建前端API调用服务 | 6h | 1.1-1.3 |
| 1.7 | 创建WebSocket Hook | 创建useWebSocket Hook | 4h | 1.4 |

#### 阶段2：管道监控功能

| 任务ID | 任务名称 | 描述 | 预估工时 | 依赖 |
|--------|----------|------|----------|------|
| 2.1 | 实现管道状态查询 | 后端实现管道状态查询接口 | 6h | 1.1 |
| 2.2 | 实现管道执行控制 | 后端实现管道启动/取消接口 | 6h | 1.1 |
| 2.3 | 开发PipelineList组件 | 前端管道列表组件 | 8h | 1.5-1.6 |
| 2.4 | 开发PipelineStage组件 | 前端管道阶段展示组件 | 8h | 1.5-1.6 |
| 2.5 | 开发ExecutionTimeline组件 | 前端执行时间线组件 | 6h | 1.5-1.6 |
| 2.6 | 开发LogViewer组件 | 前端日志查看组件 | 6h | 1.5-1.6 |
| 2.7 | 创建管道监控页面 | 整合组件创建完整页面 | 6h | 2.3-2.6 |

#### 阶段3：性能监控功能

| 任务ID | 任务名称 | 描述 | 预估工时 | 依赖 |
|--------|----------|------|----------|------|
| 3.1 | 实现指标收集API | 后端实现指标查询接口 | 6h | 1.2 |
| 3.2 | 实现指标历史API | 后端实现指标历史查询 | 6h | 1.2 |
| 3.3 | 集成图表库 | 集成ECharts/Recharts图表库 | 4h | - |
| 3.4 | 开发MetricsChart组件 | 指标图表组件 | 8h | 3.3 |
| 3.5 | 开发RealtimeMetrics组件 | 实时指标展示组件 | 6h | 1.7, 3.4 |
| 3.6 | 开发ModelMetricsCard组件 | 模型性能卡片组件 | 6h | 3.4 |
| 3.7 | 创建性能监控页面 | 整合组件创建完整页面 | 6h | 3.4-3.6 |

#### 阶段4：告警中心功能

| 任务ID | 任务名称 | 描述 | 预估工时 | 依赖 |
|--------|----------|------|----------|------|
| 4.1 | 实现告警查询API | 后端实现告警查询接口 | 6h | 1.3 |
| 4.2 | 实现告警操作API | 后端实现告警确认/解决接口 | 6h | 1.3 |
| 4.3 | 实现漂移检测API | 后端实现漂移检测查询接口 | 6h | 1.2 |
| 4.4 | 开发AlertList组件 | 告警列表组件 | 8h | 1.5-1.6 |
| 4.5 | 开发AlertDetail组件 | 告警详情组件 | 6h | 1.5-1.6 |
| 4.6 | 开发DriftReport组件 | 漂移报告组件 | 6h | 1.5-1.6 |
| 4.7 | 开发NotificationBadge组件 | 通知徽章组件 | 4h | 1.5-1.6 |
| 4.8 | 创建告警中心页面 | 整合组件创建完整页面 | 6h | 4.4-4.7 |

#### 阶段5：集成测试和优化

| 任务ID | 任务名称 | 描述 | 预估工时 | 依赖 |
|--------|----------|------|----------|------|
| 5.1 | 编写API测试 | 编写后端API单元测试 | 8h | 2-4 |
| 5.2 | 编写组件测试 | 编写前端组件单元测试 | 8h | 2-4 |
| 5.3 | 端到端测试 | 编写端到端测试用例 | 8h | 2-4 |
| 5.4 | 性能优化 | 优化前端性能（懒加载、缓存等） | 8h | 2-4 |
| 5.5 | UI/UX优化 | 优化用户界面和交互体验 | 8h | 2-4 |
| 5.6 | 文档编写 | 编写用户操作文档 | 4h | 2-4 |

---

## 7. 技术选型

### 7.1 前端技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| Next.js | 14.x | React框架 |
| React | 18.x | UI库 |
| TypeScript | 5.x | 类型系统 |
| Tailwind CSS | 3.x | CSS框架 |
| Recharts | 2.x | 图表库 |
| React Query | 5.x | 数据获取和缓存 |
| Zustand | 4.x | 状态管理 |
| date-fns | 3.x | 日期处理 |

### 7.2 后端技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| FastAPI | 0.100+ | Web框架 |
| WebSocket | - | 实时通信 |
| Pydantic | 2.x | 数据验证 |
| SQLAlchemy | 2.x | ORM（如需要持久化） |

---

## 8. 风险评估

### 8.1 风险列表

| 风险ID | 风险描述 | 可能性 | 影响 | 缓解措施 |
|--------|----------|--------|------|----------|
| R1 | WebSocket连接不稳定 | 中 | 高 | 实现自动重连机制和降级到轮询 |
| R2 | 大数据量图表渲染性能问题 | 中 | 中 | 实现数据采样和虚拟滚动 |
| R3 | API响应延迟影响用户体验 | 低 | 中 | 实现加载状态和骨架屏 |
| R4 | 实时数据推送频率过高 | 中 | 中 | 实现节流和批量更新 |
| R5 | 浏览器兼容性问题 | 低 | 低 | 使用Polyfill和特性检测 |

### 8.2 应急预案

1. **WebSocket故障**：自动降级到HTTP轮询
2. **API服务不可用**：显示离线状态和缓存数据
3. **前端性能问题**：启用懒加载和代码分割

---

## 9. 验收标准

### 9.1 功能验收

| 功能 | 验收标准 |
|------|----------|
| 管道监控 | 能够查看所有管道执行状态，实时更新进度 |
| 性能监控 | 能够查看模型性能指标，支持历史趋势分析 |
| 告警中心 | 能够查看、确认、解决告警，接收实时通知 |
| 操作控制 | 能够手动触发管道执行和回滚操作 |

### 9.2 性能验收

| 指标 | 目标值 |
|------|--------|
| 页面加载时间 | < 3秒 |
| API响应时间 | < 500ms |
| WebSocket延迟 | < 100ms |
| 图表渲染时间 | < 1秒 |

### 9.3 可用性验收

- 支持主流浏览器（Chrome, Firefox, Safari, Edge）
- 响应式设计，支持桌面和平板
- 符合WCAG 2.1 AA级无障碍标准

---

## 10. 相关文档

| 文档 | 路径 | 说明 |
|------|------|------|
| 管道层架构设计 | `docs/architecture/pipeline_architecture_design.md` | 后端管道层架构 |
| 前端项目结构 | `src/web/README.md` | 前端项目说明 |
| API规范 | `docs/api/` | API接口文档 |

---

## 11. 版本历史

| 版本 | 日期 | 作者 | 变更说明 |
|------|------|------|----------|
| 1.0.0 | 2026-02-26 | 前端架构组 | 初始版本，完整集成计划 |

---

*文档结束*
