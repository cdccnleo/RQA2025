# 统一调度器架构设计文档

## 文档信息

| 属性 | 值 |
|------|-----|
| 文档编号 | ARCH-UNIFIED-SCHEDULER-001 |
| 版本 | 2.0.0 |
| 创建日期 | 2026-03-05 |
| 更新日期 | 2026-03-06 |
| 作者 | RQA2025 Team |
| 状态 | 已更新 |

---

## 1. 概述

### 1.1 设计目标

统一调度器是RQA2025量化交易系统的核心调度组件，旨在：

- **统一调度管理**：整合数据采集、特征工程、模型训练、模型推理、交易执行、风险控制等多种任务的调度
- **消除冗余**：替代原有的多个调度器实现（service_scheduler、distributed/unified_scheduler）
- **简化架构**：提供单一、清晰、可维护的调度系统
- **支持扩展**：易于添加新的任务类型和调度策略
- **企业级能力**：支持持久化、告警、事件驱动、安全等企业级特性

### 1.2 核心特性

| 特性 | 说明 | 状态 |
|------|------|------|
| 多任务类型支持 | 25+种任务类型覆盖量化交易全流程 | ✅ 已完成 |
| 多种触发方式 | Interval间隔、Cron定时、Date指定日期、Once一次性 | ✅ 已完成 |
| 工作进程池 | 支持多工作进程并发执行任务 | ✅ 已完成 |
| 任务状态管理 | 完整的任务生命周期管理（6种状态） | ✅ 已完成 |
| 历史记录 | 自动维护任务执行历史，支持查询和统计 | ✅ 已完成 |
| 优先级队列 | 支持任务优先级设置（5级优先级） | ✅ 已完成 |
| 任务超时控制 | 支持任务超时自动取消 | ✅ 已完成 |
| 任务重试机制 | 支持自动重试和指数退避 | ✅ 已完成 |
| 数据库持久化 | 支持SQLAlchemy数据库持久化 | ✅ 已完成 |
| 多通道告警 | 支持邮件、Webhook、短信、日志告警 | ✅ 已完成 |
| 事件总线集成 | 支持事件驱动任务触发 | ✅ 已完成 |
| Prometheus指标 | 支持标准Prometheus格式指标导出 | ✅ 已完成 |
| 任务数据加密 | 支持敏感数据加密存储 | ✅ 已完成 |
| 访问控制 | 支持RBAC权限控制和API密钥管理 | ✅ 已完成 |
| 批量处理 | 支持任务批量处理提高吞吐量 | ✅ 已完成 |
| 任务缓存 | 支持任务结果缓存和智能预取 | ✅ 已完成 |

---

## 2. 架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         API Gateway Layer                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    scheduler_routes.py                           │   │
│  │  • Dashboard API    • Task Management API    • Job Management   │   │
│  │  • Auto Collection  • Config Management      • Alert Config     │   │
│  │  • Analytics API    • Status API             • Event Triggers   │   │
│  └────────────────────────┬────────────────────────────────────────┘   │
└───────────────────────────┼─────────────────────────────────────────────┘
                            │
┌───────────────────────────┼─────────────────────────────────────────────┐
│                           ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Unified Scheduler Core                        │   │
│  │  ┌─────────────────────────────────────────────────────────┐   │   │
│  │  │              UnifiedScheduler (单例)                     │   │   │
│  │  │  • start() / stop() / is_running()                      │   │   │
│  │  │  • submit_task() / create_job() / retry_task()          │   │   │
│  │  │  • cancel_task() / pause_task() / resume_task()         │   │   │
│  │  │  • Event Bus Integration / Alert Integration            │   │   │
│  │  └────────────────────────┬────────────────────────────────┘   │   │
│  │                           │                                    │   │
│  │  ┌────────────────────────┴────────────────────────────────┐   │   │
│  │  │                    Components                            │   │   │
│  │  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │   │   │
│  │  │  │ TaskManager  │ │ WorkerManager│ │ JobScheduler │   │   │   │
│  │  │  │ • 任务CRUD   │ │ • 工作进程池 │ │ • 定时任务   │   │   │   │
│  │  │  │ • 状态管理   │ │ • 任务分配   │ │ • 触发器管理 │   │   │   │
│  │  │  │ • 重试逻辑   │ │ • 心跳检测   │ │ • 调度循环   │   │   │   │
│  │  │  └──────────────┘ └──────────────┘ └──────────────┘   │   │   │
│  │  └────────────────────────────────────────────────────────┘   │   │
│  │                                                                │   │
│  │  ┌────────────────────────────────────────────────────────┐   │   │
│  │  │              Enterprise Features                        │   │   │
│  │  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐  │   │   │
│  │  │  │Persistence │ │   Alerting  │ │   Event Bus     │  │   │   │
│  │  │  │• SQLAlchemy │ │• Multi-chan │ │• Event Publish  │  │   │   │
│  │  │  │• Repository │ │• Thresholds │ │• Event Trigger  │  │   │   │
│  │  │  └─────────────┘ └─────────────┘ └─────────────────┘  │   │   │
│  │  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐  │   │   │
│  │  │  │  Security   │ │  Metrics    │ │   Performance   │  │   │   │
│  │  │  │• Encryption │ │• Prometheus │ │• Priority Queue │  │   │   │
│  │  │  │• RBAC/ACL   │ │• Histograms │ │• Batch Process  │  │   │   │
│  │  │  └─────────────┘ └─────────────┘ └─────────────────┘  │   │   │
│  │  └────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 模块结构

```
src/core/orchestration/scheduler/
├── __init__.py                    # 模块导出，提供统一入口
├── base.py                        # 基础类和接口定义
├── task_manager.py                # 任务管理器
├── worker_manager.py              # 工作进程管理器
├── unified_scheduler.py           # 统一调度器核心实现
├── persistence/                   # 持久化模块
│   ├── __init__.py
│   ├── models.py                  # SQLAlchemy数据模型
│   └── repository.py              # 仓库模式实现
├── alerting/                      # 告警模块
│   ├── __init__.py
│   ├── alert_manager.py           # 告警管理器
│   ├── handlers.py                # 告警处理器
│   └── config.py                  # 告警配置
├── integration/                   # 集成模块
│   ├── __init__.py
│   └── event_bus_integration.py   # 事件总线集成
├── performance/                   # 性能优化模块
│   ├── __init__.py
│   ├── priority_queue.py          # 优先级队列
│   ├── batch_processor.py         # 批量处理器
│   └── task_cache.py              # 任务缓存
├── metrics/                       # 指标监控模块
│   ├── __init__.py
│   └── prometheus_metrics.py      # Prometheus指标
└── security/                      # 安全模块
    ├── __init__.py
    ├── encryption.py              # 任务加密
    └── access_control.py          # 访问控制
```

---

## 3. 核心组件详解

### 3.1 基础类 (base.py)

#### 3.1.1 任务状态枚举

```python
class TaskStatus(Enum):
    PENDING = "pending"           # 等待中
    RUNNING = "running"           # 运行中
    PAUSED = "paused"             # 已暂停
    COMPLETED = "completed"       # 已完成
    FAILED = "failed"             # 失败
    CANCELLED = "cancelled"       # 已取消
    TIMEOUT = "timeout"           # 已超时 (新增)
```

#### 3.1.2 任务类型枚举 (25+种)

```python
class JobType(Enum):
    # 数据采集与处理
    DATA_COLLECTION = "data_collection"           # 数据采集
    DATA_PROCESSING = "data_processing"           # 数据处理
    DATA_CLEANING = "data_cleaning"               # 数据清洗
    DATA_VALIDATION = "data_validation"           # 数据验证
    
    # 特征工程
    FEATURE_EXTRACTION = "feature_extraction"     # 特征提取
    FEATURE_CALCULATION = "feature_calculation"   # 特征计算
    FEATURE_SELECTION = "feature_selection"       # 特征选择
    FEATURE_VALIDATION = "feature_validation"     # 特征验证
    
    # 模型相关
    MODEL_TRAINING = "model_training"             # 模型训练
    MODEL_INFERENCE = "model_inference"           # 模型推理
    MODEL_VALIDATION = "model_validation"         # 模型验证
    MODEL_DEPLOYMENT = "model_deployment"         # 模型部署
    
    # 策略相关
    STRATEGY_BACKTEST = "strategy_backtest"       # 策略回测
    STRATEGY_OPTIMIZATION = "strategy_optimization" # 策略优化
    STRATEGY_DEPLOYMENT = "strategy_deployment"   # 策略部署
    
    # 交易执行 (新增)
    SIGNAL_GENERATION = "signal_generation"       # 信号生成
    ORDER_EXECUTION = "order_execution"           # 订单执行
    TRADE_PROCESSING = "trade_processing"         # 成交处理
    PORTFOLIO_REBALANCE = "portfolio_rebalance"   # 组合再平衡
    
    # 风险控制 (新增)
    RISK_CALCULATION = "risk_calculation"         # 风险计算
    RISK_MONITORING = "risk_monitoring"           # 风险监控
    COMPLIANCE_CHECK = "compliance_check"         # 合规检查
    RISK_REPORTING = "risk_reporting"             # 风险报告
    
    # 系统任务
    SYSTEM_MAINTENANCE = "system_maintenance"     # 系统维护
    SYSTEM_BACKUP = "system_backup"               # 系统备份
    REPORT_GENERATION = "report_generation"       # 报告生成
```

#### 3.1.3 触发器类型

```python
class TriggerType(Enum):
    INTERVAL = "interval"         # 间隔触发
    CRON = "cron"                 # Cron表达式
    DATE = "date"                 # 指定日期
    ONCE = "once"                 # 一次性
    EVENT = "event"               # 事件触发 (新增)
```

#### 3.1.4 任务数据类 (增强)

```python
@dataclass
class Task:
    id: str
    type: str
    status: TaskStatus
    priority: int
    created_at: datetime
    payload: Dict[str, Any]
    
    # 时间跟踪
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    deadline: Optional[datetime] = None          # 截止时间 (新增)
    
    # 执行结果
    result: Any = None
    error: Optional[str] = None
    worker_id: Optional[str] = None
    
    # 重试机制 (新增)
    max_retries: int = 0
    retry_count: int = 0
    retry_delay_seconds: int = 0
    
    # 超时控制 (新增)
    timeout_seconds: Optional[int] = None
    
    def should_retry(self) -> bool:
        """检查是否应该重试"""
        return self.retry_count < self.max_retries
    
    def is_timeout(self) -> bool:
        """检查是否超时"""
        if not self.deadline:
            return False
        return datetime.now() > self.deadline
```

---

## 4. 企业级功能详解

### 4.1 持久化层 (persistence/)

#### 4.1.1 数据模型

```python
# TaskModel - 任务持久化模型
class TaskModel(Base):
    __tablename__ = 'scheduler_tasks'
    
    id = Column(String(64), primary_key=True)
    type = Column(String(64), nullable=False)
    status = Column(String(32), nullable=False)
    priority = Column(Integer, default=5)
    payload = Column(JSON)
    result = Column(JSON)
    error = Column(Text)
    created_at = Column(DateTime)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    timeout_seconds = Column(Integer)
    max_retries = Column(Integer, default=0)
    retry_count = Column(Integer, default=0)

# JobModel - 定时任务持久化模型
class JobModel(Base):
    __tablename__ = 'scheduler_jobs'
    
    id = Column(String(64), primary_key=True)
    name = Column(String(256), nullable=False)
    job_type = Column(String(64), nullable=False)
    trigger_type = Column(String(32), nullable=False)
    trigger_config = Column(JSON)
    config = Column(JSON)
    enabled = Column(Boolean, default=True)
    created_at = Column(DateTime)
    last_run = Column(DateTime)
    next_run = Column(DateTime)
    run_count = Column(Integer, default=0)
```

#### 4.1.2 仓库模式

```python
class TaskRepository:
    def save(self, task: Task) -> bool
    def get_by_id(self, task_id: str) -> Optional[Task]
    def get_by_status(self, status: TaskStatus) -> List[Task]
    def get_by_type(self, task_type: str) -> List[Task]
    def update_status(self, task_id: str, status: TaskStatus) -> bool
    def delete(self, task_id: str) -> bool
    def get_statistics(self) -> Dict[str, Any]
```

### 4.2 告警系统 (alerting/)

#### 4.2.1 告警级别

```python
class AlertLevel(Enum):
    INFO = 0      # 信息
    WARNING = 1   # 警告
    ERROR = 2     # 错误
    CRITICAL = 3  # 严重
```

#### 4.2.2 告警渠道

```python
class AlertChannel(Enum):
    EMAIL = "email"       # 邮件
    WEBHOOK = "webhook"   # Webhook
    LOG = "log"           # 日志
    SMS = "sms"           # 短信
```

#### 4.2.3 告警场景

| 场景 | 级别 | 说明 |
|------|------|------|
| 任务失败 | ERROR | 任务执行失败 |
| 任务超时 | WARNING | 任务执行超时 |
| 重试耗尽 | ERROR | 任务重试次数耗尽 |
| 调度器错误 | CRITICAL | 调度器内部错误 |
| 任务提交 | INFO | 新任务提交 |

### 4.3 事件总线集成 (integration/)

#### 4.3.1 事件类型

```python
class SchedulerEventType:
    TASK_CREATED = "scheduler.task.created"
    TASK_STARTED = "scheduler.task.started"
    TASK_COMPLETED = "scheduler.task.completed"
    TASK_FAILED = "scheduler.task.failed"
    TASK_CANCELLED = "scheduler.task.cancelled"
    TASK_TIMEOUT = "scheduler.task.timeout"
    TASK_RETRIED = "scheduler.task.retried"
```

#### 4.3.2 事件驱动任务触发

```python
class EventDrivenTaskTrigger:
    def register_event_trigger(
        self,
        event_type: str,                    # 监听的事件类型
        task_type: str,                     # 触发的任务类型
        payload_template: Dict[str, Any],   # 任务数据模板
        priority: int = 5,
        timeout_seconds: Optional[int] = None
    )
    
    def unregister_event_trigger(self, event_type: str)
```

### 4.4 性能优化 (performance/)

#### 4.4.1 优先级队列

```python
class TaskPriority(Enum):
    CRITICAL = 1    # 关键任务（如风险控制）
    HIGH = 2        # 高优先级（如交易执行）
    NORMAL = 3      # 普通优先级（如数据同步）
    LOW = 4         # 低优先级（如报表生成）
    BACKGROUND = 5  # 后台任务（如数据清理）

class PriorityTaskQueue:
    def enqueue(self, task_id, task_type, priority, payload, timeout)
    def dequeue(self) -> Optional[PrioritizedTask]
    def update_priority(self, task_id, new_priority) -> bool
```

#### 4.4.2 批量处理器

```python
class BatchStrategy(Enum):
    SIZE_BASED = "size"      # 基于任务数量
    TIME_BASED = "time"      # 基于时间窗口
    HYBRID = "hybrid"        # 混合策略

class BatchProcessor:
    async def submit(self, task_id, task_type, payload, priority)
    def get_statistics(self) -> Dict[str, Any]
```

#### 4.4.3 任务缓存

```python
class TaskCache:
    async def get(self, task_type, payload, ttl_seconds) -> Optional[Any]
    async def set(self, task_type, payload, value, ttl_seconds)
    def get_statistics(self) -> Dict[str, Any]  # 命中率等
```

### 4.5 Prometheus指标 (metrics/)

#### 4.5.1 标准指标

| 指标名称 | 类型 | 说明 |
|----------|------|------|
| scheduler_tasks_submitted_total | Counter | 任务提交总数 |
| scheduler_tasks_completed_total | Counter | 任务完成总数 |
| scheduler_tasks_failed_total | Counter | 任务失败总数 |
| scheduler_tasks_running | Gauge | 当前运行中任务数 |
| scheduler_workers_active | Gauge | 活跃工作进程数 |
| scheduler_task_execution_duration_seconds | Histogram | 任务执行时间分布 |

### 4.6 安全模块 (security/)

#### 4.6.1 加密模块

```python
class EncryptionLevel(Enum):
    NONE = "none"
    PAYLOAD = "payload"
    FULL = "full"

class TaskEncryption:
    def encrypt_payload(self, payload: Dict) -> Dict
    def decrypt_payload(self, payload: Dict) -> Dict
```

#### 4.6.2 访问控制

```python
class Permission(Enum):
    TASK_SUBMIT = "task:submit"
    TASK_CANCEL = "task:cancel"
    TASK_VIEW = "task:view"
    ADMIN = "admin:*"

class Role(Enum):
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"
    SERVICE = "service"

class AccessControl:
    def create_user(self, user_id, name, role) -> User
    def create_api_key(self, name, role) -> tuple
    def validate_api_key(self, raw_key) -> Optional[APIKey]
    def check_permission(self, user_or_key, permission) -> bool
```

---

## 5. API设计

### 5.1 RESTful API端点

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/api/v1/scheduler/dashboard` | 调度器仪表板数据 |
| GET | `/api/v1/scheduler/status` | 调度器状态 |
| GET | `/api/v1/scheduler/metrics` | Prometheus指标 |
| POST | `/api/v1/scheduler/tasks` | 提交任务 |
| GET | `/api/v1/scheduler/tasks/running` | 运行中任务列表 |
| GET | `/api/v1/scheduler/tasks/completed` | 已完成任务列表 |
| GET | `/api/v1/scheduler/tasks/{task_id}` | 任务详情 |
| POST | `/api/v1/scheduler/tasks/{task_id}/pause` | 暂停任务 |
| POST | `/api/v1/scheduler/tasks/{task_id}/resume` | 恢复任务 |
| POST | `/api/v1/scheduler/tasks/{task_id}/cancel` | 取消任务 |
| POST | `/api/v1/scheduler/tasks/{task_id}/retry` | 重试任务 |
| GET | `/api/v1/scheduler/jobs` | 定时任务列表 |
| POST | `/api/v1/scheduler/jobs` | 创建定时任务 |
| DELETE | `/api/v1/scheduler/jobs/{job_id}` | 删除定时任务 |
| GET | `/api/v1/scheduler/config` | 获取配置 |
| POST | `/api/v1/scheduler/config` | 更新配置 |
| GET | `/api/v1/scheduler/alerts/config` | 获取告警配置 |
| POST | `/api/v1/scheduler/alerts/config` | 更新告警配置 |
| GET | `/api/v1/scheduler/events/triggers` | 获取事件触发器 |
| POST | `/api/v1/scheduler/events/triggers` | 注册事件触发器 |

---

## 6. 配置说明

### 6.1 完整配置

```python
{
    # 基础配置
    "max_workers": 4,                    # 最大工作进程数
    "max_task_history": 1000,            # 最大任务历史记录数
    "timezone": "Asia/Shanghai",         # 时区
    "check_interval": 1,                 # 调度检查间隔（秒）
    
    # 功能开关
    "enable_persistence": True,          # 启用数据库持久化
    "enable_alerting": True,             # 启用告警
    "enable_event_bus": True,            # 启用事件总线
    
    # 告警配置
    "alert_config": {
        "enabled": True,
        "channels": ["email", "webhook", "log"],
        "level_threshold": "warning",
        "rate_limit_seconds": 60
    },
    
    # 事件总线配置
    "event_bus_config": {
        "enabled": True
    }
}
```

---

## 7. 使用指南

### 7.1 基础使用

```python
from src.core.orchestration.scheduler import get_unified_scheduler

# 获取调度器实例（单例）
scheduler = get_unified_scheduler(
    max_workers=4,
    enable_persistence=True,
    enable_alerting=True,
    enable_event_bus=True
)

# 启动调度器
await scheduler.start()

# 提交任务（带超时和重试）
task_id = await scheduler.submit_task(
    task_type="data_collection",
    payload={"source": "alpha_vantage"},
    priority=5,
    timeout_seconds=3600,      # 1小时超时
    max_retries=3,             # 最多重试3次
    retry_delay_seconds=60     # 每次重试间隔60秒
)

# 停止调度器
await scheduler.stop()
```

### 7.2 注册事件触发器

```python
# 当市场数据更新时自动触发信号生成
scheduler.register_event_trigger(
    event_type="market.data.updated",
    task_type="signal_generation",
    payload_template={"strategy_id": "strategy_001"},
    priority=2,  # 高优先级
    timeout_seconds=30
)
```

### 7.3 使用优先级队列

```python
from src.core.orchestration.scheduler.performance import TaskPriority

# 提交关键任务（最高优先级）
task_id = await scheduler.submit_task(
    task_type="risk_calculation",
    payload={"portfolio_id": "portfolio_001"},
    priority=TaskPriority.CRITICAL.value
)
```

---

## 8. 性能指标

### 8.1 目标性能

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 调度器启动时间 | < 5秒 | 从start()调用到就绪 |
| 任务提交延迟 | < 10ms | 从submit到进入队列 |
| 任务执行吞吐量 | > 1000/秒 | 每秒处理任务数（批量模式） |
| 内存占用 | < 200MB | 空闲状态 |
| 数据库查询 | < 50ms | 单次查询延迟 |

---

## 9. 版本历史

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| 1.0.0 | 2026-03-05 | 初始版本，基础调度功能 |
| 2.0.0 | 2026-03-06 | 重大更新：添加持久化、告警、事件总线、性能优化、安全模块 |
| 2.1.0 | 2026-03-08 | 添加自动采集功能：支持按活跃数据源配置自动执行数据采集任务 |

---

## 10. 自动数据采集

### 10.1 功能概述

自动采集功能是统一调度器的数据采集自动化模块，支持按照数据源配置自动执行数据采集任务。

### 10.2 核心特性

| 特性 | 说明 | 状态 |
|------|------|------|
| 按配置采集 | 根据 `data_sources_config.json` 中的活跃数据源自动采集 | ✅ 已启用 |
| 定时检查 | 支持配置检查间隔（默认60秒） | ✅ 运行中 |
| 任务自动提交 | 自动为每个活跃数据源创建并提交采集任务 | ✅ 正常 |
| 状态监控 | 提供完整的自动采集状态API | ✅ 可用 |

### 10.3 自动采集流程

```
┌─────────────────────────────────────────────────────────────┐
│                    自动采集服务 (Auto Collection)              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ 定时检查循环  │───▶│ 读取活跃数据源 │───▶│ 创建采集任务 │  │
│  │ (60秒间隔)   │    │ (enabled=true)│    │ (TaskManager)│  │
│  └──────────────┘    └──────────────┘    └──────┬───────┘  │
│                                                  │          │
│  ┌──────────────┐    ┌──────────────┐           │          │
│  │ 记录采集历史  │◀───│ 提交到调度器  │◀──────────┘          │
│  │ (PostgreSQL) │    │ (UnifiedScheduler)                   │
│  └──────────────┘    └──────────────┘                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 10.4 API接口

#### 10.4.1 启动自动采集

```http
POST /api/v1/data/scheduler/auto-collection/start
```

**响应示例**:
```json
{
  "success": true,
  "message": "自动采集已启动",
  "timestamp": 1772974680.98
}
```

#### 10.4.2 停止自动采集

```http
POST /api/v1/data/scheduler/auto-collection/stop
```

#### 10.4.3 获取自动采集状态

```http
GET /api/v1/data/scheduler/auto-collection/status
```

**响应示例**:
```json
{
  "success": true,
  "data": {
    "running": true,
    "check_interval": 60,
    "total_checks": 1,
    "tasks_submitted": 0,
    "sources_checked": 2,
    "last_check_time": "2026-03-08T20:58:00",
    "next_check_time": "2026-03-08T20:59:00",
    "pending_tasks_count": 0
  },
  "timestamp": 1772974681.0
}
```

### 10.5 数据源配置

自动采集服务读取 `data/data_sources_config.json` 配置文件：

```json
[
  {
    "id": "akshare_stock_a",
    "name": "AKShare A股数据",
    "type": "股票数据",
    "enabled": true,           // 启用状态
    "status": "连接正常",
    "config": {
      "akshare_function": "stock_zh_a_spot_em"
    }
  },
  {
    "id": "baostock_stock_a",
    "name": "BaoStock A股数据",
    "type": "股票数据",
    "enabled": true,           // 启用状态
    "status": "HTTP 200 - 连接正常",
    "config": {
      "api_key": "xxx"
    }
  }
]
```

**关键字段**:
- `enabled`: 是否启用该数据源（`true` 表示活跃）
- `status`: 数据源连接状态
- `config`: 数据源特定配置

### 10.6 运行状态（2026-03-08）

| 指标 | 数值 | 状态 |
|------|------|------|
| 自动采集服务 | 运行中 | ✅ 正常 |
| 配置文件数据源 | 16个 | ✅ 正常 |
| 活跃数据源 | 2个 | ✅ 正常 |
| 已检查数据源 | 2个 | ✅ 正常 |
| 检查间隔 | 60秒 | ✅ 正常 |

**活跃数据源**:
1. AKShare A股数据 - 连接正常
2. BaoStock A股数据 - HTTP 200 - 连接正常

### 10.7 故障排查

#### 问题1: 自动采集显示活跃数据源为0

**症状**: 调度器 dashboard 显示 `active_sources: 0`

**原因**: 调度器统计的是数据采集器工作节点数量，而不是配置文件中的活跃数据源数量。

**解决**: 
- 检查自动采集状态API获取真实的活跃数据源数量
- 或启动自动采集服务，让数据采集器工作节点注册到调度器

#### 问题2: 自动采集已启动但没有提交任务

**症状**: `tasks_submitted: 0`

**排查步骤**:
1. 检查配置文件是否有 `enabled: true` 的数据源
2. 检查数据源配置是否正确加载
3. 检查调度器是否正常运行
4. 查看日志确认任务创建是否成功

---

## 11. 参考资料

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Python asyncio](https://docs.python.org/3/library/asyncio.html)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Prometheus Metrics](https://prometheus.io/docs/concepts/metric_types/)

---

*文档结束*
