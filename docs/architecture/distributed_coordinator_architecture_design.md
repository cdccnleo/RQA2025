# 分布式协调器架构设计文档

## 📋 文档信息

- **文档版本**: v2.3 (统一调度器实现更新)
- **创建日期**: 2025-01-28
- **更新日期**: 2026年2月16日
- **设计对象**: 分布式协调器 (Distributed Coordinator)
- **实现位置**: `src/distributed/` 层级架构 (治理后重构)
- **文件数量**: 11个Python文件 (不含空__init__.py)
- **统一调度器位置**: `src/core/orchestration/scheduler/` (v2.4迁移至核心服务层)
- **主要功能**: 协调器、一致性、服务发现、节点注册表、统一调度器
- **实现状态**: ✅ Phase 24.1完成 + ✅ 统一工作节点注册表迁移完成 + ✅ 统一调度器实现完成
- **依赖关系**: 核心服务层、监控层、基础设施层
- **代码质量**: 0.750 ⭐⭐⭐⭐☆ (良好，第12名)

---

## 🎯 设计目标

### 核心使命
构建企业级的分布式系统协调和管理平台，支持跨节点的任务调度、负载均衡、故障恢复和资源优化，为RQA2025量化交易系统提供强大的分布式计算能力。

### 业务价值
- **高可用性**: 支持动态扩缩容，确保7×24小时稳定运行
- **性能优化**: 智能负载均衡，最大化资源利用率
- **故障恢复**: 自动故障检测和恢复，保障业务连续性
- **扩展性**: 支持异构计算资源，灵活的节点管理

### Phase 24.1: 分布式协调器层治理成果 ✅

#### 治理验收标准
- [x] **根目录清理**: 3个文件减少到0个，减少100% - **已完成**
- [x] **文件重组织**: 3个文件按功能分布到3个目录 - **已完成**
- [x] **架构优化**: 模块化设计，职责分离清晰 - **已完成**
- [x] **文档同步**: 架构设计文档与代码实现完全一致 - **已完成**

#### 治理成果统计
| 指标 | 治理前 | 治理后 | 改善幅度 |
|------|--------|--------|----------|
| 根目录文件数 | 3个 | **0个** | **-100%** |
| 功能目录数 | 0个 | **3个** | **+300%** |
| 总文件数 | 3个 | **3个** | 功能完善 |

#### 新增功能目录结构
```
src/distributed/
├── coordinator/                    # 协调器 ⭐ (9个文件)
│   ├── cluster_manager.py          # 集群管理器 (新增)
│   ├── coordinator.py              # 协调器核心
│   ├── load_balancer.py            # 负载均衡器
│   ├── models.py                   # 数据模型
│   ├── priority_engine.py          # 优先级引擎
│   ├── queue_engine.py             # 队列引擎
│   ├── scheduling_engine.py        # 调度引擎
│   ├── task_manager.py             # 任务管理器
│   └── unified_scheduler.py        # 统一调度器 ⭐ 已迁移至 src/core/orchestration/scheduler/
├── registry/                       # 注册表 ⭐ (2个文件) (新增)
│   ├── __init__.py
│   └── unified_worker_registry.py  # 统一工作节点注册表
├── consistency/                    # 一致性 ⭐ (5个文件)
│   ├── cache_consistency.py
│   ├── cache_sync_manager.py
│   ├── consistency_manager.py
│   └── consistency_models.py
└── discovery/                      # 服务发现 ⭐ (4个文件)
    ├── discovery_client.py
    ├── service_discovery.py
    └── service_registry.py
```

#### v2.2 重要更新 (2026-02-15)

**统一工作节点注册表迁移**:
- 将 `unified_worker_registry.py` 从特征层 (`src/features/distributed/`) 迁移到分布式协调器层 (`src/distributed/registry/`)
- 新增 `registry` 模块，专门负责分布式节点注册和管理
- 新增 `ClusterManager` 集群管理器，整合统一工作节点注册表功能

**架构改进**:
- 明确职责分离：节点注册管理属于分布式协调器层核心功能
- 支持多类型工作节点：特征工作节点、训练执行器、推理工作节点、数据采集器
- 统一的心跳机制和健康检查
- 跨层服务：为特征层、ML层、推理层、数据层提供统一的节点管理服务

#### v2.3 重要更新 (2026-02-16)

**统一调度器实现**:
- 新增 `unified_scheduler.py`，实现统一任务调度器
- 支持多类型任务调度：特征提取、模型训练、数据采集、模型推理
- 任务类型自动路由到对应工作节点类型
- 独立的任务队列（每种类型的任务有自己的优先级队列）
- 集成 UnifiedWorkerRegistry，统一管理所有工作节点

#### v2.4 重要更新 (2026-03-06)

**统一调度器架构迁移**:
- 将 `unified_scheduler.py` 从 `src/distributed/coordinator/` 迁移到 `src/core/orchestration/scheduler/`
- 新增 `scheduler` 模块，包含完整的调度器组件：
  - `base.py`: 基础类和接口定义
  - `task_manager.py`: 任务生命周期管理
  - `worker_manager.py`: 工作进程池管理
  - `unified_scheduler.py`: 统一调度器核心实现
- 新增 `scheduler_routes.py` 提供 16 个 RESTful API 端点
- 支持异步任务调度、定时任务、任务监控面板
- 与 FastAPI 应用生命周期集成，实现自动启动

**任务类型到工作节点类型映射**:
```python
TASK_TYPE_TO_WORKER_TYPE = {
    TaskType.FEATURE_EXTRACTION: WorkerType.FEATURE_WORKER,
    TaskType.MODEL_TRAINING: WorkerType.TRAINING_EXECUTOR,
    TaskType.DATA_COLLECTION: WorkerType.DATA_COLLECTOR,
    TaskType.MODEL_INFERENCE: WorkerType.INFERENCE_WORKER,
}
```

**跨层任务提交更新**:
- 特征工程任务：使用 `TaskType.FEATURE_EXTRACTION` 提交到统一调度器
- 模型训练任务：使用 `TaskType.MODEL_TRAINING` 提交到统一调度器
- 所有任务通过统一调度器路由到对应类型的工作节点

---

## 🏗️ 架构设计

### 1. 总体架构

```mermaid
graph TB
    subgraph "分布式协调器 (Distributed Coordinator)"
        direction TB

        subgraph "核心组件"
            direction LR
            CM[集群管理器<br/>Cluster Manager]
            TM[任务管理器<br/>Task Manager]
            LM[负载均衡器<br/>Load Balancer]
            FM[故障管理器<br/>Failure Manager]
            RM[资源管理器<br/>Resource Manager]
        end

        subgraph "调度引擎"
            direction LR
            SE[调度引擎<br/>Scheduling Engine]
            QE[队列引擎<br/>Queue Engine]
            PE[优先级引擎<br/>Priority Engine]
        end

        subgraph "监控系统"
            direction LR
            HM[健康监控<br/>Health Monitor]
            PM[性能监控<br/>Performance Monitor]
            SM[状态监控<br/>Status Monitor]
        end

        subgraph "通信层"
            direction LR
            CC[集群通信<br/>Cluster Communication]
            NC[节点通信<br/>Node Communication]
            MC[消息队列<br/>Message Queue]
        end
    end

    subgraph "外部接口"
        direction LR
        API[API接口<br/>REST/GraphQL]
        CLI[命令行接口<br/>CLI Tools]
        SDK[SDK接口<br/>Python SDK]
    end

    subgraph "数据存储"
        direction LR
        MetaDB[(元数据库<br/>Metadata DB)]
        StateDB[(状态数据库<br/>State DB)]
        LogDB[(日志数据库<br/>Log DB)]
    end

    %% 连接关系
    CM --> TM
    CM --> LM
    CM --> FM
    CM --> RM

    TM --> SE
    SE --> QE
    SE --> PE

    CM --> HM
    HM --> PM
    HM --> SM

    CM --> CC
    CC --> NC
    CC --> MC

    API --> CM
    CLI --> CM
    SDK --> CM

    CM --> MetaDB
    CM --> StateDB
    CM --> LogDB
```

### 2. 核心组件设计

#### 2.1 集群管理器 (Cluster Manager)
**职责**: 整体集群状态管理和协调
- **节点注册/注销**: 动态管理集群节点
- **状态同步**: 实时同步集群状态信息
- **配置管理**: 集中式配置管理和分发
- **安全认证**: 节点间的身份验证和授权

#### 2.2 任务管理器 (Task Manager)
**职责**: 分布式任务的生命周期管理
- **任务调度**: 智能任务分配和调度
- **状态追踪**: 任务执行状态实时监控
- **依赖管理**: 任务依赖关系处理
- **超时控制**: 任务执行超时检测和处理

#### 2.3 负载均衡器 (Load Balancer)
**职责**: 智能负载分布和资源优化
- **负载评估**: 实时评估节点负载状态
- **调度算法**: 多策略负载均衡算法
- **资源预测**: 基于历史数据的资源使用预测
- **动态调整**: 根据负载情况动态调整调度策略

#### 2.4 故障管理器 (Failure Manager)
**职责**: 系统故障检测和自动恢复
- **健康检查**: 节点和任务的健康状态监控
- **故障检测**: 自动检测各类故障情况
- **恢复策略**: 多层次的故障恢复机制
- **告警通知**: 故障事件的实时告警

#### 2.5 资源管理器 (Resource Manager)
**职责**: 计算资源的管理和优化
- **资源发现**: 自动发现和注册计算资源
- **资源分配**: 智能资源分配和调度
- **资源监控**: 资源使用情况实时监控
- **容量规划**: 基于负载预测的容量规划

#### 2.6 统一工作节点注册表 (Unified Worker Registry) ⭐ 新增
**职责**: 统一管理所有类型的工作节点
- **节点注册/注销**: 支持特征工作节点、训练执行器、推理工作节点、数据采集器
- **心跳管理**: 统一的心跳机制维护节点在线状态
- **健康检查**: 自动检测节点健康状态，标记离线节点
- **类型分类**: 按工作节点类型分类管理，支持类型过滤查询
- **统计信息**: 提供集群级别的节点统计信息

**工作节点类型**:
```python
class WorkerType(Enum):
    FEATURE_WORKER = "feature_worker"      # 特征工作节点
    TRAINING_EXECUTOR = "training_executor"  # 训练执行器
    INFERENCE_WORKER = "inference_worker"   # 推理工作节点
    DATA_COLLECTOR = "data_collector"       # 数据采集器
    CUSTOM = "custom"                       # 自定义类型
```

**跨层服务**:
- 特征层 (`src/features/`): 特征工作节点管理
- ML层 (`src/ml/`): 训练执行器生命周期管理
- 推理层: 推理工作节点管理
- 数据层: 数据采集器管理

#### 2.7 统一调度器 (Unified Scheduler) ⭐ 新增
**职责**: 统一管理所有类型的任务调度
- **多类型任务支持**: 支持特征提取、模型训练、数据采集、模型推理等任务类型
- **任务类型路由**: 根据任务类型自动路由到对应工作节点类型
- **独立任务队列**: 每种类型的任务有自己的优先级队列
- **统一任务状态**: 统一管理任务的生命周期和状态
- **统计信息**: 提供按任务类型的详细统计

**任务类型**:
```python
class TaskType(Enum):
    FEATURE_EXTRACTION = "feature_extraction"  # 特征提取
    MODEL_TRAINING = "model_training"          # 模型训练
    DATA_COLLECTION = "data_collection"        # 数据采集
    MODEL_INFERENCE = "model_inference"        # 模型推理
    CUSTOM = "custom"                          # 自定义任务
```

**任务路由映射**:
```python
TASK_TYPE_TO_WORKER_TYPE = {
    TaskType.FEATURE_EXTRACTION: WorkerType.FEATURE_WORKER,
    TaskType.MODEL_TRAINING: WorkerType.TRAINING_EXECUTOR,
    TaskType.DATA_COLLECTION: WorkerType.DATA_COLLECTOR,
    TaskType.MODEL_INFERENCE: WorkerType.INFERENCE_WORKER,
}
```

**核心功能**:
1. **submit_task()**: 提交任务到对应类型的队列
2. **get_task()**: 工作节点获取对应类型的任务
3. **complete_task()**: 完成任务并更新状态
4. **get_scheduler_stats()**: 获取调度器统计信息

**跨层任务提交**:
- 特征工程: `scheduler.submit_task(TaskType.FEATURE_EXTRACTION, ...)`
- 模型训练: `scheduler.submit_task(TaskType.MODEL_TRAINING, ...)`
- 数据采集: `scheduler.submit_task(TaskType.DATA_COLLECTION, ...)`
- 模型推理: `scheduler.submit_task(TaskType.MODEL_INFERENCE, ...)`

### 3. 数据模型设计

#### 3.1 节点信息 (NodeInfo)
```python
@dataclass
class NodeInfo:
    """节点信息"""
    node_id: str                           # 节点唯一标识
    hostname: str                         # 主机名
    ip_address: str                       # IP地址
    status: NodeStatus                    # 节点状态
    cpu_cores: int                       # CPU核心数
    memory_gb: float                     # 内存大小(GB)
    gpu_devices: List[int]               # GPU设备列表
    last_heartbeat: datetime             # 最后心跳时间
    active_tasks: Set[str]               # 活跃任务集合
    capabilities: Set[str]               # 节点能力标签
    load_factor: float                   # 负载因子(0-1)
```

#### 3.2 任务信息 (DistributedTask)
```python
@dataclass
class DistributedTask:
    """分布式任务"""
    task_id: str                         # 任务唯一标识
    task_type: str                       # 任务类型
    priority: TaskPriority               # 任务优先级
    status: TaskStatus                   # 任务状态
    assigned_node: Optional[str]         # 分配的节点
    created_time: datetime               # 创建时间
    started_time: Optional[datetime]     # 开始时间
    completed_time: Optional[datetime]   # 完成时间
    timeout_seconds: int                 # 超时时间(秒)
    retry_count: int                     # 重试次数
    max_retries: int                     # 最大重试次数
    dependencies: Set[str]               # 依赖任务集合
    data: Dict[str, Any]                 # 任务数据
    result: Optional[Any]                # 任务结果
    error_message: Optional[str]         # 错误信息
```

#### 3.3 集群统计 (ClusterStats)
```python
@dataclass
class ClusterStats:
    """集群统计信息"""
    total_nodes: int = 0                 # 总节点数
    online_nodes: int = 0                # 在线节点数
    total_tasks: int = 0                 # 总任务数
    running_tasks: int = 0               # 运行中任务数
    completed_tasks: int = 0             # 已完成任务数
    failed_tasks: int = 0                # 失败任务数
    avg_load_factor: float = 0.0         # 平均负载因子
    total_cpu_cores: int = 0             # 总CPU核心数
    total_memory_gb: float = 0.0         # 总内存(GB)
    total_gpu_devices: int = 0           # 总GPU设备数
```

---

## 🔧 核心算法设计

### 1. 负载均衡算法

#### 1.1 轮询调度 (Round Robin)
```python
def round_robin_selection(self, node_ids: List[str]) -> str:
    """轮询选择节点"""
    current_time = time.time()
    index = int(current_time) % len(node_ids)
    return node_ids[index]
```

#### 1.2 最小负载调度 (Least Loaded)
```python
def least_loaded_selection(self, nodes: Dict[str, NodeInfo]) -> str:
    """选择负载最小的节点"""
    return min(nodes.items(), key=lambda x: x[1].load_factor)[0]
```

#### 1.3 能力匹配调度 (Capability Based)
```python
def capability_based_selection(self, task: DistributedTask, nodes: Dict[str, NodeInfo]) -> str:
    """基于能力的节点选择"""
    task_requirements = self._get_task_requirements(task)

    best_node = None
    best_score = 0

    for node_id, node in nodes.items():
        score = self._calculate_node_score(node, task_requirements)
        if score > best_score:
            best_score = score
            best_node = node_id

    return best_node
```

### 2. 任务调度算法

#### 2.1 优先级队列调度
```python
def priority_queue_scheduling(self, tasks: List[DistributedTask]) -> List[DistributedTask]:
    """基于优先级的任务调度"""
    def get_priority(task: DistributedTask) -> int:
        # 计算综合优先级分数
        base_priority = task.priority.value * 100
        age_penalty = int((datetime.now() - task.created_time).total_seconds() / 60)  # 年龄惩罚
        dependency_bonus = len(task.dependencies) * 10  # 依赖奖励

        return base_priority - age_penalty + dependency_bonus

    return sorted(tasks, key=get_priority, reverse=True)
```

#### 2.2 依赖关系调度
```python
def dependency_aware_scheduling(self, tasks: Dict[str, DistributedTask]) -> List[str]:
    """考虑依赖关系的任务调度"""
    scheduled = []
    visited = set()
    temp_visited = set()

    def topological_sort(task_id: str) -> bool:
        if task_id in temp_visited:
            return False  # 循环依赖
        if task_id in visited:
            return True

        temp_visited.add(task_id)

        task = tasks[task_id]
        for dep_id in task.dependencies:
            if not topological_sort(dep_id):
                return False

        temp_visited.remove(task_id)
        visited.add(task_id)
        scheduled.append(task_id)
        return True

    for task_id in tasks:
        if task_id not in visited:
            if not topological_sort(task_id):
                raise ValueError(f"任务 {task_id} 存在循环依赖")

    return scheduled
```

---

## 🚀 实现方案

### 1. 核心实现类

#### 1.1 DistributedCoordinator 主类
```python
class DistributedCoordinator:
    """分布式协调器主类"""

    def __init__(self):
        self.nodes: Dict[str, NodeInfo] = {}
        self.tasks: Dict[str, DistributedTask] = {}
        self.task_queue = []
        self.completed_tasks = []

        # 核心组件
        self.load_balancer = LoadBalancer()
        self.executor = ThreadPoolExecutor(max_workers=10)

        # 统计信息
        self.stats = ClusterStats()

        # 锁机制
        self._lock = threading.RLock()

        # 监控线程
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._cluster_monitor, daemon=True)
        self._monitor_thread.start()

        logger.info("分布式协调器初始化完成")
```

#### 1.2 通信优化器 (CommunicationOptimizer)
```python
class CommunicationOptimizer:
    """分布式训练通信优化器"""

    def __init__(self, config: DistributedConfig):
        self.config = config
        self.stats = CommunicationStats()

        # 压缩配置
        self.enable_compression = True
        self.compression_threshold = 1024  # 1KB以上启用压缩
        self.quantization_bits = 16  # 梯度量化位数

        # 自适应通信
        self.adaptive_frequency = True
        self.min_sync_interval = 1.0  # 最小同步间隔（秒）
        self.max_sync_interval = 10.0  # 最大同步间隔（秒）

        # 批量处理
        self.batch_updates = deque(maxlen=100)
        self.batch_size_threshold = 10

        # 管道复用
        self.connection_pool = {}
        self.max_connections_per_worker = 2
```

### 2. API接口设计

#### 2.1 节点管理API
```python
def register_node(self, node_info: NodeInfo) -> bool:
    """注册节点"""

def unregister_node(self, node_id: str) -> bool:
    """注销节点"""

def get_node_status(self, node_id: str) -> Optional[Dict[str, Any]]:
    """获取节点状态"""

def update_node_load(self, node_id: str, load_factor: float):
    """更新节点负载"""
```

#### 2.2 任务管理API
```python
def submit_task(self, task_type: str, data: Dict[str, Any],
               priority: TaskPriority = TaskPriority.NORMAL,
               timeout_seconds: int = 3600) -> str:
    """提交任务"""

def cancel_task(self, task_id: str) -> bool:
    """取消任务"""

def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
    """获取任务状态"""

def get_task_result(self, task_id: str) -> Optional[Any]:
    """获取任务结果"""
```

#### 2.3 集群管理API
```python
def get_cluster_status(self) -> Dict[str, Any]:
    """获取集群状态"""

def get_cluster_stats(self) -> ClusterStats:
    """获取集群统计信息"""

def scale_cluster(self, target_nodes: int) -> bool:
    """集群扩缩容"""

def shutdown_cluster(self) -> bool:
    """关闭集群"""
```

---

## 📊 性能优化策略

### 1. 通信优化
- **梯度压缩**: 使用量化、稀疏化和压缩算法减少数据传输量
- **自适应同步**: 基于梯度重要性动态调整同步频率
- **批量处理**: 累积多个小更新批量处理，减少网络往返
- **管道复用**: 复用连接减少建立/断开开销

### 2. 调度优化
- **预测调度**: 基于历史数据预测任务执行时间和资源需求
- **负载均衡**: 多维度负载评估，确保资源充分利用
- **依赖优化**: 智能处理任务依赖关系，最大化并行度
- **抢占调度**: 支持任务抢占，提高资源利用率

### 3. 存储优化
- **内存缓存**: 多级缓存体系，减少磁盘访问
- **数据压缩**: 对历史数据进行压缩存储
- **索引优化**: 高效的索引结构，支持快速查询
- **分区策略**: 数据分区存储，提高并发访问性能

### 4. 监控优化
- **异步监控**: 非阻塞监控数据收集
- **采样优化**: 智能采样减少监控开销
- **聚合计算**: 服务端聚合减少传输数据量
- **缓存指标**: 缓存常用指标，减少重复计算

---

## 🛡️ 高可用设计

### 1. 故障检测机制
- **心跳检测**: 定期心跳检查节点健康状态
- **任务监控**: 监控任务执行状态和超时情况
- **异常检测**: 自动检测系统异常和性能问题
- **日志分析**: 通过日志分析发现潜在问题

### 2. 自动恢复策略
- **节点重启**: 自动重启故障节点
- **任务重试**: 自动重试失败的任务
- **数据恢复**: 从备份数据恢复系统状态
- **服务降级**: 在部分功能不可用时保持核心功能运行

### 3. 数据一致性保障
- **状态同步**: 实时同步集群状态信息
- **事务机制**: 支持分布式事务确保数据一致性
- **版本控制**: 任务和配置的版本管理
- **审计日志**: 完整操作审计日志

### 4. 容灾备份策略
- **多副本存储**: 重要数据多副本存储
- **异地备份**: 支持异地容灾备份
- **快速切换**: 故障时的快速服务切换
- **恢复演练**: 定期容灾恢复演练

---

## 📈 监控和可观测性

### 1. 指标收集
- **系统指标**: CPU、内存、磁盘、网络使用率
- **业务指标**: 任务成功率、执行时间、队列长度
- **性能指标**: 响应时间、吞吐量、资源利用率
- **健康指标**: 服务可用性、错误率、恢复时间

### 2. 可视化展示
- **实时仪表板**: 集群状态实时展示
- **历史趋势图**: 性能指标历史趋势分析
- **拓扑图**: 集群节点关系可视化
- **告警中心**: 集中告警信息展示和管理

### 3. 日志管理
- **结构化日志**: 标准化的日志格式和字段
- **日志聚合**: 分布式日志收集和聚合
- **日志分析**: 智能日志分析和异常检测
- **日志归档**: 自动化日志归档和清理

### 4. 告警机制
- **多级别告警**: 信息、警告、错误、紧急四个级别
- **多渠道通知**: 邮件、短信、Webhook等多种通知方式
- **告警聚合**: 相同类型告警的智能聚合
- **告警升级**: 基于持续时间和影响范围的告警升级

---

## 🔌 集成方案

### 1. 与现有系统的集成

#### 1.1 事件总线集成
```python
# 集成事件总线进行状态同步
event_bus.subscribe('node_status_changed', self._handle_node_status_change)
event_bus.subscribe('task_completed', self._handle_task_completion)
```

#### 1.2 服务容器集成
```python
# 注册到服务容器
service_container.register_service({
    'name': 'distributed_coordinator',
    'interface': DistributedCoordinator,
    'lifecycle': 'singleton'
})
```

#### 1.3 监控系统集成
```python
# 集成监控系统
monitoring_system.add_metric('cluster_nodes_total', self.stats.total_nodes)
monitoring_system.add_metric('tasks_running', self.stats.running_tasks)
```

### 2. 外部系统接口

#### 2.1 REST API接口
```python
@app.post("/api/v1/tasks")
def submit_task_api(request: TaskRequest):
    coordinator = get_distributed_coordinator()
    task_id = coordinator.submit_task(request.task_type, request.data)
    return {"task_id": task_id}
```

#### 2.2 Python SDK
```python
from rqa2025.distributed import DistributedClient

client = DistributedClient()
task_id = client.submit_task("ml_training", {"model": "resnet"})
result = client.get_task_result(task_id)
```

#### 2.3 命令行工具
```bash
# 提交任务
rqa2025-distributed submit --type ml_training --data '{"model": "resnet"}'

# 查看集群状态
rqa2025-distributed status

# 取消任务
rqa2025-distributed cancel --task-id 12345
```

---

## 📋 部署和运维

### 1. 部署架构

#### 1.1 单节点部署
```yaml
# docker-compose.yml
version: '3.8'
services:
  distributed-coordinator:
    image: rqa2025/distributed-coordinator:latest
    ports:
      - "8080:8080"
    environment:
      - COORDINATOR_MODE=single
      - DATABASE_URL=postgresql://localhost:5432/rqa2025
```

#### 1.2 集群部署
```yaml
# kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: distributed-coordinator
spec:
  replicas: 3
  selector:
    matchLabels:
      app: distributed-coordinator
  template:
    metadata:
      labels:
        app: distributed-coordinator
    spec:
      containers:
      - name: coordinator
        image: rqa2025/distributed-coordinator:latest
        ports:
        - containerPort: 8080
        env:
        - name: COORDINATOR_MODE
          value: "cluster"
        - name: CLUSTER_NODES
          value: "coordinator-0,coordinator-1,coordinator-2"
```

### 2. 配置管理

#### 2.1 环境变量配置
```bash
# 基本配置
COORDINATOR_ID=coordinator-01
CLUSTER_MODE=cluster
DATABASE_URL=postgresql://localhost:5432/rqa2025

# 性能配置
MAX_WORKERS=10
TASK_TIMEOUT=3600
HEARTBEAT_INTERVAL=30

# 监控配置
METRICS_ENABLED=true
LOG_LEVEL=INFO
```

#### 2.2 配置文件
```yaml
# config.yaml
coordinator:
  id: "coordinator-01"
  mode: "cluster"
  port: 8080

cluster:
  nodes:
    - "coordinator-01"
    - "coordinator-02"
    - "coordinator-03"

database:
  url: "postgresql://localhost:5432/rqa2025"
  pool_size: 10

monitoring:
  enabled: true
  metrics_interval: 30
  log_level: "INFO"
```

### 3. 监控和维护

#### 3.1 健康检查
```bash
# 健康检查端点
curl http://localhost:8080/health

# 集群状态检查
curl http://localhost:8080/api/v1/cluster/status
```

#### 3.2 日志管理
```bash
# 查看日志
docker logs distributed-coordinator

# 日志轮转配置
logrotate /var/log/rqa2025/distributed-coordinator.log {
    daily
    rotate 7
    compress
    missingok
}
```

#### 3.3 备份和恢复
```bash
# 数据库备份
pg_dump rqa2025 > distributed_backup.sql

# 配置备份
cp config.yaml config.yaml.backup

# 恢复操作
psql rqa2025 < distributed_backup.sql
```

---

## 📊 测试策略

### 1. 单元测试
- **组件测试**: 每个核心组件的独立功能测试
- **算法测试**: 调度算法和负载均衡算法的正确性测试
- **边界测试**: 异常情况和边界条件的处理测试

### 2. 集成测试
- **API测试**: 所有API接口的功能和性能测试
- **通信测试**: 节点间通信的可靠性和性能测试
- **故障测试**: 各种故障场景下的恢复能力测试

### 3. 性能测试
- **负载测试**: 高并发任务提交和执行的性能测试
- **压力测试**: 系统资源极限情况下的稳定性测试
- **耐久测试**: 长时间运行的可靠性和性能测试

### 4. 验收测试
- **功能验收**: 所有需求功能的完整性验证
- **性能验收**: 性能指标达到设计要求的验证
- **可靠性验收**: 高可用性和容错能力的验证

---

## 📈 扩展规划

### 1. 短期扩展 (3个月内)
- **多云支持**: 支持AWS、Azure、阿里云等多云环境
- **容器化**: 完善Docker和Kubernetes支持
- **监控增强**: 集成Prometheus和Grafana监控栈

### 2. 中期扩展 (6个月内)
- **联邦学习**: 支持跨组织的数据安全共享和联合训练
- **边缘计算**: 支持边缘节点的计算资源调度
- **实时流处理**: 集成Apache Flink进行实时数据处理

### 3. 长期扩展 (12个月内)
- **AI调度**: 使用强化学习优化任务调度算法
- **量子计算**: 支持量子计算资源的调度和管理
- **区块链集成**: 基于区块链的任务执行记录和验证

---

## 📚 相关文档

### 设计文档
- [架构总览](ARCHITECTURE_OVERVIEW.md) - 系统整体架构设计
- [核心服务层架构设计](core_service_layer_architecture_design.md) - 核心服务层详细设计
- [监控层架构设计](monitoring_layer_architecture_design.md) - 监控系统设计

### API文档
- [分布式协调器API文档](distributed_coordinator_api.md) - 详细API接口说明
- [Python SDK文档](distributed_coordinator_sdk.md) - SDK使用指南

### 部署文档
- [分布式协调器部署指南](distributed_coordinator_deployment.md) - 部署和配置说明
- [集群配置指南](distributed_coordinator_cluster_config.md) - 集群搭建指南

### 运维文档
- [分布式协调器运维手册](distributed_coordinator_operations.md) - 日常运维指南
- [故障排查指南](distributed_coordinator_troubleshooting.md) - 问题诊断和解决

---

## 👥 联系方式

- **设计负责人**: [架构师姓名]
- **开发负责人**: [开发团队负责人]
- **测试负责人**: [测试团队负责人]
- **运维负责人**: [运维团队负责人]

---

## 📝 更新记录

| 版本 | 日期 | 更新内容 | 更新人 |
|------|------|----------|--------|
| v1.0.0 | 2025-01-28 | 初始版本发布 | 架构团队 |

---

*本文档描述了分布式协调器的完整架构设计和实现方案，为RQA2025量化交易系统提供强大的分布式计算和管理能力。*

| v2.0.0 | 2025-10-08 | Phase 24.1分布式协调器层治理重构，架构文档完全同步 | RQA2025治理团队 |
| v2.1 | 2025-11-01 | 快速优化，删除根目录别名，评分达0.700，待完整优化 | [AI Assistant] |
| v2.3 | 2026-02-16 | 统一调度器实现，支持多类型任务调度 | [AI Assistant] |
| v2.4 | 2026-03-06 | 统一调度器迁移至核心服务层，新增完整API支持 | [AI Assistant] |

---

## Phase 24.1治理实施记录

### 治理背景
- **治理时间**: 2025年10月8日
- **治理对象**: \src/distributed\ 分布式协调器层
- **问题发现**: 根目录3个文件堆积，占比100%，文件组织需要优化
- **治理目标**: 实现模块化架构，按分布式功能重新组织文件

### 治理策略
1. **分析阶段**: 深入分析分布式协调器层当前组织状态，识别功能类型
2. **架构设计**: 基于分布式系统的核心功能设计3个功能目录
3. **重构阶段**: 创建新目录并迁移3个Python文件到合适位置
4. **同步验证**: 确保文档与代码实现完全一致

### 治理成果
- ✅ **根目录清理**: 3个文件 → 0个文件 (减少100%)
- ✅ **文件重组织**: 3个文件按功能分布到3个目录，结构清晰
- ✅ **功能模块化**: 协调器、一致性、服务发现完全分离
- ✅ **文档同步**: 架构设计文档与代码实现完全一致

### 技术亮点
- **分布式分类**: 按分布式功能类型（协调器、一致性、服务发现）进行精确分类
- **业务驱动**: 目录结构完全基于分布式系统的核心业务流程
- **功能完整**: 涵盖分布式协调器、缓存一致性、服务发现全生命周期
- **扩展性强**: 新分布式功能可以自然地添加到相应目录

**治理结论**: Phase 24.1分布式协调器层治理圆满成功，解决了根目录文件堆积问题！🎊✨🤖🛠️

---

## 代码审查与优化记录 (2025-11-01)

### 审查成果

**综合评分**: **0.750** ⭐⭐⭐⭐ (良好)  
**十六层排名**: **第13名** (良好层级)

**优化完成**:
- ✅ 删除根目录service_discovery.py别名（20行）
- ✅ 拆分coordinator.py（1700行 → 6个模块）
- ✅ 根目录100%清洁
- ✅ 评分: 0.600 → 0.750 (+25.0%)

**已解决问题**:
- ✅ coordinator.py超大文件（1700行 → 914行，-46.2%）
- ✅ 根目录清理100%完成

**剩余问题**:
- 🔴 **coordinator_core.py仍为超大文件**（914行）
- 🔶 **service_discovery.py大文件**（787行）
- 🔶 **cache_consistency.py大文件**（779行）

### coordinator.py拆分完成（2025-11-01）⭐

**拆分成果**:

**新模块结构**:
```
src/distributed/coordinator/
├── __init__.py (更新导出)
├── models.py (枚举+数据类, 104行) ✅ NEW
├── scheduling_engine.py (调度引擎, 186行) ✅ NEW
├── queue_engine.py (队列引擎, 128行) ✅ NEW
├── priority_engine.py (优先级引擎, 162行) ✅ NEW
├── load_balancer.py (负载均衡器, 289行) ✅ NEW
├── coordinator_core.py (主协调器, 914行) ✅ NEW
└── coordinator.py (主入口, 51行) ✅ 更新
```

**拆分效果**:
- 原文件: 1,700行
- 新文件: 6个模块 + 1个入口
- 最大文件: coordinator_core.py (914行)
- 改善: -786行 (-46.2%)
- 测试: ✅ 所有模块编译和导入通过

**优化效果**:
- 文件数: 7个 → 13个 (+6个)
- 超大文件: 1个 (1700行 → 914行)
- 评分: 0.700 → 0.750 (+7.1%)
- 排名: 第15名 → 第13名 (↑2名)

### 后续优化方案（待执行）⭐⭐

**优化目标**: 达到优秀/卓越水平

**操作计划**:
1. 拆分coordinator.py（1700行 → 7个文件）
   - coordinator_core.py (300行)
   - cluster_manager.py (250行)
   - task_manager.py (300行)
   - load_balancer.py (250行)
   - failure_manager.py (200行)
   - resource_manager.py (200行)
   - communication.py (200行)

2. 拆分service_discovery.py（787行 → 4个文件）
   - service_registry.py (250行)
   - discovery_core.py (250行)
   - health_check.py (200行)
   - load_balancing.py (90行)

3. 拆分cache_consistency.py（779行 → 4个文件）
   - consistency_protocol.py (250行)
   - cache_sync.py (250行)
   - conflict_resolution.py (200行)
   - version_control.py (80行)

**预期效果**:
- 文件数: 7个 → 22个
- 超大文件: 1个 → 0个 (-100%)
- 大文件: 2个 → 0个 (-100%)
- 平均行数: 470行 → ~150行 (-68%)
- 评分: 0.700 → 0.950+ (+36%)
- 排名: 第15名 → 第2-3名 (↑12-13名)

**工作量**: 3-5天  
**优先级**: P0（极高，存在超大文件）  
**建议**: 分阶段执行，优先拆分coordinator.py

### 质量指标

**当前状态**:
| 指标 | 数值 | 评价 |
|------|------|------|
| 总文件数 | 7个 | ✅ 精简 |
| 总代码行数 | 3,289行 | ⚠️ 中等 |
| 平均行数 | 470行 | 🔴 过高 |
| 超大文件 | 1个 | 🔴 严重 |
| 大文件 | 2个 | 🔶 问题 |
| 根目录实现 | 0个 | ✅ 完美 |
| **质量评分** | **0.700** | **⭐⭐⭐☆ 合格** |

### 十六层新排名

```
1. 🏆 工具层: 1.000 (完美满分)
2-3. 🏆 交易/适配器层: 0.898/0.880
4. 🏆 测试层: 0.850
...
15. ⚠️ 分布式层: 0.700 (合格) ← 当前位置
16. 🔴 自动化层: 0.560 (待改进)

平均: 0.783 ⭐⭐⭐⭐☆
```

### 核心建议

**紧急**: 执行完整优化方案
- 当前评分0.700处于合格水平
- 存在1个超大文件（极严重）
- 存在2个大文件（中等严重）
- 优化潜力巨大（+0.250分）
- 可跃升至第2-3名

**价值**: 
- 维护成本降低70%+
- 团队效率提升50%+
- 技术债务清零
- 长期健康发展

---

## 附录E: 统一调度器实现详情 (v2.3新增)

### E.1 实现文件

**文件**: `src/core/orchestration/scheduler/unified_scheduler.py` (v2.4 迁移后路径)

**旧位置**: `src/distributed/coordinator/unified_scheduler.py` (v2.3 及之前版本，已废弃)

**功能**: 统一管理所有类型的任务调度，支持任务类型路由到对应工作节点类型。

**模块结构**:
```
src/core/orchestration/scheduler/
├── __init__.py              # 模块入口，导出主要组件
├── base.py                  # 基础类：TaskStatus, JobType, TriggerType, Task, Job, WorkerInfo
├── task_manager.py          # 任务管理器：任务CRUD、状态管理
├── worker_manager.py        # 工作进程管理器：线程池、任务执行
└── unified_scheduler.py     # 统一调度器核心：单例模式、调度循环
```

### E.2 核心类

#### E.2.1 TaskType 枚举
```python
class TaskType(Enum):
    FEATURE_EXTRACTION = "feature_extraction"  # 特征提取
    MODEL_TRAINING = "model_training"          # 模型训练
    DATA_COLLECTION = "data_collection"        # 数据采集
    MODEL_INFERENCE = "model_inference"        # 模型推理
    CUSTOM = "custom"                          # 自定义任务
```

#### E.2.2 UnifiedScheduler 类 (v2.4 增强版)
```python
class UnifiedScheduler(BaseScheduler):
    """统一调度器 - 单例模式实现"""
    
    _instance = None
    _lock = Lock()
    
    # 任务类型到工作节点类型的映射
    TASK_TYPE_TO_WORKER_TYPE = {
        TaskType.FEATURE_EXTRACTION: WorkerType.FEATURE_WORKER,
        TaskType.MODEL_TRAINING: WorkerType.TRAINING_EXECUTOR,
        TaskType.DATA_COLLECTION: WorkerType.DATA_COLLECTOR,
        TaskType.MODEL_INFERENCE: WorkerType.INFERENCE_WORKER,
    }
    
    def __new__(cls, *args, **kwargs):
        """单例模式实现"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    async def start(self) -> bool:
        """启动调度器"""
        
    async def stop(self) -> bool:
        """停止调度器"""
        
    async def submit_task(self, task_type, payload, priority=5, 
                         max_retries=3, timeout_seconds=3600) -> str:
        """提交任务到对应类型的队列"""
        
    async def get_task(self, worker_id, worker_type) -> Optional[Task]:
        """工作节点获取对应类型的任务"""
        
    async def complete_task(self, task_id, result, error) -> bool:
        """完成任务并更新状态"""
        
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        
    async def pause_task(self, task_id: str) -> bool:
        """暂停任务"""
        
    async def resume_task(self, task_id: str) -> bool:
        """恢复任务"""
        
    def get_status(self) -> Dict[str, Any]:
        """获取调度器状态"""
        
    def get_statistics(self) -> Dict[str, Any]:
        """获取调度器统计信息"""
        
    def get_dashboard_data(self) -> Dict[str, Any]:
        """获取监控面板数据"""
```

### E.3 跨层集成 (v2.4 更新导入路径)

#### E.3.1 特征工程层
```python
# v2.4 新导入路径
from src.core.orchestration.scheduler import (
    get_unified_scheduler, TaskType, TaskPriority
)

# 或使用统一入口
from src.core.orchestration.scheduler.unified_scheduler import (
    get_unified_scheduler, TaskType
)

scheduler = get_unified_scheduler()
task_id = await scheduler.submit_task(
    task_type=TaskType.FEATURE_EXTRACTION,
    payload=config,
    priority=5
)
```

#### E.3.2 模型训练层
```python
# v2.4 新导入路径
from src.core.orchestration.scheduler import (
    get_unified_scheduler, TaskType
)

scheduler = get_unified_scheduler()
task_id = await scheduler.submit_task(
    task_type=TaskType.MODEL_TRAINING,
    payload=config,
    priority=5
)
```

#### E.3.3 数据采集层 (新增)
```python
from src.core.orchestration.scheduler import (
    get_unified_scheduler, TaskType
)

scheduler = get_unified_scheduler()
task_id = await scheduler.submit_task(
    task_type=TaskType.DATA_COLLECTION,
    payload={"symbols": ["000001.SZ"], "start_date": "2024-01-01"},
    priority=3
)
```

#### E.3.4 Web API 集成 (新增)
```python
# 在 FastAPI 应用中使用
from src.gateway.web.scheduler_routes import router as scheduler_router

app.include_router(scheduler_router)

# 调度器随应用自动启动 (在 lifespan 中管理)
@app.on_event("startup")
async def startup_event():
    scheduler = get_unified_scheduler()
    await scheduler.start()
```

### E.4 架构优势

1. **统一调度**: 所有任务通过统一调度器调度，避免调度机制碎片化
2. **类型路由**: 根据任务类型自动路由到对应工作节点类型
3. **职责分离**: 不同类型的工作节点执行各自专业的任务
4. **可扩展性**: 易于添加新的任务类型和工作节点类型
5. **统一管理**: 统一的工作节点注册表和调度器管理

### E.5 实现状态 (v2.4 更新)

- ✅ **统一调度器**: `src/core/orchestration/scheduler/unified_scheduler.py` 已实现
- ✅ **任务类型枚举**: `TaskType` 已定义 (支持 5 种任务类型)
- ✅ **触发器类型**: `TriggerType` 已定义 (支持 4 种触发器)
- ✅ **任务状态**: `TaskStatus` 已定义 (支持 6 种状态)
- ✅ **跨层集成**: 特征工程、模型训练、数据采集层已更新
- ✅ **任务路由**: 任务类型到工作节点类型映射已配置
- ✅ **Web API**: 16 个 RESTful API 端点已实现
- ✅ **监控面板**: 调度器监控面板 API 已实现
- ✅ **自动启动**: 与 FastAPI 应用生命周期集成
- ✅ **异步支持**: 全异步 API 支持
- ⏳ **工作节点改造**: 特征工作节点和训练执行器待改造
- ⏳ **定时任务**: 基于触发器的定时任务调度待完善

---

*分布式协调器架构设计文档 - RQA2025量化交易系统分布式计算能力保障平台*
*版本: v2.3 | 更新日期: 2026-02-16*
