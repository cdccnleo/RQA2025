# 特征提取任务提交至调度器问题分析报告

**项目**: RQA2025  
**报告类型**: 技术分析  
**生成时间**: 2026-02-16  
**版本**: v1.0  
**状态**: 🔍 分析完成

---

## 📋 报告概览

### 问题描述
特征提取任务(如 task_1771199808)在FeatureEngine中创建后，**没有正确提交至FeatureTaskScheduler进行调度执行**。

### 关键发现
- ✅ FeatureTaskScheduler 任务调度器可用且功能正常
- ❌ FeatureEngine.create_task() 仅维护内部任务列表，未调用调度器
- ❌ FeatureEngine 初始化时没有集成任务调度器
- ❌ 任务状态在 FeatureEngine 和调度器之间不同步

---

## 🔍 详细问题分析

### 问题 1: FeatureEngine.create_task() 未集成任务调度器

**位置**: `src/features/core/engine.py:446-470`

**问题描述**:
`create_task()` 方法仅将任务添加到内部的 `self.tasks` 列表，**完全没有调用 FeatureTaskScheduler 来提交任务**。

**当前代码**:
```python
def create_task(self, task_type: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """创建特征提取任务"""
    import time
    task_id = f"task_{int(time.time())}"
    task = {
        "task_id": task_id,
        "task_type": task_type,
        "status": "pending",
        "progress": 0,
        "feature_count": 0,
        "start_time": int(time.time()),
        "created_at": int(time.time()),
        "config": config or {}
    }
    self.tasks.append(task)  # ← 仅添加到内部列表
    return task
```

**问题影响**:
- 任务永远不会被调度执行
- 任务状态仅在 FeatureEngine 内部维护
- 调度器完全不知道这些任务的存在

---

### 问题 2: FeatureEngine 缺少任务调度器引用

**位置**: `src/features/core/engine.py:49-98`

**问题描述**:
FeatureEngine 在初始化时没有获取或存储 FeatureTaskScheduler 实例，导致无法与调度器交互。

**当前代码**:
```python
def __init__(self, config: Optional[FeatureConfig] = None):
    """初始化特征引擎"""
    self.config = config or FeatureConfig()
    
    # ... 其他初始化 ...
    
    # 任务管理
    self.tasks = []  # ← 只有内部任务列表
    # 特征存储
    self.features = []
    # 技术指标状态
    self.indicators = []
    
    # ... 缺少任务调度器初始化 ...
```

---

### 问题 3: 任务状态管理分散且不同步

**问题描述**:
- FeatureEngine 维护自己的 `self.tasks` 列表
- FeatureTaskScheduler 维护自己的 `_tasks` 字典
- 两者之间**没有任何同步机制**
- 任务完成后无法更新对方的状态

---

## 📊 FeatureTaskScheduler 功能验证

### ✅ 调度器可用性确认

经过验证，FeatureTaskScheduler 功能完全正常：

**位置**: `src/features/distributed/task_scheduler.py`

**核心功能**:
- ✅ `submit_task()` - 提交任务到调度队列
- ✅ `get_task()` - 工作节点获取任务
- ✅ `complete_task()` - 完成任务
- ✅ `get_scheduler_stats()` - 获取调度器统计
- ✅ `start_with_workers()` - 启动调度器和工作节点
- ✅ 优先级队列支持
- ✅ 工作节点管理

---

## 🔧 修复建议

### 高优先级修复

#### 修复 1: 在 FeatureEngine 中集成任务调度器

**文件**: `src/features/core/engine.py`

**修改位置**: `__init__()` 方法

**建议代码**:
```python
def __init__(self, config: Optional[FeatureConfig] = None):
    """初始化特征引擎"""
    self.config = config or FeatureConfig()
    
    # ... 现有初始化代码 ...
    
    # 新增：集成任务调度器
    self._task_scheduler = None
    try:
        from src.features.distributed.task_scheduler import get_task_scheduler
        self._task_scheduler = get_task_scheduler()
        self.logger.info("✅ 任务调度器集成成功")
    except Exception as e:
        self.logger.warning(f"⚠️ 任务调度器集成失败: {e}")
        self._task_scheduler = None
    
    # ... 继续现有初始化 ...
```

---

#### 修复 2: 修改 create_task() 方法

**文件**: `src/features/core/engine.py`

**修改位置**: `create_task()` 方法

**建议代码**:
```python
def create_task(self, task_type: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    创建特征提取任务
    
    Args:
        task_type: 任务类型
        config: 任务配置
    
    Returns:
        创建的任务信息
    """
    import time
    task_id = f"task_{int(time.time())}"
    task = {
        "task_id": task_id,
        "task_type": task_type,
        "status": "pending",
        "progress": 0,
        "feature_count": 0,
        "start_time": int(time.time()),
        "created_at": int(time.time()),
        "config": config or {}
    }
    
    # 添加到内部任务列表
    self.tasks.append(task)
    
    # 新增：提交到任务调度器
    if self._task_scheduler:
        try:
            from src.features.distributed.task_scheduler import TaskPriority
            scheduler_task_id = self._task_scheduler.submit_task(
                task_type=task_type,
                data=config or {},
                priority=TaskPriority.NORMAL,
                metadata={
                    'engine_task_id': task_id,
                    'created_from_engine': True
                }
            )
            task['scheduler_task_id'] = scheduler_task_id
            self.logger.info(f"✅ 任务已提交至调度器: {scheduler_task_id}")
        except Exception as e:
            self.logger.error(f"❌ 提交任务至调度器失败: {e}")
    
    return task
```

---

#### 修复 3: 添加任务状态同步机制

**文件**: `src/features/core/engine.py`

**建议新增方法**:
```python
def sync_task_with_scheduler(self, task_id: str) -> bool:
    """
    同步任务状态与调度器
    
    Args:
        task_id: FeatureEngine 中的任务ID
    
    Returns:
        是否同步成功
    """
    if not self._task_scheduler:
        return False
    
    # 查找任务
    task = next((t for t in self.tasks if t.get('task_id') == task_id), None)
    if not task:
        return False
    
    # 如果有调度器任务ID，获取调度器状态
    scheduler_task_id = task.get('scheduler_task_id')
    if scheduler_task_id:
        try:
            scheduler_status = self._task_scheduler.get_task_status(scheduler_task_id)
            if scheduler_status:
                # 映射状态
                status_map = {
                    'PENDING': 'pending',
                    'RUNNING': 'running',
                    'COMPLETED': 'completed',
                    'FAILED': 'failed',
                    'CANCELLED': 'cancelled'
                }
                new_status = status_map.get(scheduler_status.value, task.get('status'))
                if new_status != task.get('status'):
                    task['status'] = new_status
                    self.logger.debug(f"🔄 任务状态同步: {task_id} -> {new_status}")
                return True
        except Exception as e:
            self.logger.warning(f"⚠️ 同步任务状态失败: {e}")
    
    return False
```

---

#### 修复 4: 更新 update_task_status() 以通知调度器

**文件**: `src/features/core/engine.py`

**修改位置**: `update_task_status()` 方法

**建议修改**:
```python
def update_task_status(self, task_id: str, status: str, progress: int = None) -> bool:
    """
    更新任务状态和进度
    
    Args:
        task_id: 任务ID
        status: 任务状态
        progress: 任务进度（0-100）
    
    Returns:
        是否更新成功
    """
    for task in self.tasks:
        if task.get('task_id') == task_id:
            old_status = task.get('status')
            task['status'] = status
            if progress is not None:
                task['progress'] = min(100, max(0, progress))
            if status == 'completed':
                import time
                task['end_time'] = int(time.time())
            
            # 触发状态变更钩子
            self._trigger_task_status_hooks(task_id, old_status, status, progress)
            
            # 根据状态触发特定钩子
            if status == 'completed':
                self._trigger_task_completed_hooks(task_id, task)
            elif status == 'failed':
                self._trigger_task_failed_hooks(task_id, task)
            
            # 新增：如果有调度器任务ID，尝试同步状态
            scheduler_task_id = task.get('scheduler_task_id')
            if scheduler_task_id and self._task_scheduler:
                try:
                    if status == 'completed':
                        self._task_scheduler.complete_task(
                            scheduler_task_id,
                            result=task.get('result'),
                            error=None
                        )
                    elif status == 'failed':
                        self._task_scheduler.complete_task(
                            scheduler_task_id,
                            result=None,
                            error=task.get('error', 'Task failed')
                        )
                except Exception as e:
                    self.logger.warning(f"⚠️ 更新调度器任务状态失败: {e}")
            
            return True
    return False
```

---

### 中优先级修复

#### 修复 5: 在任务完成钩子中集成监控系统

**文件**: `src/features/core/engine.py`

**修改位置**: `_on_task_completed_default()` 方法

**建议添加**:
```python
def _on_task_completed_default(self, task_id: str, task: Dict[str, Any]) -> None:
    """默认的任务完成处理函数"""
    try:
        self.logger.info(f"🎯 任务完成钩子触发: {task_id}")
        
        # 1. 更新持久化存储
        self._persist_task_completion(task_id, task)
        
        # 2. WebSocket广播
        self._broadcast_task_completion(task_id, task)
        
        # 3. 发布到事件总线
        self._publish_task_event('TASK_COMPLETED', task_id, task)
        
        # 4. 触发数据归档（如果配置）
        if self.config.auto_archive:
            self._archive_task_data(task_id, task)
        
        # 新增：5. 更新监控系统指标
        self._update_monitoring_on_task_completion(task_id, task)
        
    except Exception as e:
        self.logger.error(f"任务完成钩子执行失败: {e}", exc_info=True)

def _update_monitoring_on_task_completion(self, task_id: str, task: Dict[str, Any]) -> None:
    """任务完成时更新监控系统"""
    try:
        from src.features.monitoring.features_monitor import get_monitor
        from src.features.monitoring.metrics_collector import get_collector
        
        monitor = get_monitor()
        collector = get_collector()
        
        # 收集任务完成指标
        task_type = task.get('task_type', 'unknown')
        feature_count = task.get('feature_count', 0)
        
        # 收集特征生成时间
        if 'start_time' in task and 'end_time' in task:
            generation_time = task['end_time'] - task['start_time']
            collector.collect_metric(
                name='feature_generation_time',
                value=generation_time,
                category=collector.MetricCategory.PERFORMANCE,
                metric_type=collector.MetricType.HISTOGRAM,
                labels={'task_type': task_type, 'task_id': task_id}
            )
        
        # 收集特征生成数量
        collector.collect_metric(
            name='feature_generation_count',
            value=feature_count,
            category=collector.MetricCategory.BUSINESS,
            metric_type=collector.MetricType.COUNTER,
            labels={'task_type': task_type}
        )
        
        self.logger.info(f"📊 监控指标已更新: 任务 {task_id}")
        
    except Exception as e:
        self.logger.warning(f"⚠️ 更新监控指标失败: {e}")
```

---

## 📝 实施计划

### 阶段 1: 基础集成 (高优先级)
1. 在 FeatureEngine.__init__() 中添加任务调度器引用
2. 修改 FeatureEngine.create_task() 提交任务到调度器
3. 测试任务创建和提交流程

### 阶段 2: 状态同步 (中优先级)
4. 添加任务状态双向同步机制
5. 修改 update_task_status() 通知调度器
6. 测试状态同步的正确性

### 阶段 3: 监控集成 (中优先级)
7. 在任务完成钩子中集成监控系统
8. 验证仪表盘数据更新
9. 端到端测试

---

## 📋 附录

### 相关文档
- [特征层架构设计](../../docs/architecture/feature_layer_architecture_design.md)
- [报告组织规范](../README.md)
- [报告索引](../INDEX.md)

### 相关文件
- `src/features/core/engine.py` - 特征引擎核心
- `src/features/distributed/task_scheduler.py` - 特征任务调度器
- `src/features/core/event_listeners.py` - 事件监听器
- `src/features/monitoring/` - 监控系统模块

---

*本报告通过代码分析生成。*
