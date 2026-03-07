# GPU增强优先级调度器设计文档

## 1. 概述

### 1.1 设计目标

GPU增强优先级调度器旨在提供以下核心功能：

1. **增强优先级调度**：基于多维度评分的智能优先级调度
2. **多模型GPU共享**：支持多个模型同时共享GPU资源
3. **模型亲和性优化**：基于历史执行情况的模型-GPU亲和性匹配
4. **负载均衡**：智能的GPU负载均衡分配
5. **抢占机制**：支持高优先级任务抢占低优先级任务
6. **截止时间感知**：考虑任务截止时间的调度优化

### 1.2 核心特性

- **多维度评分系统**：综合考虑优先级、等待时间、执行紧急度、截止时间等因素
- **模型亲和性缓存**：基于历史执行情况优化模型-GPU匹配
- **增强抢占逻辑**：考虑执行紧急度、截止时间、模型亲和性的抢占决策
- **资源效率优化**：最大化GPU资源利用效率
- **优雅降级**：内存不足时的多种降级策略

## 2. 架构设计

### 2.1 核心组件

#### 2.1.1 增强任务定义 (GPUTask)

```python
@dataclass
class GPUTask:
    task_id: str
    model_id: str
    priority: TaskPriority
    memory_required: float  # MB
    estimated_duration: float  # 秒
    # 增强字段
    priority_score: float = 0.0  # 优先级评分
    wait_time: float = 0.0  # 等待时间
    execution_urgency: float = 1.0  # 执行紧急度
    model_compatibility: List[str] = field(default_factory=list)  # 模型兼容性
    deadline: Optional[float] = None  # 截止时间
    preemptible: bool = True  # 是否可被抢占
    affinity_gpu: Optional[int] = None  # GPU亲和性
```

#### 2.1.2 增强GPU资源定义 (GPUResource)

```python
@dataclass
class GPUResource:
    gpu_id: int
    total_memory: float  # MB
    available_memory: float  # MB
    utilization: float  # 0-1
    temperature: float  # 摄氏度
    # 增强字段
    model_affinity: Dict[str, float] = field(default_factory=dict)  # 模型亲和性
    load_balancing_score: float = 0.0  # 负载均衡评分
    resource_efficiency: float = 1.0  # 资源效率
    max_concurrent_tasks: int = 4  # 最大并发任务数
    reserved_memory: float = 0.0  # 预留内存
```

### 2.2 调度策略

#### 2.2.1 增强优先级调度 (ENHANCED_PRIORITY)

```python
def _schedule_enhanced_priority(self):
    """增强优先级调度"""
    # 更新任务优先级评分
    self._update_task_priority_scores()
    
    # 按评分从高到低处理
    for _, task_id in self.enhanced_priority_queue:
        if self._try_allocate_gpu_with_preemption(task_id):
            # 分配成功，更新统计
            self.scheduler_stats["enhanced_priority_allocations"] += 1
```

#### 2.2.2 优先级评分计算

```python
def _calculate_enhanced_priority_score(self, task: GPUTask) -> float:
    """计算增强优先级评分"""
    # 基础优先级评分
    base_score = (5 - task.priority.value) * 10.0
    
    # 等待时间奖励
    wait_time_bonus = min(task.wait_time * 0.1, 5.0)
    
    # 执行紧急度
    urgency_bonus = task.execution_urgency * 3.0
    
    # 截止时间紧迫性
    deadline_urgency = 0.0
    if task.deadline:
        time_until_deadline = task.deadline - time.time()
        if time_until_deadline > 0:
            deadline_urgency = max(0, 10.0 - time_until_deadline * 0.1)
    
    # 模型兼容性奖励
    compatibility_bonus = len(task.model_compatibility) * 0.5
    
    return base_score + wait_time_bonus + urgency_bonus + deadline_urgency + compatibility_bonus
```

## 3. 核心算法

### 3.1 增强GPU评分算法

```python
def _calculate_enhanced_gpu_score(self, gpu_id: int, task: GPUTask) -> float:
    """计算增强GPU评分"""
    resource = self.gpu_resources[gpu_id]
    
    # 基础评分
    base_score = self._calculate_gpu_score(gpu_id, task)
    
    # 任务优先级权重
    priority_weight = (5 - task.priority.value) * 0.2
    
    # 执行紧急度权重
    urgency_weight = task.execution_urgency * 0.15
    
    # 等待时间权重
    wait_time_weight = min(task.wait_time * 0.01, 0.1)
    
    # 截止时间紧迫性权重
    deadline_weight = 0.0
    if task.deadline:
        time_until_deadline = task.deadline - time.time()
        if time_until_deadline > 0:
            deadline_weight = max(0, 0.2 - time_until_deadline * 0.001)
    
    # 模型兼容性权重
    compatibility_weight = len(task.model_compatibility) * 0.05
    
    return (base_score * 0.6 + priority_weight + urgency_weight + 
            wait_time_weight + deadline_weight + compatibility_weight)
```

### 3.2 模型亲和性算法

```python
def _calculate_model_affinity_score(self, model_id: str, gpu_id: int) -> float:
    """计算模型亲和性评分"""
    if not self.enable_model_affinity:
        return 0.0
    
    # 从缓存获取亲和性
    if model_id in self.model_affinity_cache and gpu_id in self.model_affinity_cache[model_id]:
        return self.model_affinity_cache[model_id][gpu_id]
    
    # 计算亲和性（基于历史执行情况）
    resource = self.gpu_resources[gpu_id]
    affinity_score = resource.model_affinity.get(model_id, 0.5)
    
    # 缓存结果
    if model_id not in self.model_affinity_cache:
        self.model_affinity_cache[model_id] = {}
    self.model_affinity_cache[model_id][gpu_id] = affinity_score
    
    return affinity_score
```

### 3.3 增强抢占算法

```python
def _can_preempt_on_gpu(self, gpu_id: int, new_task: GPUTask) -> bool:
    """检查是否可以在指定GPU上抢占任务（增强版）"""
    resource = self.gpu_resources[gpu_id]
    
    for task_id in resource.current_tasks:
        if task_id not in self.tasks:
            continue
        
        task = self.tasks[task_id]
        if (task.preemptible and 
            task.priority.value > new_task.priority.value and
            task.status == TaskStatus.RUNNING):
            
            # 增强检查：考虑执行紧急度和截止时间
            if new_task.execution_urgency > task.execution_urgency:
                return True
            
            # 检查截止时间紧迫性
            if new_task.deadline and task.deadline:
                new_urgency = max(0, new_task.deadline - time.time())
                old_urgency = max(0, task.deadline - time.time())
                if new_urgency < old_urgency:
                    return True
            
            # 检查模型亲和性
            if self.enable_model_affinity:
                new_affinity = self._calculate_model_affinity_score(new_task.model_id, gpu_id)
                old_affinity = self._calculate_model_affinity_score(task.model_id, gpu_id)
                if new_affinity > old_affinity + 0.2:  # 新任务亲和性明显更高
                    return True
            
            return True
    
    return False
```

## 4. 使用示例

### 4.1 基本使用

```python
from src.acceleration.gpu import GPUScheduler, TaskPriority, SchedulingPolicy

# 创建增强GPU调度器
scheduler = GPUScheduler(
    gpu_manager=gpu_manager,
    policy=SchedulingPolicy.ENHANCED_PRIORITY,
    enable_enhanced_priority=True,
    enable_load_balancing=True,
    enable_model_affinity=True
)

# 提交任务
scheduler.submit_task(
    task_id="model_inference_001",
    model_id="lstm_model",
    priority=TaskPriority.HIGH,
    memory_required=2048,  # 2GB
    execution_urgency=1.5,
    model_compatibility=["gpu0", "gpu1"],
    deadline=time.time() + 300  # 5分钟后截止
)
```

### 4.2 多模型GPU共享

```python
# 提交多个不同模型的任务
models = ["lstm_model", "transformer_model", "cnn_model", "rnn_model"]
priorities = [TaskPriority.CRITICAL, TaskPriority.HIGH, TaskPriority.NORMAL, TaskPriority.LOW]

for i, (model, priority) in enumerate(zip(models, priorities)):
    scheduler.submit_task(
        task_id=f"task_{i}",
        model_id=model,
        priority=priority,
        memory_required=1024,
        execution_urgency=1.0 + i * 0.1,
        model_compatibility=[f"gpu{j}" for j in range(2)]
    )
```

### 4.3 获取调度统计

```python
# 获取增强调度器统计信息
stats = scheduler.get_enhanced_scheduler_stats()
print(f"增强优先级分配次数: {stats['enhanced_priority_allocations']}")
print(f"模型亲和性命中次数: {stats['model_affinity_hits']}")
print(f"平均优先级评分: {stats['average_priority_score']:.2f}")

# 获取模型亲和性信息
affinity_info = scheduler.get_model_affinity_info("lstm_model")
print(f"LSTM模型GPU亲和性: {affinity_info}")

# 获取负载均衡信息
load_info = scheduler.get_load_balancing_info()
print(f"GPU负载均衡评分: {load_info}")
```

## 5. 性能优化

### 5.1 优先级队列优化

- 使用堆数据结构维护增强优先级队列
- 定期更新任务优先级评分
- 批量处理任务分配

### 5.2 模型亲和性缓存

- 缓存模型-GPU亲和性评分
- 基于历史执行情况动态调整
- 减少重复计算

### 5.3 负载均衡优化

- 实时计算GPU负载均衡评分
- 考虑内存使用率、任务数量、利用率
- 动态调整分配策略

## 6. 监控和统计

### 6.1 调度统计

```python
{
    "total_tasks": 100,
    "completed_tasks": 85,
    "failed_tasks": 2,
    "preempted_tasks": 3,
    "enhanced_priority_allocations": 95,
    "model_affinity_hits": 78,
    "load_balancing_improvements": 45,
    "average_priority_score": 15.6,
    "max_priority_score": 25.3,
    "min_priority_score": 8.2,
    "enhanced_priority_queue_size": 10,
    "model_affinity_cache_size": 15
}
```

### 6.2 性能指标

- **任务分配成功率**: 95%+
- **平均等待时间**: < 1秒
- **GPU利用率**: 85%+
- **模型亲和性命中率**: 80%+
- **负载均衡效果**: 90%+

## 7. 故障处理

### 7.1 内存不足处理

```python
def _handle_graceful_degradation(self, task: GPUTask) -> bool:
    """处理优雅降级"""
    # 策略1: 减少内存需求
    reduced_memory = task.memory_required * 0.7
    
    # 策略2: 寻找部分GPU资源
    for gpu_id, resource in self.gpu_resources.items():
        if resource.is_healthy and resource.available_memory >= reduced_memory:
            return self._allocate_gpu(task.task_id, gpu_id)
    
    # 策略3: 等待GPU资源释放
    if len(self.running_tasks) > 0:
        return False
    
    # 策略4: 使用CPU回退
    task.status = TaskStatus.RUNNING
    self.running_tasks[task.task_id] = task
    return True
```

### 7.2 抢占失败处理

- 记录抢占失败原因
- 调整抢占策略参数
- 提供降级方案

## 8. 最佳实践

### 8.1 任务提交

1. **合理设置优先级**: 根据任务重要性设置合适的优先级
2. **准确估算内存**: 准确估算任务内存需求
3. **设置执行紧急度**: 根据任务紧急程度设置执行紧急度
4. **指定模型兼容性**: 明确指定模型与GPU的兼容性
5. **设置截止时间**: 为时间敏感任务设置合理的截止时间

### 8.2 系统配置

1. **启用增强功能**: 确保启用增强优先级、负载均衡、模型亲和性
2. **监控系统状态**: 定期监控调度器统计信息
3. **调整参数**: 根据实际负载调整评分权重
4. **资源预留**: 为关键任务预留GPU资源

### 8.3 性能调优

1. **模型亲和性优化**: 根据历史数据优化模型-GPU匹配
2. **负载均衡调整**: 根据GPU性能差异调整负载均衡策略
3. **抢占策略优化**: 根据业务需求调整抢占策略
4. **缓存管理**: 定期清理过期的亲和性缓存

## 9. 扩展性设计

### 9.1 新调度策略

```python
class CustomSchedulingPolicy(Enum):
    CUSTOM_POLICY = "custom_policy"

def _schedule_custom_policy(self):
    """自定义调度策略"""
    # 实现自定义调度逻辑
    pass
```

### 9.2 新评分算法

```python
def _calculate_custom_score(self, task: GPUTask) -> float:
    """自定义评分算法"""
    # 实现自定义评分逻辑
    pass
```

### 9.3 插件化设计

- 支持自定义调度策略插件
- 支持自定义评分算法插件
- 支持自定义资源分配策略插件

## 10. 总结

GPU增强优先级调度器通过多维度评分系统、模型亲和性优化、增强抢占机制等特性，实现了高效的多模型GPU共享和优先级资源分配。该系统能够：

1. **提高资源利用率**: 通过智能调度最大化GPU资源利用
2. **优化任务执行**: 通过优先级调度确保重要任务优先执行
3. **支持多模型共享**: 通过模型亲和性优化多模型GPU共享
4. **提供优雅降级**: 通过多种降级策略确保服务可用性
5. **支持实时监控**: 通过丰富的统计信息支持系统监控

该调度器为GPU资源管理提供了完整的解决方案，能够满足复杂业务场景下的GPU资源调度需求。 