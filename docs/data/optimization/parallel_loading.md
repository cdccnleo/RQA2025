# 并行加载优化指南

## 动态线程池功能

### 核心特性
1. **智能资源分配**：根据CPU核心数自动配置
2. **负载自适应**：动态调整线程池大小
3. **队列管理**：智能任务排队策略
4. **实时监控**：提供运行时统计信息

## 基础使用

### 创建线程池

```python
from src.data.parallel.thread_pool import create_default_pool

# 使用默认配置
pool = create_default_pool()

# 自定义配置
from src.data.parallel.thread_pool import ThreadPoolConfig, DynamicThreadPool
config = ThreadPoolConfig(
    core_pool_size=8,
    max_pool_size=32,
    queue_capacity=500
)
custom_pool = DynamicThreadPool(config)
```

### 提交任务

```python
future = pool.submit(lambda x: x*2, 21)
result = future.result()  # 阻塞获取结果
```

## 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|-------|------|
| core_pool_size | int | CPU核心数×2 | 核心线程数 |
| max_pool_size | int | 50 | 最大线程数 |
| queue_capacity | int | 1000 | 任务队列容量 |
| keep_alive | int | 60 | 空闲线程存活时间(秒) |

## 性能监控

```python
stats = pool.get_stats()
"""
{
    'active_threads': 当前活跃线程数,
    'pending_tasks': 排队任务数,
    'core_pool_size': 当前核心线程数,
    'max_pool_size': 最大线程数,
    'queue_capacity': 队列容量
}
"""
```

## 最佳实践

### CPU密集型任务
```python
ThreadPoolConfig(
    core_pool_size=os.cpu_count(),
    max_pool_size=os.cpu_count() * 2,
    queue_capacity=500
)
```

### IO密集型任务
```python
ThreadPoolConfig(
    core_pool_size=os.cpu_count() * 4,
    max_pool_size=50,
    queue_capacity=2000
)
```

### 混合型任务
```python
ThreadPoolConfig(
    core_pool_size=os.cpu_count() * 3,
    max_pool_size=os.cpu_count() * 6,
    queue_capacity=1000
)
```

## 调优建议

1. **监控指标**：
   - CPU使用率保持在70%-80%
   - 队列长度不超过容量的80%

2. **异常处理**：
   ```python
   try:
       future = pool.submit(risky_task)
       result = future.result(timeout=10)
   except TimeoutError:
       print("任务执行超时")
   except Exception as e:
       print(f"任务失败: {e}")
   ```

3. **关闭清理**：
   ```python
   pool.shutdown(wait=True)  # 等待所有任务完成
   # 或
   pool.shutdown(wait=False) # 立即关闭
   ```
