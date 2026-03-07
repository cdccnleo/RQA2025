# 数据层优化功能使用指南

## 概述

数据层优化模块提供了全面的数据加载和处理优化功能，包括并行加载、缓存优化、质量监控、性能监控和数据预加载等。本指南将详细介绍如何使用这些功能。

## 核心组件

### 1. 数据优化器 (DataOptimizer)

数据优化器是核心组件，整合了所有优化功能，提供统一的数据加载接口。

#### 基本使用

```python
from src.data.optimization.data_optimizer import DataOptimizer, OptimizationConfig

# 创建优化配置
config = OptimizationConfig(
    max_workers=4,                    # 最大工作线程数
    enable_parallel_loading=True,     # 启用并行加载
    enable_cache=True,                # 启用缓存
    enable_quality_monitor=True,      # 启用质量监控
    enable_performance_monitor=True,  # 启用性能监控
    enable_preload=True,              # 启用预加载
    preload_symbols=['600519.SH'],    # 预加载股票列表
    preload_days=30                   # 预加载天数
)

# 创建数据优化器
optimizer = DataOptimizer(config)

# 优化数据加载
result = await optimizer.optimize_data_loading(
    data_type='stock',
    start_date='2024-01-01',
    end_date='2024-01-31',
    frequency='1d',
    symbols=['600519.SH', '000858.SZ']
)

# 检查结果
if result.success:
    print(f"数据加载成功，耗时: {result.load_time_ms}ms")
    print(f"缓存命中: {result.cache_hit}")
    print(f"性能指标: {result.performance_metrics}")
else:
    print(f"数据加载失败: {result.error_message}")
```

#### 配置选项详解

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `max_workers` | int | 4 | 并行加载的最大工作线程数 |
| `enable_parallel_loading` | bool | True | 是否启用并行加载 |
| `enable_cache` | bool | True | 是否启用缓存 |
| `enable_quality_monitor` | bool | True | 是否启用质量监控 |
| `enable_performance_monitor` | bool | True | 是否启用性能监控 |
| `enable_preload` | bool | False | 是否启用预加载 |
| `preload_symbols` | List[str] | None | 预加载的股票列表 |
| `preload_days` | int | 30 | 预加载的天数 |

### 2. 性能监控器 (DataPerformanceMonitor)

性能监控器提供实时的性能指标跟踪和告警功能。

#### 基本使用

```python
from src.data.optimization.performance_monitor import DataPerformanceMonitor

# 创建性能监控器
monitor = DataPerformanceMonitor()

# 记录操作
monitor.record_operation(
    operation='data_load',
    duration_ms=150.0,
    success=True,
    metadata={'symbols': ['600519.SH']}
)

# 获取性能报告
report = monitor.get_performance_report(hours=24)
print(f"总操作数: {report['total_operations']}")
print(f"平均耗时: {report['avg_load_time_ms']:.2f}ms")
print(f"成功率: {report['success_rate']:.2%}")

# 添加告警回调
def alert_callback(alert):
    print(f"性能告警: {alert.alert_type} - {alert.message}")

monitor.add_alert_callback(alert_callback)
```

#### 告警阈值配置

```python
config = {
    'alert_thresholds': {
        'load_time_ms': 5000,    # 加载时间超过5秒告警
        'memory_percent': 80,    # 内存使用超过80%告警
        'cpu_percent': 90,       # CPU使用超过90%告警
        'error_rate': 0.1        # 错误率超过10%告警
    }
}
monitor = DataPerformanceMonitor(config)
```

### 3. 数据预加载器 (DataPreloader)

数据预加载器在后台预先加载可能使用的数据，提高响应速度。

#### 基本使用

```python
from src.data.optimization.data_preloader import DataPreloader, PreloadConfig

# 创建预加载配置
config = PreloadConfig(
    max_concurrent_tasks=3,      # 最大并发任务数
    max_queue_size=50,           # 最大队列大小
    task_timeout_seconds=300,    # 任务超时时间
    enable_auto_preload=True,    # 启用自动预加载
    auto_preload_symbols=['600519.SH', '000858.SZ'],
    auto_preload_days=30
)

# 创建预加载器
preloader = DataPreloader(config)

# 添加预加载任务
task_id = preloader.add_preload_task(
    data_type='stock',
    start_date='2024-01-01',
    end_date='2024-01-31',
    frequency='1d',
    symbols=['600519.SH'],
    priority=3  # 优先级1-5，数字越大优先级越高
)

# 检查任务状态
status = preloader.get_task_status(task_id)
print(f"任务状态: {status['status']}")

# 获取任务结果
result = preloader.get_task_result(task_id)
if result:
    print("预加载完成")

# 获取统计信息
stats = preloader.get_stats()
print(f"总任务数: {stats['total_tasks']}")
print(f"完成任务数: {stats['completed_tasks']}")
```

### 4. 质量监控器 (AdvancedQualityMonitor)

质量监控器提供数据质量检查和报告功能。

#### 基本使用

```python
from src.data.quality.advanced_quality_monitor import AdvancedQualityMonitor
import pandas as pd

# 创建质量监控器
quality_monitor = AdvancedQualityMonitor()

# 检查数据质量
test_data = pd.DataFrame({
    'symbol': ['600519.SH', '000858.SZ'],
    'close': [100.0, 50.0],
    'volume': [1000000, 500000],
    'date': ['2024-01-01', '2024-01-01']
})

quality_metrics = quality_monitor.check_data_quality(test_data)

print(f"数据完整性: {quality_metrics['completeness']:.2%}")
print(f"数据准确性: {quality_metrics['accuracy']:.2%}")
print(f"数据一致性: {quality_metrics['consistency']:.2%}")
print(f"总体质量: {quality_metrics['overall_quality']:.2%}")

# 检查质量问题
if quality_metrics.get('issues'):
    for issue in quality_metrics['issues']:
        print(f"质量问题: {issue}")
```

## 高级用法

### 1. 自定义缓存策略

```python
from src.data.cache.multi_level_cache import CacheConfig

# 创建自定义缓存配置
cache_config = CacheConfig(
    max_size=1000,           # 最大缓存条目数
    ttl=3600,               # 缓存有效期（秒）
    enable_disk_cache=True,  # 启用磁盘缓存
    disk_cache_dir='cache',  # 磁盘缓存目录
    compression=True,        # 启用压缩
    encryption=False,        # 禁用加密
    enable_stats=True,       # 启用统计
    cleanup_interval=300     # 清理间隔（秒）
)

# 在优化器中使用自定义缓存
optimizer = DataOptimizer(OptimizationConfig(cache_config=cache_config))
```

### 2. 并行加载优化

```python
from src.data.parallel.parallel_loader import ParallelLoadingManager

# 创建并行加载管理器
parallel_manager = ParallelLoadingManager(max_workers=8)

# 自定义加载任务
async def custom_load_task(symbol):
    # 自定义加载逻辑
    return await load_stock_data(symbol)

# 并行执行多个任务
results = await parallel_manager.execute_parallel_tasks(
    tasks=[custom_load_task(symbol) for symbol in symbols],
    timeout=30
)
```

### 3. 性能监控集成

```python
# 使用装饰器自动监控性能
from src.data.optimization.performance_monitor import monitor_performance

@monitor_performance("data_processing")
async def process_data(data):
    # 数据处理逻辑
    return processed_data

# 或者手动监控
monitor = DataPerformanceMonitor()

async def optimized_data_loading():
    start_time = time.time()
    try:
        result = await load_data()
        duration = (time.time() - start_time) * 1000
        monitor.record_operation(
            operation='data_loading',
            duration_ms=duration,
            success=True
        )
        return result
    except Exception as e:
        duration = (time.time() - start_time) * 1000
        monitor.record_operation(
            operation='data_loading',
            duration_ms=duration,
            success=False,
            error_message=str(e)
        )
        raise
```

## 最佳实践

### 1. 配置优化

```python
# 根据系统资源调整配置
import psutil

# 根据CPU核心数设置工作线程数
cpu_count = psutil.cpu_count()
max_workers = min(cpu_count * 2, 8)  # 不超过8个线程

# 根据内存大小设置缓存大小
memory_gb = psutil.virtual_memory().total / (1024**3)
cache_size = int(memory_gb * 100)  # 每GB内存100个缓存条目

config = OptimizationConfig(
    max_workers=max_workers,
    cache_config=CacheConfig(max_size=cache_size)
)
```

### 2. 错误处理

```python
async def safe_data_loading():
    try:
        result = await optimizer.optimize_data_loading(
            data_type='stock',
            start_date='2024-01-01',
            end_date='2024-01-31',
            symbols=['600519.SH']
        )
        
        if not result.success:
            # 记录错误并尝试降级策略
            logger.error(f"数据加载失败: {result.error_message}")
            return await fallback_data_loading()
        
        return result.data_model
        
    except Exception as e:
        logger.error(f"数据加载异常: {e}")
        # 返回缓存数据或空数据
        return get_cached_data() or create_empty_data()
```

### 3. 资源管理

```python
# 确保正确清理资源
async def managed_data_loading():
    optimizer = DataOptimizer()
    preloader = DataPreloader()
    
    try:
        # 执行数据加载
        result = await optimizer.optimize_data_loading(...)
        return result
    finally:
        # 清理资源
        optimizer.cleanup()
        preloader.shutdown()
```

## 性能调优

### 1. 缓存命中率优化

```python
# 分析缓存使用模式
report = optimizer.get_optimization_report()
cache_hit_rate = report['cache_hit_rate']

if cache_hit_rate < 0.7:
    # 缓存命中率低，增加缓存大小或调整TTL
    optimizer.config.cache_config.max_size *= 2
    optimizer.config.cache_config.ttl *= 2
```

### 2. 并行度调优

```python
# 根据数据量调整并行度
symbol_count = len(symbols)
if symbol_count <= 10:
    max_workers = 2
elif symbol_count <= 50:
    max_workers = 4
else:
    max_workers = 8

optimizer.config.max_workers = max_workers
```

### 3. 预加载策略

```python
# 根据使用模式设置预加载
frequently_used_symbols = ['600519.SH', '000858.SZ', '601318.SH']
preloader.config.auto_preload_symbols = frequently_used_symbols
preloader.config.auto_preload_days = 30  # 预加载30天数据
```

## 监控和告警

### 1. 性能指标监控

```python
# 定期检查性能指标
def check_performance_health():
    report = monitor.get_performance_report(hours=1)
    
    if report['avg_load_time_ms'] > 5000:
        send_alert("数据加载性能下降")
    
    if report['error_rate'] > 0.1:
        send_alert("数据加载错误率过高")
    
    if report['success_rate'] < 0.9:
        send_alert("数据加载成功率过低")
```

### 2. 资源使用监控

```python
# 监控系统资源使用
def check_system_resources():
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    
    if cpu_percent > 90:
        send_alert("CPU使用率过高")
    
    if memory_percent > 80:
        send_alert("内存使用率过高")
```

## 故障排除

### 1. 常见问题

**问题**: 数据加载速度慢
- 检查网络连接
- 增加并行度
- 优化缓存策略
- 启用数据预加载

**问题**: 内存使用过高
- 减少缓存大小
- 调整并行度
- 启用数据压缩
- 定期清理缓存

**问题**: 缓存命中率低
- 增加缓存大小
- 延长缓存TTL
- 分析访问模式
- 优化缓存键生成

### 2. 调试技巧

```python
# 启用详细日志
import logging
logging.getLogger('src.data.optimization').setLevel(logging.DEBUG)

# 导出性能指标进行分析
monitor.export_metrics('performance_metrics.json', format='json')

# 检查缓存状态
cache_stats = optimizer.cache_manager.get_stats()
print(f"缓存命中率: {cache_stats['hit_rate']:.2%}")
print(f"缓存大小: {cache_stats['size']}")
```

## 总结

数据层优化模块提供了全面的优化功能，通过合理配置和使用这些功能，可以显著提升数据加载和处理性能。建议根据实际使用场景和系统资源情况，调整配置参数以获得最佳性能。 