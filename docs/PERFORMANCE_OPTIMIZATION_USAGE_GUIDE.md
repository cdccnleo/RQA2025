# RQA2025 数据层性能优化使用指南

## 📋 概述

本指南介绍如何使用阶段六实现的三大性能优化功能：
- **智能缓存优化器** - 智能缓存失效和预加载
- **异步处理优化器** - 并发处理和资源优化
- **数据压缩优化器** - 自适应压缩和传输优化

## 🔄 智能缓存优化器使用指南

### 基本使用

```python
from src.data.cache.smart_cache_optimizer import SmartCacheOptimizer
from src.data.cache.data_cache import DataCache
from src.data.interfaces.standard_interfaces import DataSourceType

# 初始化缓存和优化器
cache = DataCache()
optimizer = SmartCacheOptimizer(cache)

# 智能缓存操作
success = optimizer.smart_set('stock_data', data, DataSourceType.STOCK, ttl_seconds=3600)
cached_data = optimizer.smart_get('stock_data', DataSourceType.STOCK)

# 获取性能指标
metrics = optimizer.get_performance_metrics()
print(f"缓存命中率: {metrics['cache_performance']['hit_rate']:.2%}")
```

### 预加载规则配置

```python
# 添加预加载规则
optimizer.add_preload_rule(
    name="market_open_preload",
    data_type=DataSourceType.STOCK,
    condition=lambda ctx: ctx.get('hour') in [9, 10, 14, 15],  # 交易时间
    preload_func=lambda: fetch_hot_stocks(),  # 预加载函数
    priority=3,
    interval_seconds=300  # 5分钟执行一次
)

# 添加热门数据预加载
optimizer.add_preload_rule(
    name="crypto_trend_preload",
    data_type=DataSourceType.CRYPTO,
    condition=lambda ctx: ctx.get('market_active', False),
    preload_func=lambda: fetch_trending_cryptos(),
    priority=3,
    interval_seconds=60  # 1分钟执行一次
)
```

### 高级配置

```python
# 自定义缓存配置
from src.data.cache.smart_data_cache import DataCacheConfig

config = DataCacheConfig(
    max_size=10000,
    default_ttl=3600,
    enable_compression=True
)

optimizer = SmartCacheOptimizer(cache, config)

# 手动触发缓存清理
optimizer.invalidate_cache_entry('old_key', DataSourceType.STOCK)
```

## ⚡ 异步处理优化器使用指南

### 基本使用

```python
import asyncio
from src.data.parallel.async_processing_optimizer import AsyncProcessingOptimizer

async def main():
    # 初始化优化器
    optimizer = AsyncProcessingOptimizer(max_workers=8, enable_process_pool=True)

    # 定义任务函数
    def process_data(data_id: int, multiplier: float = 1.0):
        # 模拟数据处理
        import time
        time.sleep(0.1)  # 模拟处理时间
        return data_id * multiplier

    # 提交单个异步任务
    task_id = await optimizer.submit_async_task(
        process_data,
        42,
        multiplier=2.0,
        task_type="data_processing"
    )

    # 批量提交任务
    batch_tasks = [
        {'func': process_data, 'args': (i,), 'kwargs': {'multiplier': 1.5}}
        for i in range(10)
    ]

    batch_ids = await optimizer.submit_batch_tasks(batch_tasks, max_concurrent=5)

    # 等待所有任务完成
    await optimizer.wait_for_completion()

    # 获取性能报告
    report = optimizer.get_performance_report()
    print(f"总任务数: {report['task_statistics']['total_tasks']}")
    print(f"平均执行时间: {report['task_statistics']['avg_execution_time']:.3f}s")

    # 关闭优化器
    optimizer.shutdown()

# 运行
asyncio.run(main())
```

### 工作负载优化

```python
# 根据工作负载类型优化
optimizer = AsyncProcessingOptimizer(max_workers=10)

# CPU密集型工作负载
optimizer.optimize_for_workload('cpu_intensive')
print("已优化为CPU密集型处理")

# IO密集型工作负载
optimizer.optimize_for_workload('io_intensive')
print("已优化为IO密集型处理")

# 混合型工作负载
optimizer.optimize_for_workload('mixed')
print("已优化为混合型处理")
```

### 任务监控

```python
# 获取任务详情
task_details = optimizer.get_task_details(task_id)
if task_details:
    print(f"任务状态: {task_details['success']}")
    print(f"执行时间: {task_details['execution_time']:.3f}s")
    print(f"队列时间: {task_details['queue_time']:.3f}s")

# 获取系统资源状态
report = optimizer.get_performance_report()
print(f"活跃线程数: {report['resource_metrics']['active_threads']}")
print(f"CPU使用率: {report['resource_metrics']['cpu_usage']:.1%}")
print(f"内存使用率: {report['resource_metrics']['memory_usage']:.1%}")
```

## 🗜️ 数据压缩优化器使用指南

### 基本使用

```python
from src.data.compression.data_compression_optimizer import DataCompressionOptimizer

# 初始化优化器
optimizer = DataCompressionOptimizer()

# 压缩数据
test_data = "这是一段测试数据，包含中文字符和重复内容。" * 100

result = optimizer.compress_data(test_data, "text")

print("压缩结果:"print(f"原始大小: {result['original_size']} bytes")
print(f"压缩大小: {result['compressed_size']} bytes")
print(f"压缩比: {result['compression_ratio']:.2f}")
print(f"使用算法: {result['algorithm']}")
print(f"压缩时间: {result['compression_time']:.4f}s")

# 解压数据
decompressed = optimizer.decompress_data(result['compressed_data'], result['algorithm'])
print(f"解压成功: {decompressed == test_data}")
```

### 高级压缩配置

```python
# 指定特定的压缩策略
result = optimizer.compress_data(
    test_data,
    "text",
    strategy_name="text_gzip_balanced"  # 指定使用平衡的gzip策略
)

# 批量压缩
data_list = [
    {'data': text_data, 'type': 'text'},
    {'data': binary_data, 'type': 'binary'},
    {'data': json_data, 'type': 'json'}
]

batch_results = optimizer.compress_batch(data_list, parallel=True)
for i, result in enumerate(batch_results):
    print(f"数据{i+1}压缩比: {result['compression_ratio']:.2f}")
```

### 策略管理

```python
# 添加自定义压缩策略
from src.data.compression.data_compression_optimizer import CompressionStrategy

custom_strategy = CompressionStrategy(
    name="custom_high_compression",
    algorithm="bz2",
    compression_level=9,
    min_size_threshold=10000,  # 10KB以上
    max_size_threshold=100000000,  # 100MB以下
    priority=5
)

optimizer.add_compression_strategy(custom_strategy)

# 启用/禁用策略
optimizer.enable_strategy("text_gzip_fast")
optimizer.disable_strategy("large_data_bz2")

# 更新策略优先级
optimizer.update_strategy_priority("custom_high_compression", 8)
```

### 性能监控

```python
# 获取压缩报告
report = optimizer.get_compression_report(time_range_hours=24)

print("压缩统计:")
print(f"总操作数: {report['summary']['total_operations']}")
print(f"平均压缩比: {report['summary']['avg_compression_ratio']:.2f}")
print(f"平均压缩时间: {report['summary']['avg_compression_time']:.4f}s")

# 获取算法性能
gzip_perf = optimizer.get_algorithm_performance('gzip')
if gzip_perf['algorithm']:
    print(f"GZIP性能得分: {gzip_perf['performance_score']:.2f}")
    print(f"趋势: {gzip_perf['recent_trend']}")

# 获取优化器状态
status = optimizer.get_optimizer_status()
print(f"活跃策略数: {status['enabled_strategies']}")
print(f"历史指标数: {status['metrics_history_count']}")
```

## 🔧 集成使用示例

### 完整的数据处理流水线

```python
import asyncio
from typing import Any, Dict, List

class OptimizedDataPipeline:
    """优化的数据处理流水线"""

    def __init__(self):
        # 初始化各个优化器
        from src.data.cache.smart_cache_optimizer import SmartCacheOptimizer
        from src.data.cache.data_cache import DataCache
        from src.data.parallel.async_processing_optimizer import AsyncProcessingOptimizer
        from src.data.compression.data_compression_optimizer import DataCompressionOptimizer

        self.cache_optimizer = SmartCacheOptimizer(DataCache())
        self.async_optimizer = AsyncProcessingOptimizer(max_workers=8)
        self.compression_optimizer = DataCompressionOptimizer()

        # 配置预加载规则
        self._setup_preload_rules()

    def _setup_preload_rules(self):
        """设置预加载规则"""
        self.cache_optimizer.add_preload_rule(
            name="pipeline_preload",
            data_type=DataSourceType.STOCK,
            condition=lambda ctx: ctx.get('hour') in [9, 14],  # 开盘和午盘
            preload_func=self._preload_market_data,
            priority=4,
            interval_seconds=600  # 10分钟
        )

    def _preload_market_data(self) -> Dict[str, Any]:
        """预加载市场数据"""
        # 这里实现具体的预加载逻辑
        return {"status": "success", "data": []}

    async def process_data_batch(self, data_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """处理数据批次"""

        # 1. 批量压缩数据
        compressed_batch = self.compression_optimizer.compress_batch(
            [{'data': item['raw_data'], 'type': item['data_type']} for item in data_batch]
        )

        # 2. 异步处理数据
        async def process_item(compressed_item: Dict, original_item: Dict):
            # 解压数据
            raw_data = self.compression_optimizer.decompress_data(
                compressed_item['compressed_data'],
                compressed_item['algorithm']
            )

            # 处理数据
            processed_data = await self.async_optimizer.submit_async_task(
                self._process_single_item,
                raw_data,
                original_item['metadata']
            )

            # 缓存处理结果
            cache_key = f"processed_{original_item['id']}"
            self.cache_optimizer.smart_set(
                cache_key,
                processed_data,
                DataSourceType.STOCK,
                ttl_seconds=3600
            )

            return processed_data

        # 3. 并发处理所有项目
        tasks = [
            process_item(compressed, original)
            for compressed, original in zip(compressed_batch, data_batch)
        ]

        results = await asyncio.gather(*tasks)
        return results

    def _process_single_item(self, raw_data: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个数据项"""
        import time
        time.sleep(0.05)  # 模拟处理时间

        # 这里实现具体的数据处理逻辑
        return {
            'processed_data': raw_data,
            'metadata': metadata,
            'processing_time': 0.05,
            'status': 'success'
        }

    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """获取流水线性能指标"""
        return {
            'cache_metrics': self.cache_optimizer.get_performance_metrics(),
            'async_metrics': self.async_optimizer.get_performance_report(),
            'compression_metrics': self.compression_optimizer.get_compression_report()
        }

    def shutdown(self):
        """关闭流水线"""
        self.cache_optimizer.shutdown()
        self.async_optimizer.shutdown()
        self.compression_optimizer.clear_metrics_history()
```

### 使用优化的流水线

```python
async def main():
    # 创建优化的数据处理流水线
    pipeline = OptimizedDataPipeline()

    try:
        # 准备测试数据
        test_data_batch = [
            {
                'id': f'data_{i}',
                'raw_data': f'测试数据内容{i}' * 100,  # 生成较大的测试数据
                'data_type': 'text',
                'metadata': {'source': 'test', 'timestamp': '2025-01-01'}
            }
            for i in range(10)
        ]

        # 处理数据批次
        print("开始处理数据批次...")
        start_time = time.time()

        results = await pipeline.process_data_batch(test_data_batch)

        processing_time = time.time() - start_time
        print(f"批次处理完成，耗时: {processing_time:.2f}s")
        print(f"处理了 {len(results)} 个数据项")

        # 获取性能指标
        metrics = pipeline.get_pipeline_metrics()

        print("\n=== 性能指标 ===")
        print(f"缓存状态: {metrics['cache_metrics']['cache_status']['current_entries']} 条目")
        print(f"异步任务: {metrics['async_metrics']['task_statistics']['completed_tasks']} 已完成")
        print(f"压缩操作: {metrics['compression_metrics']['summary']['total_operations']} 次")

    finally:
        # 清理资源
        pipeline.shutdown()

# 运行示例
if __name__ == '__main__':
    asyncio.run(main())
```

## 📊 监控和调优

### 性能监控

```python
# 实时监控所有优化器的性能
def monitor_performance():
    """性能监控函数"""
    cache_metrics = cache_optimizer.get_performance_metrics()
    async_report = async_optimizer.get_performance_report()
    compression_report = compression_optimizer.get_compression_report()

    print("=== 性能监控报告 ===")
    print(f"缓存命中率: {cache_metrics['cache_performance']['hit_rate']:.2%}")
    print(f"活跃线程数: {async_report['resource_metrics']['active_threads']}")
    print(f"平均压缩比: {compression_report['summary']['avg_compression_ratio']:.2f}")
    print(f"CPU使用率: {async_report['resource_metrics']['cpu_usage']:.1%}")

# 设置定期监控
import threading
def start_monitoring():
    def monitor_loop():
        while True:
            monitor_performance()
            time.sleep(60)  # 每分钟监控一次

    monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    monitor_thread.start()

start_monitoring()
```

### 自适应调优

```python
# 基于性能指标自动调优
def auto_tune():
    """自动调优函数"""
    async_report = async_optimizer.get_performance_report()

    # CPU使用率过高，增加线程数
    if async_report['resource_metrics']['cpu_usage'] > 0.8:
        current_workers = async_report['resource_metrics']['total_threads']
        async_optimizer.update_adaptive_params({
            'max_workers': min(current_workers + 2, 20)
        })
        print("CPU使用率过高，已增加线程数")

    # 内存使用率过高，启用压缩
    if async_report['resource_metrics']['memory_usage'] > 0.8:
        compression_report = compression_optimizer.get_compression_report()
        if compression_report['summary']['avg_compression_ratio'] < 2.0:
            # 启用更高压缩级别的策略
            compression_optimizer.update_strategy_priority("large_data_bz2", 10)
            print("内存使用率过高，已启用更强的压缩策略")

    # 响应时间过长，启用缓存预加载
    if async_report['task_statistics']['avg_execution_time'] > 2.0:
        cache_optimizer.add_preload_rule(
            name="emergency_preload",
            data_type=DataSourceType.STOCK,
            condition=lambda ctx: True,  # 总是执行
            preload_func=lambda: preload_hot_data(),
            priority=5,
            interval_seconds=60
        )
        print("响应时间过长，已启用紧急预加载")

# 设置自动调优
def start_auto_tuning():
    def tuning_loop():
        while True:
            auto_tune()
            time.sleep(300)  # 每5分钟调优一次

    tuning_thread = threading.Thread(target=tuning_loop, daemon=True)
    tuning_thread.start()

start_auto_tuning()
```

## 🔧 故障排除

### 常见问题

1. **psutil不可用**
   ```python
   # 系统会自动降级使用默认的资源监控
   # 不影响核心功能
   ```

2. **内存不足**
   ```python
   # 减少缓存大小
   config = DataCacheConfig(max_size=5000)
   optimizer = SmartCacheOptimizer(cache, config)

   # 或增加压缩
   optimizer.optimize_for_workload('io_intensive')
   ```

3. **CPU使用率过高**
   ```python
   # 减少线程数
   optimizer.update_adaptive_params({'max_workers': 4})

   # 或切换到IO优化模式
   optimizer.optimize_for_workload('io_intensive')
   ```

4. **压缩效果不佳**
   ```python
   # 检查数据类型配置
   report = optimizer.get_algorithm_performance('gzip')
   print(f"GZIP性能: {report}")

   # 尝试其他算法
   optimizer.update_strategy_priority("binary_lzma", 10)
   ```

### 最佳实践

1. **缓存优化**
   - 为热点数据设置适当的TTL
   - 定期清理过期缓存
   - 监控缓存命中率

2. **异步处理**
   - 根据工作负载类型选择合适的配置
   - 设置合理的最大并发数
   - 监控资源使用情况

3. **数据压缩**
   - 为不同类型的数据选择合适的压缩算法
   - 监控压缩性能和效果
   - 定期调整压缩策略

## 📞 技术支持

如遇到问题或需要进一步优化，请参考：
- 各优化器的详细API文档
- 性能监控指标说明
- 自适应调整参数配置

---

**版本**: v6.0.0
**更新时间**: 2025年8月30日
**适用环境**: Python 3.7+
