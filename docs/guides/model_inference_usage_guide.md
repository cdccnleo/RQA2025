# 模型推理使用指南

## 概述

本指南详细介绍了如何在RQA2025项目中使用模型推理功能，包括基本使用、高级配置、性能优化和故障排除。

## 快速开始

### 1. 基本使用

#### 初始化推理管理器
```python
from src.models.inference import ModelInferenceManager
import numpy as np

# 基本配置
config = {
    'enable_gpu': True,
    'enable_cache': True,
    'max_batch_size': 32,
    'cache_size': 100
}

# 创建推理管理器
inference_manager = ModelInferenceManager(config)
```

#### 执行推理
```python
# 准备输入数据
input_data = np.random.randn(100, 10)

# 执行推理
result = inference_manager.predict('my_model', input_data)
print(f"推理结果形状: {result.shape}")
```

#### 批量推理
```python
# 大批量数据
batch_data = np.random.randn(1000, 10)

# 批量推理
batch_result = inference_manager.batch_predict('my_model', batch_data, batch_size=64)
print(f"批量推理结果形状: {batch_result.shape}")
```

### 2. 模型加载

#### 使用模型加载器
```python
from src.models.inference import ModelLoader

# 创建模型加载器
loader = ModelLoader({
    'model_storage_path': './models',
    'cache_models': True
})

# 加载不同格式的模型
pytorch_model = loader.load_model('model.pth', 'pytorch')
tf_model = loader.load_model('model.h5', 'tensorflow')
onnx_model = loader.load_model('model.onnx', 'onnx')

# 自动检测模型类型
auto_model = loader.load_model('model.pth')
```

#### 验证模型
```python
# 验证模型文件
is_valid = loader.validate_model_file('model.pth')
if is_valid:
    print("模型文件有效")
else:
    print("模型文件无效")

# 获取模型元数据
metadata = loader.get_model_metadata('model.pth')
print(f"模型类型: {metadata['model_type']}")
print(f"文件大小: {metadata['file_size']} bytes")
```

## 高级配置

### 1. GPU配置

#### 启用GPU加速
```python
# GPU配置
gpu_config = {
    'enable_gpu': True,
    'memory_limit': 0.8,  # 使用80%的GPU内存
    'batch_size_optimization': True,
    'parallel_processing': True
}

inference_manager = ModelInferenceManager(gpu_config)
```

#### 监控GPU使用
```python
from src.models.inference import GPUInferenceEngine

gpu_engine = GPUInferenceEngine()

# 获取GPU使用率
gpu_usage = gpu_engine.get_gpu_usage()
print(f"GPU使用率: {gpu_usage:.2%}")

# 获取GPU内存信息
memory_info = gpu_engine.get_gpu_memory_info()
print(f"GPU内存: {memory_info['used']}/{memory_info['total']} MB")
```

### 2. 缓存配置

#### 配置推理缓存
```python
from src.models.inference import InferenceCache

# 缓存配置
cache_config = {
    'cache_size': 200,
    'ttl_seconds': 7200,  # 2小时过期
    'enable_disk_cache': True,
    'disk_cache_path': './cache'
}

cache = InferenceCache(cache_config)
```

#### 使用缓存
```python
# 检查缓存
cached_result = cache.get('my_model', input_data)
if cached_result is not None:
    print("使用缓存结果")
else:
    # 执行推理并缓存
    result = inference_manager.predict('my_model', input_data)
    cache.put('my_model', input_data, result)

# 获取缓存统计
hit_rate = cache.get_hit_rate()
print(f"缓存命中率: {hit_rate:.2%}")
```

### 3. 批量处理配置

#### 配置批量处理器
```python
from src.models.inference import BatchInferenceProcessor

# 批量处理配置
batch_config = {
    'max_batch_size': 64,
    'enable_parallel': True,
    'memory_limit': 0.8,
    'retry_count': 3
}

batch_processor = BatchInferenceProcessor(batch_config)
```

#### 优化批处理大小
```python
# 自动优化批处理大小
sample_data = np.random.randn(100, 10)
optimal_batch_size = batch_processor.optimize_batch_size('my_model', sample_data, inference_manager)
print(f"最优批处理大小: {optimal_batch_size}")

# 使用优化的批处理大小
result = batch_processor.process('my_model', large_data, optimal_batch_size, inference_manager)
```

## 性能监控

### 1. 获取性能指标

```python
# 获取性能指标
metrics = inference_manager.get_performance_metrics()

# 打印关键指标
print(f"平均推理时间: {metrics['avg_inference_time']:.4f}s")
print(f"最大推理时间: {metrics['max_inference_time']:.4f}s")
print(f"最小推理时间: {metrics['min_inference_time']:.4f}s")
print(f"总请求数: {metrics['total_requests']}")
print(f"错误率: {metrics['error_rate']:.2%}")
print(f"吞吐量: {metrics['throughput']:.2f} requests/s")
print(f"缓存命中率: {metrics['cache_hit_rate']:.2%}")
```

### 2. 系统资源监控

```python
# 系统内存信息
system_memory = metrics['system_memory']
print(f"系统内存使用: {system_memory['percent']:.1f}%")
print(f"可用内存: {system_memory['available'] / 1024**3:.1f} GB")

# CPU使用率
cpu_usage = metrics['cpu_usage']
print(f"CPU使用率: {cpu_usage:.1f}%")

# GPU可用性
gpu_available = metrics['gpu_available']
print(f"GPU可用: {gpu_available}")
```

### 3. 性能分析

```python
import matplotlib.pyplot as plt

# 分析推理时间分布
inference_times = metrics.get('inference_times', [])
if inference_times:
    plt.figure(figsize=(10, 6))
    plt.hist(inference_times, bins=20, alpha=0.7)
    plt.xlabel('推理时间 (秒)')
    plt.ylabel('频次')
    plt.title('推理时间分布')
    plt.show()
```

## 错误处理

### 1. 常见错误及解决方案

#### 模型加载错误
```python
try:
    result = inference_manager.predict('my_model', input_data)
except ValueError as e:
    if "模型加载失败" in str(e):
        print("检查模型文件是否存在")
        print("检查模型格式是否支持")
        print("检查模型文件是否损坏")
    else:
        print(f"其他错误: {e}")
```

#### GPU内存不足
```python
try:
    result = inference_manager.predict('my_model', large_data)
except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        print("GPU内存不足，尝试以下解决方案:")
        print("1. 减小批处理大小")
        print("2. 使用CPU推理")
        print("3. 清理GPU内存")
        
        # 使用CPU推理
        result = inference_manager.predict('my_model', large_data, use_gpu=False)
```

#### 缓存错误
```python
try:
    cached_result = cache.get('my_model', input_data)
except Exception as e:
    print(f"缓存错误: {e}")
    print("清理缓存并重试")
    cache.clear()
    result = inference_manager.predict('my_model', input_data)
```

### 2. 调试技巧

#### 启用详细日志
```python
import logging

# 设置日志级别
logging.basicConfig(level=logging.DEBUG)

# 查看推理管理器日志
logger = logging.getLogger('src.models.inference.inference_manager')
logger.setLevel(logging.DEBUG)
```

#### 性能分析
```python
import time
import cProfile
import pstats

# 性能分析
def profile_inference():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # 执行推理
    result = inference_manager.predict('my_model', input_data)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # 显示前10个最耗时的函数

profile_inference()
```

## 最佳实践

### 1. 模型管理

#### 模型版本控制
```python
from src.models.version_manager import ModelVersionManager

# 创建版本管理器
version_manager = ModelVersionManager()

# 创建模型版本
version_id = version_manager.create_version(
    model=my_model,
    model_name='my_model',
    description='优化后的模型',
    created_by='developer',
    metadata={'accuracy': 0.95},
    performance_metrics={'inference_time': 0.01}
)

# 激活版本
version_manager.activate_version(version_id)
```

#### 模型验证
```python
# 验证模型性能
def validate_model(model_id, test_data, expected_accuracy=0.9):
    result = inference_manager.predict(model_id, test_data)
    
    # 计算准确率
    accuracy = calculate_accuracy(result, test_labels)
    
    if accuracy >= expected_accuracy:
        print(f"模型验证通过，准确率: {accuracy:.2%}")
        return True
    else:
        print(f"模型验证失败，准确率: {accuracy:.2%}")
        return False
```

### 2. 性能优化

#### 批处理优化
```python
# 动态调整批处理大小
def optimize_batch_size(model_id, sample_data):
    batch_sizes = [16, 32, 64, 128]
    best_batch_size = 32
    best_throughput = 0
    
    for batch_size in batch_sizes:
        start_time = time.time()
        result = inference_manager.batch_predict(model_id, sample_data, batch_size)
        end_time = time.time()
        
        throughput = len(sample_data) / (end_time - start_time)
        if throughput > best_throughput:
            best_throughput = throughput
            best_batch_size = batch_size
    
    return best_batch_size
```

#### 缓存优化
```python
# 预热缓存
def warm_up_cache(model_id, common_inputs):
    print("预热缓存...")
    for input_data in common_inputs:
        result = inference_manager.predict(model_id, input_data)
        # 结果会自动缓存
    print("缓存预热完成")
```

### 3. 监控告警

#### 设置性能告警
```python
def check_performance_alerts(metrics):
    alerts = []
    
    # 推理时间告警
    if metrics['avg_inference_time'] > 0.1:
        alerts.append(f"推理时间过长: {metrics['avg_inference_time']:.4f}s")
    
    # 错误率告警
    if metrics['error_rate'] > 0.01:
        alerts.append(f"错误率过高: {metrics['error_rate']:.2%}")
    
    # 内存使用告警
    memory_percent = metrics['system_memory']['percent']
    if memory_percent > 90:
        alerts.append(f"内存使用过高: {memory_percent:.1f}%")
    
    return alerts

# 定期检查
alerts = check_performance_alerts(metrics)
for alert in alerts:
    print(f"告警: {alert}")
```

## 故障排除

### 1. 常见问题

#### 问题1: 模型加载失败
**症状**: ValueError: 模型加载失败
**解决方案**:
1. 检查模型文件路径是否正确
2. 确认模型格式是否支持
3. 验证模型文件是否完整
4. 检查依赖库是否安装

#### 问题2: GPU内存不足
**症状**: RuntimeError: CUDA out of memory
**解决方案**:
1. 减小批处理大小
2. 使用CPU推理
3. 清理GPU内存
4. 优化模型大小

#### 问题3: 推理速度慢
**症状**: 推理时间过长
**解决方案**:
1. 启用GPU加速
2. 优化批处理大小
3. 使用缓存机制
4. 检查数据预处理

#### 问题4: 缓存命中率低
**症状**: 缓存命中率低于预期
**解决方案**:
1. 增加缓存大小
2. 调整TTL设置
3. 优化缓存策略
4. 预热缓存

### 2. 调试工具

#### 性能分析工具
```python
import time
import psutil
import numpy as np

def performance_profiler(func):
    """性能分析装饰器"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.virtual_memory().used
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.virtual_memory().used
        
        print(f"函数: {func.__name__}")
        print(f"执行时间: {end_time - start_time:.4f}s")
        print(f"内存使用: {(end_memory - start_memory) / 1024**2:.2f} MB")
        
        return result
    return wrapper

# 使用装饰器
@performance_profiler
def my_inference_function(input_data):
    return inference_manager.predict('my_model', input_data)
```

#### 内存监控
```python
def monitor_memory():
    """监控内存使用"""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    print(f"进程内存使用: {memory_info.rss / 1024**2:.2f} MB")
    print(f"虚拟内存使用: {memory_info.vms / 1024**2:.2f} MB")
    
    return memory_info

# 定期监控
memory_info = monitor_memory()
```

## 总结

本指南提供了模型推理功能的完整使用方法，包括：

1. **基本使用**: 快速上手和基本配置
2. **高级配置**: GPU加速、缓存、批量处理
3. **性能监控**: 实时监控和性能分析
4. **错误处理**: 常见错误和解决方案
5. **最佳实践**: 性能优化和最佳实践
6. **故障排除**: 调试工具和问题解决

通过遵循本指南，您可以充分利用RQA2025项目的模型推理功能，实现高效、稳定的模型推理服务。 