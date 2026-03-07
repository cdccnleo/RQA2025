# 模型推理API文档

## 概述

本文档描述了RQA2025项目中模型推理模块的API接口，包括推理管理器、模型加载器、GPU推理引擎等组件的详细使用方法。

## ModelInferenceManager API

### 类定义
```python
class ModelInferenceManager:
    """模型推理管理器"""
```

### 初始化
```python
def __init__(self, config: Optional[Dict[str, Any]] = None):
    """初始化推理管理器
    
    Args:
        config: 配置字典，包含以下可选参数：
            - enable_gpu: 是否启用GPU (默认: True)
            - enable_cache: 是否启用缓存 (默认: True)
            - enable_batch_processing: 是否启用批量处理 (默认: True)
            - max_batch_size: 最大批处理大小 (默认: 32)
            - cache_size: 缓存大小 (默认: 100)
            - enable_monitoring: 是否启用性能监控 (默认: True)
            - model_storage_path: 模型存储路径 (默认: './models')
            - log_performance: 是否记录性能日志 (默认: True)
    """
```

### 主要方法

#### predict
```python
def predict(self, model_id: str, input_data: np.ndarray, 
            use_cache: bool = True, use_gpu: bool = True) -> np.ndarray:
    """执行模型推理
    
    Args:
        model_id: 模型ID
        input_data: 输入数据，numpy数组
        use_cache: 是否使用缓存 (默认: True)
        use_gpu: 是否使用GPU (默认: True)
    
    Returns:
        推理结果，numpy数组
    
    Raises:
        ValueError: 模型加载失败
        RuntimeError: 推理执行失败
    """
```

#### batch_predict
```python
def batch_predict(self, model_id: str, input_data: np.ndarray, 
                 batch_size: Optional[int] = None) -> np.ndarray:
    """批量推理
    
    Args:
        model_id: 模型ID
        input_data: 输入数据，numpy数组
        batch_size: 批处理大小，如果为None则使用默认值
    
    Returns:
        批量推理结果，numpy数组
    """
```

#### get_performance_metrics
```python
def get_performance_metrics(self) -> Dict[str, Any]:
    """获取性能指标
    
    Returns:
        性能指标字典，包含以下信息：
        - avg_inference_time: 平均推理时间
        - max_inference_time: 最大推理时间
        - min_inference_time: 最小推理时间
        - avg_memory_usage: 平均内存使用
        - avg_gpu_usage: 平均GPU使用率
        - total_requests: 总请求数
        - error_rate: 错误率
        - throughput: 吞吐量
        - system_memory: 系统内存信息
        - cpu_usage: CPU使用率
        - gpu_available: GPU是否可用
        - cache_hit_rate: 缓存命中率
    """
```

### 使用示例

```python
# 初始化推理管理器
config = {
    'enable_gpu': True,
    'enable_cache': True,
    'max_batch_size': 64,
    'cache_size': 200
}
inference_manager = ModelInferenceManager(config)

# 单次推理
input_data = np.random.randn(100, 10)
result = inference_manager.predict('my_model', input_data)

# 批量推理
batch_data = np.random.randn(1000, 10)
batch_result = inference_manager.batch_predict('my_model', batch_data, batch_size=32)

# 获取性能指标
metrics = inference_manager.get_performance_metrics()
print(f"平均推理时间: {metrics['avg_inference_time']:.4f}s")
print(f"错误率: {metrics['error_rate']:.2%}")
```

## ModelLoader API

### 类定义
```python
class ModelLoader:
    """模型加载器"""
```

### 初始化
```python
def __init__(self, config: Optional[Dict[str, Any]] = None):
    """初始化模型加载器
    
    Args:
        config: 配置字典，包含以下可选参数：
            - model_storage_path: 模型存储路径 (默认: './models')
            - cache_models: 是否缓存模型 (默认: True)
            - validate_models: 是否验证模型 (默认: True)
            - enable_versioning: 是否启用版本控制 (默认: True)
            - max_cache_size: 最大缓存大小 (默认: 10)
    """
```

### 主要方法

#### load_model
```python
def load_model(self, model_path: str, model_type: str = 'auto') -> Optional[Any]:
    """加载模型
    
    Args:
        model_path: 模型文件路径
        model_type: 模型类型 ('auto', 'pytorch', 'tensorflow', 'onnx', 'tensorrt', 'pickle')
    
    Returns:
        加载的模型对象，如果加载失败返回None
    """
```

#### validate_model_file
```python
def validate_model_file(self, model_path: str) -> bool:
    """验证模型文件
    
    Args:
        model_path: 模型文件路径
    
    Returns:
        验证是否通过
    """
```

#### get_model_metadata
```python
def get_model_metadata(self, model_path: str) -> Dict[str, Any]:
    """获取模型元数据
    
    Args:
        model_path: 模型文件路径
    
    Returns:
        模型元数据字典，包含以下信息：
        - model_path: 模型路径
        - model_type: 模型类型
        - file_size: 文件大小
        - file_hash: 文件哈希
        - created_time: 创建时间
        - last_modified: 最后修改时间
    """
```

#### list_available_models
```python
def list_available_models(self) -> List[Dict[str, Any]]:
    """列出可用的模型
    
    Returns:
        模型信息列表
    """
```

### 使用示例

```python
# 初始化模型加载器
loader = ModelLoader({
    'model_storage_path': './models',
    'cache_models': True
})

# 加载PyTorch模型
pytorch_model = loader.load_model('model.pth', 'pytorch')

# 加载TensorFlow模型
tf_model = loader.load_model('model.h5', 'tensorflow')

# 加载ONNX模型
onnx_model = loader.load_model('model.onnx', 'onnx')

# 自动检测模型类型
auto_model = loader.load_model('model.pth')

# 验证模型文件
is_valid = loader.validate_model_file('model.pth')

# 获取模型元数据
metadata = loader.get_model_metadata('model.pth')
print(f"模型类型: {metadata['model_type']}")
print(f"文件大小: {metadata['file_size']} bytes")

# 列出所有可用模型
models = loader.list_available_models()
for model in models:
    print(f"模型: {model['model_path']}, 类型: {model['model_type']}")
```

## GPUInferenceEngine API

### 类定义
```python
class GPUInferenceEngine:
    """GPU推理引擎"""
```

### 主要方法

#### inference
```python
def inference(self, model: Any, input_data: np.ndarray) -> np.ndarray:
    """GPU推理
    
    Args:
        model: 模型对象
        input_data: 输入数据
    
    Returns:
        推理结果
    """
```

#### get_gpu_usage
```python
def get_gpu_usage(self) -> Optional[float]:
    """获取GPU使用率
    
    Returns:
        GPU使用率 (0-1)，如果GPU不可用返回None
    """
```

#### get_gpu_memory_info
```python
def get_gpu_memory_info(self) -> Dict[str, Any]:
    """获取GPU内存信息
    
    Returns:
        GPU内存信息字典
    """
```

### 使用示例

```python
# 初始化GPU推理引擎
gpu_engine = GPUInferenceEngine()

# 执行GPU推理
result = gpu_engine.inference(model, input_data)

# 获取GPU使用率
gpu_usage = gpu_engine.get_gpu_usage()
print(f"GPU使用率: {gpu_usage:.2%}")

# 获取GPU内存信息
memory_info = gpu_engine.get_gpu_memory_info()
print(f"GPU内存使用: {memory_info['used']}/{memory_info['total']} MB")
```

## BatchInferenceProcessor API

### 类定义
```python
class BatchInferenceProcessor:
    """批量推理处理器"""
```

### 初始化
```python
def __init__(self, config: Optional[Dict[str, Any]] = None):
    """初始化批量推理处理器
    
    Args:
        config: 配置字典，包含以下可选参数：
            - max_batch_size: 最大批处理大小 (默认: 32)
            - enable_parallel: 是否启用并行处理 (默认: True)
            - memory_limit: 内存使用限制 (默认: 0.8)
            - retry_count: 重试次数 (默认: 3)
    """
```

### 主要方法

#### process
```python
def process(self, model_id: str, input_data: np.ndarray, 
            batch_size: int, inference_manager) -> np.ndarray:
    """批量处理
    
    Args:
        model_id: 模型ID
        input_data: 输入数据
        batch_size: 批处理大小
        inference_manager: 推理管理器实例
    
    Returns:
        批量处理结果
    """
```

#### optimize_batch_size
```python
def optimize_batch_size(self, model_id: str, sample_data: np.ndarray, 
                       inference_manager) -> int:
    """优化批处理大小
    
    Args:
        model_id: 模型ID
        sample_data: 样本数据
        inference_manager: 推理管理器实例
    
    Returns:
        优化的批处理大小
    """
```

### 使用示例

```python
# 初始化批量推理处理器
batch_processor = BatchInferenceProcessor({
    'max_batch_size': 64,
    'enable_parallel': True
})

# 批量处理
result = batch_processor.process('my_model', large_input_data, 32, inference_manager)

# 优化批处理大小
optimal_batch_size = batch_processor.optimize_batch_size('my_model', sample_data, inference_manager)
print(f"最优批处理大小: {optimal_batch_size}")
```

## InferenceCache API

### 类定义
```python
class InferenceCache:
    """推理缓存"""
```

### 初始化
```python
def __init__(self, config: Optional[Dict[str, Any]] = None):
    """初始化推理缓存
    
    Args:
        config: 配置字典，包含以下可选参数：
            - cache_size: 缓存大小 (默认: 100)
            - ttl_seconds: 缓存过期时间 (默认: 3600)
            - enable_disk_cache: 是否启用磁盘缓存 (默认: True)
            - disk_cache_path: 磁盘缓存路径 (默认: './cache')
    """
```

### 主要方法

#### get
```python
def get(self, model_id: str, input_data: np.ndarray) -> Optional[np.ndarray]:
    """获取缓存结果
    
    Args:
        model_id: 模型ID
        input_data: 输入数据
    
    Returns:
        缓存的结果，如果未命中返回None
    """
```

#### put
```python
def put(self, model_id: str, input_data: np.ndarray, result: np.ndarray):
    """存储缓存结果
    
    Args:
        model_id: 模型ID
        input_data: 输入数据
        result: 推理结果
    """
```

#### get_hit_rate
```python
def get_hit_rate(self) -> float:
    """获取缓存命中率
    
    Returns:
        缓存命中率 (0-1)
    """
```

#### clear
```python
def clear(self):
    """清空缓存"""
```

### 使用示例

```python
# 初始化推理缓存
cache = InferenceCache({
    'cache_size': 200,
    'ttl_seconds': 7200
})

# 检查缓存
cached_result = cache.get('my_model', input_data)
if cached_result is not None:
    print("使用缓存结果")
else:
    # 执行推理
    result = inference_manager.predict('my_model', input_data)
    # 存储到缓存
    cache.put('my_model', input_data, result)

# 获取缓存命中率
hit_rate = cache.get_hit_rate()
print(f"缓存命中率: {hit_rate:.2%}")

# 清空缓存
cache.clear()
```

## 错误处理

### 常见异常

1. **ModelLoadError**: 模型加载失败
2. **InferenceError**: 推理执行失败
3. **GPUError**: GPU相关错误
4. **CacheError**: 缓存相关错误
5. **ValidationError**: 数据验证失败

### 错误处理示例

```python
try:
    result = inference_manager.predict('my_model', input_data)
except ValueError as e:
    print(f"模型加载失败: {e}")
except RuntimeError as e:
    print(f"推理执行失败: {e}")
except Exception as e:
    print(f"未知错误: {e}")
```

## 性能优化建议

### 1. 批处理优化
- 使用合适的批处理大小
- 启用并行处理
- 监控内存使用

### 2. 缓存优化
- 设置合理的缓存大小
- 使用TTL过期机制
- 监控缓存命中率

### 3. GPU优化
- 监控GPU使用率
- 优化内存分配
- 使用混合精度

### 4. 监控和调优
- 定期检查性能指标
- 根据负载调整配置
- 优化模型加载策略

## 总结

模型推理API提供了完整的推理功能，包括：

1. **统一接口**: 通过ModelInferenceManager提供统一的推理接口
2. **多格式支持**: 支持PyTorch、TensorFlow、ONNX、TensorRT等多种格式
3. **性能优化**: GPU加速、批量处理、缓存机制
4. **监控告警**: 完整的性能监控和错误处理
5. **易于使用**: 简洁的API设计，丰富的使用示例

这些API为RQA2025项目提供了强大而灵活的模型推理能力，支持各种复杂的推理场景。 