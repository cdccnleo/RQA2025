# 特征层错误处理指南

## 概述

本文档详细介绍了RQA特征层的错误处理机制、异常类型、错误恢复策略和最佳实践。通过统一的错误处理，确保特征处理流程的稳定性和可靠性。

## 异常类型定义

### 1. 数据验证异常

#### FeatureDataValidationError
```python
class FeatureDataValidationError(ValueError):
    """特征数据验证错误"""
    
    def __init__(self, message: str, missing_columns: List[str] = None, 
                 invalid_types: List[str] = None):
        self.message = message
        self.missing_columns = missing_columns or []
        self.invalid_types = invalid_types or []
        super().__init__(self.message)
```

**触发条件：**
- 输入数据为空
- 缺失必要列（close, high, low, volume）
- 数据类型不正确
- 包含缺失值

**使用示例：**
```python
from src.features.exceptions import FeatureDataValidationError

def validate_stock_data(data: pd.DataFrame) -> bool:
    """验证股票数据"""
    if data.empty:
        raise FeatureDataValidationError("输入数据为空")
    
    required_columns = ['close', 'high', 'low', 'volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        raise FeatureDataValidationError(
            "缺失必要列",
            missing_columns=missing_columns
        )
    
    return True
```

### 2. 配置验证异常

#### FeatureConfigValidationError
```python
class FeatureConfigValidationError(ValueError):
    """特征配置验证错误"""
    
    def __init__(self, message: str, config_field: str = None, 
                 expected_value: Any = None, actual_value: Any = None):
        self.message = message
        self.config_field = config_field
        self.expected_value = expected_value
        self.actual_value = actual_value
        super().__init__(self.message)
```

**触发条件：**
- 配置参数无效
- 特征类型不支持
- 参数范围超出限制

**使用示例：**
```python
from src.features.exceptions import FeatureConfigValidationError

def validate_feature_config(config: FeatureConfig) -> bool:
    """验证特征配置"""
    if config.max_features < config.min_features:
        raise FeatureConfigValidationError(
            "最大特征数不能小于最小特征数",
            config_field="max_features",
            expected_value=f">= {config.min_features}",
            actual_value=config.max_features
        )
    
    return True
```

### 3. 处理异常

#### FeatureProcessingError
```python
class FeatureProcessingError(RuntimeError):
    """特征处理错误"""
    
    def __init__(self, message: str, processor_name: str = None, 
                 step: str = None, original_error: Exception = None):
        self.message = message
        self.processor_name = processor_name
        self.step = step
        self.original_error = original_error
        super().__init__(self.message)
```

**触发条件：**
- 特征计算失败
- 处理器执行错误
- 内存不足

**使用示例：**
```python
from src.features.exceptions import FeatureProcessingError

def safe_feature_computation(data: pd.DataFrame, feature_name: str) -> pd.Series:
    """安全的特征计算"""
    try:
        if feature_name == "sma":
            return data['close'].rolling(window=20).mean()
        elif feature_name == "rsi":
            return calculate_rsi(data['close'])
        else:
            raise ValueError(f"不支持的特征: {feature_name}")
            
    except Exception as e:
        raise FeatureProcessingError(
            f"特征计算失败: {feature_name}",
            processor_name="technical_processor",
            step="feature_computation",
            original_error=e
        )
```

### 4. 标准化异常

#### FeatureStandardizationError
```python
class FeatureStandardizationError(RuntimeError):
    """特征标准化错误"""
    
    def __init__(self, message: str, method: str = None, 
                 scaler_path: str = None):
        self.message = message
        self.method = method
        self.scaler_path = scaler_path
        super().__init__(self.message)
```

**触发条件：**
- 标准化器未拟合
- 标准化方法不支持
- 模型文件损坏

**使用示例：**
```python
from src.features.exceptions import FeatureStandardizationError

def safe_standardization(features: pd.DataFrame, standardizer: FeatureStandardizer) -> pd.DataFrame:
    """安全的特征标准化"""
    try:
        return standardizer.transform(features)
    except NotFittedError:
        raise FeatureStandardizationError(
            "标准化器尚未拟合",
            method=standardizer.method
        )
    except Exception as e:
        raise FeatureStandardizationError(
            f"标准化失败: {str(e)}",
            method=standardizer.method
        )
```

## 错误处理策略

### 1. 防御性编程

#### 输入验证
```python
def robust_feature_processing(data: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """健壮的特征处理函数"""
    
    # 1. 输入验证
    try:
        validate_input_data(data)
    except FeatureDataValidationError as e:
        logger.error(f"数据验证失败: {e}")
        # 返回空DataFrame或抛出异常
        return pd.DataFrame()
    
    # 2. 配置验证
    try:
        validate_feature_config(config)
    except FeatureConfigValidationError as e:
        logger.error(f"配置验证失败: {e}")
        # 使用默认配置
        config = DefaultConfigs.basic_technical()
    
    # 3. 特征处理
    try:
        engine = FeatureEngine()
        features = engine.process_features(data, config)
        return features
    except FeatureProcessingError as e:
        logger.error(f"特征处理失败: {e}")
        # 返回原始数据或空DataFrame
        return data
```

#### 空值处理
```python
def handle_missing_values(data: pd.DataFrame, strategy: str = "forward") -> pd.DataFrame:
    """处理缺失值"""
    if data.empty:
        return data
    
    try:
        if strategy == "forward":
            return data.fillna(method='ffill')
        elif strategy == "backward":
            return data.fillna(method='bfill')
        elif strategy == "interpolate":
            return data.interpolate()
        else:
            return data.dropna()
    except Exception as e:
        logger.warning(f"缺失值处理失败: {e}")
        return data
```

### 2. 重试机制

#### 指数退避重试
```python
import time
from functools import wraps

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """指数退避重试装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (FeatureProcessingError, FeatureStandardizationError) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"第{attempt + 1}次尝试失败，{delay}秒后重试: {e}")
                        time.sleep(delay)
                    else:
                        logger.error(f"重试{max_retries}次后仍然失败: {e}")
                        raise last_exception
                except Exception as e:
                    # 对于其他异常，不重试
                    raise e
            
            raise last_exception
        return wrapper
    return decorator

# 使用重试装饰器
@retry_with_backoff(max_retries=3, base_delay=1.0)
def robust_feature_engine_processing(data: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """带重试机制的特征处理"""
    engine = FeatureEngine()
    return engine.process_features(data, config)
```

#### 条件重试
```python
def conditional_retry(func, retry_conditions: List[Type[Exception]], max_retries: int = 3):
    """条件重试函数"""
    def wrapper(*args, **kwargs):
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except tuple(retry_conditions) as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(f"第{attempt + 1}次尝试失败，重试: {e}")
                    time.sleep(1)
                else:
                    logger.error(f"重试{max_retries}次后仍然失败: {e}")
                    raise last_exception
            except Exception as e:
                # 对于其他异常，不重试
                raise e
        
        raise last_exception
    return wrapper

# 使用条件重试
def safe_standardization_with_retry(features: pd.DataFrame, standardizer: FeatureStandardizer) -> pd.DataFrame:
    """带重试的安全标准化"""
    return conditional_retry(
        standardizer.transform,
        retry_conditions=[FeatureStandardizationError],
        max_retries=3
    )(features)
```

### 3. 降级策略

#### 特征处理降级
```python
class FeatureProcessingFallback:
    """特征处理降级策略"""
    
    def __init__(self):
        self.fallback_strategies = {
            "technical": self._fallback_technical,
            "sentiment": self._fallback_sentiment,
            "standardization": self._fallback_standardization
        }
    
    def process_with_fallback(self, data: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
        """带降级的特征处理"""
        try:
            engine = FeatureEngine()
            return engine.process_features(data, config)
        except FeatureProcessingError as e:
            logger.warning(f"特征处理失败，使用降级策略: {e}")
            return self._apply_fallback_strategy(data, config)
    
    def _apply_fallback_strategy(self, data: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
        """应用降级策略"""
        # 1. 尝试基本技术指标
        try:
            basic_config = DefaultConfigs.basic_technical()
            engine = FeatureEngine()
            return engine.process_features(data, basic_config)
        except Exception as e:
            logger.warning(f"基本技术指标失败: {e}")
        
        # 2. 返回原始数据
        logger.warning("所有降级策略失败，返回原始数据")
        return data
    
    def _fallback_technical(self, data: pd.DataFrame) -> pd.DataFrame:
        """技术指标降级策略"""
        # 只计算简单的移动平均
        result = pd.DataFrame(index=data.index)
        result['sma_5'] = data['close'].rolling(window=5).mean()
        result['sma_10'] = data['close'].rolling(window=10).mean()
        return result
    
    def _fallback_sentiment(self, data: pd.DataFrame) -> pd.DataFrame:
        """情感分析降级策略"""
        # 返回中性情感
        result = pd.DataFrame(index=data.index)
        result['sentiment_score'] = 0.0
        return result
    
    def _fallback_standardization(self, features: pd.DataFrame) -> pd.DataFrame:
        """标准化降级策略"""
        # 使用简单的Z-score标准化
        numeric_features = features.select_dtypes(include=[np.number])
        if not numeric_features.empty:
            mean = numeric_features.mean()
            std = numeric_features.std()
            return (numeric_features - mean) / std
        return features
```

### 4. 错误恢复

#### 优雅的错误恢复
```python
class FeatureErrorRecovery:
    """特征错误恢复机制"""
    
    def __init__(self):
        self.recovery_strategies = {
            "data_validation": self._recover_data_validation,
            "config_validation": self._recover_config_validation,
            "processing_error": self._recover_processing_error,
            "standardization_error": self._recover_standardization_error
        }
    
    def recover_from_error(self, error: Exception, data: pd.DataFrame, 
                          config: FeatureConfig) -> pd.DataFrame:
        """从错误中恢复"""
        error_type = type(error).__name__
        
        if error_type in self.recovery_strategies:
            logger.info(f"尝试恢复错误: {error_type}")
            return self.recovery_strategies[error_type](error, data, config)
        else:
            logger.error(f"未知错误类型，无法恢复: {error_type}")
            return pd.DataFrame()
    
    def _recover_data_validation(self, error: FeatureDataValidationError, 
                                data: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
        """从数据验证错误中恢复"""
        # 尝试清理数据
        cleaned_data = self._clean_data(data)
        if not cleaned_data.empty:
            engine = FeatureEngine()
            return engine.process_features(cleaned_data, config)
        return pd.DataFrame()
    
    def _recover_config_validation(self, error: FeatureConfigValidationError, 
                                  data: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
        """从配置验证错误中恢复"""
        # 使用默认配置
        default_config = DefaultConfigs.basic_technical()
        engine = FeatureEngine()
        return engine.process_features(data, default_config)
    
    def _recover_processing_error(self, error: FeatureProcessingError, 
                                 data: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
        """从处理错误中恢复"""
        # 尝试简化处理
        try:
            engine = FeatureEngine()
            # 只使用基本处理器
            basic_config = DefaultConfigs.basic_technical()
            return engine.process_features(data, basic_config)
        except Exception:
            return data
    
    def _recover_standardization_error(self, error: FeatureStandardizationError, 
                                      data: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
        """从标准化错误中恢复"""
        # 跳过标准化，返回原始特征
        return data
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """清理数据"""
        if data.empty:
            return data
        
        # 移除完全为空的列
        data = data.dropna(axis=1, how='all')
        
        # 填充缺失值
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        return data
```

## 日志记录

### 1. 结构化日志

```python
import logging
import json
from datetime import datetime

class FeatureErrorLogger:
    """特征错误日志记录器"""
    
    def __init__(self, logger_name: str = "features.error"):
        self.logger = logging.getLogger(logger_name)
        self.error_count = 0
    
    def log_error(self, error: Exception, context: dict = None):
        """记录错误日志"""
        self.error_count += 1
        
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "error_id": f"FE_{self.error_count:06d}",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {},
            "stack_trace": self._get_stack_trace(error)
        }
        
        self.logger.error(f"特征处理错误: {json.dumps(error_info, ensure_ascii=False)}")
        
        # 记录到文件
        self._write_error_log(error_info)
    
    def _get_stack_trace(self, error: Exception) -> str:
        """获取堆栈跟踪"""
        import traceback
        return traceback.format_exc()
    
    def _write_error_log(self, error_info: dict):
        """写入错误日志文件"""
        log_file = f"logs/feature_errors_{datetime.now().strftime('%Y%m%d')}.json"
        
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(error_info, ensure_ascii=False) + '\n')
        except Exception as e:
            self.logger.error(f"写入错误日志失败: {e}")
    
    def get_error_summary(self) -> dict:
        """获取错误摘要"""
        return {
            "total_errors": self.error_count,
            "last_error_time": getattr(self, '_last_error_time', None)
        }
```

### 2. 性能监控

```python
import time
from contextlib import contextmanager

class FeaturePerformanceMonitor:
    """特征性能监控器"""
    
    def __init__(self):
        self.metrics = {
            "processing_time": [],
            "memory_usage": [],
            "error_count": 0,
            "success_count": 0
        }
    
    @contextmanager
    def monitor_processing(self, operation_name: str):
        """监控处理性能"""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            yield
            # 成功
            self.metrics["success_count"] += 1
        except Exception as e:
            # 失败
            self.metrics["error_count"] += 1
            raise e
        finally:
            # 记录性能指标
            processing_time = time.time() - start_time
            memory_usage = self._get_memory_usage() - start_memory
            
            self.metrics["processing_time"].append(processing_time)
            self.metrics["memory_usage"].append(memory_usage)
            
            logger.info(f"{operation_name} 完成，耗时: {processing_time:.2f}秒，内存: {memory_usage:.2f}MB")
    
    def _get_memory_usage(self) -> float:
        """获取内存使用量（MB）"""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def get_performance_summary(self) -> dict:
        """获取性能摘要"""
        if not self.metrics["processing_time"]:
            return {}
        
        return {
            "avg_processing_time": sum(self.metrics["processing_time"]) / len(self.metrics["processing_time"]),
            "max_processing_time": max(self.metrics["processing_time"]),
            "avg_memory_usage": sum(self.metrics["memory_usage"]) / len(self.metrics["memory_usage"]),
            "success_rate": self.metrics["success_count"] / (self.metrics["success_count"] + self.metrics["error_count"]),
            "total_operations": self.metrics["success_count"] + self.metrics["error_count"]
        }
```

## 最佳实践

### 1. 统一的错误处理

```python
class FeatureErrorHandler:
    """统一的特征错误处理器"""
    
    def __init__(self):
        self.logger = FeatureErrorLogger()
        self.monitor = FeaturePerformanceMonitor()
        self.recovery = FeatureErrorRecovery()
        self.fallback = FeatureProcessingFallback()
    
    def process_with_error_handling(self, data: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
        """带错误处理的特征处理"""
        
        with self.monitor.monitor_processing("feature_processing"):
            try:
                # 主要处理流程
                engine = FeatureEngine()
                return engine.process_features(data, config)
                
            except FeatureDataValidationError as e:
                self.logger.log_error(e, {"data_shape": data.shape})
                return self.recovery.recover_from_error(e, data, config)
                
            except FeatureConfigValidationError as e:
                self.logger.log_error(e, {"config": config.to_dict()})
                return self.recovery.recover_from_error(e, data, config)
                
            except FeatureProcessingError as e:
                self.logger.log_error(e, {"processor": e.processor_name})
                return self.fallback.process_with_fallback(data, config)
                
            except FeatureStandardizationError as e:
                self.logger.log_error(e, {"method": e.method})
                return self.recovery.recover_from_error(e, data, config)
                
            except Exception as e:
                self.logger.log_error(e)
                return pd.DataFrame()
    
    def get_processing_summary(self) -> dict:
        """获取处理摘要"""
        return {
            "performance": self.monitor.get_performance_summary(),
            "errors": self.logger.get_error_summary()
        }
```

### 2. 错误处理装饰器

```python
def handle_feature_errors(func):
    """特征错误处理装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        handler = FeatureErrorHandler()
        
        # 提取数据和配置参数
        data = None
        config = None
        
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                data = arg
            elif isinstance(arg, FeatureConfig):
                config = arg
        
        for key, value in kwargs.items():
            if isinstance(value, pd.DataFrame):
                data = value
            elif isinstance(value, FeatureConfig):
                config = value
        
        if data is not None and config is not None:
            return handler.process_with_error_handling(data, config)
        else:
            # 如果没有找到合适的参数，直接调用原函数
            return func(*args, **kwargs)
    
    return wrapper

# 使用装饰器
@handle_feature_errors
def process_features(data: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """处理特征（带错误处理）"""
    engine = FeatureEngine()
    return engine.process_features(data, config)
```

### 3. 错误报告

```python
class FeatureErrorReporter:
    """特征错误报告器"""
    
    def __init__(self):
        self.error_logger = FeatureErrorLogger()
        self.performance_monitor = FeaturePerformanceMonitor()
    
    def generate_error_report(self, start_time: datetime, end_time: datetime) -> dict:
        """生成错误报告"""
        return {
            "report_period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "performance_summary": self.performance_monitor.get_performance_summary(),
            "error_summary": self.error_logger.get_error_summary(),
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        performance = self.performance_monitor.get_performance_summary()
        errors = self.error_logger.get_error_summary()
        
        if performance.get("success_rate", 1.0) < 0.95:
            recommendations.append("建议检查数据质量和配置参数")
        
        if performance.get("avg_processing_time", 0) > 10:
            recommendations.append("建议优化特征计算性能")
        
        if errors.get("total_errors", 0) > 10:
            recommendations.append("建议增加错误处理和恢复机制")
        
        return recommendations
```

## 总结

通过统一的错误处理机制，特征层能够：

1. **提高稳定性**: 通过防御性编程和错误恢复机制
2. **改善可观测性**: 通过结构化日志和性能监控
3. **增强可维护性**: 通过标准化的异常类型和错误处理策略
4. **支持调试**: 通过详细的错误信息和上下文

建议在实际使用中：

1. 根据具体需求选择合适的错误处理策略
2. 定期监控错误率和性能指标
3. 及时处理异常和错误情况
4. 持续改进错误处理机制 