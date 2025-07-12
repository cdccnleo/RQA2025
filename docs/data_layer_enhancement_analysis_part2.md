# 数据层功能增强分析报告（第二部分）

## 功能实现建议（续）

### 2. 功能扩展（续）

#### 2.2 数据导出功能（续）

```python
def export_parquet(
    self,
    data: pd.DataFrame,
    filename: str,
    **kwargs
) -> str:
    """
    导出为Parquet格式
    
    Args:
        data: 数据框
        filename: 文件名
        **kwargs: 其他参数传递给to_parquet
        
    Returns:
        str: 导出文件路径
    """
    filepath = self.export_dir / filename
    
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be DataFrame for Parquet export")
    
    data.to_parquet(filepath, **kwargs)
    return str(filepath)

def export_feather(
    self,
    data: pd.DataFrame,
    filename: str,
    **kwargs
) -> str:
    """
    导出为Feather格式
    
    Args:
        data: 数据框
        filename: 文件名
        **kwargs: 其他参数传递给to_feather
        
    Returns:
        str: 导出文件路径
    """
    filepath = self.export_dir / filename
    
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be DataFrame for Feather export")
    
    data.to_feather(filepath, **kwargs)
    return str(filepath)

def export_hdf(
    self,
    data: pd.DataFrame,
    filename: str,
    key: str = 'data',
    **kwargs
) -> str:
    """
    导出为HDF5格式
    
    Args:
        data: 数据框
        filename: 文件名
        key: HDF5键名
        **kwargs: 其他参数传递给to_hdf
        
    Returns:
        str: 导出文件路径
    """
    filepath = self.export_dir / filename
    
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be DataFrame for HDF export")
    
    data.to_hdf(filepath, key=key, **kwargs)
    return str(filepath)

def export_sql(
    self,
    data: pd.DataFrame,
    table_name: str,
    conn_string: str,
    if_exists: str = 'replace',
    **kwargs
) -> bool:
    """
    导出到SQL数据库
    
    Args:
        data: 数据框
        table_name: 表名
        conn_string: 数据库连接字符串
        if_exists: 如果表已存在的处理方式
        **kwargs: 其他参数传递给to_sql
        
    Returns:
        bool: 是否成功
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be DataFrame for SQL export")
    
    try:
        import sqlalchemy
        engine = sqlalchemy.create_engine(conn_string)
        data.to_sql(table_name, engine, if_exists=if_exists, **kwargs)
        return True
    except Exception as e:
        logger.error(f"Failed to export to SQL: {e}")
        return False
```

在 `DataManager` 中集成数据导出功能：

```python
def __init__(self, config: Dict[str, Any]):
    # ... 其他初始化代码 ...
    
    # 初始化数据导出器
    export_dir = config.get('export_dir', './data/exports')
    self.exporter = DataExporter(export_dir)

def export_data(
    self,
    format: str,
    filename: str = None,
    data_model: Optional[DataModel] = None,
    **kwargs
) -> Union[str, bool]:
    """
    导出数据
    
    Args:
        format: 导出格式，支持 'csv', 'excel', 'json', 'parquet', 'feather', 'hdf', 'sql'
        filename: 文件名，如果为None则自动生成
        data_model: 数据模型，默认为当前模型
        **kwargs: 其他参数传递给对应的导出方法
        
    Returns:
        Union[str, bool]: 导出文件路径或是否成功
    """
    if data_model is None:
        data_model = self.current_model
    
    if data_model is None:
        raise ValueError("No data model available")
    
    # 如果没有提供文件名，则自动生成
    if filename is None:
        metadata = data_model.get_metadata()
        data_type = metadata.get('type', 'data')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{data_type}_{timestamp}.{format}"
    
    # 根据格式选择导出方法
    if format == 'csv':
        return self.exporter.export_csv(data_model.data, filename, **kwargs)
    elif format == 'excel':
        return self.exporter.export_excel(data_model.data, filename, **kwargs)
    elif format == 'json':
        return self.exporter.export_json(data_model.data, filename, **kwargs)
    elif format == 'parquet':
        return self.exporter.export_parquet(data_model.data, filename, **kwargs)
    elif format == 'feather':
        return self.exporter.export_feather(data_model.data, filename, **kwargs)
    elif format == 'hdf':
        return self.exporter.export_hdf(data_model.data, filename, **kwargs)
    elif format == 'sql':
        if 'table_name' not in kwargs:
            metadata = data_model.get_metadata()
            data_type = metadata.get('type', 'data')
            kwargs['table_name'] = f"{data_type}_table"
        
        if 'conn_string' not in kwargs:
            raise ValueError("conn_string is required for SQL export")
        
        return self.exporter.export_sql(data_model.data, **kwargs)
    else:
        raise ValueError(f"Unsupported export format: {format}")
```

### 3. 监控告警

#### 3.1 性能监控

建议实现一个 `PerformanceMonitor` 类，用于监控数据加载和处理性能：

```python
import time
from typing import Dict, List, Any, Optional, Callable
from functools import wraps
import logging
import psutil
import threading
from datetime import datetime

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, log_level: int = logging.INFO):
        """
        初始化性能监控器
        
        Args:
            log_level: 日志级别
        """
        self.log_level = log_level
        self.metrics = {}
        self.running = False
        self.monitor_thread = None
        self.lock = threading.Lock()
    
    def start_monitoring(self, interval: float = 1.0):
        """
        开始系统资源监控
        
        Args:
            interval: 监控间隔（秒）
        """
        if self.monitor_thread is not None and self.monitor_thread.is_alive():
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_resources,
            args=(interval,)
        )
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止系统资源监控"""
        self.running = False
        if self.monitor_thread is not None:
            self.monitor_thread.join(timeout=1.0)
    
    def _monitor_resources(self, interval: float):
        """
        监控系统资源
        
        Args:
            interval: 监控间隔（秒）
        """
        resource_metrics = []
        
        while self.running:
            try:
                # 收集系统资源指标
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                metrics = {
                    'timestamp': datetime.now().isoformat(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_used_gb': memory.used / (1024 ** 3),
                    'disk_percent': disk.percent,
                    'disk_used_gb': disk.used / (1024 ** 3)
                }
                
                resource_metrics.append(metrics)
                
                # 每10个样本记录一次
                if len(resource_metrics) >= 10:
                    with self.lock:
                        self.metrics['system_resources'] = resource_metrics
                    resource_metrics = []
                
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error monitoring resources: {e}")
    
    def time_function(self, func_name: Optional[str] = None):
        """
        函数执行时间装饰器
        
        Args:
            func_name: 函数名，如果为None则使用函数的__name__
            
        Returns:
            Callable: 装饰器
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                name = func_name or func.__name__
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                execution_time = end_time - start_time
                
                # 记录执行时间
                with self.lock:
                    if 'function_times' not in self.metrics:
                        self.metrics['function_times'] = {}
                    
                    if name not in self.metrics['function_times']:
                        self.metrics['function_times'][name] = []
                    
                    self.metrics['function_times'][name].append({
                        'timestamp': datetime.now().isoformat(),
                        'execution_time': execution_time
                    })
                
                logger.log(
                    self.log_level,
                    f"Function '{name}' executed in {execution_time:.4f} seconds"
                )
                
                return result
            return wrapper
        return decorator
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        获取性能指标
        
        Returns:
            Dict[str, Any]: 性能指标
        """
        with self.lock:
            return self.metrics.copy()
    
    def clear_metrics(self):
        """清除性能指标"""
        with self.lock:
            self.metrics = {}
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取性能指标摘要
        
        Returns:
            Dict[str, Any]: 性能指标摘要
        """
        with self.lock:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'function_times': {}
            }
            
            # 计算函数执行时间统计
            if 'function_times' in self.metrics:
                for func_name, times in self.metrics['function_times'].items():
                    execution_times = [t['execution_time'] for t in times]
                    if execution_times:
                        summary['function_times'][func_name] = {
                            'count': len(execution_times),
                            'min': min(execution_times),
                            'max': max(execution_times),
                            'avg': sum(execution_times) / len(execution_times)
                        }
            
            # 计算系统资源统计
            if 'system_resources' in self.metrics:
                resources = self.metrics['system_resources']
                if resources:
                    summary['system_resources'] = {
                        'samples': len(resources),
                        'cpu_percent': {
                            'min': min(r['cpu_percent'] for r in resources),
                            'max': max(r['cpu_percent'] for r in resources),
                            'avg': sum(r['cpu_percent'] for r in resources) / len(resources)
                        },
                        'memory_percent': {
                            'min': min(r['memory_percent'] for r in resources),
                            'max': max(r['memory_percent'] for r in resources),
                            'avg': sum(r['memory_percent'] for r in resources) / len(resources)
                        }
                    }
            
            return summary
```

在 `DataManager` 中集成性能监控功能：

```python
def __init__(self, config: Dict[str, Any]):
    # ... 其他初始化代码 ...
    
    # 初始化性能监控器
    self.performance_monitor = PerformanceMonitor()
    
    # 如果配置中启用了性能监控，则开始监控
    if config.get('enable_performance_monitoring', False):
        self.performance_monitor.start_monitoring()

# 使用装饰器监控关键方法的性能
@performance_monitor.time_function()
def load_data(self, data_type: str, start_date: str, end_date: str, symbols: Optional[List[str]] = None, **kwargs) -> DataModel:
    # 原有的load_data实现...
    pass

@performance_monitor.time_function()
def merge_data(self, data_types: List[str], start_date: str, end_date: str, symbols: Optional[List[str]] = None, **kwargs) -> DataModel:
    # 原有的merge_data实现...
    pass

def get_performance_metrics(self) -> Dict[str, Any]:
    """
    获取性能指标
    
    Returns:
        Dict[str, Any]: 性能指标
    """
    return self.performance_monitor.get_metrics()

def get_performance_summary(self) -> Dict[str, Any]:
    """
    获取性能指标摘要
    
    Returns:
        Dict[str, Any]: 性能指标摘要
    """
    return self.performance_monitor.get_summary()
```

#### 3.2 异常告警

建议实现一个 `AlertManager` 类，用于管理异常告警：

```python
from typing import Dict, List, Any, Optional, Callable, Union
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import requests
from datetime import datetime
import threading
import time

logger = logging.getLogger(__name__)

class AlertManager:
    """告警管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化告警管理器
        
        Args:
            config: 配置信息，包含告警阈值、通知#