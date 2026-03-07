# 数据层API文档

## 概述

数据层（src/data）提供了统一的数据管理接口，包括数据加载、验证、缓存、处理等核心功能。采用分层架构设计，确保接口清晰、职责明确、易于扩展。

## 架构分层

### 1. 接口定义层
提供标准化的接口定义，确保各组件间的解耦和可替换性。

### 2. 核心实现层  
提供基础的数据管理、加载、验证和注册功能。

### 3. 缓存系统
提供多级缓存支持，优化数据访问性能。

### 4. 数据处理层
提供数据清洗、转换和标准化功能。

### 5. 数据验证层
提供专业的数据质量验证和一致性检查。

### 6. 监控与质量层
提供性能监控和数据质量评估功能。

### 7. 数据加载器层
提供各种数据源的加载器实现。

## 核心接口

### IDataModel

数据模型接口，定义了所有数据模型必须实现的方法。

```python
from src.data import IDataModel

class DataModel(IDataModel):
    def validate(self) -> bool:
        """数据有效性验证"""
        pass
    
    def get_frequency(self) -> str:
        """获取数据频率"""
        pass
    
    def get_metadata(self, user_only: bool = False) -> Dict[str, Any]:
        """获取元数据信息"""
        pass
```

### IDataLoader

数据加载器接口，定义了所有数据加载器必须实现的方法。

```python
from src.data import IDataLoader

class BaseDataLoader(IDataLoader):
    def load(self, start_date: str, end_date: str, frequency: str, **kwargs) -> IDataModel:
        """统一的数据加载接口"""
        pass
    
    def get_required_config_fields(self) -> List[str]:
        """获取必需的配置字段列表"""
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """验证配置有效性"""
        pass
```

### IDataValidator

数据验证器接口，定义了所有数据验证器必须实现的方法。

```python
from src.data import IDataValidator

class DataValidator(IDataValidator):
    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """验证数据的基本有效性"""
        pass
    
    def validate_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """验证数据质量"""
        pass
    
    def validate_data_model(self, model: IDataModel) -> Dict[str, Any]:
        """验证数据模型"""
        pass
```

### IDataCache

数据缓存接口，定义了所有数据缓存必须实现的方法。

```python
from src.data import IDataCache

class CacheManager(IDataCache):
    def get(self, key: str) -> Optional[Any]:
        """获取缓存数据"""
        pass
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存数据"""
        pass
    
    def delete(self, key: str) -> bool:
        """删除缓存数据"""
        pass
    
    def clear(self) -> bool:
        """清空所有缓存"""
        pass
```

### DiskCache

磁盘缓存实现，提供持久化缓存功能。

```python
from src.data.cache import DiskCache, DiskCacheConfig

# 配置磁盘缓存
config = DiskCacheConfig(
    cache_dir="cache",
    max_file_size=10 * 1024 * 1024,  # 10MB
    compression=False,
    encryption=False,
    backup_enabled=False,
    cleanup_interval=300  # 5分钟清理一次
)

# 初始化磁盘缓存
disk_cache = DiskCache(config)

# 设置缓存
success = disk_cache.set("key1", {"data": "value"}, ttl=3600)

# 获取缓存
data = disk_cache.get("key1")

# 检查缓存是否存在
exists = disk_cache.exists("key1")

# 删除缓存
deleted = disk_cache.delete("key1")

# 清空所有缓存
cleared = disk_cache.clear()

# 获取统计信息
stats = disk_cache.get_stats()

# 健康检查
health = disk_cache.health_check()

# 关闭缓存
disk_cache.close()
```

## 核心实现

### DataManager

数据管理器，负责协调数据加载、验证和缓存。

```python
from src.data import DataManager

# 初始化数据管理器
manager = DataManager(config_dict={
    'Stock': {
        'save_path': 'data/stock',
        'max_retries': 3,
        'cache_days': 30
    }
})

# 加载数据
data_model = manager.load_data(
    data_type='stock',
    start_date='2023-01-01',
    end_date='2023-01-31',
    frequency='1d',
    symbols=['000001', '000002']
)

# 加载多源数据
multi_data = manager.load_multi_source(
    stock_symbols=['000001', '000002'],
    index_symbols=['000300'],
    start='2023-01-01',
    end='2023-01-31'
)

# 获取缓存统计
stats = manager.get_cache_stats()

# 清理过期缓存
cleaned_count = manager.clean_expired_cache()

# 关闭数据管理器
manager.shutdown()
```

### DataModel

数据模型，用于封装数据和元数据。

```python
from src.data import DataModel
import pandas as pd

# 创建数据模型
data = pd.DataFrame({
    'open': [100, 101, 102],
    'high': [105, 106, 107],
    'low': [95, 96, 97],
    'close': [103, 104, 105],
    'volume': [1000, 1100, 1200]
})

metadata = {
    'source': 'akshare',
    'symbol': '000001',
    'frequency': '1d'
}

model = DataModel(data, '1d', metadata)

# 验证数据
is_valid = model.validate()

# 获取频率
frequency = model.get_frequency()

# 获取元数据
user_metadata = model.get_metadata(user_only=True)
full_metadata = model.get_metadata()
```

### DataValidator

数据验证器，负责验证数据模型的有效性。

```python
from src.data import DataValidator
import pandas as pd

validator = DataValidator()

# 验证数据
result = validator.validate_data(data)
if result['is_valid']:
    print("数据有效")
else:
    print("数据无效:", result['errors'])

# 验证数据质量
quality_report = validator.validate_quality(data)
print(f"质量评分: {quality_report['score']}")
print(f"问题: {quality_report['issues']}")

# 验证数据模型
model_result = validator.validate_data_model(model)

# 验证日期范围
is_valid_range = validator.validate_date_range(
    data, 'date', '2023-01-01', '2023-01-31'
)

# 验证数值列
is_valid_numeric = validator.validate_numeric_columns(
    data, ['open', 'high', 'low', 'close']
)

# 验证无缺失值
no_missing = validator.validate_no_missing_values(data)

# 验证无重复值
no_duplicates = validator.validate_no_duplicates(data)

# 验证异常值
outlier_report = validator.validate_outliers(data, 'close', 'iqr')
print(f"异常值数量: {outlier_report.outlier_count}")

# 验证数据一致性
consistency_report = validator.validate_data_consistency(data)
if consistency_report.is_consistent:
    print("数据一致")
else:
    print("数据不一致:", consistency_report.inconsistencies)

# 添加自定义验证规则
def custom_rule(data):
    if len(data) > 10000:
        raise ValueError("数据行数过多")

validator.add_custom_rule(custom_rule)
```

### CacheManager

缓存管理器，负责数据缓存的管理和调度。

```python
from src.data import CacheManager
from src.data.cache import CacheConfig

# 配置缓存
config = CacheConfig(
    max_size=1000,
    enable_disk_cache=True,
    disk_cache_dir="cache",
    ttl=3600
)

cache = CacheManager(config)

# 设置缓存
cache.set('key1', {'data': 'value'}, ttl=1800)

# 获取缓存
data = cache.get('key1')

# 检查缓存是否存在
exists = cache.exists('key1')

# 删除缓存
cache.delete('key1')

# 清空所有缓存
cache.clear()

# 获取缓存统计
stats = cache.get_stats()
print(f"内存缓存大小: {stats['memory_cache']['size']}")
print(f"磁盘缓存文件数: {stats['disk_cache']['file_count']}")

# 清理过期缓存
cleaned_count = cache.clean_expired()
```

### DataRegistry

数据注册器，负责管理所有数据加载器实例。

```python
from src.data import DataRegistry, BaseDataLoader

registry = DataRegistry()

# 注册加载器实例
loader = BaseDataLoader(config)
registry.register('my_loader', loader)

# 注册加载器类
registry.register_class('my_loader_class', BaseDataLoader)

# 获取加载器
loader = registry.get_loader('my_loader')

# 创建加载器实例
loader = registry.create_loader('my_loader_class', config)

# 列出已注册的加载器
loaders = registry.list_registered_loaders()
loader_classes = registry.list_registered_loader_classes()

# 检查是否已注册
is_registered = registry.is_registered('my_loader')
```

## 处理模块

### DataProcessor

数据处理器，负责数据的清洗、转换和标准化。

```python
from src.data import DataProcessor, DataModel

processor = DataProcessor()

# 处理数据
processed_model = processor.process(
    data_model,
    fill_method='forward',
    outlier_method='iqr',
    normalize_method='minmax'
)

# 获取处理信息
info = processor.get_processing_info()
print(f"处理步骤数: {len(info['steps'])}")

# 获取处理统计
stats = processor.get_processing_stats()
print(f"处理时间: {stats['processing_time_seconds']}秒")
```

### UnifiedDataProcessor

统一数据处理器，提供更高级的数据处理功能。

```python
from src.data import UnifiedDataProcessor

processor = UnifiedDataProcessor()

# 处理数据
processed_model = processor.process(
    data_model,
    fill_method='interpolate',
    outlier_method='zscore',
    normalize_method='robust',
    index_col='date',
    time_col='timestamp'
)

# 获取处理统计
stats = processor.get_processing_stats()
```

## 验证模块

### ChinaStockValidator

中国股票数据验证器，专门针对A股数据进行验证。

```python
from src.data import ChinaStockValidator, DataType, ValidationRule

validator = ChinaStockValidator()

# 验证融资融券数据
margin_data = pd.DataFrame({
    'date': ['2023-01-01', '2023-01-02'],
    'symbol': ['000001', '000001'],
    'margin_balance': [1000000, 1100000],
    'short_balance': [500000, 550000]
})

result = validator.validate(margin_data, DataType.MARGIN_TRADING)
if result['is_valid']:
    print("融资融券数据有效")
else:
    print("融资融券数据无效:", result['errors'])

# 验证龙虎榜数据
dragon_data = pd.DataFrame({
    'symbol': ['000001'],
    'buy_seats': ['机构专用'],
    'sell_seats': ['机构专用']
})

result = validator.validate(dragon_data, DataType.DRAGON_BOARD)

# 验证Level2数据
level2_data = pd.DataFrame({
    'symbol': ['000001'],
    'price': [10.5],
    'bids': [[10.4, 10.3]],
    'asks': [[10.6, 10.7]]
})

result = validator.validate(level2_data, DataType.LEVEL2)
```

## 监控与质量模块

### PerformanceMonitor

性能监控器，监控数据加载和处理性能。

```python
from src.data import PerformanceMonitor, PerformanceMetric, PerformanceAlert

monitor = PerformanceMonitor()

# 监控数据加载性能
with monitor.track_operation('data_loading'):
    data_model = manager.load_data('stock', '2023-01-01', '2023-01-31', '1d')

# 获取性能指标
metrics = monitor.get_metrics()
print(f"平均加载时间: {metrics['avg_loading_time']}ms")
print(f"成功率: {metrics['success_rate']}%")

# 设置告警
monitor.set_alert(PerformanceAlert(
    metric='loading_time',
    threshold=5000,  # 5秒
    condition='gt'
))
```

### DataQualityMonitor

数据质量监控器，监控数据质量指标。

```python
from src.data import DataQualityMonitor, QualityMetric, QualityReport, QualityLevel

quality_monitor = DataQualityMonitor()

# 监控数据质量
quality_report = quality_monitor.monitor_data_quality(data_model)

# 获取质量指标
metrics = quality_monitor.get_quality_metrics()
print(f"完整性: {metrics['completeness']}")
print(f"准确性: {metrics['accuracy']}")
print(f"一致性: {metrics['consistency']}")

# 设置质量告警
quality_monitor.set_quality_alert(QualityAlert(
    dimension=QualityDimension.COMPLETENESS,
    threshold=0.95,
    level=QualityLevel.WARNING
))
```

## 数据加载器

### CryptoDataLoader

加密货币数据加载器。

```python
from src.data import CryptoDataLoader

loader = CryptoDataLoader(config={
    'api_key': 'your_api_key',
    'base_url': 'https://api.coingecko.com'
})

# 加载比特币数据
btc_data = loader.load(
    start_date='2023-01-01',
    end_date='2023-01-31',
    frequency='1d',
    symbol='bitcoin'
)
```

### CoinGeckoLoader

CoinGecko数据加载器。

```python
from src.data import CoinGeckoLoader

loader = CoinGeckoLoader(config={
    'api_key': 'your_api_key',
    'rate_limit': 50  # 每分钟请求数
})

# 加载以太坊数据
eth_data = loader.load(
    start_date='2023-01-01',
    end_date='2023-01-31',
    frequency='1d',
    symbol='ethereum'
)
```

### BinanceLoader

币安数据加载器。

```python
from src.data import BinanceLoader

loader = BinanceLoader(config={
    'api_key': 'your_api_key',
    'secret_key': 'your_secret_key'
})

# 加载BTC/USDT数据
btc_usdt_data = loader.load(
    start_date='2023-01-01',
    end_date='2023-01-31',
    frequency='1h',
    symbol='BTCUSDT'
)
```

### MacroDataLoader

宏观经济数据加载器。

```python
from src.data import MacroDataLoader, FREDLoader, WorldBankLoader

# FRED数据加载器
fred_loader = FREDLoader(config={
    'api_key': 'your_fred_api_key'
})

# 加载GDP数据
gdp_data = fred_loader.load(
    start_date='2020-01-01',
    end_date='2023-01-01',
    frequency='1q',
    series_id='GDP'
)

# 世界银行数据加载器
wb_loader = WorldBankLoader()

# 加载人口数据
population_data = wb_loader.load(
    start_date='2020-01-01',
    end_date='2023-01-01',
    frequency='1y',
    country_code='CHN',
    indicator='SP.POP.TOTL'
)
```

## 典型用法与集成建议

### 1. 基础数据流程

```python
from src.data import DataManager, DataValidator, CacheManager

# 初始化组件
manager = DataManager()
validator = DataValidator()
cache = CacheManager()

# 加载数据
data_model = manager.load_data(
    'stock', '2023-01-01', '2023-01-31', '1d',
    symbols=['000001']
)

# 验证数据
validation_result = validator.validate_data_model(data_model)
if validation_result['is_valid']:
    print("数据加载成功")
    
    # 缓存数据
    cache.set('stock_000001_20230101_20230131', data_model)
else:
    print("数据验证失败:", validation_result['errors'])
```

### 2. 高级数据处理流程

```python
from src.data import DataManager, UnifiedDataProcessor, ChinaStockValidator

# 初始化
manager = DataManager()
processor = UnifiedDataProcessor()
validator = ChinaStockValidator()

# 加载多源数据
multi_data = manager.load_multi_source(
    stock_symbols=['000001', '000002'],
    index_symbols=['000300'],
    start='2023-01-01',
    end='2023-01-31'
)

# 处理数据
for data_type, data_model in multi_data.items():
    # 验证数据
    validation_result = validator.validate_data_model(data_model)
    
    if validation_result['is_valid']:
        # 处理数据
        processed_model = processor.process(
            data_model,
            fill_method='forward',
            outlier_method='iqr',
            normalize_method='minmax'
        )
        
        print(f"{data_type} 数据处理完成")
    else:
        print(f"{data_type} 数据验证失败")
```

### 3. 性能优化集成

```python
from src.data import DataManager, PerformanceMonitor, DataQualityMonitor

# 初始化监控
perf_monitor = PerformanceMonitor()
quality_monitor = DataQualityMonitor()

# 监控数据加载性能
with perf_monitor.track_operation('data_loading'):
    data_model = manager.load_data('stock', '2023-01-01', '2023-01-31', '1d')

# 监控数据质量
quality_report = quality_monitor.monitor_data_quality(data_model)

# 获取监控指标
perf_metrics = perf_monitor.get_metrics()
quality_metrics = quality_monitor.get_quality_metrics()

print(f"加载时间: {perf_metrics['avg_loading_time']}ms")
print(f"数据质量评分: {quality_metrics['overall_score']}")
```

### 4. 缓存优化集成

```python
from src.data import CacheManager, CacheConfig

# 配置多级缓存
config = CacheConfig(
    memory_max_size=2000,
    memory_ttl=300,
    disk_enabled=True,
    disk_cache_dir="cache",
    disk_ttl=3600,
    disk_max_size_mb=2048
)

cache = CacheManager(config)

# 批量缓存操作
cache_keys = [
    'stock_000001_20230101_20230131',
    'stock_000002_20230101_20230131',
    'index_000300_20230101_20230131'
]

for key in cache_keys:
    if not cache.exists(key):
        data_model = manager.load_data('stock', '2023-01-01', '2023-01-31', '1d')
        cache.set(key, data_model, ttl=3600)

# 获取缓存统计
stats = cache.get_stats()
print(f"内存缓存命中率: {stats['memory_cache']['hit_rate']}")
print(f"磁盘缓存文件数: {stats['disk_cache']['file_count']}")
```

## 配置说明

### 数据管理器配置

```ini
[Stock]
save_path = data/stock
max_retries = 3
cache_days = 30

[Index]
save_path = data/index
max_retries = 3
cache_days = 30

[News]
save_path = data/news
max_retries = 3
cache_days = 7

[Financial]
save_path = data/financial
max_retries = 3
cache_days = 30
```

### 缓存配置

```python
from src.data.cache import CacheConfig

config = CacheConfig(
    memory_max_size=1000,      # 内存缓存最大条目数
    memory_ttl=300,            # 内存缓存TTL（秒）
    disk_enabled=True,         # 是否启用磁盘缓存
    disk_cache_dir="cache",    # 磁盘缓存目录
    disk_ttl=3600,            # 磁盘缓存TTL（秒）
    disk_max_size_mb=1024     # 磁盘缓存最大大小（MB）
)
```

## 错误处理

### 常见异常

```python
from src.data import DataLoaderError, ValidationError

try:
    data_model = manager.load_data('stock', '2023-01-01', '2023-01-31', '1d')
except DataLoaderError as e:
    print(f"数据加载失败: {e}")
except ValidationError as e:
    print(f"数据验证失败: {e}")
except Exception as e:
    print(f"未知错误: {e}")
```

### 错误恢复

```python
# 重试机制
max_retries = 3
for attempt in range(max_retries):
    try:
        data_model = manager.load_data('stock', '2023-01-01', '2023-01-31', '1d')
        break
    except Exception as e:
        if attempt == max_retries - 1:
            raise e
        print(f"第{attempt + 1}次尝试失败，重试...")
        time.sleep(1)
```

## 性能优化

### 缓存优化

```python
# 使用多级缓存
from src.data.cache import MultiLevelCache

cache = MultiLevelCache(CacheConfig(
    memory_max_size=2000,
    disk_enabled=True,
    disk_max_size_mb=2048
))

# 批量操作
cache.set('key1', data1)
cache.set('key2', data2)
cache.set('key3', data3)

# 获取统计
stats = cache.get_stats()
print(f"命中率: {stats['performance']['hit_rate']}")
```

### 并行处理优化

```python
# 使用优化的并行加载器
from src.data.loader import OptimizedParallelLoader

loader = OptimizedParallelLoader(
    max_workers=16,
    timeout=60,
    max_retries=3
)

# 批量加载
tasks = [
    ('000001', {'symbol': '000001', 'start_date': '2023-01-01', 'end_date': '2023-01-31'}),
    ('000002', {'symbol': '000002', 'start_date': '2023-01-01', 'end_date': '2023-01-31'}),
    ('000300', {'symbol': '000300', 'start_date': '2023-01-01', 'end_date': '2023-01-31'})
]

results = loader.batch_load(tasks, priority=True)

# 获取性能统计
stats = loader.get_stats()
print(f"成功率: {stats['success_rate']}")
print(f"平均时间: {stats['avg_time_ms']}ms")
```

## 最佳实践

### 1. 接口编程
始终使用接口而不是具体实现，确保代码的可扩展性和可测试性。

```python
# 推荐
from src.data import IDataLoader, DataManager
loader: IDataLoader = manager.get_loader('stock')

# 不推荐
from src.data.loader import StockDataLoader
loader = StockDataLoader()
```

### 2. 错误处理
正确处理所有可能的异常，提供有意义的错误信息。

```python
try:
    data_model = manager.load_data('stock', start_date, end_date, frequency)
except DataLoaderError as e:
    logger.error(f"数据加载失败: {e}")
    # 实现降级策略
    data_model = load_from_cache_or_fallback()
except ValidationError as e:
    logger.warning(f"数据验证失败: {e}")
    # 实现数据修复策略
    data_model = repair_data(data_model)
```

### 3. 资源管理
及时关闭和清理资源，避免内存泄漏。

```python
# 使用上下文管理器
with DataManager() as manager:
    data_model = manager.load_data('stock', start_date, end_date, frequency)
    # 自动清理资源

# 或手动清理
manager = DataManager()
try:
    data_model = manager.load_data('stock', start_date, end_date, frequency)
finally:
    manager.shutdown()
```

### 4. 性能监控
定期检查缓存命中率和处理时间，优化性能瓶颈。

```python
# 监控关键指标
perf_monitor = PerformanceMonitor()
quality_monitor = DataQualityMonitor()

# 定期检查性能
def check_performance():
    perf_metrics = perf_monitor.get_metrics()
    quality_metrics = quality_monitor.get_quality_metrics()
    
    if perf_metrics['avg_loading_time'] > 5000:
        logger.warning("数据加载时间过长")
    
    if quality_metrics['overall_score'] < 0.8:
        logger.warning("数据质量下降")
```

### 5. 数据验证
在关键节点验证数据质量，确保数据可靠性。

```python
# 加载后立即验证
data_model = manager.load_data('stock', start_date, end_date, frequency)
validation_result = validator.validate_data_model(data_model)

if not validation_result['is_valid']:
    logger.error(f"数据验证失败: {validation_result['errors']}")
    # 实现数据修复或告警策略
```

### 6. 配置管理
使用配置文件管理参数，支持不同环境的配置。

```python
# 使用配置文件
import configparser

config = configparser.ConfigParser()
config.read('data_config.ini')

manager = DataManager(config_dict=dict(config))
```

### 7. 日志记录
记录关键操作和错误信息，便于问题排查。

```python
import logging

logger = logging.getLogger(__name__)

def load_stock_data(symbol, start_date, end_date):
    logger.info(f"开始加载股票数据: {symbol}")
    try:
        data_model = manager.load_data('stock', start_date, end_date, '1d', symbols=[symbol])
        logger.info(f"股票数据加载成功: {symbol}")
        return data_model
    except Exception as e:
        logger.error(f"股票数据加载失败: {symbol}, 错误: {e}")
        raise
```

## 集成建议

### 1. 与基础设施层集成
数据层与基础设施层的缓存、监控、日志系统深度集成。

```python
from src.infrastructure import UnifiedConfigManager, ICacheManager, AutomationMonitor

# 使用统一配置管理
config_manager = UnifiedConfigManager()
data_config = config_manager.get('data')

# 使用统一缓存接口
cache: ICacheManager = config_manager.get_cache_manager()

# 使用统一监控
monitor = AutomationMonitor()
monitor.track_metric('data_loading_time', loading_time_ms)
```

### 2. 与特征层集成
数据层为特征层提供高质量的数据源。

```python
from src.data import DataManager
from src.features import FeatureProcessor

# 数据层提供原始数据
data_model = manager.load_data('stock', start_date, end_date, frequency)

# 特征层处理数据
feature_processor = FeatureProcessor()
features = feature_processor.extract_features(data_model.data)
```

### 3. 与模型层集成
数据层为模型层提供训练和预测所需的数据。

```python
from src.data import DataManager, UnifiedDataProcessor
from src.models import ModelTrainer

# 数据层提供处理后的数据
data_model = manager.load_data('stock', start_date, end_date, frequency)
processor = UnifiedDataProcessor()
processed_data = processor.process(data_model)

# 模型层使用数据
trainer = ModelTrainer()
model = trainer.train(processed_data.data, processed_data.metadata)
```

## 总结

数据层作为RQA系统的核心组件，提供了完整的数据管理解决方案。通过分层架构设计，确保了接口的清晰性和实现的灵活性。通过缓存优化、并行处理、质量监控等功能，为上层应用提供了高性能、高质量的数据服务。

建议在实际使用中：
1. 根据具体需求选择合适的缓存策略
2. 定期监控数据质量和性能指标
3. 及时处理异常和错误情况
4. 遵循最佳实践，确保代码的可维护性和可扩展性

## 新增功能模块

### 数据质量自动修复

数据层新增了`DataRepairer`组件，提供自动化的数据质量问题修复功能。

```python
from src.data.repair import DataRepairer, RepairConfig, RepairStrategy

# 配置修复策略
config = RepairConfig(
    null_strategy=RepairStrategy.FILL_FORWARD,
    outlier_strategy=RepairStrategy.REMOVE_OUTLIERS,
    duplicate_strategy=RepairStrategy.DROP,
    time_series_enabled=True
)

# 初始化修复器
repairer = DataRepairer(config)

# 修复数据质量问题
repaired_data, repair_result = repairer.repair_data(data, "stock")

# 检查修复结果
if repair_result.success:
    print(f"修复成功，原始形状: {repair_result.original_shape}")
    print(f"修复后形状: {repair_result.repaired_shape}")
    print(f"修复操作: {repair_result.operations}")
```

**主要功能**：
- 空值处理：支持前向填充、后向填充、插值等方法
- 异常值处理：支持移除、替换、截断等方法
- 重复值处理：支持删除、保留等方法
- 时间序列处理：支持时间序列特定的修复策略

### 数据版本管理

数据层新增了`DataVersionManager`组件，提供完整的数据版本管理功能。

```python
from src.data.version_control import DataVersionManager

# 初始化版本管理器
version_manager = DataVersionManager("./versions")

# 创建版本
version_id = version_manager.create_version(data_model, "v1.0")

# 获取版本
retrieved_model = version_manager.get_version(version_id)

# 列出版本
versions = version_manager.list_versions()

# 比较版本
comparison = version_manager.compare_versions(version1, version2)

# 回滚版本
rolled_back = version_manager.rollback_to_version(version1)

# 获取血缘关系
lineage = version_manager.get_lineage(version_id)
```

**主要功能**：
- 版本创建：支持创建带标签的版本
- 版本比较：支持数据内容和元数据比较
- 版本回滚：支持回滚到指定版本
- 血缘追踪：支持版本血缘关系追踪
- 元数据管理：支持版本元数据的更新和查询

### 数据湖架构支持

数据层新增了数据湖架构支持，包括`DataLakeManager`、`PartitionManager`、`MetadataManager`等组件。

```python
from src.data.lake import DataLakeManager, PartitionManager, MetadataManager

# 初始化数据湖管理器
lake_manager = DataLakeManager("./data_lake")

# 存储数据
lake_manager.store_data("stock_data", data, format="parquet")

# 查询数据
result = lake_manager.query_data("stock_data", filters={"symbol": "000001.SZ"})

# 管理分区
partition_manager = PartitionManager("./data_lake")
partition_manager.create_partition("stock_data", "date=2024-01-01")

# 管理元数据
metadata_manager = MetadataManager("./data_lake")
metadata_manager.update_metadata("stock_data", {"description": "股票数据"})
```

**主要功能**：
- 多格式存储：支持Parquet、CSV、JSON等格式
- 分区管理：支持按时间、字段等维度分区
- 元数据管理：支持数据集的元数据管理
- 查询优化：支持分区裁剪和谓词下推

### 智能缓存策略

数据层新增了智能缓存策略，支持多种缓存淘汰算法。

```python
from src.data.cache import ICacheStrategy, LFUStrategy, LRUStrategy

# 使用LFU策略
lfu_cache = LFUStrategy(max_size=1000)

# 使用LRU策略
lru_cache = LRUStrategy(max_size=1000)

# 自定义缓存策略
class CustomStrategy(ICacheStrategy):
    def get(self, key: str) -> Any:
        # 自定义获取逻辑
        pass
    
    def put(self, key: str, value: Any) -> None:
        # 自定义存储逻辑
        pass
    
    def evict(self) -> None:
        # 自定义淘汰逻辑
        pass
```

**主要功能**：
- LFU策略：最少使用频率淘汰
- LRU策略：最近最少使用淘汰
- 自定义策略：支持自定义淘汰算法
- 性能监控：支持缓存命中率和性能统计

### 分布式数据加载

数据层新增了`MultiprocessDataLoader`组件，支持多进程分布式数据加载。

```python
from src.data.loader import MultiprocessDataLoader

# 初始化分布式加载器
loader = MultiprocessDataLoader(
    base_loader=MyDataLoader(),
    num_processes=4,
    chunk_size=1000
)

# 分布式加载数据
data = loader.load("2024-01-01", "2024-01-31", frequency="1d")
```

**主要功能**：
- 多进程并行：支持多进程并行数据加载
- 任务分发：支持智能的任务分发策略
- 结果聚合：支持分布式结果的聚合
- 错误处理：支持分布式环境下的错误处理

### 实时数据流处理

数据层新增了实时数据流处理功能，包括`InMemoryStream`和`SimpleStreamProcessor`。

```python
from src.data.stream import InMemoryStream, SimpleStreamProcessor

# 创建内存流
stream = InMemoryStream()

# 创建流处理器
processor = SimpleStreamProcessor()

# 注册处理器
stream.register_processor("stock_data", processor)

# 发送数据
stream.send("stock_data", {"symbol": "000001.SZ", "price": 100.0})

# 处理数据
def handle_data(data):
    print(f"处理数据: {data}")

processor.register_handler(handle_data)
```

**主要功能**：
- 内存流：支持高性能的内存数据流
- 流处理器：支持可扩展的流处理管道
- 事件驱动：支持事件驱动的数据处理
- 实时监控：支持实时数据监控和告警

### 机器学习质量评估

数据层新增了`MLQualityAssessor`组件，支持机器学习驱动的数据质量评估。

```python
from src.data.quality import MLQualityAssessor

# 初始化质量评估器
assessor = MLQualityAssessor()

# 评估数据质量
quality_report = assessor.assess_quality(data)

# 检测异常
anomalies = assessor.detect_anomalies(data)

# 生成建议
suggestions = assessor.generate_suggestions(data)
```

**主要功能**：
- 异常检测：支持基于机器学习的异常检测
- 数据完整性评估：支持数据完整性评估
- 智能建议：支持基于ML的数据质量改进建议
- 模型管理：支持ML模型的版本管理和更新 