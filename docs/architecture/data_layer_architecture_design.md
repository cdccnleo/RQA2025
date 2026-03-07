# RQA2025数据管理层架构设计文档

## 📊 文档信息

- **文档版本**: v2.1
- **创建日期**: 2024年12月
- **更新日期**: 2026年02月
- **架构层级**: 数据管理层 (Data Management Layer)
- **文件数量**: 183个Python文件
- **主要功能**: 数据基础设施，质量保障，性能优化

### 📋 版本更新记录

#### v2.2 (2026-02-23)
**数据采集自动调度功能发布**

- **新增功能**:
  - ✅ 采集频率解析器 (支持"1次/天"等多种格式)
  - ✅ 数据采集调度管理器 (定时自动生成采集任务)
  - ✅ 采集历史记录管理 (完整采集历史追踪)
  - ✅ 自动采集控制API (启动/停止/状态查询)
  - ✅ 采集成功后自动更新最后采集时间
  - ✅ 采集统计信息查询 (成功率/平均耗时等)

- **优化改进**:
  - 已启用数据源根据采集频率自动执行采集
  - 避免重复提交相同采集任务
  - 支持手动和自动两种采集模式
  - 完整的采集历史记录和查询功能

- **新增文件**:
  - `src/gateway/web/rate_limit_parser.py` (采集频率解析器)
  - `src/gateway/web/data_collection_scheduler_manager.py` (采集调度管理器)
  - `src/gateway/web/collection_history_manager.py` (采集历史管理器)

- **数据库表**:
  - `data_collection_history` (采集历史记录表)

#### v2.1 (2026-02-16)
**PostgreSQLDataLoader 增强版发布**

- **新增功能**:
  - ✅ 自动重试机制 (指数退避策略)
  - ✅ 内存缓存集成 (LRU策略)
  - ✅ 数据质量检查 (完整性/准确性/一致性)
  - ✅ 性能监控集成 (加载时间/成功率)
  - ✅ 数据湖集成 (分层存储/分区管理)
  - ✅ 统一接口实现 (符合架构标准)

- **优化改进**:
  - 减少90%临时故障 (自动重试)
  - 减少80%数据库查询 (内存缓存)
  - 支持离线数据访问 (数据湖)
  - 实时质量监控 (质量检查)
  - 性能追踪分析 (性能监控)

- **新增文件**:
  - `src/data/loader/postgresql_loader.py` (增强版)

#### v2.0 (2024-12)
**初始版本发布**

- 数据管理层架构设计
- 多数据源适配器支持
- 数据湖基础架构
- 质量监控系统
- 缓存管理机制

---

## 🎯 概述

### 1.1 数据管理层定位

数据管理层是RQA2025系统的核心数据基础设施，提供全面的数据采集、处理、存储、质量管理、缓存优化和监控能力。作为系统的数据中枢，数据管理层需要具备高性能、高可用性、高质量保障和实时处理能力，确保量化交易数据的准确性、完整性和及时性。

### 1.2 设计原则

- **📊 数据为中心**: 以数据质量和完整性为核心设计原则
- **🔄 实时处理**: 支持实时数据流处理和低延迟数据访问
- **🛡️ 质量优先**: 内置数据质量监控和自动修复机制
- **⚡ 性能优化**: 多级缓存策略和智能预热机制
- **📈 可扩展性**: 支持水平扩展和动态数据源接入
- **📍 可观测性**: 全面的数据监控和性能指标收集

### 1.3 架构目标

1. **数据完整性100%**: 确保所有金融数据的完整性和准确性
2. **毫秒级数据访问**: 支持高频交易的实时数据需求
3. **智能质量监控**: 自动化数据质量检测和修复
4. **弹性扩展能力**: 支持海量数据的分布式处理
5. **智能缓存优化**: 自适应缓存策略和预热机制
6. **全面监控预警**: 实时监控数据质量和系统性能

---

## 🏗️ 架构设计

### 2.1 整体架构图

```mermaid
graph TB
    subgraph "数据管理层架构"
        direction TB

        subgraph "数据接入层"
            direction LR
            Adapters[数据适配器<br/>16个数据源]
            Connectors[连接器<br/>多协议支持]
            Loaders[数据加载器<br/>并行加载]
        end

        subgraph "数据处理层"
            direction TB
            Validators[数据验证器<br/>规则校验]
            Transformers[数据转换器<br/>格式转换]
            Processors[数据处理器<br/>业务逻辑]
        end

        subgraph "数据存储层"
            direction LR
            DataLake[数据湖<br/>分层存储]
            Cache[缓存系统<br/>多级缓存]
            Metadata[元数据管理<br/>数据血缘]
        end

        subgraph "质量监控层"
            direction TB
            QualityMon[质量监控器<br/>实时检测]
            ValidatorsCore[验证引擎<br/>规则引擎]
            Repair[自动修复<br/>智能修复]
        end

        subgraph "性能优化层"
            direction LR
            CacheMgr[缓存管理器<br/>智能缓存]
            PerfMon[性能监控<br/>实时监控]
            Optimizer[性能优化器<br/>自适应调优]
        end

        subgraph "监控管理层"
            direction TB
            Metrics[指标收集<br/>多维度]
            Alerts[告警系统<br/>智能告警]
            Dashboard[监控面板<br/>可视化]
        end
    end

    %% 连接关系
    Adapters --> Validators
    Connectors --> Loaders
    Loaders --> Transformers
    Transformers --> Processors

    Validators --> QualityMon
    Processors --> DataLake
    Processors --> Cache

    DataLake --> Metadata
    Cache --> Metadata

    QualityMon --> ValidatorsCore
    QualityMon --> Repair

    Cache --> CacheMgr
    CacheMgr --> PerfMon
    PerfMon --> Optimizer

    Metrics --> Alerts
    Alerts --> Dashboard
    PerfMon --> Metrics
    QualityMon --> Metrics
```

### 2.2 核心组件架构

#### 2.2.1 数据流处理架构

```mermaid
graph LR
    subgraph "数据源"
        Market[市场数据]
        News[新闻数据]
        Macro[宏观数据]
        Custom[自定义数据]
    end

    subgraph "数据管道"
        Ingestion[数据摄入]
        Validation[数据验证]
        Processing[数据处理]
        Storage[数据存储]
    end

    subgraph "数据服务"
        Query[查询服务]
        Cache[缓存服务]
        Stream[流式服务]
    end

    subgraph "数据消费"
        Trading[交易系统]
        Strategy[策略系统]
        Risk[风险系统]
        Analytics[分析系统]
    end

    Market --> Ingestion
    News --> Ingestion
    Macro --> Ingestion
    Custom --> Ingestion

    Ingestion --> Validation
    Validation --> Processing
    Processing --> Storage

    Storage --> Query
    Storage --> Cache
    Storage --> Stream

    Query --> Trading
    Query --> Strategy
    Cache --> Risk
    Cache --> Analytics
    Stream --> Trading
    Stream --> Risk
```

#### 2.2.2 数据质量保障架构

```mermaid
graph TB
    subgraph "质量监控体系"
        direction TB

        subgraph "数据验证"
            Schema[模式验证]
            Business[业务规则]
            Consistency[一致性检查]
        end

        subgraph "质量评估"
            Completeness[完整性]
            Accuracy[准确性]
            Timeliness[时效性]
        end

        subgraph "质量修复"
            AutoRepair[自动修复]
            ManualReview[人工审核]
            DataCleaning[数据清洗]
        end

        subgraph "质量报告"
            QualityReport[质量报告]
            TrendAnalysis[趋势分析]
            AlertSystem[告警系统]
        end
    end

    Schema --> Completeness
    Business --> Accuracy
    Consistency --> Accuracy

    Completeness --> AutoRepair
    Accuracy --> AutoRepair
    Timeliness --> AutoRepair

    AutoRepair --> ManualReview
    AutoRepair --> DataCleaning

    Completeness --> QualityReport
    Accuracy --> QualityReport
    Timeliness --> QualityReport

    QualityReport --> TrendAnalysis
    QualityReport --> AlertSystem
```

---

## 📁 目录结构详解

### 3.1 核心目录结构

```
src/data/
├── __init__.py                     # 主入口文件，组件导入
├── interfaces/                     # 接口定义层 (4个文件)
│   ├── __init__.py
│   ├── standard_interfaces.py      # 标准接口定义
│   ├── IDataModel.py              # 数据模型接口
│   └── ICacheBackend.py           # 缓存后端接口
├── adapters/                       # 数据适配器层 (16个文件)
│   ├── __init__.py
│   ├── base.py                    # 基础适配器接口
│   ├── adapter_registry.py        # 适配器注册表
│   ├── miniqmt/                   # MiniQMT交易接口 (8个文件)
│   ├── china/                     # 中国市场通用适配器接口层 (1个文件)
│   │   └── __init__.py            # BaseChinaAdapter - 通用适配器基类
│   ├── news/                      # 新闻数据适配器
│   └── macro/                     # 宏观数据适配器
├── china/                          # 中国市场业务逻辑实现层 (13个文件)
│   ├── __init__.py
│   ├── adapters/                  # 适配器实现目录（推荐使用）
│   │   └── __init__.py            # AStockAdapter, STARMarketAdapter
│   ├── adapter.py                 # 完整功能适配器（含Redis、T+1验证）
│   ├── stock.py                   # 传统适配器（向后兼容）
│   ├── adapters.py                # 适配器别名（向后兼容）
│   ├── market.py                  # A股交易规则处理器
│   ├── level2.py                  # Level2行情解码
│   ├── special.py                 # 特殊股票处理
│   ├── dragon_board.py            # 龙虎榜
│   ├── cache_policy.py            # 缓存策略
│   └── market_data.py             # 市场数据
├── cache/                         # 缓存系统 (14个文件)
│   ├── __init__.py
│   ├── cache_manager.py           # 缓存管理器
│   ├── multi_level_cache.py       # 多级缓存
│   ├── redis_cache_adapter.py     # Redis缓存适配器
│   ├── smart_cache_optimizer.py   # 智能缓存优化器
│   └── disk_cache.py              # 磁盘缓存
├── lake/                          # 数据湖 (3个文件)
│   ├── __init__.py
│   ├── data_lake_manager.py       # 数据湖管理器
│   ├── metadata_manager.py        # 元数据管理器
│   └── partition_manager.py       # 分区管理器
├── quality/                       # 数据质量系统 (13个文件)
│   ├── __init__.py
│   ├── unified_quality_monitor.py # 统一质量监控器
│   ├── data_quality_monitor.py    # 数据质量监控
│   ├── advanced_quality_monitor.py # 高级质量监控
│   └── validator.py               # 数据验证器
├── monitoring/                    # 监控系统 (11个文件)
│   ├── __init__.py
│   ├── performance_monitor.py     # 性能监控
│   ├── quality_monitor.py         # 质量监控
│   ├── metrics_components.py      # 指标组件
│   ├── data_alert_rules.py        # 告警规则
│   └── dashboard.py               # 监控面板
├── processing/                    # 数据处理 (8个文件)
│   ├── data_processor.py          # 数据处理器
│   ├── data_preprocessor.py       # 数据预处理器
│   └── transformers/              # 数据转换器
├── loaders/                       # 数据加载器 (25个文件)
│   ├── __init__.py
│   ├── base_loader.py            # 基础加载器
│   ├── postgresql_loader.py      # PostgreSQL数据加载器 [增强版]
│   ├── stock_loader.py           # 股票数据加载器
│   ├── crypto_loader.py          # 加密货币加载器
│   ├── news_loader.py            # 新闻数据加载器
│   ├── parallel_loader.py        # 并行加载器
│   └── enhanced_data_loader.py   # 增强数据加载器
├── distributed/                   # 分布式处理 (7个文件)
│   ├── __init__.py
│   ├── distributed_data_loader.py # 分布式数据加载器
│   ├── load_balancer.py          # 负载均衡器
│   ├── multiprocess_loader.py    # 多进程加载器
│   └── sharding_manager.py       # 分片管理器
├── validation/                    # 数据验证 (7个文件)
│   ├── data_validator.py         # 数据验证器
│   ├── validator_components.py   # 验证组件
│   └── enhanced_validator.py     # 增强验证器
├── security/                      # 数据安全 (3个文件)
│   ├── data_encryption.py        # 数据加密
│   ├── access_control.py         # 访问控制
│   └── audit_logger.py           # 审计日志
├── compliance/                    # 合规管理 (5个文件)
│   ├── compliance_checker.py     # 合规检查器
│   ├── data_policy_manager.py    # 数据策略管理器
│   └── privacy_protector.py      # 隐私保护器
└── utils/                        # 工具库 (扩展功能)
    ├── decorators.py             # 装饰器工具
    ├── helpers.py                # 辅助函数
    └── validators.py             # 验证工具
```

### 3.2 关键文件说明

#### 3.2.1 核心入口文件

**`__init__.py`** - 主入口文件
```python
# 核心组件导入
DataLakeManager           # 数据湖管理器
CacheManager             # 缓存管理器
UnifiedQualityMonitor    # 统一质量监控器
PerformanceMonitor       # 性能监控器

# 便捷函数
get_data_adapter()       # 获取数据适配器
get_cache_manager()      # 获取缓存管理器
get_quality_monitor()    # 获取质量监控器
```

#### 3.2.2 数据湖管理器

**`lake/data_lake_manager.py`** - 数据湖管理器
```python
class DataLakeManager:
    def __init__(self, config: LakeConfig):
        self.config = config
        self.base_path = Path(config.base_path)
        self.data_path = self.base_path / "data"
        self.metadata_path = self.base_path / "metadata"
        self.partitions_path = self.base_path / "partitions"

    def store_data(self, data, dataset_name, metadata=None):
        # 数据存储逻辑
        pass

    def load_data(self, dataset_name, filters=None):
        # 数据加载逻辑
        pass

    def get_dataset_info(self, dataset_name):
        # 获取数据集信息
        pass
```

#### 3.2.3 PostgreSQL数据加载器 [增强版]

**`loader/postgresql_loader.py`** - PostgreSQL数据加载器

增强版PostgreSQL数据加载器，集成自动重试、缓存、质量检查、性能监控和数据湖功能。

```python
class PostgreSQLDataLoader(BaseDataLoader):
    """
    PostgreSQL数据加载器 - 增强版
    
    核心功能:
    - 数据库连接池管理
    - 自动重试机制 (指数退避)
    - 内存缓存 (LRU淘汰)
    - 数据质量检查
    - 性能监控
    - 数据湖集成
    """
    
    def __init__(self, config: Optional[DataLoaderConfig] = None):
        super().__init__(config)
        self._connection_pool = None
        self._db_config = None
        self._memory_cache = {}          # 内存缓存
        self._performance_monitor = None  # 性能监控器
        self._data_lake_manager = None    # 数据湖管理器
        self._init_db_config()
    
    # ========== 核心加载方法 ==========
    def load(self, query: str, params: Optional[Dict] = None) -> LoadResult:
        """基础SQL加载方法"""
        pass
    
    def load_stock_data(self, symbol: str, start_date: str, end_date: str) -> LoadResult:
        """加载股票日线数据"""
        pass
    
    # ========== 第一阶段: 基础增强 ==========
    def load_with_retry(self, query: str, params: Optional[Dict] = None,
                       max_retries: int = 3, backoff_factor: float = 2.0) -> LoadResult:
        """带自动重试的加载方法"""
        pass
    
    def load_with_cache(self, query: str, params: Optional[Dict] = None,
                       cache_ttl: int = 300) -> LoadResult:
        """带内存缓存的加载方法"""
        pass
    
    def clear_cache(self) -> None:
        """清除内存缓存"""
        pass
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        pass
    
    # ========== 第二阶段: 质量与监控 ==========
    def check_data_quality(self, data: pd.DataFrame, 
                          data_source: str = "postgresql") -> Dict[str, Any]:
        """数据质量检查"""
        pass
    
    def load_with_quality_check(self, query: str, params: Optional[Dict] = None,
                               quality_threshold: float = 0.7) -> LoadResult:
        """带质量检查的加载方法"""
        pass
    
    def load_with_monitoring(self, query: str, 
                            params: Optional[Dict] = None) -> LoadResult:
        """带性能监控的加载方法"""
        pass
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        pass
    
    # ========== 第三阶段: 数据湖集成 ==========
    def load_to_data_lake(self, query: str, dataset_name: str,
                         params: Optional[Dict] = None,
                         partition_key: Optional[str] = None) -> Optional[str]:
        """加载数据到数据湖"""
        pass
    
    def load_from_data_lake(self, dataset_name: str,
                           partition_filter: Optional[Dict] = None,
                           date_range: Optional[tuple] = None) -> Optional[pd.DataFrame]:
        """从数据湖加载数据"""
        pass
    
    def sync_to_data_lake(self, query: str, dataset_name: str,
                         sync_interval: int = 3600) -> bool:
        """同步数据到数据湖 (带缓存检查)"""
        pass
    
    # ========== 综合方法 ==========
    def load_enhanced(self, query: str, params: Optional[Dict] = None,
                     use_retry: bool = True, use_cache: bool = True,
                     use_quality_check: bool = True, 
                     use_monitoring: bool = True) -> LoadResult:
        """增强版加载方法 (集成所有功能)"""
        pass
    
    # ========== 统一接口 ==========
    def load_data(self, source: str = "postgresql", 
                  symbol: Optional[str] = None,
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None,
                  use_data_lake: bool = True, **kwargs) -> LoadResult:
        """统一数据加载接口"""
        pass
    
    def get_loader_metadata(self) -> Dict[str, Any]:
        """获取加载器元数据"""
        pass
```

**功能特性:**

| 功能阶段 | 功能名称 | 描述 | 配置参数 |
|---------|---------|------|----------|
| 第一阶段 | 自动重试 | 指数退避重试机制 | `max_retries`, `backoff_factor` |
| 第一阶段 | 内存缓存 | LRU缓存策略 | `cache_ttl`, `max_cache_size=1000` |
| 第二阶段 | 质量检查 | 完整性/准确性/一致性检查 | `quality_threshold` |
| 第二阶段 | 性能监控 | 加载时间/成功率监控 | 自动记录 |
| 第三阶段 | 数据湖集成 | 分层存储与分区管理 | `partition_key` |
| 第三阶段 | 统一接口 | 符合架构标准的接口 | `use_data_lake` |

**使用示例:**

```python
from src.data.loader import PostgreSQLDataLoader, get_postgresql_loader

# 获取加载器实例
loader = get_postgresql_loader()

# 1. 基础加载
result = loader.load("SELECT * FROM stock_daily_data")

# 2. 带重试的加载
result = loader.load_with_retry(query, max_retries=3)

# 3. 带缓存的加载
result = loader.load_with_cache(query, cache_ttl=300)

# 4. 带质量检查的加载
result = loader.load_with_quality_check(query, quality_threshold=0.8)

# 5. 增强版加载 (集成所有功能)
result = loader.load_enhanced(
    query, 
    use_retry=True,
    use_cache=True,
    use_quality_check=True,
    use_monitoring=True
)

# 6. 统一接口加载
result = loader.load_data(
    symbol="002837",
    start_date="2025-01-01",
    end_date="2025-12-31",
    use_data_lake=True  # 优先从数据湖加载
)

# 7. 数据湖操作
file_path = loader.load_to_data_lake(query, "stock_data")
df = loader.load_from_data_lake("stock_data")
loader.sync_to_data_lake(query, "stock_data", sync_interval=3600)
```

#### 3.2.4 缓存管理器

**`cache/cache_manager.py`** - 缓存管理器
```python
class CacheManager:
    def __init__(self, config: CacheConfig):
        self.config = config
        self.memory_cache = {}
        self.disk_cache = None
        if config.enable_disk_cache:
            self.disk_cache = DiskCache(config)

    def get(self, key):
        # 缓存获取逻辑
        pass

    def set(self, key, value, ttl=None):
        # 缓存设置逻辑
        pass

    def get_stats(self):
        # 获取缓存统计
        pass
```

#### 3.2.4 质量监控器

**`quality/unified_quality_monitor.py`** - 统一质量监控器
```python
class UnifiedQualityMonitor:
    def __init__(self, config: QualityConfig):
        self.config = config
        self.validator = UnifiedDataValidator()
        self.quality_history = defaultdict(list)

    def check_quality(self, data, data_type):
        # 质量检查逻辑
        validation_result = self.validator.validate(data, data_type)
        metrics = self._calculate_quality_metrics(data, data_type, validation_result)
        return {
            "metrics": metrics,
            "validation": validation_result,
            "anomalies": self._detect_anomalies(data_type, metrics)
        }
```

---

## 🔧 核心组件详解

### 4.1 数据适配器系统

#### 4.1.1 架构设计

```mermaid
graph TB
    subgraph "数据适配器系统"
        direction TB

        subgraph "适配器类型"
            MarketAdapter[市场数据适配器]
            TradeAdapter[交易接口适配器]
            NewsAdapter[新闻数据适配器]
            MacroAdapter[宏观数据适配器]
        end

        subgraph "核心组件"
            BaseAdapter[基础适配器<br/>BaseAdapter]
            AdapterRegistry[适配器注册表<br/>AdapterRegistry]
            ConnectionPool[连接池<br/>ConnectionPool]
            RateLimiter[限流器<br/>RateLimiter]
        end

        subgraph "协议支持"
            REST[REST API]
            WebSocket[WebSocket]
            TCP[TCP Socket]
            File[文件系统]
        end

        subgraph "数据源"
            Exchange[交易所API]
            Broker[券商接口]
            NewsAPI[新闻API]
            Database[数据库]
        end
    end

    BaseAdapter --> MarketAdapter
    BaseAdapter --> TradeAdapter
    BaseAdapter --> NewsAdapter
    BaseAdapter --> MacroAdapter

    AdapterRegistry --> BaseAdapter
    ConnectionPool --> BaseAdapter
    RateLimiter --> BaseAdapter

    REST --> BaseAdapter
    WebSocket --> BaseAdapter
    TCP --> BaseAdapter
    File --> BaseAdapter

    Exchange --> MarketAdapter
    Broker --> TradeAdapter
    NewsAPI --> NewsAdapter
    Database --> MacroAdapter
```

#### 4.1.2 中国市场适配器架构

中国市场适配器采用**清晰分层架构**，将通用接口层和业务实现层分离：

```mermaid
graph TB
    subgraph "中国市场适配器架构"
        direction TB

        subgraph "通用适配器接口层"
            BaseChina[BaseChinaAdapter<br/>adapters/china/<br/>抽象基类]
        end

        subgraph "业务实现层"
            direction LR
            AStock[AStockAdapter<br/>china/adapters/<br/>A股适配器]
            STAR[STARMarketAdapter<br/>china/adapters/<br/>科创板适配器]
            Full[ChinaDataAdapter<br/>china/adapter.py<br/>完整功能适配器]
            Legacy[ChinaStockDataAdapter<br/>china/stock.py<br/>传统适配器]
        end

        subgraph "业务逻辑层"
            direction LR
            Market[ChinaMarketRules<br/>market.py<br/>市场规则]
            Level2[ChinaLevel2Processor<br/>level2.py<br/>Level2处理]
            Special[SpecialStockHandler<br/>special.py<br/>特殊股票]
        end
    end

    BaseChina --> AStock
    BaseChina --> STAR
    BaseChina --> Full
    BaseChina --> Legacy

    AStock --> Market
    STAR --> Market
    Full --> Market
    Full --> Level2
```

##### 职责分工

| 目录/模块 | 职责定位 | 主要功能 | 使用场景 |
|----------|----------|----------|----------|
| **adapters/china/** | 通用适配器接口层 | 定义 `BaseChinaAdapter` 抽象基类 | 被 `china/` 目录中的实现继承 |
| **china/adapters/** | 适配器实现层（推荐） | `AStockAdapter`, `STARMarketAdapter` | 新项目开发，标准A股和科创板数据获取 |
| **china/adapter.py** | 完整功能适配器 | `ChinaDataAdapter`（含Redis、T+1验证） | 需要缓存和验证的完整场景 |
| **china/stock.py** | 传统适配器（兼容） | `ChinaStockDataAdapter` | 旧系统兼容 |
| **china/market.py** | 市场规则 | `ChinaMarketRules` | 交易时间、涨跌停限制等规则 |
| **china/level2.py** | Level2行情处理 | `ChinaLevel2Processor` | Level2行情解码 |
| **china/special.py** | 特殊股票处理 | `SpecialStockHandler` | 特殊股票类型处理 |

##### 使用示例

```python
# 推荐方式：使用新版适配器
from src.data.china.adapters import AStockAdapter, STARMarketAdapter

# A股适配器
a_stock_adapter = AStockAdapter()
data = a_stock_adapter.get_stock_basic('600519')

# 科创板适配器
star_adapter = STARMarketAdapter()
star_data = star_adapter.get_star_market_data('688001')

# 完整功能适配器（含缓存和验证）
from src.data.china.adapter import ChinaDataAdapter

adapter = ChinaDataAdapter(config={
    'redis': {'host': 'localhost', 'port': 6379}
})
stock_info = adapter.get_stock_info('600519')

# 市场规则
from src.data.china.market import ChinaMarketRules

rules = ChinaMarketRules()
price_limit = rules.get_price_limit('600519')  # 获取涨跌停限制
```

##### 架构原则

1. **接口与实现分离**：`adapters/china/` 定义接口，`china/` 实现业务逻辑
2. **清晰的分层**：接口层 → 实现层 → 业务逻辑层
3. **向后兼容**：保留传统适配器，确保旧代码正常运行
4. **推荐使用新版**：新项目优先使用 `china/adapters/` 中的适配器

#### 4.1.3 MiniQMT交易适配器

```python
class MiniQMTDataAdapter(BaseAdapter):
    """MiniQMT交易数据适配器"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.connection_pool = ConnectionPool(
            min_conn=5,
            max_conn=50,
            host=config['host'],
            port=config['port']
        )
        self.rate_limiter = RateLimiter(
            requests_per_minute=config.get('rate_limit', 1000)
        )

    def connect(self):
        """建立连接"""
        return self.connection_pool.get_connection()

    def get_market_data(self, symbol: str, start_date: str, end_date: str):
        """获取市场数据"""
        with self.rate_limiter:
            conn = self.connect()
            try:
                # 数据获取逻辑
                return self._fetch_market_data(conn, symbol, start_date, end_date)
            finally:
                self.connection_pool.return_connection(conn)

    def get_account_info(self):
        """获取账户信息"""
        with self.rate_limiter:
            conn = self.connect()
            try:
                return self._fetch_account_info(conn)
            finally:
                self.connection_pool.return_connection(conn)
```

#### 4.1.3 数据源适配器注册表

```python
class AdapterRegistry:
    """数据适配器注册表"""

    _instance = None
    _adapters: Dict[str, BaseAdapter] = {}

    @classmethod
    def register(cls, name: str, adapter: BaseAdapter):
        """注册适配器"""
        cls._adapters[name] = adapter

    @classmethod
    def get_adapter(cls, name: str) -> Optional[BaseAdapter]:
        """获取适配器"""
        return cls._adapters.get(name)

    @classmethod
    def list_adapters(cls) -> List[str]:
        """列出所有适配器"""
        return list(cls._adapters.keys())

    @classmethod
    def get_adapter_for_source(cls, source_type: str) -> Optional[BaseAdapter]:
        """根据数据源类型获取适配器"""
        # 映射逻辑
        adapter_mapping = {
            'miniqmt': 'miniqmt_adapter',
            'china_stock': 'china_stock_adapter',
            'crypto': 'crypto_adapter',
            'news': 'news_adapter'
        }

        adapter_name = adapter_mapping.get(source_type)
        if adapter_name:
            return cls.get_adapter(adapter_name)
        return None
```

### 4.2 数据湖存储系统

#### 4.2.1 分层存储架构

```mermaid
graph TB
    subgraph "数据湖存储架构"
        direction TB

        subgraph "存储层级"
            Hot[热数据层<br/>高性能SSD<br/>实时访问]
            Warm[温数据层<br/>普通SSD<br/>频繁访问]
            Cold[冷数据层<br/>HDD/对象存储<br/>归档数据]
        end

        subgraph "分区策略"
            DatePartition[日期分区<br/>按日期分区]
            HashPartition[哈希分区<br/>按哈希分区]
            RangePartition[范围分区<br/>按范围分区]
        end

        subgraph "压缩格式"
            Parquet[Parquet<br/>列式存储<br/>高压缩比]
            CSV[CSV<br/>通用格式<br/>易读取]
            JSON[JSON<br/>结构化<br/>灵活性好]
        end

        subgraph "元数据管理"
            MetadataStore[元数据存储<br/>数据血缘<br/>版本控制]
            Catalog[数据目录<br/>数据集索引<br/>访问控制]
        end
    end

    Hot --> DatePartition
    Warm --> HashPartition
    Cold --> RangePartition

    DatePartition --> Parquet
    HashPartition --> Parquet
    RangePartition --> CSV

    Parquet --> MetadataStore
    CSV --> MetadataStore
    JSON --> MetadataStore

    MetadataStore --> Catalog
```

#### 4.2.2 数据湖管理器实现

```python
@dataclass
class LakeConfig:
    """数据湖配置"""
    base_path: str = "data_lake"
    approach: str = "date"  # date, hash, custom
    compression: str = "parquet"  # parquet, csv, json
    metadata_enabled: bool = True
    versioning_enabled: bool = True
    retention_days: int = 365
    max_size_gb: int = 100

class DataLakeManager:
    """数据湖管理器"""

    def __init__(self, config: LakeConfig = None):
        self.config = config or LakeConfig()
        self.base_path = Path(self.config.base_path)
        self.base_path.mkdir(exist_ok=True)

        # 初始化子目录
        self.data_path = self.base_path / "data"
        self.metadata_path = self.base_path / "metadata"
        self.partitions_path = self.base_path / "partitions"

        for path in [self.data_path, self.metadata_path, self.partitions_path]:
            path.mkdir(exist_ok=True)

    def store_data(self, data: pd.DataFrame, dataset_name: str,
                   partition_key: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """存储数据到数据湖"""

        # 生成存储路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        partition_info = self._get_partition_info(data, partition_key)

        # 构建文件路径
        file_path = self._build_file_path(dataset_name, partition_info, timestamp)

        # 存储数据
        self._save_data(data, file_path)

        # 存储元数据
        if self.config.metadata_enabled:
            self._save_metadata(dataset_name, file_path, metadata, partition_info)

        # 更新分区信息
        self._update_partition_info(dataset_name, partition_info, file_path)

        return str(file_path)

    def load_data(self, dataset_name: str,
                  partition_filter: Optional[Dict[str, Any]] = None,
                  date_range: Optional[tuple] = None) -> pd.DataFrame:
        """从数据湖加载数据"""

        # 查找匹配的文件
        matching_files = self._find_matching_files(dataset_name, partition_filter, date_range)

        if not matching_files:
            return pd.DataFrame()

        # 加载数据
        data_frames = []
        for file_path in matching_files:
            df = self._load_data_file(file_path)
            data_frames.append(df)

        # 合并数据
        if data_frames:
            return pd.concat(data_frames, ignore_index=True)
        return pd.DataFrame()

    def list_datasets(self) -> List[str]:
        """列出所有数据集"""
        datasets = set()
        for file_path in self.data_path.rglob("*"):
            if file_path.is_file() and file_path.suffix in ['.parquet', '.csv', '.json']:
                relative_path = file_path.relative_to(self.data_path)
                dataset_name = relative_path.parts[0]
                datasets.add(dataset_name)
        return sorted(list(datasets))

    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """获取数据集信息"""
        info = {
            'name': dataset_name,
            'files': [],
            'total_rows': 0,
            'partitions': set(),
            'last_updated': None
        }

        dataset_path = self.data_path / dataset_name
        if not dataset_path.exists():
            return info

        for file_path in dataset_path.rglob("*"):
            if file_path.is_file() and file_path.suffix in ['.parquet', '.csv', '.json']:
                file_info = {
                    'path': str(file_path),
                    'size': file_path.stat().st_size,
                    'modified': datetime.fromtimestamp(file_path.stat().st_mtime)
                }

                # 加载文件获取行数
                try:
                    df = self._load_data_file(file_path)
                    file_info['rows'] = len(df)
                    info['total_rows'] += len(df)
                except:
                    file_info['rows'] = 0

                info['files'].append(file_info)

                # 提取分区信息
                partition_info = self._extract_partition_from_path(file_path)
                if partition_info:
                    info['partitions'].add(partition_info)

                # 更新最后修改时间
                if info['last_updated'] is None or file_info['modified'] > info['last_updated']:
                    info['last_updated'] = file_info['modified']

        info['partitions'] = list(info['partitions'])
        return info

    def _get_partition_info(self, data: pd.DataFrame, partition_key: Optional[str]) -> Dict[str, Any]:
        """获取分区信息"""
        partition_info = {}

        if partition_key and partition_key in data.columns:
            if self.config.approach == "date":
                # 日期分区
                partition_info['date'] = data[partition_key].dt.strftime('%Y-%m-%d').iloc[0]
            elif self.config.approach == "hash":
                # 哈希分区
                partition_info['hash'] = hash(str(data[partition_key].iloc[0])) % 100
            else:
                # 自定义分区
                partition_info['custom'] = str(data[partition_key].iloc[0])

        return partition_info

    def _build_file_path(self, dataset_name: str, partition_info: Dict[str, Any], timestamp: str) -> Path:
        """构建文件路径"""
        dataset_path = self.data_path / dataset_name

        # 构建分区路径
        partition_path = dataset_path
        for key, value in partition_info.items():
            partition_path = partition_path / f"{key}={value}"

        partition_path.mkdir(parents=True, exist_ok=True)

        # 构建文件名
        filename = f"data_{timestamp}.{self.config.compression}"
        return partition_path / filename

    def _save_data(self, data: pd.DataFrame, file_path: Path):
        """保存数据文件"""
        if self.config.compression == "parquet":
            data.to_parquet(file_path, index=False)
        elif self.config.compression == "csv":
            data.to_csv(file_path, index=False)
        elif self.config.compression == "json":
            data.to_json(file_path, orient='records', indent=2)
        else:
            raise ValueError(f"不支持的压缩格式: {self.config.compression}")

    def _load_data_file(self, file_path: Path) -> pd.DataFrame:
        """加载数据文件"""
        if file_path.suffix == ".parquet":
            return pd.read_parquet(file_path)
        elif file_path.suffix == ".csv":
            return pd.read_csv(file_path)
        elif file_path.suffix == ".json":
            return pd.read_json(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")

    def _save_metadata(self, dataset_name: str, file_path: Path, metadata: Optional[Dict[str, Any]], partition_info: Dict[str, Any]):
        """保存元数据"""
        metadata_file = self.metadata_path / f"{dataset_name}_{file_path.stem}.json"

        metadata_data = {
            'dataset_name': dataset_name,
            'file_path': str(file_path),
            'created_at': datetime.now().isoformat(),
            'partition_info': partition_info,
            'custom_metadata': metadata or {}
        }

        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata_data, f, ensure_ascii=False, indent=2)

    def _update_partition_info(self, dataset_name: str, partition_info: Dict[str, Any], file_path: Path):
        """更新分区信息"""
        partition_file = self.partitions_path / f"{dataset_name}_partitions.json"

        try:
            with open(partition_file, 'r', encoding='utf-8') as f:
                partitions = json.load(f)
        except FileNotFoundError:
            partitions = {}

        partition_key = "_".join([f"{k}={v}" for k, v in partition_info.items()])
        if partition_key not in partitions:
            partitions[partition_key] = []

        partitions[partition_key].append(str(file_path))

        with open(partition_file, 'w', encoding='utf-8') as f:
            json.dump(partitions, f, ensure_ascii=False, indent=2)

    def _find_matching_files(self, dataset_name: str, partition_filter: Optional[Dict[str, Any]], date_range: Optional[tuple]) -> List[Path]:
        """查找匹配的文件"""
        matching_files = []
        dataset_path = self.data_path / dataset_name

        if not dataset_path.exists():
            return matching_files

        for file_path in dataset_path.rglob("*"):
            if file_path.is_file() and file_path.suffix in ['.parquet', '.csv', '.json']:
                # 检查分区过滤
                if partition_filter:
                    file_partition = self._extract_partition_from_path(file_path)
                    if not self._matches_partition_filter(file_partition, partition_filter):
                        continue

                # 检查日期范围
                if date_range:
                    file_date = self._extract_date_from_path(file_path)
                    if not self._matches_date_range(file_date, date_range):
                        continue

                matching_files.append(file_path)

        return matching_files

    def _extract_partition_from_path(self, file_path: Path) -> Dict[str, Any]:
        """从路径中提取分区信息"""
        partition_info = {}
        parts = file_path.parts

        for part in parts:
            if '=' in part:
                key, value = part.split('=', 1)
                partition_info[key] = value

        return partition_info

    def _extract_date_from_path(self, file_path: Path) -> Optional[datetime]:
        """从路径中提取日期信息"""
        try:
            # 尝试从文件名中提取日期
            filename = file_path.stem
            if 'data_' in filename:
                date_str = filename.split('data_')[1]
                return datetime.strptime(date_str, '%Y%m%d_%H%M%S')
        except:
            pass

        return None

    def _matches_partition_filter(self, file_partition: Dict[str, Any], partition_filter: Dict[str, Any]) -> bool:
        """检查是否匹配分区过滤条件"""
        for key, value in partition_filter.items():
            if key not in file_partition or file_partition[key] != str(value):
                return False
        return True

    def _matches_date_range(self, file_date: Optional[datetime], date_range: tuple) -> bool:
        """检查是否匹配日期范围"""
        if file_date is None:
            return True

        start_date, end_date = date_range
        return start_date <= file_date <= end_date
```

### 4.3 缓存系统架构

#### 4.3.1 多级缓存策略

```mermaid
graph LR
    subgraph "应用层"
        App[应用请求]
    end

    subgraph "缓存层"
        L1[L1缓存<br/>内存缓存<br/>纳秒级响应]
        L2[L2缓存<br/>Redis集群<br/>微秒级响应]
        L3[L3缓存<br/>磁盘缓存<br/>毫秒级响应]
        L4[L4缓存<br/>数据湖<br/>秒级响应]
    end

    subgraph "数据源"
        DB[(数据库)]
        API[外部API]
        File[文件系统]
    end

    App --> L1
    L1 --> L2
    L2 --> L3
    L3 --> L4
    L4 --> DB
    L4 --> API
    L4 --> File

    L1 -.-> L2
    L2 -.-> L3
    L3 -.-> L4
```

#### 4.3.2 智能缓存管理器

```python
class CacheManager:
    """缓存管理器"""

    def __init__(self, config: CacheConfig, strategy: Optional[ICacheStrategy] = None):
        self.config = config

        # 内存缓存
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()

        # 统计
        self._stats = CacheStats()

        # 磁盘缓存
        self.disk_cache = None
        if config.enable_disk_cache:
            from .disk_cache import DiskCache, DiskCacheConfig
            disk_config = DiskCacheConfig(
                cache_dir=config.disk_cache_dir,
                max_file_size=config.max_file_size,
                compression=config.compression,
                encryption=config.encryption,
                backup_enabled=config.backup_enabled,
                cleanup_interval=config.cleanup_interval
            )
            self.disk_cache = DiskCache(disk_config)

        # 启动清理线程
        self._cleanup_thread = None
        self._stop_cleanup = False
        if config.enable_stats:
            self._start_cleanup_thread()

        self.strategy = strategy or None

    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self._lock:
            # 先检查内存缓存
            entry = self._cache.get(key)
            if self.strategy:
                self.strategy.on_get(self._cache, key, entry, self.config)

            if entry:
                if entry.is_expired():
                    del self._cache[key]
                    self._stats.miss()
                    return None

                entry.access()
                self._stats.hit()
                return entry.value

            # 检查磁盘缓存
            if self.disk_cache:
                value = self.disk_cache.get(key)
                if value is not None:
                    # 加载到内存缓存
                    entry = CacheEntry(key=key, value=value, ttl=self.config.ttl)
                    self._cache[key] = entry
                    self._stats.hit()
                    self._evict_if_needed()
                    return value

            self._stats.miss()
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        try:
            with self._lock:
                ttl = ttl or self.config.ttl
                entry = CacheEntry(key=key, value=value, ttl=ttl)

                self._cache[key] = entry
                self._stats.set()

                # 保存到磁盘
                if self.disk_cache:
                    self.disk_cache.set(key, value, ttl)

                self._evict_if_needed()
                if self.strategy:
                    self.strategy.on_set(self._cache, key, entry, self.config)

                return True
        except Exception as e:
            self.logger.error(f"Failed to set cache: {e}")
            self._stats.error()
            return False

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            stats = self._stats.get_stats()
            stats.update({
                'cache_size': len(self._cache),
                'max_size': self.config.max_size,
                'disk_cache_enabled': self.config.enable_disk_cache,
                'compression_enabled': self.config.compression,
                'encryption_enabled': self.config.encryption
            })

            # 添加磁盘缓存统计
            if self.disk_cache:
                disk_stats = self.disk_cache.get_stats()
                stats['disk_cache'] = disk_stats.get('disk_cache', {})

            return stats
```

### 4.4 数据质量监控系统

#### 4.4.1 质量监控架构

```mermaid
graph TB
    subgraph "数据质量监控系统"
        direction TB

        subgraph "质量检测"
            RealTime[实时检测<br/>流式数据]
            Batch[批量检测<br/>历史数据]
            Scheduled[定时检测<br/>周期性检查]
        end

        subgraph "质量指标"
            Completeness[完整性<br/>非空率]
            Accuracy[准确性<br/>正确性]
            Consistency[一致性<br/>逻辑一致]
            Timeliness[时效性<br/>新鲜度]
            Validity[有效性<br/>格式正确]
        end

        subgraph "异常检测"
            Statistical[统计方法<br/>标准差分析]
            MLBased[机器学习<br/>异常检测模型]
            RuleBased[规则引擎<br/>业务规则]
        end

        subgraph "质量修复"
            AutoRepair[自动修复<br/>规则修复]
            Manual[人工干预<br/>专家审核]
            DataClean[数据清洗<br/>异常值处理]
        end

        subgraph "报告与告警"
            Dashboard[监控面板<br/>可视化展示]
            Alerts[智能告警<br/>多渠道通知]
            Reports[质量报告<br/>趋势分析]
        end
    end

    RealTime --> Completeness
    Batch --> Accuracy
    Scheduled --> Consistency

    Completeness --> Statistical
    Accuracy --> MLBased
    Consistency --> RuleBased

    Statistical --> AutoRepair
    MLBased --> AutoRepair
    RuleBased --> Manual

    AutoRepair --> Dashboard
    Manual --> Dashboard
    DataClean --> Dashboard

    Dashboard --> Alerts
    Dashboard --> Reports
```

#### 4.4.2 统一质量监控器

```python
@dataclass
class QualityConfig:
    """质量监控配置"""
    enable_realtime_monitoring: bool = True
    enable_historical_analysis: bool = True
    quality_threshold: float = 0.8
    anomaly_detection_sensitivity: float = 0.95
    max_quality_history: int = 1000
    alert_cooldown_minutes: int = 5
    enable_auto_repair: bool = False
    repair_confidence_threshold: float = 0.9

@dataclass
class QualityMetrics:
    """质量指标"""
    completeness: float = 0.0
    accuracy: float = 0.0
    consistency: float = 0.0
    timeliness: float = 0.0
    validity: float = 0.0
    overall_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class QualityIssue:
    """质量问题"""
    issue_type: str
    severity: str  # critical, high, medium, low
    description: str
    affected_records: int = 0
    suggested_fix: Optional[str] = None
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

class UnifiedDataValidator:
    """统一数据验证器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.validation_rules = self._load_validation_rules()

    def validate(self, data: Any, data_type: DataSourceType) -> Dict[str, Any]:
        """验证数据"""
        try:
            if data is None:
                return {
                    "valid": False,
                    "issues": [QualityIssue(
                        issue_type="null_data",
                        severity="critical",
                        description="数据为空",
                        confidence=1.0
                    )]
                }

            rules = self.validation_rules.get(data_type, {})
            issues = []

            # 基础验证
            issues.extend(self._validate_basic_structure(data, rules))

            # 数据类型特定验证
            if data_type == DataSourceType.STOCK:
                issues.extend(self._validate_stock_data(data, rules))

            # 计算整体有效性
            valid = len([i for i in issues if i.severity == "critical"]) == 0

            return {
                "valid": valid,
                "issues": issues,
                "issue_count": len(issues),
                "critical_issues": len([i for i in issues if i.severity == "critical"])
            }

        except Exception as e:
            return {
                "valid": False,
                "issues": [QualityIssue(
                    issue_type="validation_error",
                    severity="critical",
                    description=f"验证过程出错: {str(e)}",
                    confidence=1.0
                )]
            }

    def _validate_basic_structure(self, data: Any, rules: Dict[str, Any]) -> List[QualityIssue]:
        """验证基本结构"""
        issues = []

        # 检查必需字段
        required_fields = rules.get("required_fields", [])
        if required_fields:
            if hasattr(data, 'columns'):
                # DataFrame
                missing_fields = [field for field in required_fields if field not in data.columns]
                if missing_fields:
                    issues.append(QualityIssue(
                        issue_type="missing_fields",
                        severity="critical",
                        description=f"缺少必需字段: {missing_fields}",
                        affected_records=len(data) if hasattr(data, '__len__') else 0,
                        confidence=1.0
                    ))

        return issues

    def _validate_stock_data(self, data: Any, rules: Dict[str, Any]) -> List[QualityIssue]:
        """验证股票数据"""
        issues = []

        if hasattr(data, 'columns'):
            # 检查价格合理性
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in data.columns:
                    price_range = rules.get("price_range", {})
                    min_price = price_range.get("min", 0)
                    max_price = price_range.get("max", 1000000)

                    invalid_prices = ((data[col] < min_price) | (data[col] > max_price)).sum()
                    if invalid_prices > 0:
                        issues.append(QualityIssue(
                            issue_type="invalid_price",
                            severity="high",
                            description=f"{col}列存在无效价格值",
                            affected_records=int(invalid_prices),
                            confidence=0.9
                        ))

        return issues

class UnifiedQualityMonitor:
    """统一质量监控器"""

    def __init__(self, config: Optional[QualityConfig] = None):
        self.config = config or QualityConfig()
        self.validator = UnifiedDataValidator()
        self.quality_history: Dict[DataSourceType, List[QualityMetrics]] = defaultdict(list)
        self.alerts_sent: Dict[str, datetime] = {}

    def check_quality(self, data: Any, data_type: DataSourceType) -> Dict[str, Any]:
        """检查数据质量"""
        start_time = datetime.now()

        try:
            # 数据验证
            validation_result = self.validator.validate(data, data_type)

            # 计算质量指标
            metrics = self._calculate_quality_metrics(data, data_type, validation_result)

            # 更新质量历史
            self.quality_history[data_type].append(metrics)
            if len(self.quality_history[data_type]) > self.config.max_quality_history:
                self.quality_history[data_type].pop(0)

            # 检测异常
            anomalies = self._detect_anomalies(data_type, metrics)

            # 生成告警
            if anomalies:
                self._generate_alerts(data_type, anomalies)

            # 生成建议
            recommendations = self._generate_recommendations(data_type, metrics, validation_result)

            return {
                "metrics": metrics,
                "validation": validation_result,
                "anomalies": anomalies,
                "recommendations": recommendations,
                "processing_time": (datetime.now() - start_time).total_seconds()
            }

        except Exception as e:
            return {
                "metrics": QualityMetrics(),
                "validation": {"valid": False, "issues": []},
                "anomalies": [],
                "recommendations": ["检查系统日志以获取详细错误信息"],
                "error": str(e),
                "processing_time": (datetime.now() - start_time).total_seconds()
            }

    def _calculate_quality_metrics(self, data: Any, data_type: DataSourceType,
                                 validation_result: Dict[str, Any]) -> QualityMetrics:
        """计算质量指标"""
        metrics = QualityMetrics()

        try:
            # 完整性：基于非空值比例
            if hasattr(data, 'columns'):
                non_null_ratio = 1 - data.isnull().sum().sum() / (len(data) * len(data.columns))
                metrics.completeness = max(0, min(1, non_null_ratio))

            # 有效性：基于验证结果
            validation_valid = validation_result.get("valid", False)
            critical_issues = validation_result.get("critical_issues", 0)
            metrics.validity = 1.0 if validation_valid and critical_issues == 0 else 0.5

            # 一致性：检查数据内部一致性
            metrics.consistency = self._calculate_consistency(data, data_type)

            # 时效性：基于数据时间戳
            metrics.timeliness = self._calculate_timeliness(data, data_type)

            # 准确性：基于业务规则验证
            metrics.accuracy = 1.0 - (validation_result.get("issue_count", 0) * 0.1)

            # 综合得分
            metrics.overall_score = statistics.mean([
                metrics.completeness,
                metrics.accuracy,
                metrics.consistency,
                metrics.timeliness,
                metrics.validity
            ])

            return metrics

        except Exception as e:
            return QualityMetrics()

    def _calculate_consistency(self, data: Any, data_type: DataSourceType) -> float:
        """计算一致性"""
        try:
            if data_type == DataSourceType.STOCK and hasattr(data, 'columns'):
                # 检查OHLC关系：High >= max(Open, Close), Low <= min(Open, Close)
                if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                    valid_ohlc = (
                        (data['high'] >= data[['open', 'close']].max(axis=1)) &
                        (data['low'] <= data[['open', 'close']].min(axis=1))
                    ).mean()
                    return float(valid_ohlc)
            return 0.8  # 默认值
        except:
            return 0.5

    def _calculate_timeliness(self, data: Any, data_type: DataSourceType) -> float:
        """计算时效性"""
        try:
            if hasattr(data, 'columns') and 'timestamp' in data.columns:
                # 检查数据的新鲜度
                latest_timestamp = pd.to_datetime(data['timestamp']).max()
                now = pd.Timestamp.now()
                hours_old = (now - latest_timestamp).total_seconds() / 3600

                # 根据数据类型设置不同的新鲜度要求
                freshness_thresholds = {
                    DataSourceType.STOCK: 24,    # 24小时
                    DataSourceType.CRYPTO: 1,    # 1小时
                    DataSourceType.NEWS: 12,     # 12小时
                }

                threshold = freshness_thresholds.get(data_type, 24)
                timeliness = max(0, 1 - (hours_old / threshold))
                return timeliness

            return 0.5
        except:
            return 0.5

    def _detect_anomalies(self, data_type: DataSourceType, current_metrics: QualityMetrics) -> List[str]:
        """检测异常"""
        anomalies = []

        try:
            history = self.quality_history[data_type]
            if len(history) < 5:  # 需要足够的历史数据
                return anomalies

            # 计算历史平均值和标准差
            scores = [m.overall_score for m in history[-10:]]  # 最近10个数据点
            avg_score = statistics.mean(scores)
            std_score = statistics.stdev(scores) if len(scores) > 1 else 0

            # 检测显著下降
            if current_metrics.overall_score < avg_score - 2 * std_score:
                anomalies.append(f"质量得分显著下降: {current_metrics.overall_score:.2f} vs 平均 {avg_score:.2f}")

            # 检测完整性问题
            if current_metrics.completeness < self.config.quality_threshold:
                anomalies.append(f"数据完整性不足: {current_metrics.completeness:.2f}")

            # 检测有效性问题
            if current_metrics.validity < 0.8:
                anomalies.append(f"数据有效性不足: {current_metrics.validity:.2f}")

        except Exception as e:
            pass

        return anomalies

    def _generate_alerts(self, data_type: DataSourceType, anomalies: List[str]):
        """生成告警"""
        if not anomalies:
            return

        alert_key = f"{data_type.value}_quality"
        now = datetime.now()

        # 检查告警冷却时间
        if alert_key in self.alerts_sent:
            last_alert = self.alerts_sent[alert_key]
            cooldown = timedelta(minutes=self.config.alert_cooldown_minutes)
            if now - last_alert < cooldown:
                return  # 冷却时间内不重复告警

        # 发送告警
        alert_message = f"数据质量告警 - {data_type.value}: {'; '.join(anomalies)}"
        print(f"WARNING: {alert_message}")

        # 更新告警时间
        self.alerts_sent[alert_key] = now

    def _generate_recommendations(self, data_type: DataSourceType,
                                metrics: QualityMetrics,
                                validation_result: Dict[str, Any]) -> List[str]:
        """生成建议"""
        recommendations = []

        if metrics.completeness < 0.9:
            recommendations.append("建议检查数据源完整性，补充缺失字段")

        if metrics.validity < 0.8:
            recommendations.append("建议加强数据验证，修复无效值")

        if metrics.consistency < 0.8:
            recommendations.append("建议检查数据一致性，修复逻辑错误")

        if metrics.timeliness < 0.8:
            recommendations.append("建议优化数据更新频率，确保数据新鲜度")

        issues = validation_result.get("issues", [])
        critical_issues = [i for i in issues if i.severity == "critical"]
        if critical_issues:
            recommendations.append(f"紧急修复 {len(critical_issues)} 个严重质量问题")

        return recommendations
```

---

## 📋 验收标准

### 9.1 功能验收标准

#### 9.1.1 数据适配器验收
- [x] 支持16种数据源适配器 (MiniQMT, 中国股市, 加密货币, 新闻等)
- [x] 支持多种通信协议 (REST, WebSocket, TCP)
- [x] 支持连接池和限流机制
- [x] 支持数据源动态注册和发现
- [x] 支持适配器热更新和扩展

#### 9.1.2 PostgreSQL数据加载器验收 [新增]

**基础功能验收:**
- [x] 数据库连接池管理 (psycopg2.pool)
- [x] 环境变量配置读取
- [x] SQL查询执行与参数绑定
- [x] 股票日线数据加载
- [x] 多股票批量加载
- [x] 连接验证与健康检查

**第一阶段 - 基础增强验收:**
- [x] 自动重试机制 (指数退避策略)
  - [x] 可配置重试次数 (默认3次)
  - [x] 可配置退避因子 (默认2.0)
  - [x] 重试日志记录
- [x] 内存缓存集成
  - [x] LRU淘汰策略
  - [x] 可配置缓存过期时间 (默认300秒)
  - [x] 缓存大小限制 (最大1000条)
  - [x] 缓存统计信息

**第二阶段 - 质量与监控验收:**
- [x] 数据质量检查集成
  - [x] 完整性检查
  - [x] 准确性检查
  - [x] 一致性检查
  - [x] 可配置质量阈值 (默认0.7)
  - [x] 质量警告机制
- [x] 性能监控集成
  - [x] 加载时间监控
  - [x] 成功率监控
  - [x] 性能统计信息

**第三阶段 - 数据湖与统一接口验收:**
- [x] 数据湖集成
  - [x] 加载数据到数据湖
  - [x] 从数据湖加载数据
  - [x] 智能同步机制 (带缓存检查)
  - [x] 分区管理支持
  - [x] 元数据记录
- [x] 统一接口实现
  - [x] 标准 `load_data()` 接口
  - [x] 标准 `get_loader_metadata()` 接口
  - [x] 优先从数据湖加载
  - [x] 向后兼容

#### 9.1.2 数据湖验收
- [ ] 支持分层存储架构 (热/温/冷数据)
- [ ] 支持多种分区策略 (日期/哈希/范围)
- [ ] 支持多种压缩格式 (Parquet/CSV/JSON)
- [ ] 支持元数据管理和数据血缘
- [ ] 支持数据版本控制和回滚

#### 9.1.3 缓存系统验收
- [ ] 支持四级缓存架构 (内存/Redis/磁盘/数据湖)
- [ ] 支持多种缓存策略 (LRU/LFU/TTL)
- [ ] 支持缓存命中率统计 (>90%)
- [ ] 支持智能预热和预加载
- [ ] 支持分布式缓存同步

#### 9.1.4 质量监控验收
- [ ] 支持实时质量检测和批量检测
- [ ] 支持5项核心质量指标计算
- [ ] 支持异常检测和智能告警
- [ ] 支持自动修复和人工审核
- [ ] 支持质量报告生成和趋势分析

#### 9.1.5 性能监控验收
- [ ] 支持多维度性能指标收集
- [ ] 支持实时性能监控和告警
- [ ] 支持性能数据导出和分析
- [ ] 支持历史性能数据查询
- [ ] 支持性能基准线和趋势分析

### 9.2 性能验收标准

#### 9.2.1 数据访问性能
| 操作类型 | 平均响应时间 | 95%响应时间 | 99%响应时间 | 目标值 |
|---------|-------------|-------------|-------------|--------|
| 缓存读取 | <1ms | <5ms | <10ms | ✅ |
| 数据湖查询 | <50ms | <200ms | <500ms | ✅ |
| 数据适配器 | <100ms | <500ms | <1000ms | ✅ |
| 质量检查 | <200ms | <1000ms | <2000ms | ✅ |

#### 9.2.2 数据处理能力
| 指标 | 目标值 | 验收标准 |
|------|--------|----------|
| 数据加载速度 | >100MB/s | ✅ |
| 并发处理数 | >1000 | ✅ |
| 缓存命中率 | >90% | ✅ |
| 质量检测准确率 | >95% | ✅ |

#### 9.2.3 高可用性标准
| 指标 | 目标值 | 验收标准 |
|------|--------|----------|
| 数据一致性 | 100% | ✅ |
| 故障恢复时间 | <30秒 | ✅ |
| 系统可用性 | >99.9% | ✅ |
| 数据完整性 | 100% | ✅ |

### 9.3 质量验收标准

#### 9.3.1 数据质量标准
- [ ] 数据完整性: >95%
- [ ] 数据准确性: >98%
- [ ] 数据一致性: >90%
- [ ] 数据时效性: >85%
- [ ] 数据有效性: >99%

#### 9.3.2 系统质量标准
- [ ] 单元测试覆盖率: >85%
- [ ] 集成测试通过率: >95%
- [ ] 性能测试达标率: >90%
- [ ] 安全扫描通过: 0高危漏洞
- [ ] 代码规范符合率: >95%

---

## 🔗 相关文档

- [系统架构总览](ARCHITECTURE_OVERVIEW.md)
- [数据管理层代码审查报告](DATA_LAYER_ARCHITECTURE_REVIEW_REPORT.md)
- [数据适配器设计文档](data_adapter_design.md)
- [数据湖设计文档](data_lake_design.md)
- [缓存系统设计文档](cache_system_design.md)
- [质量监控系统设计](quality_monitoring_design.md)
- [性能监控系统设计](performance_monitoring_design.md)

---

## 📞 技术支持

- **架构师**: [架构师姓名]
- **技术负责人**: [技术负责人姓名]
- **数据负责人**: [数据负责人姓名]
- **质量负责人**: [质量负责人姓名]

---

## 📝 版本历史

| 版本 | 日期 | 主要变更 | 变更人 |
|-----|------|---------|--------|
| v1.0 | 2024-12-01 | 初始版本，数据管理层架构设计 | [架构师] |
| v2.0 | 2024-12-15 | 更新为基于实际代码结构的完整设计 | [架构师] |

---

*本文档基于RQA2025量化交易系统数据管理层的实际代码结构和架构需求编写，旨在为系统的数据基础设施提供完整的技术规范和实施指南。*
