# 策略回测历史数据采集 - 技术方案设计

## 📋 设计概述

基于需求规格分析，本文档详细阐述策略回测历史数据采集系统的技术实现方案。采用双轨并行架构，确保历史数据采集不影响日常业务，同时提供高效可靠的数据采集服务。

## 🏗️ 系统架构设计

### 双轨并行架构详解

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           策略回测历史数据采集系统                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐     ┌─────────────────────────────────────────────┐   │
│  │    日常补全轨       │     │        历史数据采集轨 (新增)               │   │
│  │  (增量采集为主)     │     │   (批量历史数据采集为主)                  │   │
│  ├─────────────────────┤     ├─────────────────────────────────────────────┤   │
│  │ • 现有系统保持不变  │     │ • HistoricalDataAcquisitionService       │   │
│  │ • 30-180天补全      │     │ • StrategyBacktestDataWorkflow            │   │
│  │ • 高频小批量        │     │ • 多数据源集成 (AKShare, Yahoo, TuShare) │   │
│  │ • 实时调度          │     │ • 按年分批并发处理                      │   │
│  └─────────────────────┘     └─────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                           统一数据存储与访问层                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ • TimescaleDB: 主数据库 (时序数据优化)                               │   │
│  │   - 按股票代码和时间分区存储                                        │   │
│  │   - 支持高效的历史数据查询                                          │   │
│  │   - 内置时间序列分析函数                                            │   │
│  │                                                                       │   │
│  │ • Redis Cluster: 缓存层 (热点数据加速)                              │   │
│  │   - 缓存最近1年的高频查询数据                                       │   │
│  │   - 支持分布式缓存，扩展性好                                        │   │
│  │   - 提供数据预热和缓存一致性保证                                    │   │
│  │                                                                       │   │
│  │ • MinIO: 对象存储 (大文件存储)                                     │   │
│  │   - 存储原始数据文件和备份数据                                      │   │
│  │   - 支持S3兼容API，易于扩展                                        │   │
│  │   - 提供数据版本管理和生命周期管理                                  │   │
│  │                                                                       │   │
│  │ • 数据标签系统: 区分数据来源和用途                                  │   │
│  │   - source: daily_incremental | historical_bulk                     │   │
│  │   - purpose: backtest | research | production                        │   │
│  │   - quality: verified | raw | repaired                              │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                           统一监控运维体系                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ • Prometheus: 指标收集                                               │   │
│  │   - 采集性能指标: 采集速度、成功率、延迟                           │   │
│  │   - 系统资源指标: CPU、内存、磁盘、网络                            │   │
│  │   - 业务指标: 数据质量、覆盖率、用户满意度                         │   │
│  │                                                                       │   │
│  │ • Grafana: 可视化监控                                               │   │
│  │   - 实时仪表板: 系统状态、采集进度、性能指标                       │   │
│  │   - 历史趋势图: 性能变化、错误统计、资源使用                        │   │
│  │   - 告警面板: 异常情况实时告警                                      │   │
│  │                                                                       │   │
│  │ • ELK Stack: 日志分析                                               │   │
│  │   - Elasticsearch: 日志存储和全文检索                              │   │
│  │   - Logstash: 日志收集和处理                                        │   │
│  │   - Kibana: 日志可视化和分析                                        │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### 双轨日期边界（避免重叠）

日常轨与历史轨在**日期上严格不重叠**，由配置项 `daily_period_days`、`max_history_days` 控制：

| 轨     | 日期范围 | 说明 |
|--------|----------|------|
| **日常轨** | `[today - daily_period_days, today]` | 默认 30 天，与数据源 `default_days` 对齐 |
| **历史轨** | `[hist_start, hist_end]` | `hist_end = today - daily_period_days - 1`，`hist_start = hist_end - max_history_days`（默认约 10 年） |

- 历史轨右端 `hist_end` 与日常轨左端错开 1 天，无交集。
- 历史轨按 `collection_period_days`（90 或 365 天）从 `hist_end` 向左分段，由 `historical_data_scheduler` 的 `_perform_periodic_collection`、`_perform_forced_collection` 及 `_create_segmented_tasks` 统一使用上述边界。

### 核心组件详细设计

#### **轨一：日常补全机制扩展**
继承现有补全系统，增加对历史补全模式的支持：

```
DataComplementScheduler (扩展)
├── 现有模式: MONTHLY, WEEKLY, QUARTERLY, SEMI_ANNUAL
├── 新增模式: FULL_HISTORY, STRATEGY_BACKTEST
└── 新增功能: 长周期补全支持，按年分批处理

BatchComplementProcessor (扩展)
├── 现有功能: 小批量并发处理
├── 新增功能: 超长周期批次管理，系统负载感知
└── 优化功能: 内存使用优化，大数据量处理支持
```

#### **轨二：历史数据采集服务 (新增)**
全新设计的历史数据采集服务：

```
HistoricalDataAcquisitionService (新增)
├── 数据源管理器: AKShare, Yahoo Finance, TuShare集成
├── 并发控制器: asyncio信号量控制，智能调度
├── 质量保障器: 数据验证、清洗、修复算法
├── 存储优化器: 分区存储、索引优化、压缩存储
└── 监控报告器: 实时进度、质量统计、性能指标

StrategyBacktestDataWorkflow (新增)
├── 工作流引擎: 采集任务编排和状态管理
├── 配置解析器: 用户需求转换为技术参数
├── 并行协调器: 多股票并发采集任务协调
├── 结果汇总器: 采集结果统计和质量报告
└── 建议生成器: 基于结果的优化建议生成
```

## 🔧 技术实现方案

### 数据采集层设计

#### **多数据源集成架构**
```python
class DataSourceManager:
    """数据源管理器"""

    def __init__(self):
        self.sources = {
            'akshare': AKShareDataSource(
                priority=1,         # 优先级
                rate_limit=60,      # 每分钟请求限制
                timeout=30,         # 超时时间
                retry_count=3       # 重试次数
            ),
            'yahoo': YahooDataSource(
                priority=2,
                rate_limit=30,
                timeout=60,
                retry_count=5
            ),
            'tushare': TuShareDataSource(
                priority=3,
                rate_limit=100,
                timeout=45,
                retry_count=3
            )
        }

    async def fetch_stock_data(self, symbol: str, start_date: str,
                              end_date: str) -> Optional[pd.DataFrame]:
        """智能数据源选择和获取"""
        # 1. 根据数据可用性和优先级选择最佳数据源
        best_source = await self._select_best_source(symbol, start_date, end_date)

        # 2. 按年份分批获取，避免单次请求过大
        yearly_batches = self._split_by_years(start_date, end_date)

        all_data = []
        for year_start, year_end in yearly_batches:
            try:
                # 尝试从最佳数据源获取
                data = await best_source.fetch_year_data(symbol, year_start, year_end)
                all_data.append(data)
            except Exception as e:
                # 如果失败，尝试备用数据源
                for backup_source in self._get_backup_sources(best_source):
                    try:
                        data = await backup_source.fetch_year_data(symbol, year_start, year_end)
                        all_data.append(data)
                        break
                    except Exception:
                        continue

        # 3. 数据合并和排序
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            combined_data = combined_data.sort_values('date').drop_duplicates()
            return combined_data

        return None
```

#### **并发控制机制**
```python
class ConcurrencyController:
    """并发控制器"""

    def __init__(self, max_concurrent: int = 10, max_per_host: int = 2):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.host_semaphores = {}  # 按主机限制并发
        self.max_per_host = max_per_host

        # 动态调整参数
        self.current_load = 0
        self.adjustment_interval = 60  # 每分钟调整一次

    async def acquire_slot(self, host: str) -> bool:
        """获取并发执行槽位"""
        # 全局并发控制
        await self.semaphore.acquire()

        # 按主机并发控制
        if host not in self.host_semaphores:
            self.host_semaphores[host] = asyncio.Semaphore(self.max_per_host)

        try:
            await self.host_semaphores[host].acquire()
            self.current_load += 1
            return True
        except Exception:
            self.semaphore.release()
            return False

    def release_slot(self, host: str):
        """释放并发执行槽位"""
        if host in self.host_semaphores:
            self.host_semaphores[host].release()

        self.semaphore.release()
        self.current_load -= 1

    async def adjust_concurrency(self):
        """动态调整并发参数"""
        while True:
            await asyncio.sleep(self.adjustment_interval)

            # 根据系统负载调整并发数
            system_load = await self._get_system_load()
            if system_load > 0.8:  # 高负载
                new_limit = max(1, self.semaphore._value - 2)
                self.semaphore = asyncio.Semaphore(new_limit)
            elif system_load < 0.3:  # 低负载
                new_limit = min(20, self.semaphore._value + 1)
                self.semaphore = asyncio.Semaphore(new_limit)
```

### 数据存储层设计

#### **TimescaleDB存储优化**
```sql
-- 股票价格数据表 (按股票代码和时间分区)
CREATE TABLE IF NOT EXISTS stock_price_data (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    open_price DECIMAL(10,4),
    high_price DECIMAL(10,4),
    low_price DECIMAL(10,4),
    close_price DECIMAL(10,4),
    volume BIGINT,
    amount DECIMAL(20,4),
    adj_close DECIMAL(10,4),

    -- 数据标签
    data_source VARCHAR(50),      -- 数据来源: akshare, yahoo, tushare
    collection_type VARCHAR(20),  -- 采集类型: historical_bulk, daily_incremental
    data_quality VARCHAR(20),     -- 数据质量: verified, raw, repaired

    -- 索引优化
    PRIMARY KEY (symbol, time)
) PARTITION BY RANGE (time);

-- 按月分区 (提高查询性能)
SELECT create_hypertable('stock_price_data', 'time', partitioning_column => 'symbol', number_partitions => 100);

-- 复合索引
CREATE INDEX idx_stock_price_symbol_time ON stock_price_data (symbol, time DESC);
CREATE INDEX idx_stock_price_time_symbol ON stock_price_data (time DESC, symbol);
CREATE INDEX idx_stock_price_source ON stock_price_data (data_source);
CREATE INDEX idx_stock_price_quality ON stock_price_data (data_quality);

-- 数据压缩 (节省存储空间)
ALTER TABLE stock_price_data SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol',
    timescaledb.compress_orderby = 'time DESC'
);

-- 自动压缩策略 (7天后压缩)
SELECT add_compression_policy('stock_price_data', INTERVAL '7 days');
```

#### **Redis缓存策略**
```python
class RedisCacheManager:
    """Redis缓存管理器"""

    def __init__(self, redis_client):
        self.redis = redis_client
        self.cache_ttl = {
            'recent_data': 3600,      # 最近数据: 1小时
            'hot_stocks': 7200,       # 热门股票: 2小时
            'index_data': 1800,       # 指数数据: 30分钟
            'metadata': 86400         # 元数据: 24小时
        }

    async def cache_stock_data(self, symbol: str, data: pd.DataFrame,
                              data_type: str = 'price'):
        """缓存股票数据"""
        cache_key = f"stock:{data_type}:{symbol}"

        # 序列化数据
        data_json = data.to_json(orient='records', date_format='iso')

        # 设置缓存 (带过期时间)
        ttl = self.cache_ttl.get('recent_data', 3600)
        await self.redis.setex(cache_key, ttl, data_json)

        # 更新缓存统计
        await self._update_cache_stats(cache_key, len(data))

    async def get_stock_data(self, symbol: str, data_type: str = 'price') -> Optional[pd.DataFrame]:
        """获取缓存的股票数据"""
        cache_key = f"stock:{data_type}:{symbol}"

        # 尝试从缓存获取
        cached_data = await self.redis.get(cache_key)
        if cached_data:
            # 反序列化数据
            data_list = json.loads(cached_data)
            df = pd.DataFrame(data_list)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()

            # 更新访问统计
            await self._update_access_stats(cache_key)

            return df

        return None

    async def preload_hot_data(self):
        """预加载热点数据"""
        # 获取热门股票列表 (基于访问频率)
        hot_symbols = await self._get_hot_symbols(limit=100)

        # 并发预加载
        semaphore = asyncio.Semaphore(5)  # 限制并发

        async def preload_symbol(symbol):
            async with semaphore:
                # 检查缓存是否存在
                if not await self.redis.exists(f"stock:price:{symbol}"):
                    # 从数据库加载并缓存
                    data = await self._load_from_database(symbol)
                    if data is not None:
                        await self.cache_stock_data(symbol, data)

        await asyncio.gather(*[preload_symbol(symbol) for symbol in hot_symbols])
```

### 数据质量保障体系

#### **多层次质量检查**
```python
class DataQualityManager:
    """数据质量管理器"""

    def __init__(self):
        self.checkers = {
            'completeness': CompletenessChecker(),
            'accuracy': AccuracyChecker(),
            'consistency': ConsistencyChecker(),
            'timeliness': TimelinessChecker()
        }
        self.repairers = {
            'missing_data': MissingDataRepairer(),
            'outlier_data': OutlierDataRepairer(),
            'inconsistent_data': InconsistentDataRepairer()
        }

    async def validate_and_repair(self, data: pd.DataFrame,
                                symbol: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """数据验证和修复"""
        quality_report = {
            'original_records': len(data),
            'quality_scores': {},
            'issues_found': [],
            'repairs_applied': [],
            'final_records': 0
        }

        # 1. 完整性检查
        completeness_score = await self.checkers['completeness'].check(data)
        quality_report['quality_scores']['completeness'] = completeness_score

        if completeness_score < 0.9:
            # 尝试修复缺失数据
            data = await self.repairers['missing_data'].repair(data, symbol)
            quality_report['repairs_applied'].append('missing_data_repair')

        # 2. 准确性检查
        accuracy_score = await self.checkers['accuracy'].check(data)
        quality_report['quality_scores']['accuracy'] = accuracy_score

        if accuracy_score < 0.95:
            # 检测和修复异常值
            data, outliers = await self.repairers['outlier_data'].detect_and_repair(data)
            if outliers:
                quality_report['issues_found'].append(f'outliers_detected: {len(outliers)}')
                quality_report['repairs_applied'].append('outlier_repair')

        # 3. 一致性检查
        consistency_score = await self.checkers['consistency'].check(data)
        quality_report['quality_scores']['consistency'] = consistency_score

        if consistency_score < 0.95:
            # 修复不一致数据
            data = await self.repairers['inconsistent_data'].repair(data)
            quality_report['repairs_applied'].append('consistency_repair')

        # 4. 时效性检查
        timeliness_score = await self.checkers['timeliness'].check(data)
        quality_report['quality_scores']['timeliness'] = timeliness_score

        # 计算综合质量分数
        weights = {'completeness': 0.3, 'accuracy': 0.3, 'consistency': 0.2, 'timeliness': 0.2}
        overall_score = sum(
            quality_report['quality_scores'].get(metric, 0) * weight
            for metric, weight in weights.items()
        )
        quality_report['quality_scores']['overall'] = overall_score
        quality_report['final_records'] = len(data)

        return data, quality_report
```

### API接口设计

#### **采集服务API**
```python
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="策略回测历史数据采集服务")

class DataAcquisitionRequest(BaseModel):
    symbols: List[str]
    start_date: str
    end_date: str
    data_types: List[str] = ["price", "volume"]
    priority: str = "normal"
    quality_threshold: float = 0.85
    max_concurrent: int = 5

class AcquisitionStatus(BaseModel):
    task_id: str
    status: str
    progress: float
    estimated_completion: Optional[str]
    result_summary: Optional[dict]

@app.post("/api/v1/acquisition/start", response_model=dict)
async def start_data_acquisition(request: DataAcquisitionRequest, background_tasks: BackgroundTasks):
    """启动数据采集任务"""
    try:
        # 参数验证
        if not request.symbols:
            raise HTTPException(status_code=400, detail="股票代码列表不能为空")

        if len(request.symbols) > 1000:
            raise HTTPException(status_code=400, detail="单次采集股票数量不能超过1000只")

        # 创建采集任务
        task_id = await acquisition_service.create_acquisition_task(
            symbols=request.symbols,
            start_date=request.start_date,
            end_date=request.end_date,
            data_types=request.data_types,
            priority=request.priority,
            quality_threshold=request.quality_threshold,
            max_concurrent=request.max_concurrent
        )

        # 启动后台采集
        background_tasks.add_task(
            acquisition_service.execute_acquisition_task,
            task_id
        )

        return {
            "task_id": task_id,
            "status": "accepted",
            "message": "数据采集任务已启动"
        }

    except Exception as e:
        logger.error(f"启动采集任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"任务启动失败: {str(e)}")

@app.get("/api/v1/acquisition/{task_id}/status", response_model=AcquisitionStatus)
async def get_acquisition_status(task_id: str):
    """获取采集任务状态"""
    try:
        status = await acquisition_service.get_task_status(task_id)
        if not status:
            raise HTTPException(status_code=404, detail="任务不存在")

        return AcquisitionStatus(**status)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取任务状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")

@app.get("/api/v1/data/stock/{symbol}")
async def get_stock_data(
    symbol: str,
    start_date: str,
    end_date: str,
    data_type: str = "price",
    adjusted: bool = True
):
    """获取股票历史数据"""
    try:
        # 参数验证
        if not symbol or not start_date or not end_date:
            raise HTTPException(status_code=400, detail="参数不完整")

        # 获取数据
        data = await data_service.get_stock_history(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            data_type=data_type,
            adjusted=adjusted
        )

        if data is None or data.empty:
            raise HTTPException(status_code=404, detail="未找到相关数据")

        return {
            "symbol": symbol,
            "data_type": data_type,
            "start_date": start_date,
            "end_date": end_date,
            "record_count": len(data),
            "data": data.to_dict('records')
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取股票数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"数据获取失败: {str(e)}")
```

## 📊 数据流设计

### 完整数据采集流程

```
用户请求 → API网关 → 请求验证 → 任务创建 → 任务入队 → 后台处理
    ↓         ↓         ↓         ↓         ↓         ↓
异常响应 ← 参数校验 ← 权限检查 ← 资源检查 ← 队列满载 ← 任务冲突
```

### 详细处理流程

#### **任务创建阶段**
1. **参数验证**: 验证股票代码、日期范围、数据类型
2. **资源评估**: 评估所需计算资源和存储空间
3. **任务分解**: 将大任务分解为可管理的子任务
4. **优先级分配**: 根据用户类型和需求分配优先级

#### **并发执行阶段**
1. **队列调度**: 基于优先级和资源可用性调度任务
2. **数据源选择**: 智能选择最佳数据源和备用方案
3. **分批采集**: 按时间段分批采集，避免单次请求过大
4. **并发控制**: 控制全局和按主机的并发数量

#### **数据处理阶段**
1. **实时验证**: 边采集边进行基础数据验证
2. **质量检查**: 多维度数据质量评估
3. **异常修复**: 自动检测和修复常见数据问题
4. **格式标准化**: 统一数据格式和字段名称

#### **存储和索引阶段**
1. **分区存储**: 按股票代码和时间分区存储
2. **索引建立**: 创建查询优化索引
3. **缓存更新**: 更新热点数据缓存
4. **备份同步**: 同步到备份存储

#### **结果汇总阶段**
1. **进度统计**: 实时更新采集进度
2. **质量报告**: 生成详细的质量评估报告
3. **性能分析**: 统计采集性能和资源使用
4. **优化建议**: 基于结果生成改进建议

## 🔒 安全与权限设计

### 数据安全保障

#### **传输安全**
- **HTTPS**: 所有API调用使用HTTPS加密
- **API密钥**: 请求身份验证和授权
- **请求限制**: 防止恶意高频请求

#### **存储安全**
- **数据加密**: 敏感数据加密存储
- **访问控制**: 基于角色的细粒度权限控制
- **审计日志**: 完整的数据访问审计记录

#### **合规要求**
- **数据脱敏**: 个人隐私数据自动脱敏
- **留存策略**: 符合监管要求的數據留存策略
- **跨境传输**: 敏感数据跨境传输合规处理

### 权限体系设计

#### **用户角色**
```python
class UserRole:
    ADMIN = "admin"           # 系统管理员
    DATA_MANAGER = "data_manager"  # 数据管理员
    RESEARCHER = "researcher"      # 策略研究员
    ANALYST = "analyst"            # 数据分析师
    VIEWER = "viewer"              # 只读用户
```

#### **权限控制**
```python
PERMISSIONS = {
    "data_acquisition": {
        "start_task": ["admin", "data_manager", "researcher"],
        "cancel_task": ["admin", "data_manager"],
        "view_status": ["admin", "data_manager", "researcher", "analyst", "viewer"],
        "modify_config": ["admin", "data_manager"]
    },
    "data_access": {
        "read_price_data": ["admin", "data_manager", "researcher", "analyst", "viewer"],
        "read_fundamental_data": ["admin", "data_manager", "researcher", "analyst"],
        "export_data": ["admin", "data_manager", "researcher"],
        "bulk_download": ["admin", "data_manager"]
    },
    "system_management": {
        "view_metrics": ["admin", "data_manager"],
        "manage_cache": ["admin", "data_manager"],
        "system_config": ["admin"],
        "emergency_stop": ["admin"]
    }
}
```

## 📈 扩展性设计

### 水平扩展策略

#### **数据源扩展**
```python
class DataSourcePlugin:
    """数据源插件接口"""

    @abstractmethod
    async def fetch_historical_data(self, symbol: str, start_date: str,
                                  end_date: str) -> pd.DataFrame:
        """获取历史数据"""
        pass

    @abstractmethod
    def get_rate_limit(self) -> int:
        """获取请求频率限制"""
        pass

    @abstractmethod
    def get_supported_data_types(self) -> List[str]:
        """获取支持的数据类型"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """健康检查"""
        pass
```

#### **存储后端扩展**
```python
class StorageBackend:
    """存储后端接口"""

    @abstractmethod
    async def store_stock_data(self, symbol: str, data: pd.DataFrame,
                             data_type: str) -> bool:
        """存储股票数据"""
        pass

    @abstractmethod
    async def get_stock_data(self, symbol: str, start_date: str,
                           end_date: str, data_type: str) -> pd.DataFrame:
        """获取股票数据"""
        pass

    @abstractmethod
    async def optimize_storage(self, symbol: str) -> bool:
        """优化存储"""
        pass
```

### 垂直扩展策略

#### **数据类型扩展**
- **衍生数据**: 技术指标、统计数据
- **新闻数据**: 文本情感分析、事件提取
- **社交数据**: 社交媒体情绪分析
- **宏观数据**: 经济指标、政策数据

#### **分析功能扩展**
- **实时分析**: 实时数据流分析
- **批量分析**: 大规模历史数据分析
- **机器学习**: AI驱动的模式识别
- **预测分析**: 基于历史数据的预测

## 🎯 部署方案

### 容器化部署

#### **Docker Compose配置**
```yaml
version: '3.8'
services:
  historical-collector:
    image: strategy-backtest-collector:latest
    environment:
      - REDIS_URL=redis://redis:6379
      - POSTGRES_URL=postgresql://user:pass@postgres:5432/db
      - MINIO_ENDPOINT=minio:9000
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs
    networks:
      - backtest-network
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G

  redis:
    image: redis:7-alpine
    command: redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    networks:
      - backtest-network

  postgres:
    image: timescale/timescaledb:latest-pg15
    environment:
      POSTGRES_DB: backtest_db
      POSTGRES_USER: backtest_user
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - backtest-network

  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ACCESS_KEY: backtest_access
      MINIO_SECRET_KEY: backtest_secret
    volumes:
      - minio_data:/data
    networks:
      - backtest-network

networks:
  backtest-network:
    driver: bridge

volumes:
  redis_data:
  postgres_data:
  minio_data:
```

### Kubernetes部署

#### **Deployment配置**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: historical-collector
  namespace: backtest
spec:
  replicas: 3
  selector:
    matchLabels:
      app: historical-collector
  template:
    metadata:
      labels:
        app: historical-collector
    spec:
      containers:
      - name: collector
        image: strategy-backtest-collector:latest
        resources:
          requests:
            cpu: 1000m
            memory: 2Gi
          limits:
            cpu: 2000m
            memory: 4Gi
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        - name: POSTGRES_URL
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: url
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

---

**技术方案版本**: v1.0
**制定日期**: 2026-01-24
**设计原则**: 双轨并行、性能隔离、质量保障、扩展性优先
**核心技术**: Python asyncio、TimescaleDB、Redis、FastAPI
**架构特点**: 微服务架构、事件驱动、云原生设计