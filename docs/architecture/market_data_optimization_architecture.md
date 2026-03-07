# 市场数据获取优化架构文档

**版本**: v1.0  
**创建日期**: 2026-02-21  
**作者**: AI Assistant

---

## 概述

本文档描述了RQA2025量化交易系统的市场数据获取优化架构，包括四个主要阶段的实现：

1. **Phase 1**: 数据采集优化
2. **Phase 2**: 多股票支持
3. **Phase 3**: 实时数据集成
4. **Phase 4**: 信号验证监控

---

## 架构总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Market Data Optimization Architecture                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Phase 4: Signal Validation & Monitoring          │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │   │
│  │  │   Signal    │  │   Signal    │  │   Signal    │                │   │
│  │  │ Validation  │  │   Filter    │  │   Monitor   │                │   │
│  │  │   Engine    │  │             │  │             │                │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                │   │
│  │         └─────────────────┼─────────────────┘                       │   │
│  └───────────────────────────┼─────────────────────────────────────────┘   │
│                              │                                              │
│  ┌───────────────────────────┼─────────────────────────────────────────┐   │
│  │              Phase 3: Real-Time Data Integration                    │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │   │
│  │  │  Real-Time  │  │  Real-Time  │  │  WebSocket  │                │   │
│  │  │   Router    │  │   Signal    │  │  Publisher  │                │   │
│  │  │             │  │ Integration │  │             │                │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                │   │
│  │         └─────────────────┼─────────────────┘                       │   │
│  └───────────────────────────┼─────────────────────────────────────────┘   │
│                              │                                              │
│  ┌───────────────────────────┼─────────────────────────────────────────┐   │
│  │              Phase 2: Multi-Stock Support                           │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │   │
│  │  │   Strategy  │  │   Symbol    │  │   Multi     │                │   │
│  │  │   Config    │  │   Mapping   │  │   Stock     │                │   │
│  │  │   Parser    │  │   Service   │  │   Manager   │                │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                │   │
│  │         └─────────────────┼─────────────────┘                       │   │
│  └───────────────────────────┼─────────────────────────────────────────┘   │
│                              │                                              │
│  ┌───────────────────────────┼─────────────────────────────────────────┐   │
│  │              Phase 1: Data Collection Optimization                  │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │   │
│  │  │   Data      │  │  Enhanced   │  │    Data     │                │   │
│  │  │ Collection  │  │   AKShare   │  │   Quality   │                │   │
│  │  │ Orchestrator│  │  Collector  │  │   Checker   │                │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                │   │
│  │         └─────────────────┼─────────────────┘                       │   │
│  └───────────────────────────┼─────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Unified Data Store                              │   │
│  │              (PostgreSQL + Redis Cache)                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: 数据采集优化

### 1.1 数据采集协调器 (DataCollectionOrchestrator)

**文件**: `src/data/collectors/data_collection_orchestrator.py`

**职责**:
- 管理多个数据采集器
- 协调采集任务调度
- 处理采集失败重试
- 监控采集状态

**核心类**:
```python
class DataCollectionOrchestrator:
    def register_collector(self, name: str, collector: Any) -> bool
    def schedule_collection(self, symbols: List[str], frequency: str) -> List[str]
    def execute_task(self, task_id: str) -> bool
    def get_status(self) -> Dict[str, Any]
```

**使用示例**:
```python
from src.data.collectors.data_collection_orchestrator import get_data_collection_orchestrator

orchestrator = get_data_collection_orchestrator()
orchestrator.register_collector("akshare", akshare_collector)
task_ids = orchestrator.schedule_collection(["000001", "000002"], "daily")
```

### 1.2 增强版AKShare采集器 (EnhancedAKShareCollector)

**文件**: `src/data/collectors/enhanced_akshare_collector.py`

**职责**:
- 支持增量采集
- 支持多股票批量采集
- 数据质量检查
- 自动补全缺失数据

**核心类**:
```python
class EnhancedAKShareCollector:
    def collect_stock_data_incremental(self, symbol: str) -> Optional[List[Dict]]
    def collect_multiple_stocks(self, symbols: List[str]) -> Dict[str, List[Dict]]
    def save_to_database(self, data: List[Dict], symbol: str) -> bool
    def fill_missing_data(self, symbol: str, expected_dates: List[str]) -> bool
```

**数据质量检查规则**:
1. 价格必须为正数
2. 成交量必须为正数
3. 最高价不能低于最低价
4. 无重复日期
5. 日期连续性检查

---

## Phase 2: 多股票支持

### 2.1 策略配置解析器 (StrategyConfigParser)

**文件**: `src/data/strategy_config_parser.py`

**职责**:
- 解析策略配置文件（YAML/JSON）
- 提取股票代码列表
- 支持动态配置更新

**核心类**:
```python
class StrategyConfigParser:
    def parse_config(self, config_path: Union[str, Path]) -> Optional[StrategyConfig]
    def get_symbols_for_strategy(self, strategy_id: str) -> List[str]
    def create_default_config(self, strategy_id: str, symbols: List[str]) -> StrategyConfig
```

**配置示例** (`config/strategies/default_strategy.yaml`):
```yaml
strategy_id: default_strategy
strategy_name: 默认策略
symbols:
  - "000001"  # 平安银行
  - "000002"  # 万科A
  - "000858"  # 五粮液
parameters:
  lookback_period: 20
  threshold: 0.05
```

### 2.2 股票代码映射服务 (SymbolMappingService)

**文件**: `src/data/symbol_mapping_service.py`

**职责**:
- 策略到股票代码的映射
- 股票代码到策略的反向映射
- 支持多对多关系

**核心类**:
```python
class SymbolMappingService:
    def register_mapping(self, strategy_id: str, symbols: List[str]) -> bool
    def get_symbols_for_strategy(self, strategy_id: str) -> List[str]
    def get_strategies_for_symbol(self, symbol: str) -> List[str]
```

### 2.3 多股票数据管理器 (MultiStockDataManager)

**文件**: `src/data/multi_stock_data_manager.py`

**职责**:
- 从策略配置获取股票代码
- 批量获取多股票数据
- 多级缓存支持

**核心类**:
```python
class MultiStockDataManager:
    def get_data_for_strategy(self, strategy_id: str) -> Dict[str, pd.DataFrame]
    def get_batch_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]
    def register_strategy_mapping(self, strategy_id: str, symbols: List[str]) -> bool
```

---

## Phase 3: 实时数据集成

### 3.1 实时数据路由器 (RealtimeDataRouter)

**文件**: `src/data/realtime_data_router.py`

**职责**:
- 多数据源聚合
- 数据去重和合并
- 优先级路由
- 实时数据分发

**核心类**:
```python
class RealtimeDataRouter:
    def register_data_source(self, name: str, handler: Callable) -> bool
    def subscribe(self, symbol: str, callback: Callable)
    def route_data(self, data: RealtimeMarketData) -> bool
    def get_latest_data(self, symbol: str) -> Optional[RealtimeMarketData]
```

### 3.2 实时信号集成 (RealtimeSignalIntegration)

**文件**: `src/data/realtime_signal_integration.py`

**职责**:
- 实时数据订阅和处理
- 实时信号生成
- WebSocket推送
- 信号缓存和去重

**核心类**:
```python
class RealtimeSignalIntegration:
    def start(self)
    def stop(self)
    def register_signal_generator(self, strategy_id: str, generator: Callable)
    def register_websocket_callback(self, callback: Callable)
```

### 3.3 WebSocket发布器 (WebSocketPublisher)

**文件**: `src/gateway/web/websocket_publisher.py`

**职责**:
- WebSocket连接管理
- 实时数据推送
- 订阅管理
- 广播和单播支持

**核心类**:
```python
class WebSocketPublisher:
    def on_connect(self, sid: str, environ: dict)
    def on_disconnect(self, sid: str)
    def subscribe_symbol(self, sid: str, symbol: str)
    def publish_to_symbol(self, symbol: str, data: Any)
    def broadcast(self, data: Any, event: str = 'message')
```

---

## Phase 4: 信号验证监控

### 4.1 信号验证引擎 (SignalValidationEngine)

**文件**: `src/trading/signal/signal_validation_engine.py`

**职责**:
- 历史回测验证
- 质量评分算法
- 风险评分算法
- 综合评分计算

**核心类**:
```python
class SignalValidationEngine:
    def validate_signal(self, signal: Dict, market_data: pd.DataFrame) -> SignalValidationResult
```

**评分算法**:

**质量评分** (4个维度):
- 信号强度 (30%): 基于置信度
- 市场条件 (25%): 基于波动率
- 成交量确认 (20%): 基于成交量比率
- 趋势一致性 (25%): 基于价格与均线关系

**风险评分** (4个维度):
- 波动率风险 (30%)
- 回撤风险 (30%)
- 集中度风险 (20%)
- 流动性风险 (20%)

**综合评分**:
- 质量评分 × 40%
- 回测评分 × 35%
- (100 - 风险评分) × 25%

### 4.2 信号过滤器 (SignalFilter)

**文件**: `src/trading/signal/signal_filter.py`

**职责**:
- 基于评分的过滤
- 可配置阈值
- 频率限制

**核心类**:
```python
class SignalFilter:
    def filter_signal(self, signal: Dict, validation_result: SignalValidationResult) -> Tuple[bool, str]
    def update_config(self, **kwargs)
```

**默认阈值**:
- 最小综合评分: 60.0
- 最小质量评分: 60.0
- 最大风险评分: 70.0
- 最小置信度: 0.5

### 4.3 信号监控器 (SignalMonitor)

**文件**: `src/trading/signal/signal_monitor.py`

**职责**:
- 指标收集和计算
- 告警引擎
- 监控面板API

**核心类**:
```python
class SignalMonitor:
    def record_signal(self, signal: Dict, validation_result: SignalValidationResult)
    def calculate_current_metrics(self) -> SignalMetrics
    def get_dashboard_data(self) -> Dict[str, Any]
    def register_alert_callback(self, callback: Callable)
```

**默认告警规则**:
1. 低质量信号告警: avg_quality_score < 50
2. 高风险信号告警: avg_risk_score > 80
3. 信号数量激增告警: total_signals > 100

---

## 数据流

### 端到端数据流

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Data       │────▶│   Real-Time  │────▶│   Signal     │
│   Sources    │     │   Router     │     │   Generator  │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                                  ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Monitor    │◀────│   Filter     │◀────│   Validation │
│   Dashboard  │     │              │     │   Engine     │
└──────────────┘     └──────────────┘     └──────────────┘
                                                  │
                                                  ▼
                                         ┌──────────────┐
                                         │   WebSocket  │
                                         │   Publisher  │
                                         └──────────────┘
```

---

## 测试

### 综合测试脚本

**文件**: `tests/test_market_data_optimization_complete.py`

运行所有四个Phase的测试:
```bash
python tests/test_market_data_optimization_complete.py
```

**测试结果** (2026-02-21):
- 总测试数: 11
- 通过: 11
- 失败: 0
- 通过率: 100.0%

---

## 配置文件

### 策略配置

**位置**: `config/strategies/`

**格式**: YAML

**示例**:
```yaml
strategy_id: default_strategy
strategy_name: 默认策略
symbols:
  - "000001"
  - "000002"
description: 默认策略描述
parameters:
  lookback_period: 20
  threshold: 0.05
enabled: true
```

---

## API接口

### 信号监控API

**获取监控面板数据**:
```
GET /api/signal-monitoring/dashboard
```

**响应**:
```json
{
  "timestamp": "2026-02-21T11:15:19",
  "current_metrics": {
    "total_signals": 6,
    "valid_signals": 6,
    "avg_quality_score": 80.0,
    "avg_risk_score": 50.0
  },
  "alert_rules": [...]
}
```

---

## 性能指标

### 目标性能

| 指标 | 目标值 | 实际值 |
|------|--------|--------|
| 数据查询响应时间 | < 100ms | 待测试 |
| 实时数据延迟 | < 1秒 | 待测试 |
| 信号生成延迟 | < 1秒 | 待测试 |
| 缓存命中率 | > 80% | 待测试 |

---

## 部署

### 依赖安装

```bash
pip install pandas numpy pyyaml
```

### 配置环境变量

```bash
# 数据库连接
DATABASE_URL=postgresql://user:password@localhost:5432/rqa2025_prod

# Redis缓存（可选）
REDIS_URL=redis://localhost:6379/0
```

### 启动服务

```bash
# 填充初始数据
python scripts/fill_stock_data.py --default

# 运行综合测试
python tests/test_market_data_optimization_complete.py
```

---

## 扩展组件 (v2.0)

### 国际市场数据适配器

**文件**: `src/data/adapters/international/`

**组件**:
- `base_international_adapter.py`: 国际数据源适配器基类
- `yahoo_finance_adapter.py`: Yahoo Finance适配器（美股、港股、加密货币）
- `alpha_vantage_adapter.py`: Alpha Vantage适配器（美股、外汇、技术指标）

**支持市场**:
- 美股 (US_STOCK)
- 港股 (HK_STOCK)
- 日股 (JP_STOCK)
- 英股 (UK_STOCK)
- 期货 (FUTURES)
- 外汇 (FOREX)
- 加密货币 (CRYPTO)

### 另类数据适配器框架

**文件**: `src/data/adapters/alternative/`

**组件**:
- `base_alternative_adapter.py`: 另类数据适配器基类
- `DataFusionEngine`: 数据融合引擎

**支持数据类型**:
- 社交媒体情绪数据
- 新闻情绪分析数据
- 搜索趋势数据
- 卫星/替代数据

### Level2行情数据适配器

**文件**: `src/data/adapters/professional/level2_market_data_adapter.py`

**功能**:
- 十档/五档盘口数据
- 逐笔成交数据
- 委托队列数据
- 订单簿失衡度计算
- 成交压力计算

### 数据压缩引擎

**文件**: `src/data/compression/advanced_compression_engine.py`

**支持算法**:
- LZ4: 高速压缩
- Snappy: Google Snappy
- Zstandard: 高压缩比
- Gzip: 标准压缩
- Brotli: Web优化

**支持格式**:
- Parquet: 列式存储
- Feather: Apache Feather
- HDF5: HDF5格式

### 智能预处理流水线

**文件**: `src/data/processing/intelligent_preprocessing_pipeline.py`

**功能**:
- 异常值检测（Z-score、IQR、MAD、孤立森林）
- 缺失值填充（均值、中位数、KNN、迭代填充）
- 数据标准化（Standard、MinMax、Robust、Log）
- 预处理质量评估

### 智能缓存预热器

**文件**: `src/data/cache/intelligent_cache_warmer.py`

**功能**:
- 基于历史访问模式的缓存预热
- 机器学习预测模型
- 自适应预热策略（时间/频率/预测/混合）
- 预热效果评估

### 多因子策略框架

**文件**: `src/trading/strategy/advanced/multi_factor_strategy.py`

**支持因子**:
- 价值因子（PE、PB）
- 成长因子
- 质量因子（ROE）
- 动量因子（收益率、趋势）
- 波动率因子
- 流动性因子
- 情绪因子

### 统计套利策略

**文件**: `src/trading/strategy/advanced/statistical_arbitrage_strategy.py`

**功能**:
- 协整性检验（Engle-Granger）
- 配对交易信号生成
- 均值回归策略
- 动态阈值调整

### 策略组合优化器

**文件**: `src/trading/portfolio/strategy_portfolio_optimizer.py`

**优化方法**:
- 风险平价（Risk Parity）
- 均值方差优化（Mean-Variance）
- 等权重（Equal Weight）
- 最小方差（Minimum Variance）
- 最大夏普比率（Maximum Sharpe）

### 自动化特征工程

**文件**: `src/ml/feature_engineering/automated_feature_engineer.py`

**功能**:
- 时序特征提取（滞后、滚动统计）
- 技术指标特征（RSI、MACD、布林带）
- 时间特征提取
- 交叉特征生成
- 特征选择和重要性分析

### XGBoost/LightGBM模型训练器

**文件**: `src/ml/models/xgboost_lightgbm_trainer.py`

**功能**:
- XGBoost模型训练
- LightGBM模型训练
- 超参数自动优化（Optuna）
- GPU支持
- 模型评估和对比

---

## 测试

### 综合测试脚本

**v1.0测试**: `tests/test_market_data_optimization_complete.py`
- 总测试数: 11
- 通过率: 100%

**v2.0测试**: `tests/test_market_data_optimization_v2.py`
- 总测试数: 23
- 通过率: 100%

---

## 变更日志

### v2.0 (2026-02-21)

**新增**:
- 国际市场数据适配器（Yahoo Finance、Alpha Vantage）
- 另类数据适配器框架
- Level2行情数据适配器
- 数据压缩引擎（LZ4、Snappy、Zstandard）
- 智能预处理流水线
- 智能缓存预热器
- 多因子策略框架
- 统计套利策略
- 策略组合优化器
- 自动化特征工程
- XGBoost/LightGBM模型训练器

### v1.0 (2026-02-21)

**新增**:
- Phase 1: 数据采集协调器、增强版AKShare采集器
- Phase 2: 策略配置解析器、股票代码映射服务、多股票数据管理器
- Phase 3: 实时数据路由器、实时信号集成、WebSocket发布器
- Phase 4: 信号验证引擎、信号过滤器、信号监控器
- 综合测试脚本

**改进**:
- 支持增量数据采集
- 多级缓存策略
- 实时数据去重
- 信号质量评分

---

*文档结束*
