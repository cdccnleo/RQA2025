# 市场数据获取优化开发计划

## 目标
按照后续优化建议，完善市场数据获取和信号生成功能。

## 优化方向

### 1. 数据采集 - 填充 akshare_stock_data 表
**优先级**: 高
**目标**: 确保数据库中有足够的市场数据供信号生成器使用

**具体任务**:
- [ ] 检查数据采集服务状态
- [ ] 启动 AKShare 数据采集
- [ ] 采集默认股票代码（000001 平安银行）的历史数据
- [ ] 验证数据已正确存入数据库
- [ ] 设置定期采集任务

**技术方案**:
```python
# 启动数据采集服务
# 1. 使用现有的数据采集器
# 2. 配置采集参数（股票代码、时间范围、频率）
# 3. 启动采集任务
# 4. 验证数据入库
```

### 2. 多股票支持 - 从策略配置获取股票代码
**优先级**: 高
**目标**: 支持根据策略配置动态获取不同股票的数据

**具体任务**:
- [ ] 修改 MarketDataService 支持多股票查询
- [ ] 从策略配置读取股票代码
- [ ] 实现股票代码到策略的映射
- [ ] 支持批量数据查询
- [ ] 优化多股票数据缓存

**技术方案**:
```python
# 修改 _get_market_data_for_signal_generation()
def _get_market_data_for_signal_generation(strategy_id=None):
    # 1. 如果提供了 strategy_id，从策略配置获取股票代码
    # 2. 否则使用默认股票代码
    # 3. 支持多股票数据获取
    # 4. 返回股票代码到数据的映射
```

### 3. 实时数据集成 - 支持实时信号生成
**优先级**: 中
**目标**: 集成实时数据流，支持实时信号生成

**具体任务**:
- [ ] 检查 RealTimeDataStream 状态
- [ ] 集成实时数据流到 TradingSignalService
- [ ] 实现实时数据缓存
- [ ] 修改信号生成器支持实时数据
- [ ] 添加实时信号推送机制

**技术方案**:
```python
# 实时数据流集成
# 1. 订阅 RealTimeDataStream
# 2. 实时更新市场数据缓存
# 3. 触发实时信号生成
# 4. 推送信号到前端
```

### 4. 信号验证和监控 - 添加信号质量检查
**优先级**: 中
**目标**: 添加信号验证机制和监控功能

**具体任务**:
- [ ] 实现信号质量评分算法
- [ ] 添加信号历史回测验证
- [ ] 实现信号准确性统计
- [ ] 添加信号监控面板
- [ ] 实现信号告警机制

**技术方案**:
```python
# 信号验证和监控
# 1. 信号质量评分（基于历史表现）
# 2. 信号回测验证
# 3. 信号准确性统计
# 4. 监控面板展示
# 5. 异常信号告警
```

## 实施阶段

### Phase 1: 数据采集（1-2天）
**目标**: 确保数据库中有市场数据

**任务清单**:
1. 检查数据采集服务状态
2. 配置 AKShare 数据源
3. 启动数据采集任务
4. 验证数据入库
5. 设置定时采集

**验收标准**:
- `akshare_stock_data` 表中有数据
- 至少包含默认股票代码（000001）的数据
- 数据时间范围覆盖最近30天

### Phase 2: 多股票支持（2-3天）
**目标**: 支持策略配置驱动的多股票数据获取

**任务清单**:
1. 修改 MarketDataService 支持批量查询
2. 实现策略配置读取
3. 添加股票代码映射
4. 优化多股票缓存
5. 测试多股票数据获取

**验收标准**:
- 支持从策略配置读取股票代码
- 支持批量获取多股票数据
- 缓存命中率 > 80%

### Phase 3: 实时数据集成（3-5天）
**目标**: 支持实时信号生成

**任务清单**:
1. 检查 RealTimeDataStream 状态
2. 集成实时数据订阅
3. 实现实时数据缓存更新
4. 修改信号生成逻辑支持实时数据
5. 添加 WebSocket 实时推送

**验收标准**:
- 实时数据流正常工作
- 信号生成延迟 < 1秒
- 前端实时接收信号更新

### Phase 4: 信号验证和监控（2-3天）
**目标**: 添加信号质量检查和监控

**任务清单**:
1. 实现信号质量评分算法
2. 添加信号回测验证
3. 实现信号准确性统计
4. 开发监控面板
5. 添加告警机制

**验收标准**:
- 信号有质量评分
- 信号准确性可统计
- 监控面板正常显示
- 异常信号触发告警

## 技术实现细节

### 数据采集实现
```python
# src/data_collection/akshare_collector.py

class AKShareDataCollector:
    """AKShare 数据采集器"""

    def collect_stock_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """采集股票历史数据"""
        import akshare as ak

        # 使用 AKShare 获取数据
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"  # 前复权
        )

        # 数据清洗和转换
        df = self._clean_data(df)

        # 保存到数据库
        self._save_to_database(df, symbol)

        return df
```

### 多股票支持实现
```python
# src/gateway/web/market_data_service.py

class MarketDataService:
    def get_multi_stock_data(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> Dict[str, pd.DataFrame]:
        """获取多股票数据"""
        result = {}
        for symbol in symbols:
            df = self.get_stock_data(symbol, start_date, end_date, limit)
            if not df.empty:
                result[symbol] = df
        return result
```

### 实时数据集成实现
```python
# src/gateway/web/realtime_data_integration.py

class RealtimeDataIntegration:
    """实时数据集成"""

    def __init__(self):
        self._stream = None
        self._cache = {}

    async def subscribe_realtime_data(self, symbols: List[str]):
        """订阅实时数据"""
        from src.strategy.realtime.real_time_processor import get_realtime_engine

        engine = await get_realtime_engine()
        if engine and hasattr(engine, 'data_stream'):
            self._stream = engine.data_stream
            for symbol in symbols:
                self._stream.subscribe(symbol, self._on_data_update)

    def _on_data_update(self, symbol: str, data: dict):
        """数据更新回调"""
        self._cache[symbol] = data
        # 触发信号生成
        self._trigger_signal_generation(symbol, data)
```

### 信号验证实现
```python
# src/trading/signal/signal_validator.py

class SignalValidator:
    """信号验证器"""

    def validate_signal(
        self,
        signal: TradingSignal,
        historical_data: pd.DataFrame
    ) -> SignalValidationResult:
        """验证信号质量"""
        # 1. 历史回测验证
        backtest_result = self._backtest_signal(signal, historical_data)

        # 2. 计算准确性评分
        accuracy_score = self._calculate_accuracy(backtest_result)

        # 3. 计算风险评分
        risk_score = self._calculate_risk_score(signal, historical_data)

        # 4. 综合评分
        overall_score = self._calculate_overall_score(
            accuracy_score, risk_score
        )

        return SignalValidationResult(
            signal=signal,
            accuracy_score=accuracy_score,
            risk_score=risk_score,
            overall_score=overall_score,
            is_valid=overall_score > 0.6
        )
```

## 验证和测试

### 单元测试
- [ ] MarketDataService 测试
- [ ] 多股票数据获取测试
- [ ] 实时数据集成测试
- [ ] 信号验证算法测试

### 集成测试
- [ ] 端到端数据流测试
- [ ] 实时信号生成测试
- [ ] 监控面板测试
- [ ] 告警机制测试

### 性能测试
- [ ] 数据查询性能测试
- [ ] 实时数据流性能测试
- [ ] 信号生成性能测试
- [ ] 系统整体性能测试

## 风险和对策

### 风险1: 数据源不稳定
**对策**: 实现多数据源备份，自动切换

### 风险2: 实时数据延迟
**对策**: 优化数据流处理，使用缓存和预加载

### 风险3: 信号准确性低
**对策**: 添加信号验证机制，过滤低质量信号

### 风险4: 系统性能瓶颈
**对策**: 性能监控，及时优化瓶颈点

## 时间计划

| 阶段 | 任务 | 预计时间 | 依赖 |
|------|------|----------|------|
| Phase 1 | 数据采集 | 1-2天 | 无 |
| Phase 2 | 多股票支持 | 2-3天 | Phase 1 |
| Phase 3 | 实时数据集成 | 3-5天 | Phase 2 |
| Phase 4 | 信号验证监控 | 2-3天 | Phase 3 |
| **总计** | | **8-13天** | |

## 成功标准

1. **数据采集**: 数据库中有充足的市场数据
2. **多股票支持**: 支持策略配置驱动的数据获取
3. **实时数据**: 支持实时信号生成和推送
4. **信号质量**: 信号有质量评分和验证机制
5. **监控完善**: 有完整的监控和告警体系
