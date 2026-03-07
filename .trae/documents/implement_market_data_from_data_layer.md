# 从数据管理层获取历史数据实施方案

## 目标
实现从数据管理层(PostgreSQL)获取历史市场数据，为信号生成器提供数据支持。

## 当前状态
- 数据库表: `akshare_stock_data` 已存在
- 信号生成器: `SimpleSignalGenerator.generate_signals(data)` 需要 pandas DataFrame
- 数据管理层: `PostgreSQLDataLoader` 已可用

## 实施步骤

### Phase 1: 数据查询接口实现
- [ ] 创建市场数据查询服务
- [ ] 实现从 PostgreSQL 查询股票数据
- [ ] 数据转换为 pandas DataFrame
- [ ] 添加数据缓存机制

### Phase 2: TradingSignalService 集成
- [ ] 修改 `get_realtime_signals()` 函数
- [ ] 集成市场数据查询服务
- [ ] 为信号生成器提供数据参数
- [ ] 添加错误处理和降级方案

### Phase 3: 数据格式标准化
- [ ] 定义标准 DataFrame 格式
- [ ] 实现数据列映射转换
- [ ] 处理缺失数据
- [ ] 数据质量验证

### Phase 4: 性能优化
- [ ] 实现数据缓存
- [ ] 优化查询性能
- [ ] 添加数据预加载
- [ ] 实现增量更新

### Phase 5: 测试验证
- [ ] 单元测试
- [ ] 集成测试
- [ ] 性能测试
- [ ] 端到端测试

## 技术实现细节

### 1. 市场数据查询服务
```python
# src/gateway/web/market_data_service.py
import pandas as pd
from typing import Optional, List
from datetime import datetime, timedelta

class MarketDataService:
    """市场数据查询服务"""
    
    def __init__(self):
        self._cache = {}
        self._cache_ttl = 300  # 5分钟缓存
    
    def get_stock_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        获取股票历史数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            limit: 限制条数
            
        Returns:
            pandas DataFrame with columns: open, high, low, close, volume
        """
        # 1. 检查缓存
        cache_key = f"{symbol}_{start_date}_{end_date}_{limit}"
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if datetime.now().timestamp() - timestamp < self._cache_ttl:
                return cached_data
        
        # 2. 从数据库查询
        data = self._query_from_database(symbol, start_date, end_date, limit)
        
        # 3. 转换为标准格式
        df = self._convert_to_dataframe(data)
        
        # 4. 更新缓存
        self._cache[cache_key] = (df, datetime.now().timestamp())
        
        return df
    
    def _query_from_database(
        self,
        symbol: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        limit: int
    ) -> List[dict]:
        """从PostgreSQL查询数据"""
        from .postgresql_persistence import get_db_connection
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = """
            SELECT date, open, high, low, close, volume
            FROM akshare_stock_data
            WHERE symbol = %s
        """
        params = [symbol]
        
        if start_date:
            query += " AND date >= %s"
            params.append(start_date)
        if end_date:
            query += " AND date <= %s"
            params.append(end_date)
        
        query += " ORDER BY date DESC LIMIT %s"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        cursor.close()
        
        return [
            {
                'date': row[0],
                'open': row[1],
                'high': row[2],
                'low': row[3],
                'close': row[4],
                'volume': row[5]
            }
            for row in rows
        ]
    
    def _convert_to_dataframe(self, data: List[dict]) -> pd.DataFrame:
        """转换为pandas DataFrame"""
        if not data:
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        # 确保列名和数据类型正确
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        
        return df
```

### 2. TradingSignalService 修改
```python
# src/gateway/web/trading_signal_service.py

def get_realtime_signals() -> List[Dict[str, Any]]:
    """获取实时交易信号"""
    signal_generator = get_signal_generator()
    
    if not signal_generator:
        logger.debug("信号生成器不可用，返回空信号列表")
        return []
    
    try:
        signals = []
        
        # 尝试不同的方法名
        if hasattr(signal_generator, 'get_realtime_signals'):
            signals = signal_generator.get_realtime_signals()
        elif hasattr(signal_generator, 'get_current_signals'):
            signals = signal_generator.get_current_signals()
        elif hasattr(signal_generator, 'generate_signals'):
            # 获取市场数据
            market_data = _get_market_data_for_signal_generation()
            if market_data is not None and not market_data.empty:
                signals = signal_generator.generate_signals(market_data)
            else:
                logger.warning("无法获取市场数据，跳过信号生成")
                return []
        elif hasattr(signal_generator, 'get_signals'):
            signals = signal_generator.get_signals()
        
        # 转换信号格式...
        return _format_signals(signals)
        
    except Exception as e:
        logger.error(f"获取实时信号失败: {e}")
        return []


def _get_market_data_for_signal_generation() -> Optional[pd.DataFrame]:
    """获取信号生成所需的市场数据"""
    try:
        from .market_data_service import MarketDataService
        
        service = MarketDataService()
        
        # 获取默认股票数据（可以从配置或策略参数中获取）
        default_symbol = "000001"  # 平安银行作为默认示例
        
        # 获取最近30天的数据
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        df = service.get_stock_data(
            symbol=default_symbol,
            start_date=start_date,
            end_date=end_date,
            limit=100
        )
        
        return df
        
    except Exception as e:
        logger.error(f"获取市场数据失败: {e}")
        return None


def _format_signals(signals) -> List[Dict[str, Any]]:
    """格式化信号列表"""
    if not signals:
        return []
    
    formatted_signals = []
    for signal in signals:
        if not isinstance(signal, dict):
            if hasattr(signal, '__dict__'):
                signal_dict = signal.__dict__
            elif hasattr(signal, 'to_dict'):
                signal_dict = signal.to_dict()
            else:
                continue
        else:
            signal_dict = signal
        
        signal_data = {
            "id": signal_dict.get('id', signal_dict.get('signal_id', '')),
            "symbol": signal_dict.get('symbol', ''),
            "type": signal_dict.get('type', signal_dict.get('signal_type', 'unknown')),
            "strength": signal_dict.get('strength', 0),
            "price": signal_dict.get('price', 0),
            "status": signal_dict.get('status', 'pending'),
            "timestamp": signal_dict.get('timestamp', int(datetime.now().timestamp())),
            "accuracy": signal_dict.get('accuracy', 0),
            "latency": signal_dict.get('latency', 0),
            "quality": signal_dict.get('quality', 0)
        }
        formatted_signals.append(signal_data)
        
        # 保存到持久化存储
        try:
            from .signal_persistence import save_signal
            save_signal(signal_data)
        except Exception as e:
            logger.debug(f"保存信号失败: {e}")
    
    return formatted_signals
```

### 3. 数据格式标准
```python
# 标准DataFrame格式
standard_columns = ['open', 'high', 'low', 'close', 'volume']

# 数据类型要求
dtype_mapping = {
    'open': float,
    'high': float,
    'low': float,
    'close': float,
    'volume': float
}

# 索引要求
# - 类型: DatetimeIndex
# - 格式: YYYY-MM-DD
# - 时区: 本地时间
```

## 数据库表结构

### akshare_stock_data 表
```sql
CREATE TABLE akshare_stock_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    open DECIMAL(10, 4),
    high DECIMAL(10, 4),
    low DECIMAL(10, 4),
    close DECIMAL(10, 4),
    volume BIGINT,
    amount DECIMAL(15, 4),
    amplitude DECIMAL(10, 4),
    pct_change DECIMAL(10, 4),
    change_amount DECIMAL(10, 4),
    turnover DECIMAL(10, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, date)
);

CREATE INDEX idx_akshare_stock_symbol ON akshare_stock_data(symbol);
CREATE INDEX idx_akshare_stock_date ON akshare_stock_data(date);
CREATE INDEX idx_akshare_stock_symbol_date ON akshare_stock_data(symbol, date);
```

## 验证检查点

### 功能验证
- [ ] 能正确查询数据库
- [ ] 数据转换为DataFrame格式正确
- [ ] 信号生成器能正常接收数据
- [ ] 信号生成结果正确

### 性能验证
- [ ] 查询响应时间 < 1秒
- [ ] 缓存命中率 > 80%
- [ ] 内存使用合理

### 异常处理
- [ ] 数据库连接失败处理
- [ ] 数据缺失处理
- [ ] 格式错误处理
- [ ] 降级方案生效

## 迭代优化计划

### 迭代1: 基础功能实现
- 实现 MarketDataService
- 修改 TradingSignalService
- 基础错误处理

### 迭代2: 性能优化
- 添加缓存机制
- 优化查询SQL
- 实现数据预加载

### 迭代3: 功能增强
- 支持多股票代码
- 支持不同时间周期
- 支持数据质量检查

### 迭代4: 监控完善
- 添加性能监控
- 添加数据质量监控
- 添加告警机制
