# 实现数据采集写入 akshare_stock_data 表的计划

## 检查结果

### 1. 数据采集服务已实现
- **文件**: `src/gateway/web/data_collectors.py`
- **功能**: 提交数据采集任务到统一调度器
- **数据流**: DataCollectionService → UnifiedScheduler → DataCollector Worker → DataAdapter → DataLayer

### 2. 数据适配器未实现写入逻辑
- **ChinaDataAdapter** (`src/data/china/china_data_adapter.py`): 空壳实现，无实际功能
- **MarketDataAdapter** (`src/data/adapters/market_data_adapter.py`): 抽象基类，只有接口
- **未发现**: 直接写入 `akshare_stock_data` 表的逻辑

### 3. 需要实现的内容

#### 3.1 实现 AKShare 数据采集器
**文件**: `src/data/collectors/akshare_collector.py`

**功能**:
- 使用 AKShare 库获取股票历史数据
- 数据清洗和转换
- 写入 `akshare_stock_data` 表

**实现代码**:
```python
import akshare as ak
import pandas as pd
from datetime import datetime
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class AKShareCollector:
    """AKShare 数据采集器"""

    def collect_stock_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str = "qfq"
    ) -> Optional[pd.DataFrame]:
        """
        采集股票历史数据

        Args:
            symbol: 股票代码 (如 "000001")
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            adjust: 复权类型 (qfq-前复权, hfq-后复权, 不复权)

        Returns:
            DataFrame with columns: date, open, high, low, close, volume, amount, ...
        """
        try:
            logger.info(f"开始采集股票数据: {symbol}, 时间范围: {start_date} 到 {end_date}")

            # 使用 AKShare 获取数据
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust=adjust
            )

            if df.empty:
                logger.warning(f"未获取到数据: {symbol}")
                return None

            # 重命名列以匹配数据库表结构
            df = df.rename(columns={
                "日期": "date",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
                "成交额": "amount",
                "振幅": "amplitude",
                "涨跌幅": "pct_change",
                "涨跌额": "change_amount",
                "换手率": "turnover"
            })

            # 添加股票代码列
            df["symbol"] = symbol

            logger.info(f"成功采集股票数据: {symbol}, 记录数: {len(df)}")
            return df

        except Exception as e:
            logger.error(f"采集股票数据失败 {symbol}: {e}")
            return None

    def save_to_database(self, df: pd.DataFrame, symbol: str) -> bool:
        """
        将数据保存到数据库

        Args:
            df: 股票数据 DataFrame
            symbol: 股票代码

        Returns:
            是否保存成功
        """
        try:
            from src.gateway.web.postgresql_persistence import get_db_connection

            conn = get_db_connection()
            cursor = conn.cursor()

            # 准备插入语句
            insert_query = """
                INSERT INTO akshare_stock_data (
                    symbol, date, open, high, low, close, volume, amount,
                    amplitude, pct_change, change_amount, turnover
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, date) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    amount = EXCLUDED.amount,
                    amplitude = EXCLUDED.amplitude,
                    pct_change = EXCLUDED.pct_change,
                    change_amount = EXCLUDED.change_amount,
                    turnover = EXCLUDED.turnover
            """

            # 批量插入数据
            for _, row in df.iterrows():
                cursor.execute(insert_query, (
                    symbol,
                    row["date"],
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    int(row["volume"]),
                    float(row["amount"]),
                    float(row.get("amplitude", 0)),
                    float(row.get("pct_change", 0)),
                    float(row.get("change_amount", 0)),
                    float(row.get("turnover", 0))
                ))

            conn.commit()
            cursor.close()

            logger.info(f"成功保存股票数据到数据库: {symbol}, 记录数: {len(df)}")
            return True

        except Exception as e:
            logger.error(f"保存股票数据到数据库失败 {symbol}: {e}")
            return False

    def collect_and_save(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str = "qfq"
    ) -> bool:
        """
        采集并保存股票数据

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            adjust: 复权类型

        Returns:
            是否成功
        """
        df = self.collect_stock_data(symbol, start_date, end_date, adjust)
        if df is not None and not df.empty:
            return self.save_to_database(df, symbol)
        return False
```

#### 3.2 修改数据采集工作器
**文件**: `src/distributed/worker/data_collector_worker.py` (需要创建或修改)

**功能**:
- 接收数据采集任务
- 调用 AKShareCollector 采集数据
- 保存到数据库

**实现代码**:
```python
from src.data.collectors.akshare_collector import AKShareCollector

class DataCollectorWorker:
    """数据采集工作器"""

    def __init__(self):
        self.collector = AKShareCollector()

    async def process_task(self, task_data: dict) -> dict:
        """处理数据采集任务"""
        symbols = task_data.get("symbols", [])
        start_date = task_data.get("start_date")
        end_date = task_data.get("end_date")

        results = []
        for symbol in symbols:
            success = self.collector.collect_and_save(symbol, start_date, end_date)
            results.append({"symbol": symbol, "success": success})

        return {"results": results}
```

#### 3.3 创建数据采集 API
**文件**: `src/gateway/web/data_collection_api.py`

**功能**:
- 提供手动触发数据采集的 API
- 查询数据采集状态
- 管理数据采集任务

**API 端点**:
```python
@router.post("/api/v1/data/collection/start")
async def start_data_collection(request: DataCollectionRequest):
    """启动数据采集"""
    # 调用数据采集服务
    result = submit_data_collection_task(
        symbols=request.symbols,
        start_date=request.start_date,
        end_date=request.end_date
    )
    return result

@router.get("/api/v1/data/collection/status/{task_id}")
async def get_collection_status(task_id: str):
    """获取数据采集状态"""
    # 查询任务状态
    pass
```

#### 3.4 创建数据库表（如果不存在）
**SQL**:
```sql
CREATE TABLE IF NOT EXISTS akshare_stock_data (
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

CREATE INDEX IF NOT EXISTS idx_akshare_stock_symbol ON akshare_stock_data(symbol);
CREATE INDEX IF NOT EXISTS idx_akshare_stock_date ON akshare_stock_data(date);
CREATE INDEX IF NOT EXISTS idx_akshare_stock_symbol_date ON akshare_stock_data(symbol, date);
```

## 实施步骤

### 步骤 1: 创建 AKShare 数据采集器
1. 创建 `src/data/collectors/akshare_collector.py`
2. 实现数据采集和保存逻辑
3. 添加错误处理和日志

### 步骤 2: 修改数据采集工作器
1. 修改 `src/distributed/worker/data_collector_worker.py`
2. 集成 AKShareCollector
3. 实现任务处理逻辑

### 步骤 3: 创建数据采集 API
1. 创建 `src/gateway/web/data_collection_api.py`
2. 实现 API 端点
3. 注册到 FastAPI 应用

### 步骤 4: 创建数据库表
1. 执行 SQL 创建表
2. 验证表结构
3. 添加索引

### 步骤 5: 测试和验证
1. 手动触发数据采集
2. 验证数据已写入数据库
3. 测试信号生成功能

## 验收标准

1. **数据采集**: 能成功采集股票数据并保存到数据库
2. **数据完整性**: 数据包含所有必要字段
3. **信号生成**: 信号生成器能使用数据库数据生成信号
4. **API 可用**: 提供手动触发数据采集的 API
5. **错误处理**: 有完善的错误处理和日志记录

## 时间估计

- 步骤 1: 2-3 小时
- 步骤 2: 1-2 小时
- 步骤 3: 1-2 小时
- 步骤 4: 0.5 小时
- 步骤 5: 1-2 小时
- **总计**: 5.5-9.5 小时
