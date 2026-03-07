## 问题分析

PostgreSQL 中已存在 `akshare_stock_data` 表，但特征提取任务查询的是 `stock_data` 表，导致任务失败。

### akshare_stock_data 表结构
- `symbol`: 股票代码
- `date`: 日期
- `open_price`: 开盘价
- `high_price`: 最高价
- `low_price`: 最低价
- `close_price`: 收盘价
- `volume`: 成交量

### 字段映射
| stock_data | akshare_stock_data |
|------------|-------------------|
| symbol | symbol |
| date | date |
| open | open_price |
| high | high_price |
| low | low_price |
| close | close_price |
| volume | volume |

## 修复方案

修改 `src/features/distributed/worker_executor.py`：
1. 将表名从 `stock_data` 改为 `akshare_stock_data`
2. 将字段名从 `open`, `high`, `low`, `close` 改为 `open_price`, `high_price`, `low_price`, `close_price`

## 修改内容

```python
# 修改前
query = """
    SELECT date, open, high, low, close, volume
    FROM stock_data
    WHERE symbol = %s AND date >= %s AND date <= %s
    ORDER BY date ASC
"""

# 修改后
query = """
    SELECT date, open_price as open, high_price as high, low_price as low, close_price as close, volume
    FROM akshare_stock_data
    WHERE symbol = %s AND date >= %s AND date <= %s
    ORDER BY date ASC
"""
```

这样修改后，特征提取任务将正确查询 `akshare_stock_data` 表中的数据。