## 问题分析

在 `postgresql_persistence.py` 文件中，`persist_akshare_fundamental_data` 函数存在一个缩进错误。具体来说，在 `else:` 块的内部有一个 `try:` 块，但是没有对应的 `except:` 块，这导致了 Python 解释器报错：`unexpected unindent (postgresql_persistence.py, line 1233)`。

## 解决方案

需要在 `try:` 块后面添加一个 `except:` 块，以捕获可能的异常。这样可以确保 Python 解释器能够正确解析代码的缩进结构。

## 修复步骤

1. **定位问题代码**：找到 `persist_akshare_fundamental_data` 函数中的 `else:` 块，特别是包含 `try:` 块但缺少 `except:` 块的部分。

2. **添加 except 块**：在 `try:` 块后面添加一个 `except:` 块，以捕获可能的异常。

3. **验证修复**：确保修复后的代码能够正确运行，并且不再出现缩进错误。

## 具体修改

在 `postgresql_persistence.py` 文件中，找到以下代码：

```python
else:
    # 如果SQL文件不存在，直接创建表（简化版）
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS akshare_fundamental_data (
                id BIGSERIAL PRIMARY KEY,
                source_id VARCHAR(50) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                report_date DATE NOT NULL,
                company_name VARCHAR(100),
                industry VARCHAR(50),
                pe DECIMAL(10, 4),
                pb DECIMAL(10, 4),
                market_cap DECIMAL(20, 2),
                revenue DECIMAL(20, 2),
                net_profit DECIMAL(20, 2),
                roe DECIMAL(10, 4),
                data_source VARCHAR(50) DEFAULT 'akshare',
                collected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                persistence_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT unique_fundamental_record UNIQUE(source_id, symbol, report_date)
            );
            
            -- 创建索引
            CREATE INDEX IF NOT EXISTS idx_akshare_fundamental_symbol_date ON akshare_fundamental_data(symbol, report_date DESC);
            CREATE INDEX IF NOT EXISTS idx_akshare_fundamental_industry ON akshare_fundamental_data(industry);
            CREATE INDEX IF NOT EXISTS idx_akshare_fundamental_source_collected ON akshare_fundamental_data(source_id, collected_at DESC);
        """)
        conn.commit()
        logger.debug("基本面数据表结构检查完成")
```

修改为：

```python
else:
    # 如果SQL文件不存在，直接创建表（简化版）
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS akshare_fundamental_data (
                id BIGSERIAL PRIMARY KEY,
                source_id VARCHAR(50) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                report_date DATE NOT NULL,
                company_name VARCHAR(100),
                industry VARCHAR(50),
                pe DECIMAL(10, 4),
                pb DECIMAL(10, 4),
                market_cap DECIMAL(20, 2),
                revenue DECIMAL(20, 2),
                net_profit DECIMAL(20, 2),
                roe DECIMAL(10, 4),
                data_source VARCHAR(50) DEFAULT 'akshare',
                collected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                persistence_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT unique_fundamental_record UNIQUE(source_id, symbol, report_date)
            );
            
            -- 创建索引
            CREATE INDEX IF NOT EXISTS idx_akshare_fundamental_symbol_date ON akshare_fundamental_data(symbol, report_date DESC);
            CREATE INDEX IF NOT EXISTS idx_akshare_fundamental_industry ON akshare_fundamental_data(industry);
            CREATE INDEX IF NOT EXISTS idx_akshare_fundamental_source_collected ON akshare_fundamental_data(source_id, collected_at DESC);
        """)
        conn.commit()
        logger.debug("基本面数据表结构检查完成")
    except Exception as e:
        logger.debug(f"基本面数据表可能已存在: {e}")
```

这样就可以修复缩进错误，确保代码能够正确运行。