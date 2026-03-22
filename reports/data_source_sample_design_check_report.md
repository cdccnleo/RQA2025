# 数据源配置样本设计深度检查报告

**报告编号**: RQA2025-DATASOURCE-SAMPLE-001  
**检查日期**: 2026-03-22  
**问题来源**: 后台错误日志  
**检查范围**: 数据源 `akshare_stock_a` 样本数据查询失败问题

---

## 一、问题描述

### 1.1 错误日志

```
2026-03-22 18:03:31,283 - src.gateway.web.postgresql_persistence - ERROR - 
查询最新股票数据失败: relation "akshare_stock_data" does not exist
LINE 5: FROM akshare_stock_data
         ^
2026-03-22 18:03:31,283 - datasource_routes - INFO - 
数据源 akshare_stock_a 在PostgreSQL中没有找到样本数据。
```

### 1.2 问题影响

- 数据源配置页面无法显示样本数据
- 用户无法预览采集的数据
- 影响数据质量验证和监控

---

## 二、问题定位分析

### 2.1 数据源配置检查

**配置文件**: [data/data_sources_config.json](file:///c:/PythonProject/RQA2025/data/data_sources_config.json)

```json
{
  "id": "akshare_stock_a",
  "name": "AKShare A股数据",
  "type": "股票数据",
  "url": "https://akshare.akfamily.xyz",
  "rate_limit": "1次/小时",
  "enabled": true,
  "config": {
    "akshare_category": "A股",
    "default_days": 30,
    "adjust_type": "qfq",
    "enable_incremental": true,
    "akshare_function": "stock_zh_a_spot_em"
  },
  "last_test": "2026-02-09 15:58:05",
  "status": "连接失败"
}
```

**发现**:
- 数据源配置正确，`id` 为 `akshare_stock_a`
- 状态显示"连接失败"，表明数据采集可能未成功

### 2.2 数据库表结构检查

**代码位置**: [postgresql_persistence.py:335-393](file:///c:/PythonProject/RQA2025/src/gateway/web/postgresql_persistence.py#L335-L393)

**表创建逻辑**:
```python
def ensure_table_exists() -> bool:
    """确保 akshare_stock_data 表存在，不存在则创建"""
    # 检查表是否存在
    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = 'akshare_stock_data'
        );
    """)
    # 如果不存在，创建表
    cursor.execute("""
        CREATE TABLE akshare_stock_data (
            id BIGSERIAL PRIMARY KEY,
            source_id VARCHAR(50) NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            date DATE NOT NULL,
            data_type VARCHAR(20) NOT NULL DEFAULT 'daily',
            open_price DECIMAL(15, 6),
            ...
            CONSTRAINT unique_akshare_record UNIQUE(source_id, symbol, date, data_type)
        );
    """)
```

**关键发现**:
- 表名硬编码为 `akshare_stock_data`
- 表创建采用"惰性创建"模式（查询时检查并创建）
- 表创建逻辑在 `persist_akshare_data_to_postgresql()` 函数中调用

### 2.3 样本查询逻辑检查

**代码位置**: [postgresql_persistence.py:544-612](file:///c:/PythonProject/RQA2025/src/gateway/web/postgresql_persistence.py#L544-L612)

```python
def query_latest_stock_data_from_postgresql(source_id: str, limit: int = 10, data_type: str = None):
    """从PostgreSQL查询最新的股票数据样本"""
    query = """
        SELECT 
            symbol, date, open_price, high_price, low_price, close_price,
            volume, amount, pct_change, change, turnover_rate, amplitude
        FROM akshare_stock_data 
        WHERE source_id = %s
    """
```

**问题**: 查询直接使用 `akshare_stock_data` 表名，但未先检查表是否存在。

### 2.4 迁移文件检查

**检查结果**: 在 `migrations/` 目录中**未找到** `akshare_stock_data` 表的迁移文件。

**现有迁移文件**:
| 文件 | 创建的表 |
|-----|---------|
| 001_create_data_tables.sql | data_cache_entries, data_lineage_records 等 |
| 002_create_feature_tables.sql | feature_store, feature_cache 等 |
| 003_create_ml_tables.sql | ml_models, ml_training_history 等 |
| 004_create_strategy_tables.sql | strategies, strategy_configs 等 |
| 005_create_trading_tables.sql | trading_orders, trading_positions 等 |
| 006_create_risk_tables.sql | risk_checks, risk_alerts 等 |

**关键发现**: `akshare_stock_data` 表缺少独立的迁移文件！

---

## 三、根本原因分析

### 3.1 问题根源

| 原因编号 | 原因描述 | 影响程度 |
|---------|---------|---------|
| R1 | **表未创建**: 数据库中确实不存在 `akshare_stock_data` 表 | 高 |
| R2 | **惰性创建机制缺陷**: 表创建依赖于数据持久化操作，但数据采集未成功执行 | 高 |
| R3 | **缺少迁移文件**: 没有独立的迁移文件来创建股票数据表 | 中 |
| R4 | **查询未做表存在性检查**: 样本查询直接使用表名，未先检查表是否存在 | 中 |
| R5 | **数据采集失败**: 数据源状态显示"连接失败"，导致无数据持久化 | 中 |

### 3.2 问题链分析

```
数据采集失败 (连接失败)
    ↓
未调用 persist_akshare_data_to_postgresql()
    ↓
ensure_table_exists() 未被调用
    ↓
akshare_stock_data 表未创建
    ↓
样本查询失败 (relation does not exist)
```

### 3.3 架构问题

**当前架构**:
```
数据采集 → 持久化 → 表创建（惰性）→ 数据查询
```

**问题**: 表创建依赖于数据采集成功，形成循环依赖。

**理想架构**:
```
数据库迁移 → 表创建 → 数据采集 → 持久化 → 数据查询
```

---

## 四、修正方案

### 4.1 短期修复（立即执行）

#### 4.1.1 创建数据库迁移文件

**文件**: `migrations/009_create_akshare_stock_data_table.sql`

```sql
-- ============================================================
-- AKShare 股票数据表
-- 用于存储从 AKShare 采集的 A股、港股等股票行情数据
-- ============================================================

CREATE TABLE IF NOT EXISTS akshare_stock_data (
    id BIGSERIAL PRIMARY KEY,
    source_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    data_type VARCHAR(20) NOT NULL DEFAULT 'daily',
    open_price DECIMAL(15, 6),
    high_price DECIMAL(15, 6),
    low_price DECIMAL(15, 6),
    close_price DECIMAL(15, 6),
    volume BIGINT,
    amount DECIMAL(20, 2),
    pct_change DECIMAL(10, 4),
    change DECIMAL(15, 6),
    turnover_rate DECIMAL(10, 4),
    amplitude DECIMAL(10, 4),
    data_source VARCHAR(50) DEFAULT 'akshare',
    collected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    persistence_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_akshare_record UNIQUE(source_id, symbol, date, data_type)
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_akshare_stock_symbol ON akshare_stock_data(symbol);
CREATE INDEX IF NOT EXISTS idx_akshare_stock_date ON akshare_stock_data(date);
CREATE INDEX IF NOT EXISTS idx_akshare_stock_source ON akshare_stock_data(source_id);
CREATE INDEX IF NOT EXISTS idx_akshare_stock_source_date ON akshare_stock_data(source_id, date);

-- 表注释
COMMENT ON TABLE akshare_stock_data IS 'AKShare股票行情数据表';
COMMENT ON COLUMN akshare_stock_data.source_id IS '数据源ID（如 akshare_stock_a）';
COMMENT ON COLUMN akshare_stock_data.symbol IS '股票代码';
COMMENT ON COLUMN akshare_stock_data.date IS '交易日期';
COMMENT ON COLUMN akshare_stock_data.data_type IS '数据类型（daily/hourly/minute）';

-- 分区（可选，按月分区）
-- CREATE TABLE akshare_stock_data_2026_01 PARTITION OF akshare_stock_data
--     FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
```

#### 4.1.2 修复样本查询函数

**修改**: [postgresql_persistence.py](file:///c:/PythonProject/RQA2025/src/gateway/web/postgresql_persistence.py)

```python
def query_latest_stock_data_from_postgresql(source_id: str, limit: int = 10, data_type: str = None):
    """从PostgreSQL查询最新的股票数据样本"""
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            logger.warning("无法获取数据库连接")
            return []
        
        cursor = conn.cursor()
        
        # 先检查表是否存在
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = 'akshare_stock_data'
            );
        """)
        
        if not cursor.fetchone()[0]:
            logger.info(f"akshare_stock_data 表不存在，返回空数据")
            cursor.close()
            return []
        
        # 表存在，执行查询
        query = """
            SELECT 
                symbol, date, open_price, high_price, low_price, close_price,
                volume, amount, pct_change, change, turnover_rate, amplitude
            FROM akshare_stock_data 
            WHERE source_id = %s
        """
        # ... 后续查询逻辑
```

### 4.2 中期改进（1周内）

#### 4.2.1 统一表名管理

创建配置文件管理数据表名：

```python
# src/gateway/web/table_config.py
DATA_TABLES = {
    "akshare_stock": "akshare_stock_data",
    "baostock_stock": "baostock_stock_data",
    "akshare_index": "akshare_index_data",
    # ...
}

def get_table_name(source_id: str) -> str:
    """根据数据源ID获取对应的表名"""
    # 映射逻辑
    source_type = source_id.split('_')[0] if '_' in source_id else source_id
    return DATA_TABLES.get(source_type, f"{source_id}_data")
```

#### 4.2.2 添加数据源状态监控

```python
# 在数据源配置中添加表状态字段
{
    "id": "akshare_stock_a",
    "table_name": "akshare_stock_data",
    "table_exists": True,  # 动态检查
    "record_count": 0,     # 动态统计
    "last_collection": null
}
```

### 4.3 长期改进（2-4周）

#### 4.3.1 实现数据库迁移自动化

```python
# scripts/run_migrations.py 增加自动检测
def check_required_tables():
    """检查必需的表是否存在"""
    required_tables = [
        'akshare_stock_data',
        'data_cache_entries',
        # ...
    ]
    
    missing = []
    for table in required_tables:
        if not table_exists(table):
            missing.append(table)
    
    return missing
```

#### 4.3.2 完善数据采集状态反馈

在数据源配置页面显示：
- 表是否存在
- 表中记录数
- 最后采集时间
- 数据时间范围

---

## 五、执行步骤

### 5.1 立即执行

1. **创建迁移文件**
   ```bash
   # 创建 migrations/009_create_akshare_stock_data_table.sql
   ```

2. **执行迁移**
   ```powershell
   cd c:\PythonProject\RQA2025
   $env:POSTGRES_HOST="localhost"
   $env:POSTGRES_PASSWORD="SecurePass123!"
   python scripts\run_migrations.py --force
   ```

3. **验证表创建**
   ```sql
   SELECT COUNT(*) FROM akshare_stock_data;
   ```

### 5.2 后续验证

1. 重新采集数据源 `akshare_stock_a` 的数据
2. 检查样本数据页面是否正常显示
3. 验证数据持久化流程

---

## 六、总结

### 6.1 问题本质

`akshare_stock_data` 表不存在是因为：
1. 表创建采用惰性模式，依赖于数据持久化操作
2. 数据采集失败导致表从未被创建
3. 缺少独立的数据库迁移文件

### 6.2 解决方案

| 方案 | 优先级 | 执行时间 |
|-----|-------|---------|
| 创建迁移文件 | 高 | 立即 |
| 修复查询函数 | 高 | 立即 |
| 统一表名管理 | 中 | 1周内 |
| 状态监控完善 | 中 | 2周内 |

### 6.3 预防措施

1. 所有数据表都应有独立的迁移文件
2. 查询函数应先检查表是否存在
3. 数据源配置应包含表名映射信息
4. 建立表存在性监控机制

---

**报告编制**: AI Assistant  
**审核状态**: 待审核  
**版本**: 1.0
