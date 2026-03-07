-- AKShare股票数据表结构
-- 支持PostgreSQL + TimescaleDB时序数据优化
-- 版本: 1.0.0
-- 创建日期: 2025-12-31

-- 创建AKShare股票数据表
CREATE TABLE IF NOT EXISTS akshare_stock_data (
    id BIGSERIAL PRIMARY KEY,
    source_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
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
    -- 防止重复数据：同一数据源、同一股票、同一日期只能有一条记录
    CONSTRAINT unique_akshare_record UNIQUE(source_id, symbol, date)
);

-- 创建索引优化查询性能
CREATE INDEX IF NOT EXISTS idx_akshare_symbol_date ON akshare_stock_data(symbol, date DESC);
CREATE INDEX IF NOT EXISTS idx_akshare_source_collected ON akshare_stock_data(source_id, collected_at DESC);
CREATE INDEX IF NOT EXISTS idx_akshare_date_range ON akshare_stock_data(date DESC);
CREATE INDEX IF NOT EXISTS idx_akshare_symbol_source ON akshare_stock_data(symbol, source_id);

-- 如果安装了TimescaleDB扩展，创建超表（时序数据优化）
-- 注意：需要先安装TimescaleDB扩展: CREATE EXTENSION IF NOT EXISTS timescaledb;
DO $$
BEGIN
    -- 检查TimescaleDB扩展是否可用
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        -- 创建超表（如果还没有创建）
        IF NOT EXISTS (
            SELECT 1 FROM _timescaledb_catalog.hypertable 
            WHERE hypertable_name = 'akshare_stock_data'
        ) THEN
            PERFORM create_hypertable(
                'akshare_stock_data',
                'date',
                chunk_time_interval => INTERVAL '1 month',
                if_not_exists => TRUE
            );
            RAISE NOTICE 'TimescaleDB超表创建成功: akshare_stock_data';
        ELSE
            RAISE NOTICE 'TimescaleDB超表已存在: akshare_stock_data';
        END IF;
    ELSE
        RAISE NOTICE 'TimescaleDB扩展未安装，使用标准PostgreSQL表';
    END IF;
END $$;

-- 创建数据统计视图
CREATE OR REPLACE VIEW v_akshare_stock_summary AS
SELECT 
    symbol,
    source_id,
    COUNT(*) as record_count,
    MIN(date) as first_date,
    MAX(date) as last_date,
    AVG(close_price) as avg_close_price,
    AVG(volume) as avg_volume,
    SUM(volume) as total_volume
FROM akshare_stock_data
GROUP BY symbol, source_id;

-- 创建数据采集统计视图
CREATE OR REPLACE VIEW v_akshare_collection_stats AS
SELECT 
    source_id,
    DATE(collected_at) as collection_date,
    COUNT(*) as records_collected,
    COUNT(DISTINCT symbol) as unique_symbols,
    MIN(collected_at) as first_collection,
    MAX(collected_at) as last_collection
FROM akshare_stock_data
GROUP BY source_id, DATE(collected_at)
ORDER BY collection_date DESC;

-- 添加注释
COMMENT ON TABLE akshare_stock_data IS 'AKShare采集的A股股票数据表';
COMMENT ON COLUMN akshare_stock_data.source_id IS '数据源ID，如akshare_stock';
COMMENT ON COLUMN akshare_stock_data.symbol IS '股票代码，如000001';
COMMENT ON COLUMN akshare_stock_data.date IS '交易日期';
COMMENT ON COLUMN akshare_stock_data.close_price IS '收盘价';
COMMENT ON COLUMN akshare_stock_data.volume IS '成交量';
COMMENT ON COLUMN akshare_stock_data.pct_change IS '涨跌幅(%)';
COMMENT ON COLUMN akshare_stock_data.turnover_rate IS '换手率(%)';
COMMENT ON COLUMN akshare_stock_data.collected_at IS '数据采集时间';
COMMENT ON COLUMN akshare_stock_data.persistence_timestamp IS '数据持久化时间';

