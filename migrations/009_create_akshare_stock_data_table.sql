-- ============================================================
-- AKShare 股票数据表
-- 用于存储从 AKShare 采集的 A股、港股等股票行情数据
-- 
-- 创建时间: 2026-03-22
-- 版本: 1.0.0
-- ============================================================

-- ============================================================
-- 1. 主表: akshare_stock_data
-- 存储股票日线行情数据
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

-- ============================================================
-- 2. 索引
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_akshare_stock_symbol ON akshare_stock_data(symbol);
CREATE INDEX IF NOT EXISTS idx_akshare_stock_date ON akshare_stock_data(date);
CREATE INDEX IF NOT EXISTS idx_akshare_stock_source ON akshare_stock_data(source_id);
CREATE INDEX IF NOT EXISTS idx_akshare_stock_source_date ON akshare_stock_data(source_id, date);
CREATE INDEX IF NOT EXISTS idx_akshare_stock_data_type ON akshare_stock_data(data_type);

-- ============================================================
-- 3. 表注释
-- ============================================================
COMMENT ON TABLE akshare_stock_data IS 'AKShare股票行情数据表，存储A股、港股等股票的日线行情数据';
COMMENT ON COLUMN akshare_stock_data.id IS '主键ID';
COMMENT ON COLUMN akshare_stock_data.source_id IS '数据源ID（如 akshare_stock_a, akshare_stock_hk）';
COMMENT ON COLUMN akshare_stock_data.symbol IS '股票代码（如 000001, 600000）';
COMMENT ON COLUMN akshare_stock_data.date IS '交易日期';
COMMENT ON COLUMN akshare_stock_data.data_type IS '数据类型（daily=日线, hourly=小时线, minute=分钟线）';
COMMENT ON COLUMN akshare_stock_data.open_price IS '开盘价';
COMMENT ON COLUMN akshare_stock_data.high_price IS '最高价';
COMMENT ON COLUMN akshare_stock_data.low_price IS '最低价';
COMMENT ON COLUMN akshare_stock_data.close_price IS '收盘价';
COMMENT ON COLUMN akshare_stock_data.volume IS '成交量（股）';
COMMENT ON COLUMN akshare_stock_data.amount IS '成交额（元）';
COMMENT ON COLUMN akshare_stock_data.pct_change IS '涨跌幅（%）';
COMMENT ON COLUMN akshare_stock_data.change IS '涨跌额（元）';
COMMENT ON COLUMN akshare_stock_data.turnover_rate IS '换手率（%）';
COMMENT ON COLUMN akshare_stock_data.amplitude IS '振幅（%）';
COMMENT ON COLUMN akshare_stock_data.data_source IS '数据来源（akshare/baostock）';
COMMENT ON COLUMN akshare_stock_data.collected_at IS '数据采集时间';
COMMENT ON COLUMN akshare_stock_data.persistence_timestamp IS '数据持久化时间';

-- ============================================================
-- 4. 触发器: 自动更新 persistence_timestamp
-- ============================================================
CREATE OR REPLACE FUNCTION update_akshare_stock_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.persistence_timestamp = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_akshare_stock_persistence_ts ON akshare_stock_data;
CREATE TRIGGER update_akshare_stock_persistence_ts
    BEFORE UPDATE ON akshare_stock_data
    FOR EACH ROW
    EXECUTE FUNCTION update_akshare_stock_timestamp();

-- ============================================================
-- 5. 视图: 按数据源统计
-- ============================================================
CREATE OR REPLACE VIEW v_akshare_stock_source_stats AS
SELECT 
    source_id,
    data_type,
    COUNT(*) as total_records,
    COUNT(DISTINCT symbol) as symbol_count,
    MIN(date) as earliest_date,
    MAX(date) as latest_date,
    MAX(collected_at) as last_collected
FROM akshare_stock_data
GROUP BY source_id, data_type
ORDER BY source_id, data_type;

COMMENT ON VIEW v_akshare_stock_source_stats IS '按数据源统计股票数据';

-- ============================================================
-- 6. 视图: 最新数据概览
-- ============================================================
CREATE OR REPLACE VIEW v_akshare_stock_latest AS
SELECT DISTINCT ON (source_id, symbol)
    source_id,
    symbol,
    date,
    close_price,
    volume,
    pct_change,
    collected_at
FROM akshare_stock_data
ORDER BY source_id, symbol, date DESC;

COMMENT ON VIEW v_akshare_stock_latest IS '每个股票的最新数据记录';

-- ============================================================
-- 7. 存储过程: 清理过期数据
-- ============================================================
CREATE OR REPLACE PROCEDURE cleanup_old_akshare_stock_data(
    p_days_old INTEGER DEFAULT 365
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_cleaned_count INTEGER;
BEGIN
    DELETE FROM akshare_stock_data 
    WHERE date < CURRENT_DATE - INTERVAL '1 day' * p_days_old;
    
    GET DIAGNOSTICS v_cleaned_count = ROW_COUNT;
    
    RAISE NOTICE 'Cleaned % stock data records older than % days', v_cleaned_count, p_days_old;
END;
$$;

COMMENT ON PROCEDURE cleanup_old_akshare_stock_data IS '清理指定天数前的股票数据';

-- ============================================================
-- 8. 存储过程: 数据质量检查
-- ============================================================
CREATE OR REPLACE PROCEDURE check_akshare_stock_data_quality(
    p_source_id VARCHAR(50) DEFAULT NULL
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_total_count BIGINT;
    v_null_open_count BIGINT;
    v_null_close_count BIGINT;
    v_null_volume_count BIGINT;
    v_duplicate_count BIGINT;
BEGIN
    -- 总记录数
    SELECT COUNT(*) INTO v_total_count FROM akshare_stock_data
    WHERE p_source_id IS NULL OR source_id = p_source_id;
    
    -- 开盘价为空的记录数
    SELECT COUNT(*) INTO v_null_open_count FROM akshare_stock_data
    WHERE open_price IS NULL
      AND (p_source_id IS NULL OR source_id = p_source_id);
    
    -- 收盘价为空的记录数
    SELECT COUNT(*) INTO v_null_close_count FROM akshare_stock_data
    WHERE close_price IS NULL
      AND (p_source_id IS NULL OR source_id = p_source_id);
    
    -- 成交量为空的记录数
    SELECT COUNT(*) INTO v_null_volume_count FROM akshare_stock_data
    WHERE volume IS NULL
      AND (p_source_id IS NULL OR source_id = p_source_id);
    
    RAISE NOTICE '=== AKShare Stock Data Quality Report ===';
    RAISE NOTICE 'Total records: %', v_total_count;
    RAISE NOTICE 'Records with NULL open_price: %', v_null_open_count;
    RAISE NOTICE 'Records with NULL close_price: %', v_null_close_count;
    RAISE NOTICE 'Records with NULL volume: %', v_null_volume_count;
    
    IF v_total_count > 0 THEN
        RAISE NOTICE 'Data completeness: %%%', 
            ROUND(100.0 * (v_total_count - v_null_close_count) / v_total_count, 2);
    END IF;
END;
$$;

COMMENT ON PROCEDURE check_akshare_stock_data_quality IS '检查股票数据质量';

-- ============================================================
-- 9. 权限设置（根据实际环境调整）
-- ============================================================
-- GRANT SELECT, INSERT, UPDATE, DELETE ON akshare_stock_data TO rqa2025_admin;
-- GRANT EXECUTE ON PROCEDURE cleanup_old_akshare_stock_data TO rqa2025_admin;
-- GRANT EXECUTE ON PROCEDURE check_akshare_stock_data_quality TO rqa2025_admin;
-- GRANT SELECT ON v_akshare_stock_source_stats TO rqa2025_admin;
-- GRANT SELECT ON v_akshare_stock_latest TO rqa2025_admin;
