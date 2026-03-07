-- AKShare外汇/数字货币数据表结构
-- 支持TimescaleDB超表

CREATE TABLE IF NOT EXISTS akshare_forex_crypto_data (
    id BIGSERIAL PRIMARY KEY,
    source_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    open_price DECIMAL(15, 6),
    high_price DECIMAL(15, 6),
    low_price DECIMAL(15, 6),
    close_price DECIMAL(15, 6),
    rate DECIMAL(15, 6),
    volume DECIMAL(20, 2),
    data_subtype VARCHAR(50),
    data_source VARCHAR(50) DEFAULT 'akshare',
    collected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    persistence_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_id, symbol, date)
);

CREATE INDEX IF NOT EXISTS idx_akshare_forex_crypto_symbol_date ON akshare_forex_crypto_data(symbol, date DESC);
CREATE INDEX IF NOT EXISTS idx_akshare_forex_crypto_source_collected ON akshare_forex_crypto_data(source_id, collected_at DESC);
CREATE INDEX IF NOT EXISTS idx_akshare_forex_crypto_date_range ON akshare_forex_crypto_data(date DESC);
CREATE INDEX IF NOT EXISTS idx_akshare_forex_crypto_subtype ON akshare_forex_crypto_data(data_subtype);

-- TimescaleDB集成
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        IF NOT (SELECT public.hypertable_relation_id('akshare_forex_crypto_data'::regclass) IS NOT NULL) THEN
            PERFORM create_hypertable('akshare_forex_crypto_data', 'date', chunk_time_interval => INTERVAL '1 month');
            RAISE NOTICE 'TimescaleDB超表已创建: akshare_forex_crypto_data';
        ELSE
            RAISE NOTICE 'akshare_forex_crypto_data已经是TimescaleDB超表';
        END IF;
    ELSE
        RAISE NOTICE 'TimescaleDB扩展未安装，使用标准PostgreSQL表';
    END IF;
END
$$;

COMMENT ON TABLE akshare_forex_crypto_data IS 'AKShare外汇/数字货币数据';
COMMENT ON COLUMN akshare_forex_crypto_data.source_id IS '数据源ID';
COMMENT ON COLUMN akshare_forex_crypto_data.symbol IS '交易对代码（如USDCNY、BTCUSDT）';
COMMENT ON COLUMN akshare_forex_crypto_data.date IS '交易日期';
COMMENT ON COLUMN akshare_forex_crypto_data.data_subtype IS '数据类型（forex/crypto）';

