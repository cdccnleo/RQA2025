-- AKShare期货/期权数据表结构
-- 支持TimescaleDB超表

CREATE TABLE IF NOT EXISTS akshare_futures_data (
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
    open_interest BIGINT,
    settlement_price DECIMAL(15, 6),
    futures_type VARCHAR(50),
    data_source VARCHAR(50) DEFAULT 'akshare',
    collected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    persistence_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_id, symbol, date)
);

CREATE INDEX IF NOT EXISTS idx_akshare_futures_symbol_date ON akshare_futures_data(symbol, date DESC);
CREATE INDEX IF NOT EXISTS idx_akshare_futures_source_collected ON akshare_futures_data(source_id, collected_at DESC);
CREATE INDEX IF NOT EXISTS idx_akshare_futures_date_range ON akshare_futures_data(date DESC);
CREATE INDEX IF NOT EXISTS idx_akshare_futures_type ON akshare_futures_data(futures_type);

-- TimescaleDB集成
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        IF NOT (SELECT public.hypertable_relation_id('akshare_futures_data'::regclass) IS NOT NULL) THEN
            PERFORM create_hypertable('akshare_futures_data', 'date', chunk_time_interval => INTERVAL '1 month');
            RAISE NOTICE 'TimescaleDB超表已创建: akshare_futures_data';
        ELSE
            RAISE NOTICE 'akshare_futures_data已经是TimescaleDB超表';
        END IF;
    ELSE
        RAISE NOTICE 'TimescaleDB扩展未安装，使用标准PostgreSQL表';
    END IF;
END
$$;

COMMENT ON TABLE akshare_futures_data IS 'AKShare期货/期权数据';
COMMENT ON COLUMN akshare_futures_data.source_id IS '数据源ID';
COMMENT ON COLUMN akshare_futures_data.symbol IS '合约代码';
COMMENT ON COLUMN akshare_futures_data.date IS '交易日期';
COMMENT ON COLUMN akshare_futures_data.futures_type IS '期货类型（商品期货/金融期货等）';
COMMENT ON COLUMN akshare_futures_data.open_interest IS '持仓量';

