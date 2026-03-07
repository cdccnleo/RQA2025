-- AKShare基金数据表结构
-- 支持TimescaleDB超表

CREATE TABLE IF NOT EXISTS akshare_fund_data (
    id BIGSERIAL PRIMARY KEY,
    source_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    net_value DECIMAL(15, 6),
    accumulated_value DECIMAL(15, 6),
    fund_type VARCHAR(50),
    daily_return DECIMAL(10, 4),
    data_source VARCHAR(50) DEFAULT 'akshare',
    collected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    persistence_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_id, symbol, date)
);

CREATE INDEX IF NOT EXISTS idx_akshare_fund_symbol_date ON akshare_fund_data(symbol, date DESC);
CREATE INDEX IF NOT EXISTS idx_akshare_fund_source_collected ON akshare_fund_data(source_id, collected_at DESC);
CREATE INDEX IF NOT EXISTS idx_akshare_fund_date_range ON akshare_fund_data(date DESC);
CREATE INDEX IF NOT EXISTS idx_akshare_fund_type ON akshare_fund_data(fund_type);

-- TimescaleDB集成
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        IF NOT (SELECT public.hypertable_relation_id('akshare_fund_data'::regclass) IS NOT NULL) THEN
            PERFORM create_hypertable('akshare_fund_data', 'date', chunk_time_interval => INTERVAL '1 month');
            RAISE NOTICE 'TimescaleDB超表已创建: akshare_fund_data';
        ELSE
            RAISE NOTICE 'akshare_fund_data已经是TimescaleDB超表';
        END IF;
    ELSE
        RAISE NOTICE 'TimescaleDB扩展未安装，使用标准PostgreSQL表';
    END IF;
END
$$;

COMMENT ON TABLE akshare_fund_data IS 'AKShare基金数据';
COMMENT ON COLUMN akshare_fund_data.source_id IS '数据源ID';
COMMENT ON COLUMN akshare_fund_data.symbol IS '基金代码';
COMMENT ON COLUMN akshare_fund_data.date IS '净值日期';
COMMENT ON COLUMN akshare_fund_data.net_value IS '单位净值';
COMMENT ON COLUMN akshare_fund_data.accumulated_value IS '累计净值';
COMMENT ON COLUMN akshare_fund_data.fund_type IS '基金类型（ETF/LOF/开放式基金等）';

