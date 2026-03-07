-- AKShare宏观经济数据表结构

CREATE TABLE IF NOT EXISTS akshare_macro_data (
    id BIGSERIAL PRIMARY KEY,
    source_id VARCHAR(50) NOT NULL,
    indicator VARCHAR(100) NOT NULL,
    date DATE NOT NULL,
    value DECIMAL(20, 6),
    unit VARCHAR(50),
    macro_type VARCHAR(50),
    period VARCHAR(50),
    data_source VARCHAR(50) DEFAULT 'akshare',
    collected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    persistence_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_id, indicator, date)
);

CREATE INDEX IF NOT EXISTS idx_akshare_macro_indicator_date ON akshare_macro_data(indicator, date DESC);
CREATE INDEX IF NOT EXISTS idx_akshare_macro_source_collected ON akshare_macro_data(source_id, collected_at DESC);
CREATE INDEX IF NOT EXISTS idx_akshare_macro_date_range ON akshare_macro_data(date DESC);
CREATE INDEX IF NOT EXISTS idx_akshare_macro_type ON akshare_macro_data(macro_type);
CREATE INDEX IF NOT EXISTS idx_akshare_macro_indicator ON akshare_macro_data(indicator);

COMMENT ON TABLE akshare_macro_data IS 'AKShare宏观经济数据';
COMMENT ON COLUMN akshare_macro_data.source_id IS '数据源ID';
COMMENT ON COLUMN akshare_macro_data.indicator IS '经济指标名称（如GDP、CPI、PPI）';
COMMENT ON COLUMN akshare_macro_data.date IS '数据日期';
COMMENT ON COLUMN akshare_macro_data.value IS '指标数值';
COMMENT ON COLUMN akshare_macro_data.unit IS '单位';
COMMENT ON COLUMN akshare_macro_data.macro_type IS '宏观经济类型（gdp/cpi/ppi/pmi等）';
COMMENT ON COLUMN akshare_macro_data.period IS '统计周期（月度/季度/年度）';

