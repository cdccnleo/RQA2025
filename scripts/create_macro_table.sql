-- Create macro economic data table
CREATE TABLE IF NOT EXISTS akshare_macro_data (
    id BIGSERIAL PRIMARY KEY,
    source_id VARCHAR(50) NOT NULL,
    country VARCHAR(20) NOT NULL,
    indicator_type VARCHAR(50) NOT NULL,
    indicator_name VARCHAR(100),
    period VARCHAR(50),
    date DATE NOT NULL,
    value NUMERIC(20, 6),
    forecast_value NUMERIC(20, 6),
    previous_value NUMERIC(20, 6),
    unit VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_id, country, indicator_type, date)
);

CREATE INDEX IF NOT EXISTS idx_macro_country_date ON akshare_macro_data(country, date DESC);
CREATE INDEX IF NOT EXISTS idx_macro_indicator ON akshare_macro_data(indicator_type, date DESC);
CREATE INDEX IF NOT EXISTS idx_macro_source ON akshare_macro_data(source_id, date DESC);
