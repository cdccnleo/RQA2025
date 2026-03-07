-- AKShare另类数据表结构
-- 支持TimescaleDB超表

CREATE TABLE IF NOT EXISTS akshare_alternative_data (
    id BIGSERIAL PRIMARY KEY,
    source_id VARCHAR(50) NOT NULL,
    data_category VARCHAR(50) NOT NULL,
    data_subtype VARCHAR(100) NOT NULL,
    keyword VARCHAR(200),
    city VARCHAR(100),
    index_value DECIMAL(20, 6),
    index_name VARCHAR(200),
    rank INTEGER,
    trend VARCHAR(50),
    date DATE,
    value DECIMAL(20, 6),
    unit VARCHAR(50),
    -- 社交媒体数据字段
    -- 消费数据字段
    movie_name VARCHAR(200),
    boxoffice DECIMAL(20, 2),
    boxoffice_unit VARCHAR(50),
    product_name VARCHAR(500),
    sales_volume BIGINT,
    price DECIMAL(15, 2),
    shop_name VARCHAR(200),
    -- 供应链数据字段
    change_value DECIMAL(15, 6),
    change_pct DECIMAL(10, 4),
    -- 环境数据字段
    aqi INTEGER,
    pm25 DECIMAL(10, 2),
    pm10 DECIMAL(10, 2),
    co DECIMAL(10, 2),
    no2 DECIMAL(10, 2),
    o3 DECIMAL(10, 2),
    so2 DECIMAL(10, 2),
    quality_level VARCHAR(50),
    temperature_high DECIMAL(5, 2),
    temperature_low DECIMAL(5, 2),
    weather VARCHAR(100),
    wind_direction VARCHAR(50),
    wind_level VARCHAR(50),
    humidity DECIMAL(5, 2),
    update_frequency VARCHAR(50),
    data_source VARCHAR(50) DEFAULT 'akshare',
    collected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    persistence_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_id, data_category, data_subtype, keyword, city, date)
);

CREATE INDEX IF NOT EXISTS idx_akshare_alt_category_subtype ON akshare_alternative_data(data_category, data_subtype);
CREATE INDEX IF NOT EXISTS idx_akshare_alt_source_collected ON akshare_alternative_data(source_id, collected_at DESC);
CREATE INDEX IF NOT EXISTS idx_akshare_alt_date_range ON akshare_alternative_data(date DESC);
CREATE INDEX IF NOT EXISTS idx_akshare_alt_keyword ON akshare_alternative_data(keyword);
CREATE INDEX IF NOT EXISTS idx_akshare_alt_city ON akshare_alternative_data(city);

-- TimescaleDB集成 (如果TimescaleDB扩展已安装)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        IF NOT (SELECT public.hypertable_relation_id('akshare_alternative_data'::regclass) IS NOT NULL) THEN
            PERFORM create_hypertable('akshare_alternative_data', 'date', chunk_time_interval => INTERVAL '1 month');
            RAISE NOTICE 'TimescaleDB超表已创建: akshare_alternative_data';
        ELSE
            RAISE NOTICE 'akshare_alternative_data已经是TimescaleDB超表';
        END IF;
    ELSE
        RAISE NOTICE 'TimescaleDB扩展未安装，使用标准PostgreSQL表';
    END IF;
END
$$;

COMMENT ON TABLE akshare_alternative_data IS 'AKShare另类数据';
COMMENT ON COLUMN akshare_alternative_data.source_id IS '数据源ID';
COMMENT ON COLUMN akshare_alternative_data.data_category IS '数据类别（社交媒体/消费数据/供应链数据/环境数据）';
COMMENT ON COLUMN akshare_alternative_data.data_subtype IS '数据子类型';
COMMENT ON COLUMN akshare_alternative_data.keyword IS '关键词（用于搜索指数、淘宝销量等）';
COMMENT ON COLUMN akshare_alternative_data.city IS '城市（用于空气质量、天气数据等）';

