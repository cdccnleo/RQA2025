-- AKShare新闻数据表结构
-- 支持TimescaleDB超表

CREATE TABLE IF NOT EXISTS akshare_news_data (
    id BIGSERIAL PRIMARY KEY,
    source_id VARCHAR(50) NOT NULL,
    title TEXT NOT NULL,
    content TEXT,
    summary TEXT,
    publish_time TIMESTAMP WITH TIME ZONE,
    url VARCHAR(500),
    category VARCHAR(100),
    tags VARCHAR(500),
    news_source VARCHAR(100),
    news_source_code VARCHAR(50),
    update_frequency VARCHAR(50),
    delay VARCHAR(50),
    hot_index INTEGER,
    data_source VARCHAR(50) DEFAULT 'akshare',
    collected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    persistence_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_id, title, publish_time)
);

CREATE INDEX IF NOT EXISTS idx_akshare_news_source_date ON akshare_news_data(news_source_code, publish_time DESC);
CREATE INDEX IF NOT EXISTS idx_akshare_news_source_collected ON akshare_news_data(source_id, collected_at DESC);
CREATE INDEX IF NOT EXISTS idx_akshare_news_date_range ON akshare_news_data(publish_time DESC);
CREATE INDEX IF NOT EXISTS idx_akshare_news_category ON akshare_news_data(category);
CREATE INDEX IF NOT EXISTS idx_akshare_news_title_search ON akshare_news_data USING gin(to_tsvector('simple', title));

-- TimescaleDB集成 (如果TimescaleDB扩展已安装)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        IF NOT (SELECT public.hypertable_relation_id('akshare_news_data'::regclass) IS NOT NULL) THEN
            PERFORM create_hypertable('akshare_news_data', 'publish_time', chunk_time_interval => INTERVAL '1 month');
            RAISE NOTICE 'TimescaleDB超表已创建: akshare_news_data';
        ELSE
            RAISE NOTICE 'akshare_news_data已经是TimescaleDB超表';
        END IF;
    ELSE
        RAISE NOTICE 'TimescaleDB扩展未安装，使用标准PostgreSQL表';
    END IF;
END
$$;

COMMENT ON TABLE akshare_news_data IS 'AKShare新闻数据';
COMMENT ON COLUMN akshare_news_data.source_id IS '数据源ID';
COMMENT ON COLUMN akshare_news_data.title IS '新闻标题';
COMMENT ON COLUMN akshare_news_data.content IS '新闻内容';
COMMENT ON COLUMN akshare_news_data.publish_time IS '发布时间';
COMMENT ON COLUMN akshare_news_data.news_source IS '新闻源名称';
COMMENT ON COLUMN akshare_news_data.news_source_code IS '新闻源代码';

