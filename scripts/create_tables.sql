-- RQA2025 数据库表创建脚本
-- 创建应用所需的数据库表

-- AKShare股票数据表
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

-- 创建索引以提高查询性能
CREATE INDEX IF NOT EXISTS idx_akshare_stock_symbol ON akshare_stock_data(symbol);
CREATE INDEX IF NOT EXISTS idx_akshare_stock_date ON akshare_stock_data(date);
CREATE INDEX IF NOT EXISTS idx_akshare_stock_source ON akshare_stock_data(source_id);
CREATE INDEX IF NOT EXISTS idx_akshare_stock_type ON akshare_stock_data(data_type);

-- AKShare指数数据表
CREATE TABLE IF NOT EXISTS akshare_index_data (
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
    data_source VARCHAR(50) DEFAULT 'akshare',
    collected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    persistence_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_akshare_index_record UNIQUE(source_id, symbol, date, data_type)
);

-- AKShare基金数据表
CREATE TABLE IF NOT EXISTS akshare_fund_data (
    id BIGSERIAL PRIMARY KEY,
    source_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    data_type VARCHAR(20) NOT NULL DEFAULT 'daily',
    net_value DECIMAL(15, 6),
    accumulative_net_value DECIMAL(15, 6),
    growth_rate DECIMAL(10, 4),
    data_source VARCHAR(50) DEFAULT 'akshare',
    collected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    persistence_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_akshare_fund_record UNIQUE(source_id, symbol, date, data_type)
);

-- AKShare宏观经济数据表
CREATE TABLE IF NOT EXISTS akshare_macro_data (
    id BIGSERIAL PRIMARY KEY,
    source_id VARCHAR(50) NOT NULL,
    indicator_name VARCHAR(200) NOT NULL,
    date DATE NOT NULL,
    value DECIMAL(20, 6),
    unit VARCHAR(50),
    data_source VARCHAR(50) DEFAULT 'akshare',
    collected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    persistence_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_akshare_macro_record UNIQUE(source_id, indicator_name, date)
);

-- AKShare新闻数据表
CREATE TABLE IF NOT EXISTS akshare_news_data (
    id BIGSERIAL PRIMARY KEY,
    source_id VARCHAR(50) NOT NULL,
    title TEXT NOT NULL,
    content TEXT,
    url VARCHAR(500),
    publish_time TIMESTAMP WITH TIME ZONE,
    data_source VARCHAR(50) DEFAULT 'akshare',
    collected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    persistence_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- AKShare另类数据表
CREATE TABLE IF NOT EXISTS akshare_alternative_data (
    id BIGSERIAL PRIMARY KEY,
    source_id VARCHAR(50) NOT NULL,
    data_type VARCHAR(50) NOT NULL,
    symbol VARCHAR(50),
    date DATE,
    json_data JSONB,
    data_source VARCHAR(50) DEFAULT 'akshare',
    collected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    persistence_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 数据采集任务表
CREATE TABLE IF NOT EXISTS data_collection_tasks (
    id BIGSERIAL PRIMARY KEY,
    task_id VARCHAR(100) NOT NULL UNIQUE,
    source_id VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    priority INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3
);

-- 数据质量监控表
CREATE TABLE IF NOT EXISTS data_quality_metrics (
    id BIGSERIAL PRIMARY KEY,
    source_id VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10, 4),
    threshold_value DECIMAL(10, 4),
    status VARCHAR(20) NOT NULL,
    measured_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    details JSONB
);

-- 应用配置表
CREATE TABLE IF NOT EXISTS app_config (
    id BIGSERIAL PRIMARY KEY,
    config_key VARCHAR(200) NOT NULL UNIQUE,
    config_value TEXT,
    config_type VARCHAR(50) DEFAULT 'string',
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(100)
);

-- 创建相应的索引
CREATE INDEX IF NOT EXISTS idx_data_collection_tasks_status ON data_collection_tasks(status);
CREATE INDEX IF NOT EXISTS idx_data_collection_tasks_source ON data_collection_tasks(source_id);
CREATE INDEX IF NOT EXISTS idx_data_quality_metrics_source ON data_quality_metrics(source_id);
CREATE INDEX IF NOT EXISTS idx_data_quality_metrics_status ON data_quality_metrics(status);
CREATE INDEX IF NOT EXISTS idx_app_config_key ON app_config(config_key);