-- RQA2025 历史数据采集表结构初始化
-- Historical Data Collection Tables Initialization

-- ===============================================
-- 历史股票数据表 (核心表)
-- ===============================================
CREATE TABLE IF NOT EXISTS historical_stock_data (
    -- 基本标识字段
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    data_source VARCHAR(50) NOT NULL DEFAULT 'unknown',

    -- OHLC价格数据
    open_price DECIMAL(10,2),
    high_price DECIMAL(10,2),
    low_price DECIMAL(10,2),
    close_price DECIMAL(10,2),
    adj_close DECIMAL(10,2),

    -- 成交量数据
    volume BIGINT,
    amount DECIMAL(15,2),

    -- 技术指标（可选）
    ma5 DECIMAL(10,2),    -- 5日均线
    ma10 DECIMAL(10,2),   -- 10日均线
    ma20 DECIMAL(10,2),   -- 20日均线
    ma30 DECIMAL(10,2),   -- 30日均线
    ma60 DECIMAL(10,2),   -- 60日均线

    -- 波动率指标
    returns DECIMAL(8,6), -- 日收益率
    volatility DECIMAL(8,6), -- 波动率

    -- 数据质量字段
    quality_score DECIMAL(3,2) DEFAULT 0.0 CHECK (quality_score >= 0.0 AND quality_score <= 1.0),
    data_quality_level VARCHAR(20) DEFAULT 'unknown',

    -- 批次和元数据
    batch_id VARCHAR(100),
    collection_timestamp TIMESTAMP WITH TIME ZONE,
    source_metadata JSONB,

    -- 审计字段
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- 主键约束
    PRIMARY KEY (symbol, date)
);

-- ===============================================
-- 历史指数数据表
-- ===============================================
CREATE TABLE IF NOT EXISTS historical_index_data (
    -- 基本标识字段
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    data_source VARCHAR(50) NOT NULL DEFAULT 'unknown',

    -- 指数OHLC数据
    open_value DECIMAL(12,2),
    high_value DECIMAL(12,2),
    low_value DECIMAL(12,2),
    close_value DECIMAL(12,2),

    -- 成交量数据
    volume BIGINT,
    amount DECIMAL(15,2),

    -- 指数特有指标
    pe_ratio DECIMAL(8,2),    -- 市盈率
    pb_ratio DECIMAL(8,2),    -- 市净率
    dividend_yield DECIMAL(6,4), -- 股息率
    turnover_rate DECIMAL(6,4),  -- 换手率

    -- 数据质量字段
    quality_score DECIMAL(3,2) DEFAULT 0.0 CHECK (quality_score >= 0.0 AND quality_score <= 1.0),
    data_quality_level VARCHAR(20) DEFAULT 'unknown',

    -- 批次和元数据
    batch_id VARCHAR(100),
    collection_timestamp TIMESTAMP WITH TIME ZONE,
    source_metadata JSONB,

    -- 审计字段
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- 主键约束
    PRIMARY KEY (symbol, date)
);

-- ===============================================
-- 历史基金数据表
-- ===============================================
CREATE TABLE IF NOT EXISTS historical_fund_data (
    -- 基本标识字段
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    data_source VARCHAR(50) NOT NULL DEFAULT 'unknown',

    -- 基金净值数据
    nav DECIMAL(10,4),        -- 单位净值
    accumulated_nav DECIMAL(10,4), -- 累计净值

    -- 基金表现数据
    daily_return DECIMAL(8,6), -- 日收益率
    weekly_return DECIMAL(8,6), -- 周收益率
    monthly_return DECIMAL(8,6), -- 月收益率

    -- 基金规模和流动性
    total_assets DECIMAL(15,2), -- 总资产
    shares_outstanding BIGINT, -- 总份额

    -- 数据质量字段
    quality_score DECIMAL(3,2) DEFAULT 0.0 CHECK (quality_score >= 0.0 AND quality_score <= 1.0),
    data_quality_level VARCHAR(20) DEFAULT 'unknown',

    -- 批次和元数据
    batch_id VARCHAR(100),
    collection_timestamp TIMESTAMP WITH TIME ZONE,
    source_metadata JSONB,

    -- 审计字段
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- 主键约束
    PRIMARY KEY (symbol, date)
);

-- ===============================================
-- 数据采集批次记录表
-- ===============================================
CREATE TABLE IF NOT EXISTS data_collection_batches (
    -- 批次标识
    batch_id VARCHAR(100) PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    year INTEGER NOT NULL,

    -- 采集配置
    data_source VARCHAR(50) NOT NULL,
    data_type VARCHAR(20) NOT NULL DEFAULT 'stock',
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,

    -- 采集结果
    status VARCHAR(20) NOT NULL DEFAULT 'pending', -- pending, processing, completed, failed
    records_collected INTEGER DEFAULT 0,
    quality_score DECIMAL(3,2),

    -- 时间戳
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,

    -- 错误信息
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,

    -- 元数据
    collection_config JSONB,
    source_metadata JSONB
);

-- ===============================================
-- 数据质量监控表
-- ===============================================
CREATE TABLE IF NOT EXISTS data_quality_metrics (
    -- 监控标识
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    data_type VARCHAR(20) NOT NULL DEFAULT 'stock',
    check_date DATE NOT NULL,

    -- 完整性指标
    completeness_score DECIMAL(3,2), -- 完整性评分 (0-1)
    total_expected_records INTEGER,
    total_actual_records INTEGER,
    missing_records INTEGER,

    -- 准确性指标
    accuracy_score DECIMAL(3,2), -- 准确性评分 (0-1)
    invalid_records INTEGER,
    outlier_records INTEGER,

    -- 时效性指标
    timeliness_score DECIMAL(3,2), -- 时效性评分 (0-1)
    oldest_record_age_days INTEGER,
    newest_record_age_days INTEGER,

    -- 一致性指标
    consistency_score DECIMAL(3,2), -- 一致性评分 (0-1)
    data_source_consistency DECIMAL(3,2),

    -- 总体质量评分
    overall_quality_score DECIMAL(3,2),

    -- 详细报告
    quality_report JSONB,

    -- 时间戳
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ===============================================
-- 数据源状态监控表
-- ===============================================
CREATE TABLE IF NOT EXISTS data_source_status (
    -- 数据源标识
    data_source VARCHAR(50) PRIMARY KEY,
    data_type VARCHAR(20) NOT NULL,

    -- 状态信息
    last_check_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    status VARCHAR(20) DEFAULT 'unknown', -- active, inactive, error
    response_time_ms INTEGER,

    -- 可用性统计
    total_checks INTEGER DEFAULT 0,
    successful_checks INTEGER DEFAULT 0,
    failed_checks INTEGER DEFAULT 0,
    uptime_percentage DECIMAL(5,2),

    -- 质量统计
    avg_quality_score DECIMAL(3,2),
    min_quality_score DECIMAL(3,2),
    max_quality_score DECIMAL(3,2),

    -- 错误统计
    last_error_message TEXT,
    consecutive_failures INTEGER DEFAULT 0,

    -- 配置信息
    config_hash VARCHAR(64), -- 配置变更检测
    version_info VARCHAR(50),

    -- 更新时间戳
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ===============================================
-- 创建TimescaleDB超表
-- ===============================================

-- 转换历史股票数据表为超表
SELECT create_hypertable(
    'historical_stock_data',
    'date',
    chunk_time_interval => INTERVAL '1 year',
    if_not_exists => TRUE
);

-- 转换历史指数数据表为超表
SELECT create_hypertable(
    'historical_index_data',
    'date',
    chunk_time_interval => INTERVAL '1 year',
    if_not_exists => TRUE
);

-- 转换历史基金数据表为超表
SELECT create_hypertable(
    'historical_fund_data',
    'date',
    chunk_time_interval => INTERVAL '1 year',
    if_not_exists => TRUE
);

-- 转换数据质量监控表为超表
SELECT create_hypertable(
    'data_quality_metrics',
    'check_date',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

-- ===============================================
-- 设置压缩策略
-- ===============================================

-- 为历史股票数据设置压缩策略（3个月后的数据自动压缩）
ALTER TABLE historical_stock_data SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol,data_source',
    timescaledb.compress_orderby = 'date DESC, quality_score DESC'
);

SELECT add_compression_policy(
    'historical_stock_data',
    INTERVAL '3 months',
    if_not_exists => TRUE
);

-- 为历史指数数据设置压缩策略
ALTER TABLE historical_index_data SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol,data_source',
    timescaledb.compress_orderby = 'date DESC, quality_score DESC'
);

SELECT add_compression_policy(
    'historical_index_data',
    INTERVAL '3 months',
    if_not_exists => TRUE
);

-- 为历史基金数据设置压缩策略
ALTER TABLE historical_fund_data SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol,data_source',
    timescaledb.compress_orderby = 'date DESC, quality_score DESC'
);

SELECT add_compression_policy(
    'historical_fund_data',
    INTERVAL '3 months',
    if_not_exists => TRUE
);

-- ===============================================
-- 创建索引
-- ===============================================

-- 历史股票数据索引
CREATE INDEX IF NOT EXISTS idx_historical_stock_symbol_date
    ON historical_stock_data (symbol, date DESC);

CREATE INDEX IF NOT EXISTS idx_historical_stock_date_symbol
    ON historical_stock_data (date DESC, symbol);

CREATE INDEX IF NOT EXISTS idx_historical_stock_source
    ON historical_stock_data (data_source);

CREATE INDEX IF NOT EXISTS idx_historical_stock_quality
    ON historical_stock_data (quality_score);

CREATE INDEX IF NOT EXISTS idx_historical_stock_batch
    ON historical_stock_data (batch_id);

-- 历史指数数据索引
CREATE INDEX IF NOT EXISTS idx_historical_index_symbol_date
    ON historical_index_data (symbol, date DESC);

CREATE INDEX IF NOT EXISTS idx_historical_index_date_symbol
    ON historical_index_data (date DESC, symbol);

CREATE INDEX IF NOT EXISTS idx_historical_index_source
    ON historical_index_data (data_source);

CREATE INDEX IF NOT EXISTS idx_historical_index_quality
    ON historical_index_data (quality_score);

-- 数据采集批次索引
CREATE INDEX IF NOT EXISTS idx_batches_symbol_year
    ON data_collection_batches (symbol, year);

CREATE INDEX IF NOT EXISTS idx_batches_status
    ON data_collection_batches (status);

CREATE INDEX IF NOT EXISTS idx_batches_created_at
    ON data_collection_batches (created_at DESC);

-- 数据质量监控索引
CREATE INDEX IF NOT EXISTS idx_quality_symbol_date
    ON data_quality_metrics (symbol, check_date DESC);

CREATE INDEX IF NOT EXISTS idx_quality_overall_score
    ON data_quality_metrics (overall_quality_score);

-- ===============================================
-- 创建分区保留策略
-- ===============================================

-- 为历史数据设置保留策略（保留10年数据）
SELECT add_retention_policy(
    'historical_stock_data',
    INTERVAL '10 years',
    if_not_exists => TRUE
);

SELECT add_retention_policy(
    'historical_index_data',
    INTERVAL '10 years',
    if_not_exists => TRUE
);

SELECT add_retention_policy(
    'historical_fund_data',
    INTERVAL '10 years',
    if_not_exists => TRUE
);

-- 为质量监控数据设置保留策略（保留2年）
SELECT add_retention_policy(
    'data_quality_metrics',
    INTERVAL '2 years',
    if_not_exists => TRUE
);

-- ===============================================
-- 创建数据完整性检查函数
-- ===============================================

-- 检查数据完整性的函数
CREATE OR REPLACE FUNCTION check_data_completeness(
    p_symbol VARCHAR(20),
    p_start_date DATE,
    p_end_date DATE,
    p_data_type VARCHAR(20) DEFAULT 'stock'
)
RETURNS TABLE (
    symbol VARCHAR(20),
    total_expected_days INTEGER,
    total_actual_days INTEGER,
    completeness_ratio DECIMAL(5,4),
    missing_days INTEGER,
    first_date DATE,
    last_date DATE
) AS $$
DECLARE
    table_name TEXT;
    total_trading_days INTEGER;
BEGIN
    -- 确定表名
    table_name := CASE
        WHEN p_data_type = 'stock' THEN 'historical_stock_data'
        WHEN p_data_type = 'index' THEN 'historical_index_data'
        WHEN p_data_type = 'fund' THEN 'historical_fund_data'
        ELSE 'historical_stock_data'
    END;

    -- 计算预期交易日数（简化版：假设每周5个交易日）
    total_trading_days := ((p_end_date - p_start_date) / 7) * 5;

    RETURN QUERY EXECUTE format('
        SELECT
            $1::VARCHAR(20) as symbol,
            $2::INTEGER as total_expected_days,
            COUNT(*)::INTEGER as total_actual_days,
            ROUND(COUNT(*)::DECIMAL / $2, 4) as completeness_ratio,
            ($2 - COUNT(*))::INTEGER as missing_days,
            MIN(date) as first_date,
            MAX(date) as last_date
        FROM %I
        WHERE symbol = $1
          AND date BETWEEN $3 AND $4
          AND quality_score >= 0.8
    ', table_name)
    USING p_symbol, total_trading_days, p_start_date, p_end_date;
END;
$$ LANGUAGE plpgsql;

-- ===============================================
-- 创建数据质量报告函数
-- ===============================================

CREATE OR REPLACE FUNCTION generate_quality_report(
    p_symbol VARCHAR(20),
    p_data_type VARCHAR(20) DEFAULT 'stock',
    p_days INTEGER DEFAULT 30
)
RETURNS JSONB AS $$
DECLARE
    table_name TEXT;
    result JSONB;
BEGIN
    -- 确定表名
    table_name := CASE
        WHEN p_data_type = 'stock' THEN 'historical_stock_data'
        WHEN p_data_type = 'index' THEN 'historical_index_data'
        WHEN p_data_type = 'fund' THEN 'historical_fund_data'
        ELSE 'historical_stock_data'
    END;

    -- 生成质量报告
    EXECUTE format('
        SELECT jsonb_build_object(
            ''symbol'', $1,
            ''data_type'', $2,
            ''period_days'', $3,
            ''metrics'', jsonb_build_object(
                ''total_records'', COUNT(*),
                ''avg_quality_score'', ROUND(AVG(quality_score)::DECIMAL, 3),
                ''min_quality_score'', MIN(quality_score),
                ''max_quality_score'', MAX(quality_score),
                ''null_values'', jsonb_build_object(
                    ''open_price_nulls'', COUNT(*) FILTER (WHERE open_price IS NULL),
                    ''close_price_nulls'', COUNT(*) FILTER (WHERE close_price IS NULL),
                    ''volume_nulls'', COUNT(*) FILTER (WHERE volume IS NULL)
                ),
                ''data_sources'', jsonb_agg(DISTINCT data_source),
                ''date_range'', jsonb_build_object(
                    ''oldest'', MIN(date),
                    ''newest'', MAX(date),
                    ''days_span'', MAX(date) - MIN(date)
                )
            ),
            ''generated_at'', NOW()
        )
        FROM %I
        WHERE symbol = $1
          AND date >= CURRENT_DATE - INTERVAL ''%s days''
    ', table_name, p_days)
    INTO result
    USING p_symbol, p_data_type, p_days;

    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- ===============================================
-- 插入初始化记录
-- ===============================================

INSERT INTO system_health (component, status, details)
VALUES (
    'historical_data_tables',
    'initialized',
    jsonb_build_object(
        'tables_created', jsonb_build_array(
            'historical_stock_data',
            'historical_index_data',
            'historical_fund_data',
            'data_collection_batches',
            'data_quality_metrics',
            'data_source_status'
        ),
        'hypertables_created', true,
        'compression_policies', true,
        'indexes_created', true,
        'retention_policies', true,
        'functions_created', jsonb_build_array(
            'check_data_completeness',
            'generate_quality_report'
        ),
        'initialized_at', NOW()
    )
)
ON CONFLICT (id) DO NOTHING;

-- ===============================================
-- 记录初始化完成
-- ===============================================

DO $$
BEGIN
    RAISE NOTICE '历史数据采集表结构初始化完成';
    RAISE NOTICE '创建的表: historical_stock_data, historical_index_data, historical_fund_data';
    RAISE NOTICE '创建的辅助表: data_collection_batches, data_quality_metrics, data_source_status';
    RAISE NOTICE '超表和压缩策略已配置';
    RAISE NOTICE '索引和保留策略已设置';
END $$;