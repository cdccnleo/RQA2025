-- RQA2025 PostgreSQL数据库表结构优化脚本
-- 优化索引、约束和性能

-- =============================================================================
-- 1. 股票数据表优化
-- =============================================================================

-- 优化akshare_stock_data表的索引（删除旧索引，创建优化索引）
DROP INDEX IF EXISTS idx_akshare_stock_symbol;
DROP INDEX IF EXISTS idx_akshare_stock_date;
DROP INDEX IF EXISTS idx_akshare_stock_source;
DROP INDEX IF EXISTS idx_akshare_stock_type;

-- 创建优化的复合索引（基于查询模式）
CREATE INDEX IF NOT EXISTS idx_akshare_stock_symbol_date_desc ON akshare_stock_data(symbol, date DESC);
CREATE INDEX IF NOT EXISTS idx_akshare_stock_source_symbol_date ON akshare_stock_data(source_id, symbol, date DESC);
CREATE INDEX IF NOT EXISTS idx_akshare_stock_date_range ON akshare_stock_data(date DESC) WHERE date >= CURRENT_DATE - INTERVAL '1 year';
CREATE INDEX IF NOT EXISTS idx_akshare_stock_data_type_date ON akshare_stock_data(data_type, date DESC);
CREATE INDEX IF NOT EXISTS idx_akshare_stock_collected_at ON akshare_stock_data(collected_at DESC);

-- 添加数据完整性约束
ALTER TABLE akshare_stock_data
ADD CONSTRAINT chk_positive_prices CHECK (
    (open_price IS NULL OR open_price > 0) AND
    (high_price IS NULL OR high_price > 0) AND
    (low_price IS NULL OR low_price > 0) AND
    (close_price IS NULL OR close_price > 0)
),
ADD CONSTRAINT chk_price_ranges CHECK (
    (high_price IS NULL OR low_price IS NULL OR high_price >= low_price) AND
    (open_price IS NULL OR high_price IS NULL OR high_price >= open_price) AND
    (close_price IS NULL OR high_price IS NULL OR high_price >= close_price) AND
    (open_price IS NULL OR low_price IS NULL OR open_price >= low_price) AND
    (close_price IS NULL OR low_price IS NULL OR close_price >= low_price)
),
ADD CONSTRAINT chk_positive_volume CHECK (volume IS NULL OR volume >= 0),
ADD CONSTRAINT chk_positive_amount CHECK (amount IS NULL OR amount >= 0),
ADD CONSTRAINT chk_valid_dates CHECK (date <= CURRENT_DATE AND date >= '1990-01-01');

-- =============================================================================
-- 2. 指数数据表优化
-- =============================================================================

-- 优化akshare_index_data表的索引
DROP INDEX IF EXISTS idx_akshare_index_symbol;
DROP INDEX IF EXISTS idx_akshare_index_date;
DROP INDEX IF EXISTS idx_akshare_index_source;
DROP INDEX IF EXISTS idx_akshare_index_type;

-- 创建优化的复合索引
CREATE INDEX IF NOT EXISTS idx_akshare_index_symbol_date_desc ON akshare_index_data(symbol, date DESC);
CREATE INDEX IF NOT EXISTS idx_akshare_index_source_symbol_date ON akshare_index_data(source_id, symbol, date DESC);
CREATE INDEX IF NOT EXISTS idx_akshare_index_date_range ON akshare_index_data(date DESC) WHERE date >= CURRENT_DATE - INTERVAL '1 year';
CREATE INDEX IF NOT EXISTS idx_akshare_index_data_type_date ON akshare_index_data(data_type, date DESC);
CREATE INDEX IF NOT EXISTS idx_akshare_index_collected_at ON akshare_index_data(collected_at DESC);

-- 添加数据完整性约束
ALTER TABLE akshare_index_data
ADD CONSTRAINT chk_index_positive_prices CHECK (
    (open_price IS NULL OR open_price > 0) AND
    (high_price IS NULL OR high_price > 0) AND
    (low_price IS NULL OR low_price > 0) AND
    (close_price IS NULL OR close_price > 0)
),
ADD CONSTRAINT chk_index_price_ranges CHECK (
    (high_price IS NULL OR low_price IS NULL OR high_price >= low_price) AND
    (open_price IS NULL OR high_price IS NULL OR high_price >= open_price) AND
    (close_price IS NULL OR high_price IS NULL OR high_price >= close_price) AND
    (open_price IS NULL OR low_price IS NULL OR open_price >= low_price) AND
    (close_price IS NULL OR low_price IS NULL OR close_price >= low_price)
),
ADD CONSTRAINT chk_index_positive_volume CHECK (volume IS NULL OR volume >= 0),
ADD CONSTRAINT chk_index_positive_amount CHECK (amount IS NULL OR amount >= 0),
ADD CONSTRAINT chk_index_valid_dates CHECK (date <= CURRENT_DATE AND date >= '1990-01-01');

-- =============================================================================
-- 3. 基金数据表优化
-- =============================================================================

-- 优化akshare_fund_data表的索引
CREATE INDEX IF NOT EXISTS idx_akshare_fund_symbol_date_desc ON akshare_fund_data(symbol, date DESC);
CREATE INDEX IF NOT EXISTS idx_akshare_fund_source_symbol_date ON akshare_fund_data(source_id, symbol, date DESC);
CREATE INDEX IF NOT EXISTS idx_akshare_fund_date_range ON akshare_fund_data(date DESC) WHERE date >= CURRENT_DATE - INTERVAL '1 year';
CREATE INDEX IF NOT EXISTS idx_akshare_fund_data_type_date ON akshare_fund_data(data_type, date DESC);

-- 添加数据完整性约束
ALTER TABLE akshare_fund_data
ADD CONSTRAINT chk_fund_positive_values CHECK (
    (net_value IS NULL OR net_value > 0) AND
    (accumulative_net_value IS NULL OR accumulative_net_value > 0)
),
ADD CONSTRAINT chk_fund_valid_dates CHECK (date <= CURRENT_DATE AND date >= '1990-01-01');

-- =============================================================================
-- 4. 宏观经济数据表优化
-- =============================================================================

-- 优化akshare_macro_data表的索引
CREATE INDEX IF NOT EXISTS idx_akshare_macro_indicator_date ON akshare_macro_data(indicator_name, date DESC);
CREATE INDEX IF NOT EXISTS idx_akshare_macro_source_indicator_date ON akshare_macro_data(source_id, indicator_name, date DESC);
CREATE INDEX IF NOT EXISTS idx_akshare_macro_date_range ON akshare_macro_data(date DESC) WHERE date >= CURRENT_DATE - INTERVAL '5 years';
CREATE INDEX IF NOT EXISTS idx_akshare_macro_collected_at ON akshare_macro_data(collected_at DESC);

-- 添加数据完整性约束
ALTER TABLE akshare_macro_data
ADD CONSTRAINT chk_macro_valid_dates CHECK (date <= CURRENT_DATE AND date >= '1990-01-01');

-- =============================================================================
-- 5. 新闻数据表优化
-- =============================================================================

-- 优化akshare_news_data表的索引
CREATE INDEX IF NOT EXISTS idx_akshare_news_publish_time ON akshare_news_data(publish_time DESC);
CREATE INDEX IF NOT EXISTS idx_akshare_news_source_publish ON akshare_news_data(source_id, publish_time DESC);
CREATE INDEX IF NOT EXISTS idx_akshare_news_title ON akshare_news_data USING gin(to_tsvector('chinese', title));
CREATE INDEX IF NOT EXISTS idx_akshare_news_content ON akshare_news_data USING gin(to_tsvector('chinese', content));
CREATE INDEX IF NOT EXISTS idx_akshare_news_collected_at ON akshare_news_data(collected_at DESC);

-- =============================================================================
-- 6. 另类数据表优化
-- =============================================================================

-- 优化akshare_alternative_data表的索引
CREATE INDEX IF NOT EXISTS idx_akshare_alt_source_type_date ON akshare_alternative_data(source_id, data_type, date DESC);
CREATE INDEX IF NOT EXISTS idx_akshare_alt_date_range ON akshare_alternative_data(date DESC) WHERE date IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_akshare_alt_json_data ON akshare_alternative_data USING gin(json_data);
CREATE INDEX IF NOT EXISTS idx_akshare_alt_collected_at ON akshare_alternative_data(collected_at DESC);

-- =============================================================================
-- 7. 数据采集任务表优化
-- =============================================================================

-- 优化data_collection_tasks表的索引
DROP INDEX IF EXISTS idx_data_collection_tasks_status;
DROP INDEX IF EXISTS idx_data_collection_tasks_source;

CREATE INDEX IF NOT EXISTS idx_data_collection_tasks_status_priority ON data_collection_tasks(status, priority DESC, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_data_collection_tasks_source_status ON data_collection_tasks(source_id, status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_data_collection_tasks_created_at ON data_collection_tasks(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_data_collection_tasks_started_at ON data_collection_tasks(started_at DESC) WHERE started_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_data_collection_tasks_completed_at ON data_collection_tasks(completed_at DESC) WHERE completed_at IS NOT NULL;

-- 添加数据完整性约束
ALTER TABLE data_collection_tasks
ADD CONSTRAINT chk_tasks_status CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
ADD CONSTRAINT chk_tasks_priority CHECK (priority >= 1 AND priority <= 100),
ADD CONSTRAINT chk_tasks_retry_count CHECK (retry_count >= 0),
ADD CONSTRAINT chk_tasks_timestamps CHECK (
    (started_at IS NULL OR started_at >= created_at) AND
    (completed_at IS NULL OR (started_at IS NOT NULL AND completed_at >= started_at))
);

-- =============================================================================
-- 8. 数据质量监控表优化
-- =============================================================================

-- 优化data_quality_metrics表的索引
DROP INDEX IF EXISTS idx_data_quality_metrics_source;
DROP INDEX IF EXISTS idx_data_quality_metrics_status;

CREATE INDEX IF NOT EXISTS idx_data_quality_metrics_source_time ON data_quality_metrics(source_id, measured_at DESC);
CREATE INDEX IF NOT EXISTS idx_data_quality_metrics_status_time ON data_quality_metrics(status, measured_at DESC);
CREATE INDEX IF NOT EXISTS idx_data_quality_metrics_metric_name ON data_quality_metrics(metric_name, measured_at DESC);
CREATE INDEX IF NOT EXISTS idx_data_quality_metrics_threshold ON data_quality_metrics(threshold_value, measured_at DESC);

-- 添加数据完整性约束
ALTER TABLE data_quality_metrics
ADD CONSTRAINT chk_quality_status CHECK (status IN ('normal', 'warning', 'critical')),
ADD CONSTRAINT chk_quality_values CHECK (
    metric_value >= 0 AND metric_value <= 100 AND
    threshold_value >= 0 AND threshold_value <= 100
);

-- =============================================================================
-- 9. 性能监控和统计表
-- =============================================================================

-- 创建数据采集性能统计表
CREATE TABLE IF NOT EXISTS data_collection_performance (
    id BIGSERIAL PRIMARY KEY,
    source_id VARCHAR(50) NOT NULL,
    collection_date DATE NOT NULL,
    total_records INTEGER NOT NULL DEFAULT 0,
    inserted_records INTEGER NOT NULL DEFAULT 0,
    updated_records INTEGER NOT NULL DEFAULT 0,
    failed_records INTEGER NOT NULL DEFAULT 0,
    processing_time_seconds DECIMAL(10, 3) NOT NULL,
    data_quality_score DECIMAL(5, 2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_collection_performance UNIQUE(source_id, collection_date)
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_collection_perf_source_date ON data_collection_performance(source_id, collection_date DESC);
CREATE INDEX IF NOT EXISTS idx_collection_perf_date ON data_collection_performance(collection_date DESC);
CREATE INDEX IF NOT EXISTS idx_collection_perf_quality ON data_collection_performance(data_quality_score DESC);

-- 添加约束
ALTER TABLE data_collection_performance
ADD CONSTRAINT chk_perf_positive_counts CHECK (
    total_records >= 0 AND inserted_records >= 0 AND
    updated_records >= 0 AND failed_records >= 0
),
ADD CONSTRAINT chk_perf_processing_time CHECK (processing_time_seconds > 0),
ADD CONSTRAINT chk_perf_quality_score CHECK (data_quality_score >= 0 AND data_quality_score <= 100),
ADD CONSTRAINT chk_perf_record_consistency CHECK (total_records = inserted_records + updated_records + failed_records);

-- =============================================================================
-- 10. 触发器和自动维护
-- =============================================================================

-- 创建更新时间戳的触发器函数
CREATE OR REPLACE FUNCTION update_persistence_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.persistence_timestamp = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 为股票数据表添加触发器
DROP TRIGGER IF EXISTS trigger_update_stock_timestamp ON akshare_stock_data;
CREATE TRIGGER trigger_update_stock_timestamp
    BEFORE UPDATE ON akshare_stock_data
    FOR EACH ROW
    EXECUTE FUNCTION update_persistence_timestamp();

-- 为其他表添加类似的触发器
DROP TRIGGER IF EXISTS trigger_update_index_timestamp ON akshare_index_data;
CREATE TRIGGER trigger_update_index_timestamp
    BEFORE UPDATE ON akshare_index_data
    FOR EACH ROW
    EXECUTE FUNCTION update_persistence_timestamp();

DROP TRIGGER IF EXISTS trigger_update_fund_timestamp ON akshare_fund_data;
CREATE TRIGGER trigger_update_fund_timestamp
    BEFORE UPDATE ON akshare_fund_data
    FOR EACH ROW
    EXECUTE FUNCTION update_persistence_timestamp();

DROP TRIGGER IF EXISTS trigger_update_macro_timestamp ON akshare_macro_data;
CREATE TRIGGER trigger_update_macro_timestamp
    BEFORE UPDATE ON akshare_macro_data
    FOR EACH ROW
    EXECUTE FUNCTION update_persistence_timestamp();

-- =============================================================================
-- 11. 性能优化建议
-- =============================================================================

-- 分析查询，建议创建分区表（如果数据量很大）
-- 可以通过以下方式创建分区表：
--
-- -- 创建分区表（按年分区）
-- CREATE TABLE akshare_stock_data_y2024 PARTITION OF akshare_stock_data
--     FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
--
-- -- 创建分区表（按月分区）
-- CREATE TABLE akshare_stock_data_y2024m01 PARTITION OF akshare_stock_data
--     FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- =============================================================================
-- 执行说明
-- =============================================================================

-- 此脚本可以安全地多次执行（使用IF NOT EXISTS和IF EXISTS）
-- 执行顺序：基础表结构 -> 索引 -> 约束 -> 触发器
-- 执行时间：根据数据量大小，通常在几秒到几分钟之间

-- 完成提示
DO $$
BEGIN
    RAISE NOTICE 'PostgreSQL数据库表结构优化脚本执行完成';
    RAISE NOTICE '优化内容：';
    RAISE NOTICE '  - 添加了优化的复合索引';
    RAISE NOTICE '  - 添加了数据完整性约束';
    RAISE NOTICE '  - 创建了性能统计表';
    RAISE NOTICE '  - 添加了自动更新时间戳的触发器';
    RAISE NOTICE '  - 优化了查询性能';
END $$;