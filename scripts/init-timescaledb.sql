-- RQA2025 TimescaleDB 初始化脚本
-- 初始化TimescaleDB扩展和基础配置

-- 启用TimescaleDB扩展
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- 验证TimescaleDB扩展是否正确加载
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_extension
        WHERE extname = 'timescaledb'
    ) THEN
        RAISE EXCEPTION 'TimescaleDB extension not found';
    END IF;

    -- 记录初始化日志
    RAISE NOTICE 'TimescaleDB extension initialized successfully';
END $$;

-- 设置TimescaleDB相关参数
-- 这些参数将在postgresql.conf中配置，但在初始化时设置默认值
ALTER SYSTEM SET timescaledb.max_background_workers = 8;
ALTER SYSTEM SET timescaledb.license = 'apache2';  -- 使用Apache 2.0许可证

-- 创建TimescaleDB元数据表检查函数
CREATE OR REPLACE FUNCTION check_timescaledb_setup()
RETURNS TABLE (
    hypertables_count INTEGER,
    chunks_count INTEGER,
    compression_enabled BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        (SELECT COUNT(*) FROM timescaledb_information.hypertables)::INTEGER as hypertables_count,
        (SELECT COUNT(*) FROM timescaledb_information.chunks)::INTEGER as chunks_count,
        (SELECT COUNT(*) > 0 FROM timescaledb_information.compression_settings)::BOOLEAN as compression_enabled;
END;
$$ LANGUAGE plpgsql;

-- 创建数据库状态监控函数
CREATE OR REPLACE FUNCTION get_database_status()
RETURNS TABLE (
    db_name TEXT,
    size_mb NUMERIC,
    active_connections INTEGER,
    total_connections INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        datname::TEXT,
        pg_database_size(datname)::NUMERIC / 1024 / 1024 as size_mb,
        (SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() AND state = 'active')::INTEGER as active_connections,
        (SELECT count(*) FROM pg_stat_activity WHERE datname = current_database())::INTEGER as total_connections
    FROM pg_database
    WHERE datname = current_database();
END;
$$ LANGUAGE plpgsql;

-- 创建TimescaleDB性能监控视图
CREATE OR REPLACE VIEW timescaledb_performance_stats AS
SELECT
    hypertable_name,
    total_chunks,
    compressed_chunks,
    uncompressed_chunks,
    compression_ratio,
    before_compression_total_bytes,
    after_compression_total_bytes
FROM timescaledb_information.compressed_hypertable_stats;

-- 设置默认TimescaleDB配置
-- 这些配置会在运行时通过ALTER SYSTEM设置
-- 但这里提供初始化时的默认值

-- 记录初始化完成
INSERT INTO public.initialization_log (component, status, message, created_at)
VALUES ('timescaledb', 'completed', 'TimescaleDB extension initialized successfully', NOW())
ON CONFLICT (component) DO UPDATE SET
    status = EXCLUDED.status,
    message = EXCLUDED.message,
    created_at = EXCLUDED.created_at;

-- 创建初始化日志表（如果不存在）
CREATE TABLE IF NOT EXISTS public.initialization_log (
    component TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);