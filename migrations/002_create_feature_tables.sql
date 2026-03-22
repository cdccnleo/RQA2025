-- ============================================================
-- 特征工程模块数据库表结构定义
-- 数据库: PostgreSQL
-- 版本: 1.0.0
-- 创建时间: 2025-01-01
-- 描述: 特征存储、特征缓存、监控指标的表结构定义
-- ============================================================

-- ============================================================
-- 1. 特征存储表 (feature_store)
-- 用于存储特征计算结果，支持特征持久化和检索
-- ============================================================
CREATE TABLE IF NOT EXISTS feature_store (
    feature_id VARCHAR(64) PRIMARY KEY,
    feature_name VARCHAR(255) NOT NULL,
    shape JSONB NOT NULL,
    columns JSONB NOT NULL,
    dtypes JSONB NOT NULL,
    format VARCHAR(20) NOT NULL DEFAULT 'parquet',
    data BYTEA,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}',
    checksum VARCHAR(64),
    size_bytes BIGINT DEFAULT 0
);

-- 特征存储索引
CREATE INDEX IF NOT EXISTS idx_feature_store_name ON feature_store(feature_name);
CREATE INDEX IF NOT EXISTS idx_feature_store_created ON feature_store(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_feature_store_format ON feature_store(format);

-- 特征存储注释
COMMENT ON TABLE feature_store IS '特征存储表，存储特征计算结果';
COMMENT ON COLUMN feature_store.feature_id IS '特征唯一标识符(MD5哈希)';
COMMENT ON COLUMN feature_store.feature_name IS '特征名称';
COMMENT ON COLUMN feature_store.shape IS '特征数据形状 [行数, 列数]';
COMMENT ON COLUMN feature_store.columns IS '特征列名列表';
COMMENT ON COLUMN feature_store.dtypes IS '特征数据类型映射';
COMMENT ON COLUMN feature_store.format IS '存储格式(parquet/csv/pickle)';
COMMENT ON COLUMN feature_store.data IS '序列化的特征数据';
COMMENT ON COLUMN feature_store.metadata IS '附加元数据(JSON)';
COMMENT ON COLUMN feature_store.checksum IS '数据校验和';
COMMENT ON COLUMN feature_store.size_bytes IS '数据大小(字节)';

-- ============================================================
-- 2. 特征缓存表 (feature_cache)
-- 用于缓存计算过的特征，支持TTL和生命周期管理
-- ============================================================
CREATE TABLE IF NOT EXISTS feature_cache (
    feature_id VARCHAR(64) PRIMARY KEY,
    feature_name VARCHAR(255) NOT NULL,
    feature_type VARCHAR(50) NOT NULL,
    params JSONB DEFAULT '{}',
    dependencies JSONB DEFAULT '[]',
    data BYTEA,
    data_shape JSONB,
    data_size_mb FLOAT DEFAULT 0,
    checksum VARCHAR(64),
    version VARCHAR(20) DEFAULT '1.0',
    description TEXT,
    tags JSONB DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 特征缓存索引
CREATE INDEX IF NOT EXISTS idx_feature_cache_name ON feature_cache(feature_name);
CREATE INDEX IF NOT EXISTS idx_feature_cache_type ON feature_cache(feature_type);
CREATE INDEX IF NOT EXISTS idx_feature_cache_updated ON feature_cache(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_feature_cache_tags ON feature_cache USING GIN(tags);

-- 特征缓存注释
COMMENT ON TABLE feature_cache IS '特征缓存表，支持TTL和生命周期管理';
COMMENT ON COLUMN feature_cache.feature_type IS '特征类型(technical/fundamental/sentiment)';
COMMENT ON COLUMN feature_cache.params IS '特征计算参数';
COMMENT ON COLUMN feature_cache.dependencies IS '特征依赖列表';
COMMENT ON COLUMN feature_cache.data_shape IS '数据形状';
COMMENT ON COLUMN feature_cache.data_size_mb IS '数据大小(MB)';
COMMENT ON COLUMN feature_cache.version IS '特征版本号';
COMMENT ON COLUMN feature_cache.tags IS '特征标签数组';

-- ============================================================
-- 3. 监控指标表 (monitoring_metrics)
-- 用于存储特征计算过程的监控指标
-- ============================================================
CREATE TABLE IF NOT EXISTS monitoring_metrics (
    id SERIAL PRIMARY KEY,
    component_name VARCHAR(255) NOT NULL,
    metric_name VARCHAR(255) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    metric_type VARCHAR(50) NOT NULL,
    timestamp DOUBLE PRECISION NOT NULL,
    labels JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ttl DOUBLE PRECISION,
    priority INTEGER DEFAULT 1,
    data_tier VARCHAR(10) DEFAULT 'hot'
);

-- 监控指标索引
CREATE INDEX IF NOT EXISTS idx_metrics_component_time ON monitoring_metrics(component_name, metric_name, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp_tier ON monitoring_metrics(timestamp, data_tier);
CREATE INDEX IF NOT EXISTS idx_metrics_type_time ON monitoring_metrics(metric_type, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_metrics_priority ON monitoring_metrics(priority DESC, timestamp DESC);

-- 监控指标注释
COMMENT ON TABLE monitoring_metrics IS '监控指标表，存储特征计算过程的监控数据';
COMMENT ON COLUMN monitoring_metrics.component_name IS '组件名称';
COMMENT ON COLUMN monitoring_metrics.metric_name IS '指标名称';
COMMENT ON COLUMN monitoring_metrics.metric_value IS '指标值';
COMMENT ON COLUMN monitoring_metrics.metric_type IS '指标类型(counter/gauge/histogram)';
COMMENT ON COLUMN monitoring_metrics.timestamp IS 'Unix时间戳';
COMMENT ON COLUMN monitoring_metrics.labels IS '指标标签';
COMMENT ON COLUMN monitoring_metrics.ttl IS '生存时间(秒)';
COMMENT ON COLUMN monitoring_metrics.priority IS '优先级(1-10)';
COMMENT ON COLUMN monitoring_metrics.data_tier IS '数据层级(hot/warm/cold)';

-- ============================================================
-- 4. 特征选择历史表 (feature_selector_history)
-- 用于存储特征选择的历史记录
-- ============================================================
CREATE TABLE IF NOT EXISTS feature_selector_history (
    id SERIAL PRIMARY KEY,
    selection_id VARCHAR(64) NOT NULL UNIQUE,
    selector_type VARCHAR(50) NOT NULL,
    original_features JSONB NOT NULL,
    selected_features JSONB NOT NULL,
    selection_scores JSONB DEFAULT '{}',
    params JSONB DEFAULT '{}',
    performance_metrics JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_type VARCHAR(100),
    dataset_hash VARCHAR(64)
);

-- 特征选择历史索引
CREATE INDEX IF NOT EXISTS idx_selector_history_type ON feature_selector_history(selector_type);
CREATE INDEX IF NOT EXISTS idx_selector_history_created ON feature_selector_history(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_selector_history_model ON feature_selector_history(model_type);

-- 特征选择历史注释
COMMENT ON TABLE feature_selector_history IS '特征选择历史表，记录特征选择过程和结果';
COMMENT ON COLUMN feature_selector_history.selection_id IS '选择操作唯一标识';
COMMENT ON COLUMN feature_selector_history.selector_type IS '选择器类型(RFECV/SelectKBest/etc)';
COMMENT ON COLUMN feature_selector_history.original_features IS '原始特征列表';
COMMENT ON COLUMN feature_selector_history.selected_features IS '选中特征列表';
COMMENT ON COLUMN feature_selector_history.selection_scores IS '特征选择得分';
COMMENT ON COLUMN feature_selector_history.params IS '选择器参数';
COMMENT ON COLUMN feature_selector_history.performance_metrics IS '性能指标';

-- ============================================================
-- 5. 特征质量评估表 (feature_quality_assessment)
-- 用于存储特征质量评估结果
-- ============================================================
CREATE TABLE IF NOT EXISTS feature_quality_assessment (
    id SERIAL PRIMARY KEY,
    assessment_id VARCHAR(64) NOT NULL UNIQUE,
    feature_name VARCHAR(255) NOT NULL,
    importance_score FLOAT,
    correlation_score FLOAT,
    stability_score FLOAT,
    missing_rate FLOAT,
    outlier_rate FLOAT,
    skewness FLOAT,
    kurtosis FLOAT,
    quality_grade VARCHAR(10),
    assessment_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 特征质量评估索引
CREATE INDEX IF NOT EXISTS idx_quality_feature_name ON feature_quality_assessment(feature_name);
CREATE INDEX IF NOT EXISTS idx_quality_grade ON feature_quality_assessment(quality_grade);
CREATE INDEX IF NOT EXISTS idx_quality_created ON feature_quality_assessment(created_at DESC);

-- 特征质量评估注释
COMMENT ON TABLE feature_quality_assessment IS '特征质量评估表，存储特征质量评估结果';
COMMENT ON COLUMN feature_quality_assessment.importance_score IS '特征重要性得分';
COMMENT ON COLUMN feature_quality_assessment.correlation_score IS '相关性得分';
COMMENT ON COLUMN feature_quality_assessment.stability_score IS '稳定性得分';
COMMENT ON COLUMN feature_quality_assessment.missing_rate IS '缺失率';
COMMENT ON COLUMN feature_quality_assessment.outlier_rate IS '异常值率';
COMMENT ON COLUMN feature_quality_assessment.quality_grade IS '质量等级(A/B/C/D/F)';

-- ============================================================
-- 6. 触发器: 自动更新 updated_at 字段
-- ============================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- feature_store 更新触发器
DROP TRIGGER IF EXISTS update_feature_store_updated_at ON feature_store;
CREATE TRIGGER update_feature_store_updated_at
    BEFORE UPDATE ON feature_store
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- feature_cache 更新触发器
DROP TRIGGER IF EXISTS update_feature_cache_updated_at ON feature_cache;
CREATE TRIGGER update_feature_cache_updated_at
    BEFORE UPDATE ON feature_cache
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================
-- 7. 视图: 特征统计视图
-- ============================================================
CREATE OR REPLACE VIEW v_feature_statistics AS
SELECT 
    feature_name,
    COUNT(*) as version_count,
    MAX(updated_at) as last_updated,
    AVG(size_bytes) as avg_size_bytes,
    SUM(size_bytes) as total_size_bytes
FROM feature_store
GROUP BY feature_name
ORDER BY total_size_bytes DESC;

COMMENT ON VIEW v_feature_statistics IS '特征统计视图，按特征名称汇总统计信息';

-- ============================================================
-- 8. 视图: 监控指标聚合视图
-- ============================================================
CREATE OR REPLACE VIEW v_metrics_summary AS
SELECT 
    component_name,
    metric_name,
    metric_type,
    COUNT(*) as metric_count,
    AVG(metric_value) as avg_value,
    MIN(metric_value) as min_value,
    MAX(metric_value) as max_value,
    STDDEV(metric_value) as std_value,
    MAX(created_at) as last_recorded
FROM monitoring_metrics
GROUP BY component_name, metric_name, metric_type
ORDER BY component_name, metric_name;

COMMENT ON VIEW v_metrics_summary IS '监控指标聚合视图，按组件和指标名称汇总';

-- ============================================================
-- 9. 数据归档存储过程
-- ============================================================
CREATE OR REPLACE PROCEDURE archive_old_metrics(
    p_hot_days INTEGER DEFAULT 7,
    p_warm_days INTEGER DEFAULT 30,
    p_cold_days INTEGER DEFAULT 365
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_hot_threshold FLOAT;
    v_warm_threshold FLOAT;
    v_cold_threshold FLOAT;
    v_deleted_count INTEGER;
BEGIN
    v_cold_threshold := EXTRACT(EPOCH FROM NOW()) - (p_cold_days * 24 * 3600);
    v_warm_threshold := EXTRACT(EPOCH FROM NOW()) - (p_warm_days * 24 * 3600);
    v_hot_threshold := EXTRACT(EPOCH FROM NOW()) - (p_hot_days * 24 * 3600);
    
    DELETE FROM monitoring_metrics 
    WHERE timestamp < v_cold_threshold AND data_tier = 'cold';
    
    GET DIAGNOSTICS v_deleted_count = ROW_COUNT;
    RAISE NOTICE 'Deleted % cold records', v_deleted_count;
    
    UPDATE monitoring_metrics 
    SET data_tier = 'cold' 
    WHERE timestamp < v_warm_threshold AND data_tier = 'warm';
    
    UPDATE monitoring_metrics 
    SET data_tier = 'warm' 
    WHERE timestamp < v_hot_threshold AND data_tier = 'hot';
    
    COMMIT;
END;
$$;

COMMENT ON PROCEDURE archive_old_metrics IS '归档旧监控数据，按时间分层管理';

-- ============================================================
-- 10. 权限设置 (根据实际环境调整)
-- ============================================================
-- GRANT SELECT, INSERT, UPDATE, DELETE ON feature_store TO rqa2025_admin;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON feature_cache TO rqa2025_admin;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON monitoring_metrics TO rqa2025_admin;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON feature_selector_history TO rqa2025_admin;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON feature_quality_assessment TO rqa2025_admin;
-- GRANT EXECUTE ON PROCEDURE archive_old_metrics TO rqa2025_admin;
