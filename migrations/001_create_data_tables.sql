-- ============================================================
-- 数据管理层表结构定义
-- 数据库: PostgreSQL
-- 版本: 1.0.0
-- 创建时间: 2026-03-22
-- 描述: 数据缓存、血缘追踪、质量管理、合规检查的表结构定义
-- ============================================================

-- ============================================================
-- 1. 数据缓存表 (data_cache_entries)
-- 用于持久化数据管理层的缓存数据
-- ============================================================
CREATE TABLE IF NOT EXISTS data_cache_entries (
    id SERIAL PRIMARY KEY,
    cache_key VARCHAR(128) NOT NULL UNIQUE,
    cache_type VARCHAR(32) NOT NULL DEFAULT 'general',
    data BYTEA,
    metadata JSONB DEFAULT '{}',
    data_size_bytes BIGINT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    last_accessed TIMESTAMP,
    access_count INTEGER DEFAULT 0,
    tags JSONB DEFAULT '[]',
    description TEXT
);

-- 数据缓存索引
CREATE INDEX IF NOT EXISTS idx_data_cache_key ON data_cache_entries(cache_key);
CREATE INDEX IF NOT EXISTS idx_data_cache_type ON data_cache_entries(cache_type);
CREATE INDEX IF NOT EXISTS idx_data_cache_created ON data_cache_entries(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_data_cache_expires ON data_cache_entries(expires_at);
CREATE INDEX IF NOT EXISTS idx_data_cache_tags ON data_cache_entries USING GIN(tags);

-- 数据缓存注释
COMMENT ON TABLE data_cache_entries IS '数据缓存表，持久化数据管理层的缓存数据';
COMMENT ON COLUMN data_cache_entries.cache_key IS '缓存键（唯一标识）';
COMMENT ON COLUMN data_cache_entries.cache_type IS '缓存类型（general/model/feature等）';
COMMENT ON COLUMN data_cache_entries.data IS '序列化的缓存数据';
COMMENT ON COLUMN data_cache_entries.metadata IS '缓存元数据';
COMMENT ON COLUMN data_cache_entries.expires_at IS '缓存过期时间';

-- ============================================================
-- 2. 数据血缘表 (data_lineage_records)
-- 用于记录数据的来源、转换和依赖关系
-- ============================================================
CREATE TABLE IF NOT EXISTS data_lineage_records (
    id SERIAL PRIMARY KEY,
    data_type VARCHAR(64) NOT NULL,
    source_info JSONB DEFAULT '{}',
    transform_info JSONB DEFAULT '{}',
    dependencies JSONB DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(128),
    version VARCHAR(32) DEFAULT '1.0'
);

-- 数据血缘索引
CREATE INDEX IF NOT EXISTS idx_lineage_data_type ON data_lineage_records(data_type);
CREATE INDEX IF NOT EXISTS idx_lineage_created ON data_lineage_records(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_lineage_dependencies ON data_lineage_records USING GIN(dependencies);

-- 数据血缘注释
COMMENT ON TABLE data_lineage_records IS '数据血缘表，记录数据的来源、转换和依赖关系';
COMMENT ON COLUMN data_lineage_records.data_type IS '数据类型（stock/index/news等）';
COMMENT ON COLUMN data_lineage_records.source_info IS '数据来源信息';
COMMENT ON COLUMN data_lineage_records.transform_info IS '数据转换信息';
COMMENT ON COLUMN data_lineage_records.dependencies IS '数据依赖列表';

-- ============================================================
-- 3. 数据质量指标表 (data_quality_metrics)
-- 用于记录数据质量检查结果
-- ============================================================
CREATE TABLE IF NOT EXISTS data_quality_metrics (
    id SERIAL PRIMARY KEY,
    data_type VARCHAR(64) NOT NULL,
    completeness FLOAT DEFAULT 0.0,
    accuracy FLOAT DEFAULT 0.0,
    timeliness FLOAT DEFAULT 0.0,
    consistency FLOAT DEFAULT 0.0,
    uniqueness FLOAT DEFAULT 0.0,
    overall_score FLOAT DEFAULT 0.0,
    issues JSONB DEFAULT '[]',
    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    check_duration_ms FLOAT DEFAULT 0.0,
    checker_version VARCHAR(32) DEFAULT '1.0'
);

-- 数据质量指标索引
CREATE INDEX IF NOT EXISTS idx_quality_data_type ON data_quality_metrics(data_type);
CREATE INDEX IF NOT EXISTS idx_quality_checked ON data_quality_metrics(checked_at DESC);
CREATE INDEX IF NOT EXISTS idx_quality_score ON data_quality_metrics(overall_score);

-- 数据质量指标注释
COMMENT ON TABLE data_quality_metrics IS '数据质量指标表，记录数据质量检查结果';
COMMENT ON COLUMN data_quality_metrics.completeness IS '数据完整性得分（0-1）';
COMMENT ON COLUMN data_quality_metrics.accuracy IS '数据准确性得分（0-1）';
COMMENT ON COLUMN data_quality_metrics.timeliness IS '数据及时性得分（0-1）';
COMMENT ON COLUMN data_quality_metrics.consistency IS '数据一致性得分（0-1）';
COMMENT ON COLUMN data_quality_metrics.uniqueness IS '数据唯一性得分（0-1）';
COMMENT ON COLUMN data_quality_metrics.overall_score IS '综合质量得分（0-1）';

-- ============================================================
-- 4. 合规检查记录表 (data_compliance_checks)
-- 用于记录数据合规性检查结果
-- ============================================================
CREATE TABLE IF NOT EXISTS data_compliance_checks (
    id SERIAL PRIMARY KEY,
    policy_id VARCHAR(64) NOT NULL,
    data_type VARCHAR(64) NOT NULL,
    is_compliant BOOLEAN DEFAULT TRUE,
    issues JSONB DEFAULT '[]',
    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    check_duration_ms FLOAT DEFAULT 0.0,
    checker_id VARCHAR(128),
    metadata JSONB DEFAULT '{}'
);

-- 合规检查记录索引
CREATE INDEX IF NOT EXISTS idx_compliance_policy ON data_compliance_checks(policy_id);
CREATE INDEX IF NOT EXISTS idx_compliance_data_type ON data_compliance_checks(data_type);
CREATE INDEX IF NOT EXISTS idx_compliance_checked ON data_compliance_checks(checked_at DESC);
CREATE INDEX IF NOT EXISTS idx_compliance_result ON data_compliance_checks(is_compliant);

-- 合规检查记录注释
COMMENT ON TABLE data_compliance_checks IS '合规检查记录表，记录数据合规性检查结果';
COMMENT ON COLUMN data_compliance_checks.policy_id IS '合规策略ID';
COMMENT ON COLUMN data_compliance_checks.is_compliant IS '是否合规';
COMMENT ON COLUMN data_compliance_checks.issues IS '发现的问题列表';

-- ============================================================
-- 5. 数据访问日志表 (data_access_logs)
-- 用于记录数据访问审计日志
-- ============================================================
CREATE TABLE IF NOT EXISTS data_access_logs (
    id SERIAL PRIMARY KEY,
    data_type VARCHAR(64) NOT NULL,
    data_key VARCHAR(256),
    operation VARCHAR(32) NOT NULL,
    user_id VARCHAR(128),
    ip_address VARCHAR(64),
    access_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    duration_ms FLOAT DEFAULT 0.0,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    metadata JSONB DEFAULT '{}'
);

-- 数据访问日志索引
CREATE INDEX IF NOT EXISTS idx_access_data_type ON data_access_logs(data_type);
CREATE INDEX IF NOT EXISTS idx_access_user ON data_access_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_access_time ON data_access_logs(access_at DESC);
CREATE INDEX IF NOT EXISTS idx_access_operation ON data_access_logs(operation);

-- 数据访问日志注释
COMMENT ON TABLE data_access_logs IS '数据访问日志表，记录数据访问审计日志';
COMMENT ON COLUMN data_access_logs.operation IS '操作类型（read/write/delete）';
COMMENT ON COLUMN data_access_logs.success IS '操作是否成功';

-- ============================================================
-- 6. 数据元数据表 (data_metadata_store)
-- 用于存储数据集的元数据信息
-- ============================================================
CREATE TABLE IF NOT EXISTS data_metadata_store (
    id SERIAL PRIMARY KEY,
    data_key VARCHAR(256) NOT NULL UNIQUE,
    data_type VARCHAR(64) NOT NULL,
    schema_info JSONB DEFAULT '{}',
    statistics JSONB DEFAULT '{}',
    tags JSONB DEFAULT '[]',
    owner VARCHAR(128),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    version INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE,
    description TEXT,
    retention_days INTEGER DEFAULT 365
);

-- 数据元数据索引
CREATE INDEX IF NOT EXISTS idx_metadata_key ON data_metadata_store(data_key);
CREATE INDEX IF NOT EXISTS idx_metadata_type ON data_metadata_store(data_type);
CREATE INDEX IF NOT EXISTS idx_metadata_owner ON data_metadata_store(owner);
CREATE INDEX IF NOT EXISTS idx_metadata_tags ON data_metadata_store USING GIN(tags);

-- 数据元数据注释
COMMENT ON TABLE data_metadata_store IS '数据元数据表，存储数据集的元数据信息';
COMMENT ON COLUMN data_metadata_store.schema_info IS '数据结构信息';
COMMENT ON COLUMN data_metadata_store.statistics IS '数据统计信息';

-- ============================================================
-- 7. 触发器: 自动更新 updated_at 字段
-- ============================================================
CREATE OR REPLACE FUNCTION update_data_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- data_cache_entries 更新触发器
DROP TRIGGER IF EXISTS update_data_cache_updated_at ON data_cache_entries;
CREATE TRIGGER update_data_cache_updated_at
    BEFORE UPDATE ON data_cache_entries
    FOR EACH ROW
    EXECUTE FUNCTION update_data_updated_at_column();

-- data_metadata_store 更新触发器
DROP TRIGGER IF EXISTS update_data_metadata_updated_at ON data_metadata_store;
CREATE TRIGGER update_data_metadata_updated_at
    BEFORE UPDATE ON data_metadata_store
    FOR EACH ROW
    EXECUTE FUNCTION update_data_updated_at_column();

-- ============================================================
-- 8. 视图: 缓存统计视图
-- ============================================================
CREATE OR REPLACE VIEW v_cache_statistics AS
SELECT 
    cache_type,
    COUNT(*) as entry_count,
    SUM(data_size_bytes) as total_size_bytes,
    AVG(data_size_bytes) as avg_size_bytes,
    SUM(access_count) as total_accesses,
    AVG(access_count) as avg_accesses,
    COUNT(CASE WHEN expires_at > CURRENT_TIMESTAMP THEN 1 END) as active_entries,
    COUNT(CASE WHEN expires_at <= CURRENT_TIMESTAMP THEN 1 END) as expired_entries
FROM data_cache_entries
GROUP BY cache_type
ORDER BY entry_count DESC;

COMMENT ON VIEW v_cache_statistics IS '缓存统计视图，按类型汇总缓存信息';

-- ============================================================
-- 9. 视图: 质量趋势视图
-- ============================================================
CREATE OR REPLACE VIEW v_quality_trend AS
SELECT 
    data_type,
    DATE(checked_at) as check_date,
    AVG(completeness) as avg_completeness,
    AVG(accuracy) as avg_accuracy,
    AVG(timeliness) as avg_timeliness,
    AVG(consistency) as avg_consistency,
    AVG(uniqueness) as avg_uniqueness,
    AVG(overall_score) as avg_overall_score,
    COUNT(*) as check_count
FROM data_quality_metrics
GROUP BY data_type, DATE(checked_at)
ORDER BY check_date DESC, data_type;

COMMENT ON VIEW v_quality_trend IS '质量趋势视图，按日期汇总质量指标';

-- ============================================================
-- 10. 视图: 合规状态视图
-- ============================================================
CREATE OR REPLACE VIEW v_compliance_status AS
SELECT 
    policy_id,
    data_type,
    COUNT(*) as total_checks,
    SUM(CASE WHEN is_compliant THEN 1 ELSE 0 END) as compliant_count,
    SUM(CASE WHEN NOT is_compliant THEN 1 ELSE 0 END) as non_compliant_count,
    ROUND(100.0 * SUM(CASE WHEN is_compliant THEN 1 ELSE 0 END) / COUNT(*), 2) as compliance_rate,
    MAX(checked_at) as last_check
FROM data_compliance_checks
GROUP BY policy_id, data_type
ORDER BY compliance_rate ASC, last_check DESC;

COMMENT ON VIEW v_compliance_status IS '合规状态视图，汇总合规检查结果';

-- ============================================================
-- 11. 存储过程: 清理过期缓存
-- ============================================================
CREATE OR REPLACE PROCEDURE cleanup_expired_data_cache()
LANGUAGE plpgsql
AS $$
DECLARE
    v_cleaned_count INTEGER;
BEGIN
    DELETE FROM data_cache_entries WHERE expires_at < CURRENT_TIMESTAMP;
    GET DIAGNOSTICS v_cleaned_count = ROW_COUNT;
    
    RAISE NOTICE 'Cleaned % expired cache entries', v_cleaned_count;
END;
$$;

COMMENT ON PROCEDURE cleanup_expired_data_cache IS '清理过期的数据缓存条目';

-- ============================================================
-- 12. 存储过程: 归档旧质量记录
-- ============================================================
CREATE OR REPLACE PROCEDURE archive_old_quality_metrics(
    p_days_old INTEGER DEFAULT 90
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_archived_count INTEGER;
BEGIN
    DELETE FROM data_quality_metrics 
    WHERE checked_at < CURRENT_TIMESTAMP - INTERVAL '1 day' * p_days_old;
    
    GET DIAGNOSTICS v_archived_count = ROW_COUNT;
    
    RAISE NOTICE 'Archived % quality metrics older than % days', v_archived_count, p_days_old;
END;
$$;

COMMENT ON PROCEDURE archive_old_quality_metrics IS '归档旧的质量指标记录';

-- ============================================================
-- 13. 存储过程: 清理旧访问日志
-- ============================================================
CREATE OR REPLACE PROCEDURE cleanup_old_access_logs(
    p_days_old INTEGER DEFAULT 30
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_cleaned_count INTEGER;
BEGIN
    DELETE FROM data_access_logs 
    WHERE access_at < CURRENT_TIMESTAMP - INTERVAL '1 day' * p_days_old;
    
    GET DIAGNOSTICS v_cleaned_count = ROW_COUNT;
    
    RAISE NOTICE 'Cleaned % access logs older than % days', v_cleaned_count, p_days_old;
END;
$$;

COMMENT ON PROCEDURE cleanup_old_access_logs IS '清理旧的数据访问日志';

-- ============================================================
-- 14. 权限设置 (根据实际环境调整)
-- ============================================================
-- GRANT SELECT, INSERT, UPDATE, DELETE ON data_cache_entries TO rqa2025_admin;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON data_lineage_records TO rqa2025_admin;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON data_quality_metrics TO rqa2025_admin;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON data_compliance_checks TO rqa2025_admin;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON data_access_logs TO rqa2025_admin;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON data_metadata_store TO rqa2025_admin;
-- GRANT EXECUTE ON PROCEDURE cleanup_expired_data_cache TO rqa2025_admin;
-- GRANT EXECUTE ON PROCEDURE archive_old_quality_metrics TO rqa2025_admin;
-- GRANT EXECUTE ON PROCEDURE cleanup_old_access_logs TO rqa2025_admin;
