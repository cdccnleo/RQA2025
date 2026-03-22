-- ============================================================
-- 机器学习模块数据库表结构定义
-- 数据库: PostgreSQL
-- 版本: 1.0.0
-- 创建时间: 2026-03-22
-- 描述: 模型存储、训练历史、推理缓存、特征缓存的表结构定义
-- ============================================================

-- ============================================================
-- 1. 模型存储表 (ml_models)
-- 用于存储训练好的机器学习模型
-- ============================================================
CREATE TABLE IF NOT EXISTS ml_models (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(128) NOT NULL,
    version VARCHAR(64) NOT NULL,
    model_type VARCHAR(64) NOT NULL,
    model_data BYTEA,
    status VARCHAR(32) DEFAULT 'active',
    metadata JSONB DEFAULT '{}',
    feature_columns JSONB DEFAULT '[]',
    metrics JSONB DEFAULT '{}',
    config JSONB DEFAULT '{}',
    size_bytes BIGINT DEFAULT 0,
    checksum VARCHAR(64),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(model_id, version)
);

-- 模型存储索引
CREATE INDEX IF NOT EXISTS idx_ml_models_id ON ml_models(model_id);
CREATE INDEX IF NOT EXISTS idx_ml_models_type ON ml_models(model_type);
CREATE INDEX IF NOT EXISTS idx_ml_models_created ON ml_models(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ml_models_status ON ml_models(status);

-- 模型存储注释
COMMENT ON TABLE ml_models IS '模型存储表，存储训练好的机器学习模型';
COMMENT ON COLUMN ml_models.model_id IS '模型唯一标识符';
COMMENT ON COLUMN ml_models.version IS '模型版本号';
COMMENT ON COLUMN ml_models.model_type IS '模型类型(RandomForest/XGBoost/LightGBM/LSTM等)';
COMMENT ON COLUMN ml_models.model_data IS '序列化的模型数据(二进制)';
COMMENT ON COLUMN ml_models.status IS '模型状态(draft/active/deprecated/archived)';
COMMENT ON COLUMN ml_models.metadata IS '模型元数据(JSON)';
COMMENT ON COLUMN ml_models.feature_columns IS '特征列名列表';
COMMENT ON COLUMN ml_models.metrics IS '模型性能指标';
COMMENT ON COLUMN ml_models.config IS '模型配置参数';
COMMENT ON COLUMN ml_models.size_bytes IS '模型大小(字节)';
COMMENT ON COLUMN ml_models.checksum IS '模型数据校验和';

-- ============================================================
-- 2. 训练历史表 (ml_training_history)
-- 用于记录模型训练过程和结果
-- ============================================================
CREATE TABLE IF NOT EXISTS ml_training_history (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(128) NOT NULL,
    version VARCHAR(64),
    training_config JSONB DEFAULT '{}',
    training_metrics JSONB DEFAULT '{}',
    hyperparameters JSONB DEFAULT '{}',
    feature_importance JSONB DEFAULT '{}',
    training_duration FLOAT,
    samples_trained INTEGER,
    status VARCHAR(32) DEFAULT 'completed',
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 训练历史索引
CREATE INDEX IF NOT EXISTS idx_training_history_model ON ml_training_history(model_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_training_history_status ON ml_training_history(status);
CREATE INDEX IF NOT EXISTS idx_training_history_created ON ml_training_history(created_at DESC);

-- 训练历史注释
COMMENT ON TABLE ml_training_history IS '训练历史表，记录模型训练过程和结果';
COMMENT ON COLUMN ml_training_history.training_config IS '训练配置参数';
COMMENT ON COLUMN ml_training_history.training_metrics IS '训练过程中的指标';
COMMENT ON COLUMN ml_training_history.hyperparameters IS '超参数配置';
COMMENT ON COLUMN ml_training_history.feature_importance IS '特征重要性得分';
COMMENT ON COLUMN ml_training_history.training_duration IS '训练时长(秒)';
COMMENT ON COLUMN ml_training_history.samples_trained IS '训练样本数量';

-- ============================================================
-- 3. 推理结果表 (ml_inference_results)
-- 用于存储模型推理结果，支持推理缓存
-- ============================================================
CREATE TABLE IF NOT EXISTS ml_inference_results (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(128) NOT NULL,
    version VARCHAR(64),
    request_id VARCHAR(64),
    input_hash VARCHAR(64),
    predictions JSONB,
    probabilities JSONB,
    confidence_scores JSONB,
    inference_time_ms FLOAT,
    batch_size INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 推理结果索引
CREATE INDEX IF NOT EXISTS idx_inference_results_model ON ml_inference_results(model_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_inference_results_hash ON ml_inference_results(input_hash);
CREATE INDEX IF NOT EXISTS idx_inference_results_request ON ml_inference_results(request_id);

-- 推理结果注释
COMMENT ON TABLE ml_inference_results IS '推理结果表，存储模型推理结果';
COMMENT ON COLUMN ml_inference_results.request_id IS '请求唯一标识';
COMMENT ON COLUMN ml_inference_results.input_hash IS '输入数据哈希值';
COMMENT ON COLUMN ml_inference_results.predictions IS '预测结果';
COMMENT ON COLUMN ml_inference_results.probabilities IS '预测概率';
COMMENT ON COLUMN ml_inference_results.confidence_scores IS '置信度分数';
COMMENT ON COLUMN ml_inference_results.inference_time_ms IS '推理耗时(毫秒)';

-- ============================================================
-- 4. 推理缓存表 (inference_cache)
-- 用于缓存推理结果，提升重复推理性能
-- ============================================================
CREATE TABLE IF NOT EXISTS inference_cache (
    cache_key VARCHAR(128) PRIMARY KEY,
    model_id VARCHAR(128) NOT NULL,
    input_hash VARCHAR(64) NOT NULL,
    result_data BYTEA,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    access_count INTEGER DEFAULT 0
);

-- 推理缓存索引
CREATE INDEX IF NOT EXISTS idx_inference_cache_model ON inference_cache(model_id);
CREATE INDEX IF NOT EXISTS idx_inference_cache_expires ON inference_cache(expires_at);
CREATE INDEX IF NOT EXISTS idx_inference_cache_input_hash ON inference_cache(input_hash);

-- 推理缓存注释
COMMENT ON TABLE inference_cache IS '推理缓存表，缓存推理结果以提升性能';
COMMENT ON COLUMN inference_cache.cache_key IS '缓存键(模型ID+输入哈希)';
COMMENT ON COLUMN inference_cache.result_data IS '序列化的推理结果';
COMMENT ON COLUMN inference_cache.expires_at IS '缓存过期时间';

-- ============================================================
-- 5. 特征缓存表 (feature_cache)
-- 用于缓存特征工程结果，避免重复计算
-- ============================================================
CREATE TABLE IF NOT EXISTS feature_cache (
    cache_key VARCHAR(64) PRIMARY KEY,
    task_id VARCHAR(128) NOT NULL,
    cache_data BYTEA,
    metadata JSONB DEFAULT '{}',
    size_bytes BIGINT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 0,
    expires_at TIMESTAMP
);

-- 特征缓存索引
CREATE INDEX IF NOT EXISTS idx_feature_cache_task ON feature_cache(task_id);
CREATE INDEX IF NOT EXISTS idx_feature_cache_expires ON feature_cache(expires_at);
CREATE INDEX IF NOT EXISTS idx_feature_cache_accessed ON feature_cache(accessed_at DESC);

-- 特征缓存注释
COMMENT ON TABLE feature_cache IS '特征缓存表，缓存特征工程结果';
COMMENT ON COLUMN feature_cache.task_id IS '特征工程任务ID';
COMMENT ON COLUMN feature_cache.cache_data IS '序列化的特征数据';
COMMENT ON COLUMN feature_cache.metadata IS '特征元数据';

-- ============================================================
-- 6. 模型版本控制表 (ml_model_versions)
-- 用于管理模型版本历史
-- ============================================================
CREATE TABLE IF NOT EXISTS ml_model_versions (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(128) NOT NULL,
    version VARCHAR(64) NOT NULL,
    parent_version VARCHAR(64),
    change_type VARCHAR(32) NOT NULL,
    change_description TEXT,
    metrics_delta JSONB DEFAULT '{}',
    created_by VARCHAR(128),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_id, version)
);

-- 模型版本索引
CREATE INDEX IF NOT EXISTS idx_model_versions_model ON ml_model_versions(model_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_model_versions_parent ON ml_model_versions(model_id, parent_version);

-- 模型版本注释
COMMENT ON TABLE ml_model_versions IS '模型版本控制表，管理模型版本历史';
COMMENT ON COLUMN ml_model_versions.change_type IS '变更类型(create/update/rollback/deprecate)';
COMMENT ON COLUMN ml_model_versions.metrics_delta IS '指标变化';

-- ============================================================
-- 7. 触发器: 自动更新 updated_at 字段
-- ============================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- ml_models 更新触发器
DROP TRIGGER IF EXISTS update_ml_models_updated_at ON ml_models;
CREATE TRIGGER update_ml_models_updated_at
    BEFORE UPDATE ON ml_models
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================
-- 8. 视图: 模型统计视图
-- ============================================================
CREATE OR REPLACE VIEW v_model_statistics AS
SELECT 
    model_id,
    model_type,
    COUNT(*) as version_count,
    MAX(created_at) as latest_created,
    MAX(updated_at) as latest_updated,
    AVG(size_bytes) as avg_size_bytes,
    SUM(size_bytes) as total_size_bytes,
    MAX((metrics->>'accuracy')::float) as best_accuracy
FROM ml_models
WHERE is_active = TRUE
GROUP BY model_id, model_type
ORDER BY latest_updated DESC;

COMMENT ON VIEW v_model_statistics IS '模型统计视图，按模型ID汇总统计信息';

-- ============================================================
-- 9. 视图: 训练统计视图
-- ============================================================
CREATE OR REPLACE VIEW v_training_statistics AS
SELECT 
    model_id,
    COUNT(*) as training_count,
    AVG(training_duration) as avg_duration,
    SUM(samples_trained) as total_samples,
    AVG((training_metrics->>'accuracy')::float) as avg_accuracy,
    MAX(created_at) as last_training
FROM ml_training_history
WHERE status = 'completed'
GROUP BY model_id
ORDER BY last_training DESC;

COMMENT ON VIEW v_training_statistics IS '训练统计视图，按模型ID汇总训练信息';

-- ============================================================
-- 10. 存储过程: 清理过期缓存
-- ============================================================
CREATE OR REPLACE PROCEDURE cleanup_expired_caches()
LANGUAGE plpgsql
AS $$
DECLARE
    v_inference_cleaned INTEGER;
    v_feature_cleaned INTEGER;
BEGIN
    DELETE FROM inference_cache WHERE expires_at < CURRENT_TIMESTAMP;
    GET DIAGNOSTICS v_inference_cleaned = ROW_COUNT;
    
    DELETE FROM feature_cache WHERE expires_at < CURRENT_TIMESTAMP;
    GET DIAGNOSTICS v_feature_cleaned = ROW_COUNT;
    
    RAISE NOTICE 'Cleaned % inference cache entries, % feature cache entries', 
                 v_inference_cleaned, v_feature_cleaned;
END;
$$;

COMMENT ON PROCEDURE cleanup_expired_caches IS '清理过期的推理缓存和特征缓存';

-- ============================================================
-- 11. 存储过程: 归档旧模型
-- ============================================================
CREATE OR REPLACE PROCEDURE archive_old_models(
    p_days_old INTEGER DEFAULT 90
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_archived_count INTEGER;
BEGIN
    UPDATE ml_models 
    SET status = 'archived', is_active = FALSE
    WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '1 day' * p_days_old
      AND status = 'active'
      AND is_active = TRUE;
    
    GET DIAGNOSTICS v_archived_count = ROW_COUNT;
    
    RAISE NOTICE 'Archived % models older than % days', v_archived_count, p_days_old;
END;
$$;

COMMENT ON PROCEDURE archive_old_models IS '归档旧模型，将其状态设为archived';

-- ============================================================
-- 12. 权限设置 (根据实际环境调整)
-- ============================================================
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ml_models TO rqa2025_admin;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ml_training_history TO rqa2025_admin;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ml_inference_results TO rqa2025_admin;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON inference_cache TO rqa2025_admin;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON feature_cache TO rqa2025_admin;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ml_model_versions TO rqa2025_admin;
-- GRANT EXECUTE ON PROCEDURE cleanup_expired_caches TO rqa2025_admin;
-- GRANT EXECUTE ON PROCEDURE archive_old_models TO rqa2025_admin;
