-- 迁移脚本：创建模型元数据表
-- 执行时间: 2026-02-16

BEGIN;

-- 创建模型元数据表
CREATE TABLE IF NOT EXISTS trained_models (
    model_id VARCHAR(255) PRIMARY KEY,
    job_id VARCHAR(255),
    model_name VARCHAR(255) NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) DEFAULT '1.0.0',
    model_path VARCHAR(500) NOT NULL,
    model_format VARCHAR(50) DEFAULT 'pickle',
    
    -- 性能指标
    accuracy FLOAT,
    loss FLOAT,
    precision FLOAT,
    recall FLOAT,
    f1_score FLOAT,
    auc_roc FLOAT,
    
    -- 训练信息
    training_time INTEGER,
    epochs INTEGER,
    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- 状态和配置
    status VARCHAR(50) DEFAULT 'active',
    is_deployed BOOLEAN DEFAULT FALSE,
    
    -- JSON字段
    hyperparameters JSONB,
    feature_columns JSONB,
    training_data_source VARCHAR(255),
    training_data_range JSONB,
    training_samples INTEGER,
    metadata JSONB,
    description TEXT,
    tags JSONB,
    
    -- 版本控制
    parent_model_id VARCHAR(255),
    version_notes TEXT,
    
    -- 审计字段
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255)
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_trained_models_job_id ON trained_models(job_id);
CREATE INDEX IF NOT EXISTS idx_trained_models_model_type ON trained_models(model_type);
CREATE INDEX IF NOT EXISTS idx_trained_models_status ON trained_models(status);
CREATE INDEX IF NOT EXISTS idx_trained_models_trained_at ON trained_models(trained_at DESC);
CREATE INDEX IF NOT EXISTS idx_trained_models_accuracy ON trained_models(accuracy DESC);
CREATE INDEX IF NOT EXISTS idx_trained_models_is_deployed ON trained_models(is_deployed);

-- 创建更新时间触发器
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_trained_models_updated_at ON trained_models;
CREATE TRIGGER update_trained_models_updated_at
    BEFORE UPDATE ON trained_models
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- 修改 model_training_jobs 表
ALTER TABLE model_training_jobs 
ADD COLUMN IF NOT EXISTS model_path VARCHAR(500),
ADD COLUMN IF NOT EXISTS model_id VARCHAR(255),
ADD COLUMN IF NOT EXISTS is_model_saved BOOLEAN DEFAULT FALSE;

COMMIT;
