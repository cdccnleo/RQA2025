-- 特征选择历史表迁移脚本
-- 创建 feature_selection_history 表，支持 PostgreSQL + 文件系统双存储

-- 创建表
CREATE TABLE IF NOT EXISTS feature_selection_history (
    id SERIAL PRIMARY KEY,
    selection_id VARCHAR(100) UNIQUE NOT NULL,
    task_id VARCHAR(100) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    input_features JSONB,
    input_feature_count INTEGER DEFAULT 0,
    selection_method VARCHAR(50),
    selection_params JSONB,
    selected_features JSONB,
    selected_feature_count INTEGER DEFAULT 0,
    selection_ratio FLOAT DEFAULT 0.0,
    evaluation_metrics JSONB,
    processing_time FLOAT DEFAULT 0.0,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_feature_selection_task_id ON feature_selection_history(task_id);
CREATE INDEX IF NOT EXISTS idx_feature_selection_timestamp ON feature_selection_history(timestamp);
CREATE INDEX IF NOT EXISTS idx_feature_selection_method ON feature_selection_history(selection_method);

-- 添加注释
COMMENT ON TABLE feature_selection_history IS '特征选择历史记录表';
COMMENT ON COLUMN feature_selection_history.selection_id IS '选择记录唯一标识';
COMMENT ON COLUMN feature_selection_history.task_id IS '关联的任务ID';
COMMENT ON COLUMN feature_selection_history.timestamp IS '选择操作时间戳';
COMMENT ON COLUMN feature_selection_history.input_features IS '输入特征列表（JSONB）';
COMMENT ON COLUMN feature_selection_history.selected_features IS '选择的特征列表（JSONB）';
COMMENT ON COLUMN feature_selection_history.selection_method IS '选择方法：correlation, importance, etc.';
COMMENT ON COLUMN feature_selection_history.selection_ratio IS '选择比例';

-- 创建更新时间触发器
CREATE OR REPLACE FUNCTION update_feature_selection_history_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_feature_selection_history ON feature_selection_history;
CREATE TRIGGER trigger_update_feature_selection_history
    BEFORE UPDATE ON feature_selection_history
    FOR EACH ROW
    EXECUTE FUNCTION update_feature_selection_history_updated_at();
