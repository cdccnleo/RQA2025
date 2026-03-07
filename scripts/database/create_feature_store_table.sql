-- 特征存储表创建脚本
-- 创建时间: 2026-02-22
-- 用途: 存储特征工程任务生成的特征元数据
-- 符合特征层架构设计：支持特征元数据管理、版本控制和质量评估

-- 特征存储表
CREATE TABLE IF NOT EXISTS feature_store (
    feature_id VARCHAR(200) PRIMARY KEY,
    task_id VARCHAR(100) NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    feature_type VARCHAR(50),
    parameters JSONB,
    symbol VARCHAR(20),
    quality_score DECIMAL(5, 4),
    importance DECIMAL(5, 4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 特征存储表索引
-- 按任务ID索引，用于快速查询某个任务的所有特征
CREATE INDEX IF NOT EXISTS idx_feature_store_task_id ON feature_store(task_id);

-- 按股票代码索引，用于快速查询某只股票的所有特征
CREATE INDEX IF NOT EXISTS idx_feature_store_symbol ON feature_store(symbol);

-- 按特征类型索引，用于快速查询某类特征（如所有SMA特征）
CREATE INDEX IF NOT EXISTS idx_feature_store_feature_type ON feature_store(feature_type);

-- 按创建时间索引，用于排序和清理旧数据
CREATE INDEX IF NOT EXISTS idx_feature_store_created_at ON feature_store(created_at DESC);

-- 复合索引：任务ID + 特征名称，用于快速检查重复特征
CREATE INDEX IF NOT EXISTS idx_feature_store_task_feature ON feature_store(task_id, feature_name);

-- 提交所有更改
COMMIT;

-- 添加表注释
COMMENT ON TABLE feature_store IS '特征存储表，存储特征工程任务生成的特征元数据';
COMMENT ON COLUMN feature_store.feature_id IS '特征唯一标识，格式：{task_id}_{feature_name}';
COMMENT ON COLUMN feature_store.task_id IS '关联的特征工程任务ID';
COMMENT ON COLUMN feature_store.feature_name IS '特征名称，如 SMA_5, EMA_10';
COMMENT ON COLUMN feature_store.feature_type IS '特征类型，如 SMA, EMA, RSI, MACD';
COMMENT ON COLUMN feature_store.parameters IS '特征参数，JSONB格式，如 {"period": 5}';
COMMENT ON COLUMN feature_store.symbol IS '股票代码';
COMMENT ON COLUMN feature_store.quality_score IS '特征质量评分，0-1之间的小数';
COMMENT ON COLUMN feature_store.importance IS '特征重要性，0-1之间的小数';
COMMENT ON COLUMN feature_store.created_at IS '特征创建时间';
COMMENT ON COLUMN feature_store.updated_at IS '特征更新时间';
