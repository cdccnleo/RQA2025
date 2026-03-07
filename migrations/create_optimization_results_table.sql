-- 创建策略优化结果表
-- 用于存储策略优化任务的持久化数据
-- 支持双写机制（文件系统 + PostgreSQL）

CREATE TABLE IF NOT EXISTS optimization_results (
    id SERIAL PRIMARY KEY,
    task_id VARCHAR(255) UNIQUE NOT NULL,
    strategy_id VARCHAR(255) NOT NULL,
    strategy_name VARCHAR(500),
    method VARCHAR(100) NOT NULL,
    target VARCHAR(100) NOT NULL,
    results JSONB NOT NULL DEFAULT '[]'::jsonb,
    completed_at TIMESTAMP WITH TIME ZONE,
    saved_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_optimization_results_task_id ON optimization_results(task_id);
CREATE INDEX IF NOT EXISTS idx_optimization_results_strategy_id ON optimization_results(strategy_id);
CREATE INDEX IF NOT EXISTS idx_optimization_results_saved_at ON optimization_results(saved_at DESC);
CREATE INDEX IF NOT EXISTS idx_optimization_results_method ON optimization_results(method);
CREATE INDEX IF NOT EXISTS idx_optimization_results_target ON optimization_results(target);

-- 创建更新时间触发器
CREATE OR REPLACE FUNCTION update_optimization_results_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_optimization_results_updated_at ON optimization_results;

CREATE TRIGGER trigger_update_optimization_results_updated_at
    BEFORE UPDATE ON optimization_results
    FOR EACH ROW
    EXECUTE FUNCTION update_optimization_results_updated_at();

-- 添加表注释
COMMENT ON TABLE optimization_results IS '策略优化结果表，存储策略参数优化的结果数据';
COMMENT ON COLUMN optimization_results.task_id IS '优化任务唯一标识';
COMMENT ON COLUMN optimization_results.strategy_id IS '关联的策略ID';
COMMENT ON COLUMN optimization_results.strategy_name IS '策略名称';
COMMENT ON COLUMN optimization_results.method IS '优化方法（grid_search, bayesian, genetic等）';
COMMENT ON COLUMN optimization_results.target IS '优化目标（sharpe, return, drawdown等）';
COMMENT ON COLUMN optimization_results.results IS '优化结果列表，JSONB格式存储参数组合和性能指标';
COMMENT ON COLUMN optimization_results.completed_at IS '优化完成时间';
COMMENT ON COLUMN optimization_results.saved_at IS '数据保存时间';
COMMENT ON COLUMN optimization_results.updated_at IS '数据更新时间';
