-- 策略开发流程数据库表结构
-- 创建时间: 2026-02-08

-- 策略构思表
CREATE TABLE IF NOT EXISTS strategy_conceptions (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(100) NOT NULL,
    description TEXT,
    target_market VARCHAR(100),
    risk_level VARCHAR(50),
    nodes JSONB NOT NULL,
    connections JSONB NOT NULL,
    parameters JSONB,
    backtest_result JSONB,
    version INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    saved_locally BOOLEAN DEFAULT FALSE
);

-- 策略构思表索引
CREATE INDEX IF NOT EXISTS idx_strategy_conceptions_type ON strategy_conceptions(type);
CREATE INDEX IF NOT EXISTS idx_strategy_conceptions_created_at ON strategy_conceptions(created_at);
CREATE INDEX IF NOT EXISTS idx_strategy_conceptions_name ON strategy_conceptions USING gin(to_tsvector('english', name));

-- 策略管理表
CREATE TABLE IF NOT EXISTS strategy_management (
    id VARCHAR(255) PRIMARY KEY,
    conception_id VARCHAR(255) REFERENCES strategy_conceptions(id),
    status VARCHAR(50) DEFAULT 'draft',
    last_modified_by VARCHAR(100),
    tags JSONB,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 策略管理表索引
CREATE INDEX IF NOT EXISTS idx_strategy_management_status ON strategy_management(status);
CREATE INDEX IF NOT EXISTS idx_strategy_management_conception_id ON strategy_management(conception_id);

-- 策略优化表
CREATE TABLE IF NOT EXISTS strategy_optimizations (
    id VARCHAR(255) PRIMARY KEY,
    strategy_id VARCHAR(255) REFERENCES strategy_conceptions(id),
    optimization_type VARCHAR(100) NOT NULL,
    parameters JSONB NOT NULL,
    results JSONB,
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 策略优化表索引
CREATE INDEX IF NOT EXISTS idx_strategy_optimizations_strategy_id ON strategy_optimizations(strategy_id);
CREATE INDEX IF NOT EXISTS idx_strategy_optimizations_status ON strategy_optimizations(status);
CREATE INDEX IF NOT EXISTS idx_strategy_optimizations_type ON strategy_optimizations(optimization_type);

-- 策略性能评估表
CREATE TABLE IF NOT EXISTS strategy_performance_evaluations (
    id VARCHAR(255) PRIMARY KEY,
    strategy_id VARCHAR(255) REFERENCES strategy_conceptions(id),
    backtest_id VARCHAR(100),
    metrics JSONB NOT NULL,
    risk_metrics JSONB,
    benchmark_comparison JSONB,
    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 策略性能评估表索引
CREATE INDEX IF NOT EXISTS idx_strategy_performance_strategy_id ON strategy_performance_evaluations(strategy_id);
CREATE INDEX IF NOT EXISTS idx_strategy_performance_backtest_id ON strategy_performance_evaluations(backtest_id);
CREATE INDEX IF NOT EXISTS idx_strategy_performance_date ON strategy_performance_evaluations(evaluation_date);

-- 策略部署表
CREATE TABLE IF NOT EXISTS strategy_lifecycle (
    id VARCHAR(255) PRIMARY KEY,
    strategy_id VARCHAR(255) REFERENCES strategy_conceptions(id),
    deployment_status VARCHAR(50) DEFAULT 'undeployed',
    deployment_environment VARCHAR(100),
    configuration JSONB,
    deployment_date TIMESTAMP,
    last_executed TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 策略部署表索引
CREATE INDEX IF NOT EXISTS idx_strategy_lifecycle_strategy_id ON strategy_lifecycle(strategy_id);
CREATE INDEX IF NOT EXISTS idx_strategy_lifecycle_status ON strategy_lifecycle(deployment_status);
CREATE INDEX IF NOT EXISTS idx_strategy_lifecycle_environment ON strategy_lifecycle(deployment_environment);

-- 策略执行监控表
CREATE TABLE IF NOT EXISTS strategy_execution_monitor (
    id VARCHAR(255) PRIMARY KEY,
    strategy_id VARCHAR(255) REFERENCES strategy_conceptions(id),
    execution_id VARCHAR(100),
    status VARCHAR(50) NOT NULL,
    signals JSONB,
    trades JSONB,
    metrics JSONB,
    errors JSONB,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 策略执行监控表索引
CREATE INDEX IF NOT EXISTS idx_strategy_execution_strategy_id ON strategy_execution_monitor(strategy_id);
CREATE INDEX IF NOT EXISTS idx_strategy_execution_status ON strategy_execution_monitor(status);
CREATE INDEX IF NOT EXISTS idx_strategy_execution_created_at ON strategy_execution_monitor(created_at);
CREATE INDEX IF NOT EXISTS idx_strategy_execution_execution_id ON strategy_execution_monitor(execution_id);

-- 回测结果表（如果不存在）
CREATE TABLE IF NOT EXISTS backtest_results (
    backtest_id VARCHAR(100) PRIMARY KEY,
    strategy_id VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    initial_capital DECIMAL(18, 2) NOT NULL,
    final_capital DECIMAL(18, 2),
    total_return DECIMAL(10, 4),
    annualized_return DECIMAL(10, 4),
    sharpe_ratio DECIMAL(10, 4),
    max_drawdown DECIMAL(10, 4),
    win_rate DECIMAL(10, 4),
    total_trades INTEGER,
    equity_curve JSONB,
    trades JSONB,
    metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 回测结果表索引
CREATE INDEX IF NOT EXISTS idx_backtest_strategy ON backtest_results(strategy_id);
CREATE INDEX IF NOT EXISTS idx_backtest_created ON backtest_results(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_backtest_status ON backtest_results(status);

-- 提交所有更改
COMMIT;