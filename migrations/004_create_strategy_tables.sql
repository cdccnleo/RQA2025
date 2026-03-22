-- ============================================================
-- 策略服务层数据库表结构定义
-- 数据库: PostgreSQL
-- 版本: 1.0.0
-- 创建时间: 2026-03-22
-- 描述: 策略存储、策略配置、策略生命周期、回测结果的表结构定义
-- ============================================================

-- ============================================================
-- 1. 策略存储表 (strategies)
-- 用于存储策略定义和元数据
-- ============================================================
CREATE TABLE IF NOT EXISTS strategies (
    id SERIAL PRIMARY KEY,
    strategy_id VARCHAR(128) NOT NULL UNIQUE,
    strategy_name VARCHAR(255) NOT NULL,
    strategy_type VARCHAR(64) NOT NULL,
    version VARCHAR(32) DEFAULT '1.0.0',
    status VARCHAR(32) DEFAULT 'draft',
    description TEXT,
    author VARCHAR(128),
    tags JSONB DEFAULT '[]',
    config JSONB DEFAULT '{}',
    strategy_data JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- 策略存储索引
CREATE INDEX IF NOT EXISTS idx_strategies_id ON strategies(strategy_id);
CREATE INDEX IF NOT EXISTS idx_strategies_type ON strategies(strategy_type);
CREATE INDEX IF NOT EXISTS idx_strategies_status ON strategies(status);
CREATE INDEX IF NOT EXISTS idx_strategies_name ON strategies(strategy_name);

-- 策略存储注释
COMMENT ON TABLE strategies IS '策略存储表，存储策略定义和元数据';
COMMENT ON COLUMN strategies.strategy_id IS '策略唯一标识符';
COMMENT ON COLUMN strategies.strategy_type IS '策略类型(trend_following/mean_reversion/ml_based/rl_based)';
COMMENT ON COLUMN strategies.status IS '策略状态(draft/active/deprecated/archived)';
COMMENT ON COLUMN strategies.strategy_data IS '策略完整数据(JSON)';

-- ============================================================
-- 2. 策略配置表 (strategy_configs)
-- 用于存储策略的运行配置
-- ============================================================
CREATE TABLE IF NOT EXISTS strategy_configs (
    id SERIAL PRIMARY KEY,
    strategy_id VARCHAR(128) NOT NULL,
    config_name VARCHAR(128) NOT NULL,
    config_data JSONB DEFAULT '{}',
    version VARCHAR(32) DEFAULT '1.0.0',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(strategy_id, config_name)
);

-- 策略配置索引
CREATE INDEX IF NOT EXISTS idx_strategy_configs_strategy ON strategy_configs(strategy_id);
CREATE INDEX IF NOT EXISTS idx_strategy_configs_name ON strategy_configs(config_name);

-- 策略配置注释
COMMENT ON TABLE strategy_configs IS '策略配置表，存储策略的运行配置';
COMMENT ON COLUMN strategy_configs.config_name IS '配置名称(default/production/development)';

-- ============================================================
-- 3. 策略生命周期表 (strategy_lifecycle)
-- 用于管理策略的生命周期状态
-- ============================================================
CREATE TABLE IF NOT EXISTS strategy_lifecycle (
    id SERIAL PRIMARY KEY,
    strategy_id VARCHAR(128) NOT NULL UNIQUE,
    current_stage VARCHAR(64) NOT NULL,
    stage_history JSONB DEFAULT '[]',
    next_allowed_actions JSONB DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- 策略生命周期索引
CREATE INDEX IF NOT EXISTS idx_strategy_lifecycle_strategy ON strategy_lifecycle(strategy_id);
CREATE INDEX IF NOT EXISTS idx_strategy_lifecycle_stage ON strategy_lifecycle(current_stage);

-- 策略生命周期注释
COMMENT ON TABLE strategy_lifecycle IS '策略生命周期表，管理策略的生命周期状态';
COMMENT ON COLUMN strategy_lifecycle.current_stage IS '当前阶段(created/developing/testing/backtesting/running/retired)';
COMMENT ON COLUMN strategy_lifecycle.stage_history IS '阶段历史记录';

-- ============================================================
-- 4. 回测配置表 (backtest_configs)
-- 用于存储回测任务配置
-- ============================================================
CREATE TABLE IF NOT EXISTS backtest_configs (
    id SERIAL PRIMARY KEY,
    backtest_id VARCHAR(128) NOT NULL UNIQUE,
    strategy_id VARCHAR(128) NOT NULL,
    start_date TIMESTAMP NOT NULL,
    end_date TIMESTAMP NOT NULL,
    initial_capital FLOAT NOT NULL,
    commission FLOAT DEFAULT 0.0003,
    slippage FLOAT DEFAULT 0.0001,
    benchmark_symbol VARCHAR(32),
    data_frequency VARCHAR(16) DEFAULT '1d',
    mode VARCHAR(32) DEFAULT 'single',
    parameters JSONB DEFAULT '{}',
    risk_limits JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- 回测配置索引
CREATE INDEX IF NOT EXISTS idx_backtest_configs_strategy ON backtest_configs(strategy_id);
CREATE INDEX IF NOT EXISTS idx_backtest_configs_dates ON backtest_configs(start_date, end_date);
CREATE INDEX IF NOT EXISTS idx_backtest_configs_mode ON backtest_configs(mode);

-- 回测配置注释
COMMENT ON TABLE backtest_configs IS '回测配置表，存储回测任务配置';
COMMENT ON COLUMN backtest_configs.mode IS '回测模式(single/multi_strategy/parameter_sweep/walk_forward)';

-- ============================================================
-- 5. 回测结果表 (backtest_results)
-- 用于存储回测执行结果
-- ============================================================
CREATE TABLE IF NOT EXISTS backtest_results (
    id SERIAL PRIMARY KEY,
    backtest_id VARCHAR(128) NOT NULL UNIQUE,
    strategy_id VARCHAR(128) NOT NULL,
    status VARCHAR(32) DEFAULT 'completed',
    execution_time FLOAT,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    returns_data BYTEA,
    positions_data BYTEA,
    trades_data BYTEA,
    metrics JSONB DEFAULT '{}',
    risk_metrics JSONB DEFAULT '{}',
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 回测结果索引
CREATE INDEX IF NOT EXISTS idx_backtest_results_strategy ON backtest_results(strategy_id);
CREATE INDEX IF NOT EXISTS idx_backtest_results_status ON backtest_results(status);
CREATE INDEX IF NOT EXISTS idx_backtest_results_created ON backtest_results(created_at DESC);

-- 回测结果注释
COMMENT ON TABLE backtest_results IS '回测结果表，存储回测执行结果';
COMMENT ON COLUMN backtest_results.returns_data IS '序列化的收益率数据';
COMMENT ON COLUMN backtest_results.positions_data IS '序列化的持仓数据';
COMMENT ON COLUMN backtest_results.trades_data IS '序列化的交易记录';

-- ============================================================
-- 6. 回测指标表 (backtest_metrics)
-- 用于存储回测性能指标
-- ============================================================
CREATE TABLE IF NOT EXISTS backtest_metrics (
    id SERIAL PRIMARY KEY,
    backtest_id VARCHAR(128) NOT NULL UNIQUE,
    total_return FLOAT,
    annual_return FLOAT,
    volatility FLOAT,
    sharpe_ratio FLOAT,
    max_drawdown FLOAT,
    win_rate FLOAT,
    profit_factor FLOAT,
    calmar_ratio FLOAT,
    sortino_ratio FLOAT,
    alpha FLOAT,
    beta FLOAT,
    information_ratio FLOAT,
    var_95 FLOAT,
    expected_shortfall FLOAT,
    recovery_time INTEGER,
    consecutive_wins INTEGER,
    consecutive_losses INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 回测指标索引
CREATE INDEX IF NOT EXISTS idx_backtest_metrics_backtest ON backtest_metrics(backtest_id);
CREATE INDEX IF NOT EXISTS idx_backtest_metrics_sharpe ON backtest_metrics(sharpe_ratio DESC);
CREATE INDEX IF NOT EXISTS idx_backtest_metrics_return ON backtest_metrics(total_return DESC);

-- 回测指标注释
COMMENT ON TABLE backtest_metrics IS '回测指标表，存储回测性能指标';
COMMENT ON COLUMN backtest_metrics.sharpe_ratio IS '夏普比率';
COMMENT ON COLUMN backtest_metrics.max_drawdown IS '最大回撤';
COMMENT ON COLUMN backtest_metrics.var_95 IS '95%置信度VaR';

-- ============================================================
-- 7. 回测交易记录表 (backtest_trades)
-- 用于存储回测过程中的交易记录
-- ============================================================
CREATE TABLE IF NOT EXISTS backtest_trades (
    id SERIAL PRIMARY KEY,
    trade_id VARCHAR(128) NOT NULL,
    backtest_id VARCHAR(128) NOT NULL,
    strategy_id VARCHAR(128) NOT NULL,
    symbol VARCHAR(32) NOT NULL,
    side VARCHAR(16) NOT NULL,
    quantity FLOAT NOT NULL,
    price FLOAT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    commission FLOAT,
    slippage FLOAT,
    pnl FLOAT,
    pnl_pct FLOAT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 回测交易记录索引
CREATE INDEX IF NOT EXISTS idx_backtest_trades_backtest ON backtest_trades(backtest_id);
CREATE INDEX IF NOT EXISTS idx_backtest_trades_symbol ON backtest_trades(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_backtest_trades_strategy ON backtest_trades(strategy_id, timestamp);

-- 回测交易记录注释
COMMENT ON TABLE backtest_trades IS '回测交易记录表，存储回测过程中的交易记录';
COMMENT ON COLUMN backtest_trades.side IS '交易方向(buy/sell)';
COMMENT ON COLUMN backtest_trades.pnl IS '盈亏金额';

-- ============================================================
-- 8. 策略信号表 (strategy_signals)
-- 用于存储策略生成的交易信号
-- ============================================================
CREATE TABLE IF NOT EXISTS strategy_signals (
    id SERIAL PRIMARY KEY,
    signal_id VARCHAR(128) NOT NULL UNIQUE,
    strategy_id VARCHAR(128) NOT NULL,
    symbol VARCHAR(32) NOT NULL,
    signal_type VARCHAR(32) NOT NULL,
    direction VARCHAR(16) NOT NULL,
    strength FLOAT DEFAULT 1.0,
    price FLOAT,
    quantity FLOAT,
    timestamp TIMESTAMP NOT NULL,
    expiry TIMESTAMP,
    metadata JSONB DEFAULT '{}',
    status VARCHAR(32) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 策略信号索引
CREATE INDEX IF NOT EXISTS idx_strategy_signals_strategy ON strategy_signals(strategy_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_strategy_signals_symbol ON strategy_signals(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_strategy_signals_status ON strategy_signals(status);

-- 策略信号注释
COMMENT ON TABLE strategy_signals IS '策略信号表，存储策略生成的交易信号';
COMMENT ON COLUMN strategy_signals.signal_type IS '信号类型(entry/exit/stop_loss/take_profit)';
COMMENT ON COLUMN strategy_signals.direction IS '信号方向(long/short)';
COMMENT ON COLUMN strategy_signals.status IS '信号状态(pending/executed/expired/cancelled)';

-- ============================================================
-- 9. 策略性能监控表 (strategy_performance)
-- 用于存储策略运行时的性能数据
-- ============================================================
CREATE TABLE IF NOT EXISTS strategy_performance (
    id SERIAL PRIMARY KEY,
    strategy_id VARCHAR(128) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    portfolio_value FLOAT,
    cash FLOAT,
    position_value FLOAT,
    unrealized_pnl FLOAT,
    realized_pnl FLOAT,
    daily_return FLOAT,
    cumulative_return FLOAT,
    drawdown FLOAT,
    positions_count INTEGER,
    trades_count INTEGER,
    win_rate FLOAT,
    metrics JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 策略性能监控索引
CREATE INDEX IF NOT EXISTS idx_strategy_performance_strategy ON strategy_performance(strategy_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_strategy_performance_timestamp ON strategy_performance(timestamp DESC);

-- 策略性能监控注释
COMMENT ON TABLE strategy_performance IS '策略性能监控表，存储策略运行时的性能数据';

-- ============================================================
-- 10. 触发器: 自动更新 updated_at 字段
-- ============================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- strategies 更新触发器
DROP TRIGGER IF EXISTS update_strategies_updated_at ON strategies;
CREATE TRIGGER update_strategies_updated_at
    BEFORE UPDATE ON strategies
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- strategy_configs 更新触发器
DROP TRIGGER IF EXISTS update_strategy_configs_updated_at ON strategy_configs;
CREATE TRIGGER update_strategy_configs_updated_at
    BEFORE UPDATE ON strategy_configs
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- strategy_lifecycle 更新触发器
DROP TRIGGER IF EXISTS update_strategy_lifecycle_updated_at ON strategy_lifecycle;
CREATE TRIGGER update_strategy_lifecycle_updated_at
    BEFORE UPDATE ON strategy_lifecycle
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================
-- 11. 视图: 策略统计视图
-- ============================================================
CREATE OR REPLACE VIEW v_strategy_statistics AS
SELECT 
    s.strategy_id,
    s.strategy_name,
    s.strategy_type,
    s.status,
    COUNT(DISTINCT bc.backtest_id) as backtest_count,
    COUNT(DISTINCT bs.signal_id) as signal_count,
    MAX(bc.created_at) as last_backtest,
    MAX(bs.timestamp) as last_signal,
    s.created_at,
    s.updated_at
FROM strategies s
LEFT JOIN backtest_configs bc ON s.strategy_id = bc.strategy_id AND bc.is_active = TRUE
LEFT JOIN strategy_signals bs ON s.strategy_id = bs.strategy_id
WHERE s.is_active = TRUE
GROUP BY s.strategy_id, s.strategy_name, s.strategy_type, s.status, s.created_at, s.updated_at
ORDER BY s.updated_at DESC;

COMMENT ON VIEW v_strategy_statistics IS '策略统计视图，按策略ID汇总统计信息';

-- ============================================================
-- 12. 视图: 回测性能排名视图
-- ============================================================
CREATE OR REPLACE VIEW v_backtest_performance_ranking AS
SELECT 
    bm.backtest_id,
    bc.strategy_id,
    s.strategy_name,
    bm.total_return,
    bm.annual_return,
    bm.sharpe_ratio,
    bm.max_drawdown,
    bm.win_rate,
    bm.profit_factor,
    bc.initial_capital,
    bc.start_date,
    bc.end_date,
    bm.created_at,
    RANK() OVER (ORDER BY bm.sharpe_ratio DESC) as sharpe_rank,
    RANK() OVER (ORDER BY bm.total_return DESC) as return_rank
FROM backtest_metrics bm
JOIN backtest_configs bc ON bm.backtest_id = bc.backtest_id
JOIN strategies s ON bc.strategy_id = s.strategy_id
ORDER BY bm.sharpe_ratio DESC;

COMMENT ON VIEW v_backtest_performance_ranking IS '回测性能排名视图，按夏普比率排序';

-- ============================================================
-- 13. 存储过程: 清理过期数据
-- ============================================================
CREATE OR REPLACE PROCEDURE cleanup_expired_strategy_data(
    p_days_old INTEGER DEFAULT 90
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_cleaned_configs INTEGER;
    v_cleaned_results INTEGER;
    v_cleaned_signals INTEGER;
BEGIN
    DELETE FROM backtest_configs 
    WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '1 day' * p_days_old
      AND is_active = FALSE;
    GET DIAGNOSTICS v_cleaned_configs = ROW_COUNT;
    
    DELETE FROM backtest_results 
    WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '1 day' * p_days_old;
    GET DIAGNOSTICS v_cleaned_results = ROW_COUNT;
    
    DELETE FROM strategy_signals 
    WHERE timestamp < CURRENT_TIMESTAMP - INTERVAL '1 day' * p_days_old
      AND status IN ('expired', 'cancelled');
    GET DIAGNOSTICS v_cleaned_signals = ROW_COUNT;
    
    RAISE NOTICE 'Cleaned: % configs, % results, % signals', 
                 v_cleaned_configs, v_cleaned_results, v_cleaned_signals;
    
    COMMIT;
END;
$$;

COMMENT ON PROCEDURE cleanup_expired_strategy_data IS '清理过期的策略数据';

-- ============================================================
-- 14. 存储过程: 归档旧策略
-- ============================================================
CREATE OR REPLACE PROCEDURE archive_old_strategies(
    p_days_old INTEGER DEFAULT 180
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_archived_count INTEGER;
BEGIN
    UPDATE strategies 
    SET status = 'archived', is_active = FALSE
    WHERE updated_at < CURRENT_TIMESTAMP - INTERVAL '1 day' * p_days_old
      AND status = 'active'
      AND is_active = TRUE;
    
    GET DIAGNOSTICS v_archived_count = ROW_COUNT;
    
    RAISE NOTICE 'Archived % strategies older than % days', v_archived_count, p_days_old;
    
    COMMIT;
END;
$$;

COMMENT ON PROCEDURE archive_old_strategies IS '归档旧策略，将其状态设为archived';

-- ============================================================
-- 15. 权限设置 (根据实际环境调整)
-- ============================================================
-- GRANT SELECT, INSERT, UPDATE, DELETE ON strategies TO rqa2025_admin;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON strategy_configs TO rqa2025_admin;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON strategy_lifecycle TO rqa2025_admin;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON backtest_configs TO rqa2025_admin;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON backtest_results TO rqa2025_admin;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON backtest_metrics TO rqa2025_admin;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON backtest_trades TO rqa2025_admin;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON strategy_signals TO rqa2025_admin;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON strategy_performance TO rqa2025_admin;
-- GRANT EXECUTE ON PROCEDURE cleanup_expired_strategy_data TO rqa2025_admin;
-- GRANT EXECUTE ON PROCEDURE archive_old_strategies TO rqa2025_admin;
