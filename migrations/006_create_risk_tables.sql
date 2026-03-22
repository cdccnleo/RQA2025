-- ============================================================
-- 风险控制层数据库表结构定义
-- 数据库: PostgreSQL
-- 版本: 1.0.0
-- 创建时间: 2026-03-22
-- 描述: 风险检查、告警记录、风险指标、风险规则的表结构定义
-- ============================================================

-- ============================================================
-- 1. 风险检查记录表 (risk_checks)
-- 用于存储风险检查的历史记录和结果
-- ============================================================
CREATE TABLE IF NOT EXISTS risk_checks (
    id SERIAL PRIMARY KEY,
    check_id VARCHAR(128) NOT NULL UNIQUE,
    check_type VARCHAR(64) NOT NULL,
    risk_level VARCHAR(32) NOT NULL,
    passed BOOLEAN NOT NULL DEFAULT FALSE,
    score DECIMAL(10, 6) DEFAULT 0.0,
    symbol VARCHAR(32),
    account_id VARCHAR(128),
    strategy_id VARCHAR(128),
    order_id VARCHAR(128),
    portfolio_id VARCHAR(128),
    details JSONB DEFAULT '{}',
    recommendations JSONB DEFAULT '[]',
    warnings JSONB DEFAULT '[]',
    check_duration_ms INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- 风险检查索引
CREATE INDEX IF NOT EXISTS idx_risk_checks_id ON risk_checks(check_id);
CREATE INDEX IF NOT EXISTS idx_risk_checks_type ON risk_checks(check_type);
CREATE INDEX IF NOT EXISTS idx_risk_checks_level ON risk_checks(risk_level);
CREATE INDEX IF NOT EXISTS idx_risk_checks_symbol ON risk_checks(symbol, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_risk_checks_account ON risk_checks(account_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_risk_checks_strategy ON risk_checks(strategy_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_risk_checks_passed ON risk_checks(passed);
CREATE INDEX IF NOT EXISTS idx_risk_checks_created ON risk_checks(created_at DESC);

-- 风险检查注释
COMMENT ON TABLE risk_checks IS '风险检查记录表，存储风险检查的历史记录和结果';
COMMENT ON COLUMN risk_checks.check_type IS '检查类型(position/market/liquidity/operational/compliance)';
COMMENT ON COLUMN risk_checks.risk_level IS '风险等级(low/medium/high/critical)';
COMMENT ON COLUMN risk_checks.passed IS '是否通过检查';
COMMENT ON COLUMN risk_checks.score IS '风险评分(0-1)';

-- ============================================================
-- 2. 风险告警表 (risk_alerts)
-- 用于存储风险告警信息
-- ============================================================
CREATE TABLE IF NOT EXISTS risk_alerts (
    id SERIAL PRIMARY KEY,
    alert_id VARCHAR(128) NOT NULL UNIQUE,
    alert_type VARCHAR(64) NOT NULL,
    alert_level VARCHAR(32) NOT NULL,
    title VARCHAR(512) NOT NULL,
    message TEXT,
    status VARCHAR(32) DEFAULT 'active',
    source VARCHAR(128),
    rule_id VARCHAR(128),
    symbol VARCHAR(32),
    account_id VARCHAR(128),
    strategy_id VARCHAR(128),
    portfolio_id VARCHAR(128),
    acknowledged_by VARCHAR(128),
    acknowledged_at TIMESTAMP,
    resolved_by VARCHAR(128),
    resolved_at TIMESTAMP,
    details JSONB DEFAULT '{}',
    notification_sent BOOLEAN DEFAULT FALSE,
    notification_channels JSONB DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- 风险告警索引
CREATE INDEX IF NOT EXISTS idx_risk_alerts_id ON risk_alerts(alert_id);
CREATE INDEX IF NOT EXISTS idx_risk_alerts_type ON risk_alerts(alert_type);
CREATE INDEX IF NOT EXISTS idx_risk_alerts_level ON risk_alerts(alert_level);
CREATE INDEX IF NOT EXISTS idx_risk_alerts_status ON risk_alerts(status);
CREATE INDEX IF NOT EXISTS idx_risk_alerts_symbol ON risk_alerts(symbol, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_risk_alerts_account ON risk_alerts(account_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_risk_alerts_rule ON risk_alerts(rule_id);
CREATE INDEX IF NOT EXISTS idx_risk_alerts_created ON risk_alerts(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_risk_alerts_active ON risk_alerts(status, alert_level) WHERE status = 'active';

-- 风险告警注释
COMMENT ON TABLE risk_alerts IS '风险告警表，存储风险告警信息';
COMMENT ON COLUMN risk_alerts.alert_type IS '告警类型(risk_threshold/position_limit/volatility/liquidity/system_error)';
COMMENT ON COLUMN risk_alerts.alert_level IS '告警级别(info/warning/error/critical)';
COMMENT ON COLUMN risk_alerts.status IS '告警状态(active/acknowledged/resolved/expired)';

-- ============================================================
-- 3. 风险指标表 (risk_metrics)
-- 用于存储风险指标的历史数据
-- ============================================================
CREATE TABLE IF NOT EXISTS risk_metrics (
    id SERIAL PRIMARY KEY,
    metric_id VARCHAR(128) NOT NULL UNIQUE,
    metric_name VARCHAR(128) NOT NULL,
    metric_type VARCHAR(64) NOT NULL,
    value DECIMAL(18, 8) NOT NULL,
    threshold_low DECIMAL(18, 8) DEFAULT 0.0,
    threshold_medium DECIMAL(18, 8) DEFAULT 0.0,
    threshold_high DECIMAL(18, 8) DEFAULT 0.0,
    risk_level VARCHAR(32) DEFAULT 'low',
    symbol VARCHAR(32),
    account_id VARCHAR(128),
    portfolio_id VARCHAR(128),
    calculation_method VARCHAR(64),
    confidence_level DECIMAL(5, 4),
    time_horizon INTEGER DEFAULT 1,
    lookback_period INTEGER DEFAULT 252,
    calculation_duration_ms INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- 风险指标索引
CREATE INDEX IF NOT EXISTS idx_risk_metrics_id ON risk_metrics(metric_id);
CREATE INDEX IF NOT EXISTS idx_risk_metrics_name ON risk_metrics(metric_name, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_risk_metrics_type ON risk_metrics(metric_type);
CREATE INDEX IF NOT EXISTS idx_risk_metrics_level ON risk_metrics(risk_level);
CREATE INDEX IF NOT EXISTS idx_risk_metrics_symbol ON risk_metrics(symbol, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_risk_metrics_portfolio ON risk_metrics(portfolio_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_risk_metrics_created ON risk_metrics(created_at DESC);

-- 风险指标注释
COMMENT ON TABLE risk_metrics IS '风险指标表，存储风险指标的历史数据';
COMMENT ON COLUMN risk_metrics.metric_name IS '指标名称(var_95/cvar/sharpe_ratio/max_drawdown/volatility)';
COMMENT ON COLUMN risk_metrics.metric_type IS '指标类型(market/liquidity/credit/operational)';
COMMENT ON COLUMN risk_metrics.calculation_method IS '计算方法(historical/monte_carlo/parametric)';

-- ============================================================
-- 4. 风险规则表 (risk_rules)
-- 用于存储风险规则配置
-- ============================================================
CREATE TABLE IF NOT EXISTS risk_rules (
    id SERIAL PRIMARY KEY,
    rule_id VARCHAR(128) NOT NULL UNIQUE,
    rule_name VARCHAR(256) NOT NULL,
    rule_type VARCHAR(64) NOT NULL,
    risk_type VARCHAR(64) NOT NULL,
    conditions JSONB DEFAULT '{}',
    actions JSONB DEFAULT '[]',
    alert_level VARCHAR(32) DEFAULT 'warning',
    enabled BOOLEAN DEFAULT TRUE,
    cooldown_minutes INTEGER DEFAULT 30,
    priority INTEGER DEFAULT 100,
    description TEXT,
    version INTEGER DEFAULT 1,
    created_by VARCHAR(128),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(128),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- 风险规则索引
CREATE INDEX IF NOT EXISTS idx_risk_rules_id ON risk_rules(rule_id);
CREATE INDEX IF NOT EXISTS idx_risk_rules_type ON risk_rules(rule_type);
CREATE INDEX IF NOT EXISTS idx_risk_rules_risk_type ON risk_rules(risk_type);
CREATE INDEX IF NOT EXISTS idx_risk_rules_enabled ON risk_rules(enabled);
CREATE INDEX IF NOT EXISTS idx_risk_rules_priority ON risk_rules(priority);

-- 风险规则注释
COMMENT ON TABLE risk_rules IS '风险规则表，存储风险规则配置';
COMMENT ON COLUMN risk_rules.rule_type IS '规则类型(threshold/range/complex)';
COMMENT ON COLUMN risk_rules.risk_type IS '风险类型(market/liquidity/credit/operational/compliance)';
COMMENT ON COLUMN risk_rules.conditions IS '触发条件JSON';
COMMENT ON COLUMN risk_rules.actions IS '触发动作列表';

-- ============================================================
-- 5. 风险限额表 (risk_limits)
-- 用于存储风险限额配置
-- ============================================================
CREATE TABLE IF NOT EXISTS risk_limits (
    id SERIAL PRIMARY KEY,
    limit_id VARCHAR(128) NOT NULL UNIQUE,
    limit_name VARCHAR(256) NOT NULL,
    limit_type VARCHAR(64) NOT NULL,
    scope VARCHAR(64) DEFAULT 'global',
    scope_id VARCHAR(128),
    soft_limit DECIMAL(18, 4),
    hard_limit DECIMAL(18, 4) NOT NULL,
    current_value DECIMAL(18, 4) DEFAULT 0.0,
    utilization_pct DECIMAL(10, 4) DEFAULT 0.0,
    unit VARCHAR(32),
    enabled BOOLEAN DEFAULT TRUE,
    breach_count INTEGER DEFAULT 0,
    last_breach_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- 风险限额索引
CREATE INDEX IF NOT EXISTS idx_risk_limits_id ON risk_limits(limit_id);
CREATE INDEX IF NOT EXISTS idx_risk_limits_type ON risk_limits(limit_type);
CREATE INDEX IF NOT EXISTS idx_risk_limits_scope ON risk_limits(scope, scope_id);
CREATE INDEX IF NOT EXISTS idx_risk_limits_enabled ON risk_limits(enabled);

-- 风险限额注释
COMMENT ON TABLE risk_limits IS '风险限额表，存储风险限额配置';
COMMENT ON COLUMN risk_limits.limit_type IS '限额类型(position/var/drawdown/exposure/turnover)';
COMMENT ON COLUMN risk_limits.scope IS '限额范围(global/account/strategy/symbol)';
COMMENT ON COLUMN risk_limits.soft_limit IS '软限额(预警阈值)';
COMMENT ON COLUMN risk_limits.hard_limit IS '硬限额(强制限制)';

-- ============================================================
-- 6. 风险事件表 (risk_events)
-- 用于记录重大风险事件
-- ============================================================
CREATE TABLE IF NOT EXISTS risk_events (
    id SERIAL PRIMARY KEY,
    event_id VARCHAR(128) NOT NULL UNIQUE,
    event_type VARCHAR(64) NOT NULL,
    event_level VARCHAR(32) NOT NULL,
    title VARCHAR(512) NOT NULL,
    description TEXT,
    impact_level VARCHAR(32) DEFAULT 'low',
    affected_entities JSONB DEFAULT '[]',
    root_cause TEXT,
    resolution TEXT,
    status VARCHAR(32) DEFAULT 'open',
    opened_by VARCHAR(128),
    opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    closed_by VARCHAR(128),
    closed_at TIMESTAMP,
    resolution_time_seconds INTEGER,
    metadata JSONB DEFAULT '{}'
);

-- 风险事件索引
CREATE INDEX IF NOT EXISTS idx_risk_events_id ON risk_events(event_id);
CREATE INDEX IF NOT EXISTS idx_risk_events_type ON risk_events(event_type);
CREATE INDEX IF NOT EXISTS idx_risk_events_level ON risk_events(event_level);
CREATE INDEX IF NOT EXISTS idx_risk_events_status ON risk_events(status);
CREATE INDEX IF NOT EXISTS idx_risk_events_opened ON risk_events(opened_at DESC);

-- 风险事件注释
COMMENT ON TABLE risk_events IS '风险事件表，记录重大风险事件';
COMMENT ON COLUMN risk_events.event_type IS '事件类型(breach/violation/incident/anomaly)';
COMMENT ON COLUMN risk_events.status IS '事件状态(open/investigating/resolved/closed)';

-- ============================================================
-- 7. 合规检查记录表 (compliance_checks)
-- 用于存储合规检查结果
-- ============================================================
CREATE TABLE IF NOT EXISTS compliance_checks (
    id SERIAL PRIMARY KEY,
    check_id VARCHAR(128) NOT NULL UNIQUE,
    rule_id VARCHAR(128) NOT NULL,
    rule_name VARCHAR(256),
    check_type VARCHAR(64) NOT NULL,
    entity_type VARCHAR(64),
    entity_id VARCHAR(128),
    compliant BOOLEAN NOT NULL,
    violations JSONB DEFAULT '[]',
    severity VARCHAR(32) DEFAULT 'low',
    remediation TEXT,
    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    checked_by VARCHAR(128),
    metadata JSONB DEFAULT '{}'
);

-- 合规检查索引
CREATE INDEX IF NOT EXISTS idx_compliance_checks_id ON compliance_checks(check_id);
CREATE INDEX IF NOT EXISTS idx_compliance_checks_rule ON compliance_checks(rule_id);
CREATE INDEX IF NOT EXISTS idx_compliance_checks_type ON compliance_checks(check_type);
CREATE INDEX IF NOT EXISTS idx_compliance_checks_entity ON compliance_checks(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_compliance_checks_compliant ON compliance_checks(compliant);
CREATE INDEX IF NOT EXISTS idx_compliance_checks_time ON compliance_checks(checked_at DESC);

-- 合规检查注释
COMMENT ON TABLE compliance_checks IS '合规检查记录表，存储合规检查结果';
COMMENT ON COLUMN compliance_checks.check_type IS '检查类型(position_limit/trading_restriction/disclosure/compliance_rule)';
COMMENT ON COLUMN compliance_checks.violations IS '违规项列表';

-- ============================================================
-- 8. 压力测试结果表 (stress_test_results)
-- 用于存储压力测试结果
-- ============================================================
CREATE TABLE IF NOT EXISTS stress_test_results (
    id SERIAL PRIMARY KEY,
    test_id VARCHAR(128) NOT NULL UNIQUE,
    test_name VARCHAR(256) NOT NULL,
    scenario_id VARCHAR(128),
    scenario_name VARCHAR(256),
    portfolio_id VARCHAR(128),
    portfolio_value DECIMAL(18, 4),
    stressed_value DECIMAL(18, 4),
    portfolio_loss DECIMAL(18, 4),
    loss_percentage DECIMAL(10, 6),
    var_impact DECIMAL(10, 6),
    liquidity_impact DECIMAL(10, 6),
    breach_count INTEGER DEFAULT 0,
    breached_limits JSONB DEFAULT '[]',
    passed BOOLEAN DEFAULT TRUE,
    test_duration_ms INTEGER DEFAULT 0,
    tested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- 压力测试结果索引
CREATE INDEX IF NOT EXISTS idx_stress_test_id ON stress_test_results(test_id);
CREATE INDEX IF NOT EXISTS idx_stress_test_scenario ON stress_test_results(scenario_id);
CREATE INDEX IF NOT EXISTS idx_stress_test_portfolio ON stress_test_results(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_stress_test_time ON stress_test_results(tested_at DESC);
CREATE INDEX IF NOT EXISTS idx_stress_test_passed ON stress_test_results(passed);

-- 压力测试结果注释
COMMENT ON TABLE stress_test_results IS '压力测试结果表，存储压力测试结果';
COMMENT ON COLUMN stress_test_results.scenario_id IS '压力测试场景ID';
COMMENT ON COLUMN stress_test_results.breached_limits IS '突破的限额列表';

-- ============================================================
-- 9. 触发器: 自动更新 updated_at 字段
-- ============================================================
CREATE OR REPLACE FUNCTION update_risk_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- risk_alerts 更新触发器
DROP TRIGGER IF EXISTS update_risk_alerts_updated_at ON risk_alerts;
CREATE TRIGGER update_risk_alerts_updated_at
    BEFORE UPDATE ON risk_alerts
    FOR EACH ROW
    EXECUTE FUNCTION update_risk_updated_at_column();

-- risk_limits 更新触发器
DROP TRIGGER IF EXISTS update_risk_limits_updated_at ON risk_limits;
CREATE TRIGGER update_risk_limits_updated_at
    BEFORE UPDATE ON risk_limits
    FOR EACH ROW
    EXECUTE FUNCTION update_risk_updated_at_column();

-- risk_rules 更新触发器
DROP TRIGGER IF EXISTS update_risk_rules_updated_at ON risk_rules;
CREATE TRIGGER update_risk_rules_updated_at
    BEFORE UPDATE ON risk_rules
    FOR EACH ROW
    EXECUTE FUNCTION update_risk_updated_at_column();

-- ============================================================
-- 10. 触发器: 告警状态变更事件记录
-- ============================================================
CREATE TABLE IF NOT EXISTS risk_alert_events (
    id SERIAL PRIMARY KEY,
    event_id VARCHAR(128) NOT NULL UNIQUE,
    alert_id VARCHAR(128) NOT NULL,
    event_type VARCHAR(32) NOT NULL,
    old_status VARCHAR(32),
    new_status VARCHAR(32),
    changed_by VARCHAR(128),
    change_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_risk_alert_events_alert ON risk_alert_events(alert_id, created_at DESC);

CREATE OR REPLACE FUNCTION record_alert_status_change()
RETURNS TRIGGER AS $$
BEGIN
    IF OLD.status IS DISTINCT FROM NEW.status THEN
        INSERT INTO risk_alert_events (
            event_id, alert_id, event_type, old_status, new_status, created_at
        ) VALUES (
            'evt_' || NEW.alert_id || '_' || EXTRACT(EPOCH FROM CURRENT_TIMESTAMP)::TEXT,
            NEW.alert_id,
            LOWER(NEW.status),
            OLD.status,
            NEW.status,
            CURRENT_TIMESTAMP
        );
    END IF;
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS trigger_alert_status_change ON risk_alerts;
CREATE TRIGGER trigger_alert_status_change
    AFTER UPDATE ON risk_alerts
    FOR EACH ROW
    EXECUTE FUNCTION record_alert_status_change();

-- ============================================================
-- 11. 视图: 活跃告警汇总视图
-- ============================================================
CREATE OR REPLACE VIEW v_active_alerts_summary AS
SELECT 
    alert_level,
    alert_type,
    COUNT(*) as alert_count,
    MIN(created_at) as earliest_alert,
    MAX(created_at) as latest_alert,
    array_agg(DISTINCT symbol) FILTER (WHERE symbol IS NOT NULL) as affected_symbols,
    array_agg(DISTINCT account_id) FILTER (WHERE account_id IS NOT NULL) as affected_accounts
FROM risk_alerts
WHERE status = 'active'
GROUP BY alert_level, alert_type
ORDER BY 
    CASE alert_level 
        WHEN 'critical' THEN 1 
        WHEN 'error' THEN 2 
        WHEN 'warning' THEN 3 
        ELSE 4 
    END,
    alert_count DESC;

COMMENT ON VIEW v_active_alerts_summary IS '活跃告警汇总视图，按级别和类型统计活跃告警';

-- ============================================================
-- 12. 视图: 风险指标趋势视图
-- ============================================================
CREATE OR REPLACE VIEW v_risk_metrics_trend AS
SELECT 
    metric_name,
    metric_type,
    created_at::date as metric_date,
    AVG(value) as avg_value,
    MIN(value) as min_value,
    MAX(value) as max_value,
    STDDEV(value) as stddev_value,
    COUNT(*) as sample_count
FROM risk_metrics
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY metric_name, metric_type, created_at::date
ORDER BY metric_name, metric_date DESC;

COMMENT ON VIEW v_risk_metrics_trend IS '风险指标趋势视图，按日期统计指标变化趋势';

-- ============================================================
-- 13. 视图: 风险检查统计视图
-- ============================================================
CREATE OR REPLACE VIEW v_risk_check_statistics AS
SELECT 
    check_type,
    DATE(created_at) as check_date,
    COUNT(*) as total_checks,
    SUM(CASE WHEN passed THEN 1 ELSE 0 END) as passed_checks,
    SUM(CASE WHEN NOT passed THEN 1 ELSE 0 END) as failed_checks,
    AVG(score) as avg_score,
    AVG(check_duration_ms) as avg_duration_ms
FROM risk_checks
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY check_type, DATE(created_at)
ORDER BY check_date DESC, check_type;

COMMENT ON VIEW v_risk_check_statistics IS '风险检查统计视图，按类型和日期统计检查结果';

-- ============================================================
-- 14. 存储过程: 清理过期告警
-- ============================================================
CREATE OR REPLACE PROCEDURE cleanup_expired_alerts(
    p_max_age_hours INTEGER DEFAULT 72
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_expired_count INTEGER;
BEGIN
    UPDATE risk_alerts 
    SET status = 'expired',
        updated_at = CURRENT_TIMESTAMP
    WHERE status = 'active'
      AND created_at < CURRENT_TIMESTAMP - INTERVAL '1 hour' * p_max_age_hours;
    
    GET DIAGNOSTICS v_expired_count = ROW_COUNT;
    
    RAISE NOTICE 'Expired % alerts older than % hours', v_expired_count, p_max_age_hours;
    
    COMMIT;
END;
$$;

COMMENT ON PROCEDURE cleanup_expired_alerts IS '清理过期告警，将长时间未处理的活跃告警标记为expired';

-- ============================================================
-- 15. 存储过程: 计算风险指标统计
-- ============================================================
CREATE OR REPLACE PROCEDURE calculate_risk_metric_statistics(
    p_metric_name VARCHAR(128),
    p_days INTEGER DEFAULT 30
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_avg_value DECIMAL(18,8);
    v_min_value DECIMAL(18,8);
    v_max_value DECIMAL(18,8);
    v_stddev DECIMAL(18,8);
BEGIN
    SELECT AVG(value), MIN(value), MAX(value), STDDEV(value)
    INTO v_avg_value, v_min_value, v_max_value, v_stddev
    FROM risk_metrics
    WHERE metric_name = p_metric_name
      AND created_at >= CURRENT_DATE - p_days;
    
    RAISE NOTICE 'Risk Metric Statistics for % (last % days):', p_metric_name, p_days;
    RAISE NOTICE '  Average: %', v_avg_value;
    RAISE NOTICE '  Min: %', v_min_value;
    RAISE NOTICE '  Max: %', v_max_value;
    RAISE NOTICE '  StdDev: %', v_stddev;
END;
$$;

COMMENT ON PROCEDURE calculate_risk_metric_statistics IS '计算风险指标统计信息';

-- ============================================================
-- 16. 存储过程: 归档历史风险数据
-- ============================================================
CREATE OR REPLACE PROCEDURE archive_historical_risk_data(
    p_days_old INTEGER DEFAULT 90
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_archived_checks INTEGER;
    v_archived_metrics INTEGER;
BEGIN
    -- 归档风险检查记录
    DELETE FROM risk_checks 
    WHERE created_at < CURRENT_DATE - p_days_old
      AND passed = TRUE;
    
    GET DIAGNOSTICS v_archived_checks = ROW_COUNT;
    
    -- 归档风险指标
    DELETE FROM risk_metrics 
    WHERE created_at < CURRENT_DATE - p_days_old;
    
    GET DIAGNOSTICS v_archived_metrics = ROW_COUNT;
    
    RAISE NOTICE 'Archived % risk checks and % risk metrics older than % days', 
                 v_archived_checks, v_archived_metrics, p_days_old;
    
    COMMIT;
END;
$$;

COMMENT ON PROCEDURE archive_historical_risk_data IS '归档历史风险数据，清理过期的检查记录和指标数据';

-- ============================================================
-- 17. 插入默认风险规则
-- ============================================================
INSERT INTO risk_rules (rule_id, rule_name, rule_type, risk_type, conditions, actions, alert_level, enabled, cooldown_minutes, priority, description)
VALUES 
    ('rule_var_limit', 'VaR限额规则', 'threshold', 'market', 
     '{"metric": "var_95", "threshold": 0.08, "operator": ">"}', 
     '["alert", "reduce_position"]', 'error', TRUE, 30, 10, 
     '当VaR超过8%时触发告警'),
    ('rule_drawdown_limit', '最大回撤规则', 'threshold', 'market', 
     '{"metric": "max_drawdown", "threshold": 0.15, "operator": ">"}', 
     '["alert", "stop_trading"]', 'critical', TRUE, 15, 5, 
     '当最大回撤超过15%时触发严重告警'),
    ('rule_position_concentration', '持仓集中度规则', 'threshold', 'market', 
     '{"metric": "concentration", "threshold": 0.3, "operator": ">"}', 
     '["alert"]', 'warning', TRUE, 60, 50, 
     '当单一持仓占比超过30%时触发警告'),
    ('rule_liquidity_ratio', '流动性比率规则', 'threshold', 'liquidity', 
     '{"metric": "liquidity_ratio", "threshold": 0.1, "operator": "<"}', 
     '["alert", "limit_trading"]', 'warning', TRUE, 30, 30, 
     '当流动性比率低于10%时触发警告'),
    ('rule_daily_loss_limit', '日损失限额规则', 'threshold', 'market', 
     '{"metric": "daily_pnl_pct", "threshold": -0.05, "operator": "<"}', 
     '["alert", "stop_trading"]', 'critical', TRUE, 5, 1, 
     '当日损失超过5%时触发严重告警并停止交易')
ON CONFLICT (rule_id) DO NOTHING;

-- ============================================================
-- 18. 权限设置 (根据实际环境调整)
-- ============================================================
-- GRANT SELECT, INSERT, UPDATE, DELETE ON risk_checks TO rqa2025_admin;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON risk_alerts TO rqa2025_admin;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON risk_metrics TO rqa2025_admin;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON risk_rules TO rqa2025_admin;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON risk_limits TO rqa2025_admin;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON risk_events TO rqa2025_admin;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON compliance_checks TO rqa2025_admin;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON stress_test_results TO rqa2025_admin;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON risk_alert_events TO rqa2025_admin;
-- GRANT EXECUTE ON PROCEDURE cleanup_expired_alerts TO rqa2025_admin;
-- GRANT EXECUTE ON PROCEDURE calculate_risk_metric_statistics TO rqa2025_admin;
-- GRANT EXECUTE ON PROCEDURE archive_historical_risk_data TO rqa2025_admin;
