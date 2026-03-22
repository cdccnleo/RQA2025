-- ============================================================
-- 交易层数据库表结构定义
-- 数据库: PostgreSQL
-- 版本: 1.0.0
-- 创建时间: 2026-03-22
-- 描述: 订单、账户、持仓、交易记录的表结构定义
-- ============================================================

-- ============================================================
-- 1. 交易账户表 (trading_accounts)
-- 用于存储交易账户信息和资金状态
-- ============================================================
CREATE TABLE IF NOT EXISTS trading_accounts (
    id SERIAL PRIMARY KEY,
    account_id VARCHAR(128) NOT NULL UNIQUE,
    account_name VARCHAR(255),
    account_type VARCHAR(32) DEFAULT 'cash',
    balance DECIMAL(18, 4) DEFAULT 0.0,
    frozen_balance DECIMAL(18, 4) DEFAULT 0.0,
    available_balance DECIMAL(18, 4) DEFAULT 0.0,
    margin_balance DECIMAL(18, 4) DEFAULT 0.0,
    maintenance_margin DECIMAL(18, 4) DEFAULT 0.0,
    initial_margin DECIMAL(18, 4) DEFAULT 0.0,
    currency VARCHAR(16) DEFAULT 'CNY',
    status VARCHAR(32) DEFAULT 'active',
    risk_level VARCHAR(16) DEFAULT 'normal',
    leverage_ratio DECIMAL(8, 4) DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_settlement_at TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- 交易账户索引
CREATE INDEX IF NOT EXISTS idx_trading_accounts_id ON trading_accounts(account_id);
CREATE INDEX IF NOT EXISTS idx_trading_accounts_status ON trading_accounts(status);
CREATE INDEX IF NOT EXISTS idx_trading_accounts_type ON trading_accounts(account_type);

-- 交易账户注释
COMMENT ON TABLE trading_accounts IS '交易账户表，存储账户信息和资金状态';
COMMENT ON COLUMN trading_accounts.account_id IS '账户唯一标识符';
COMMENT ON COLUMN trading_accounts.account_type IS '账户类型(cash/margin/credit)';
COMMENT ON COLUMN trading_accounts.balance IS '账户总余额';
COMMENT ON COLUMN trading_accounts.frozen_balance IS '冻结资金';
COMMENT ON COLUMN trading_accounts.available_balance IS '可用资金';
COMMENT ON COLUMN trading_accounts.status IS '账户状态(active/frozen/closed)';

-- ============================================================
-- 2. 交易订单表 (trading_orders)
-- 用于存储订单全生命周期数据
-- ============================================================
CREATE TABLE IF NOT EXISTS trading_orders (
    id SERIAL PRIMARY KEY,
    order_id VARCHAR(128) NOT NULL UNIQUE,
    account_id VARCHAR(128) NOT NULL,
    strategy_id VARCHAR(128),
    symbol VARCHAR(32) NOT NULL,
    exchange VARCHAR(16),
    side VARCHAR(16) NOT NULL,
    order_type VARCHAR(32) NOT NULL,
    quantity DECIMAL(18, 4) NOT NULL,
    price DECIMAL(18, 4),
    stop_price DECIMAL(18, 4),
    trailing_offset DECIMAL(18, 4),
    status VARCHAR(32) DEFAULT 'pending',
    filled_quantity DECIMAL(18, 4) DEFAULT 0.0,
    remaining_quantity DECIMAL(18, 4),
    avg_fill_price DECIMAL(18, 4) DEFAULT 0.0,
    broker_order_id VARCHAR(128),
    broker VARCHAR(64),
    time_in_force VARCHAR(16) DEFAULT 'GTC',
    parent_order_id VARCHAR(128),
    error_message TEXT,
    commission DECIMAL(18, 4) DEFAULT 0.0,
    slippage DECIMAL(18, 4) DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    submitted_at TIMESTAMP,
    filled_at TIMESTAMP,
    cancelled_at TIMESTAMP,
    expired_at TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- 交易订单索引
CREATE INDEX IF NOT EXISTS idx_trading_orders_id ON trading_orders(order_id);
CREATE INDEX IF NOT EXISTS idx_trading_orders_account ON trading_orders(account_id);
CREATE INDEX IF NOT EXISTS idx_trading_orders_symbol ON trading_orders(symbol);
CREATE INDEX IF NOT EXISTS idx_trading_orders_status ON trading_orders(status);
CREATE INDEX IF NOT EXISTS idx_trading_orders_strategy ON trading_orders(strategy_id);
CREATE INDEX IF NOT EXISTS idx_trading_orders_created ON trading_orders(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_trading_orders_broker ON trading_orders(broker_order_id);

-- 交易订单注释
COMMENT ON TABLE trading_orders IS '交易订单表，存储订单全生命周期数据';
COMMENT ON COLUMN trading_orders.order_id IS '订单唯一标识符';
COMMENT ON COLUMN trading_orders.side IS '订单方向(buy/sell)';
COMMENT ON COLUMN trading_orders.order_type IS '订单类型(market/limit/stop/stop_limit/trailing_stop)';
COMMENT ON COLUMN trading_orders.status IS '订单状态(pending/submitted/partial/filled/cancelled/rejected/expired)';
COMMENT ON COLUMN trading_orders.time_in_force IS '有效期类型(GTC/IOC/FOK/Day)';

-- ============================================================
-- 3. 交易记录表 (trading_trades)
-- 用于存储成交记录
-- ============================================================
CREATE TABLE IF NOT EXISTS trading_trades (
    id SERIAL PRIMARY KEY,
    trade_id VARCHAR(128) NOT NULL UNIQUE,
    order_id VARCHAR(128) NOT NULL,
    account_id VARCHAR(128) NOT NULL,
    strategy_id VARCHAR(128),
    symbol VARCHAR(32) NOT NULL,
    exchange VARCHAR(16),
    side VARCHAR(16) NOT NULL,
    quantity DECIMAL(18, 4) NOT NULL,
    price DECIMAL(18, 4) NOT NULL,
    amount DECIMAL(18, 4) NOT NULL,
    commission DECIMAL(18, 4) DEFAULT 0.0,
    stamp_duty DECIMAL(18, 4) DEFAULT 0.0,
    transfer_fee DECIMAL(18, 4) DEFAULT 0.0,
    total_fees DECIMAL(18, 4) DEFAULT 0.0,
    net_amount DECIMAL(18, 4),
    pnl DECIMAL(18, 4),
    pnl_pct DECIMAL(10, 6),
    broker_trade_id VARCHAR(128),
    broker VARCHAR(64),
    traded_at TIMESTAMP NOT NULL,
    settlement_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- 交易记录索引
CREATE INDEX IF NOT EXISTS idx_trading_trades_id ON trading_trades(trade_id);
CREATE INDEX IF NOT EXISTS idx_trading_trades_order ON trading_trades(order_id);
CREATE INDEX IF NOT EXISTS idx_trading_trades_account ON trading_trades(account_id);
CREATE INDEX IF NOT EXISTS idx_trading_trades_symbol ON trading_trades(symbol, traded_at DESC);
CREATE INDEX IF NOT EXISTS idx_trading_trades_strategy ON trading_trades(strategy_id);
CREATE INDEX IF NOT EXISTS idx_trading_trades_time ON trading_trades(traded_at DESC);

-- 交易记录注释
COMMENT ON TABLE trading_trades IS '交易记录表，存储成交记录';
COMMENT ON COLUMN trading_trades.trade_id IS '成交唯一标识符';
COMMENT ON COLUMN trading_trades.side IS '交易方向(buy/sell)';
COMMENT ON COLUMN trading_trades.amount IS '成交金额';
COMMENT ON COLUMN trading_trades.net_amount IS '净金额(扣除费用)';
COMMENT ON COLUMN trading_trades.pnl IS '已实现盈亏';

-- ============================================================
-- 4. 持仓表 (trading_positions)
-- 用于存储当前持仓信息
-- ============================================================
CREATE TABLE IF NOT EXISTS trading_positions (
    id SERIAL PRIMARY KEY,
    position_id VARCHAR(128) NOT NULL UNIQUE,
    account_id VARCHAR(128) NOT NULL,
    strategy_id VARCHAR(128),
    symbol VARCHAR(32) NOT NULL,
    exchange VARCHAR(16),
    position_type VARCHAR(16) DEFAULT 'long',
    quantity DECIMAL(18, 4) DEFAULT 0.0,
    available_quantity DECIMAL(18, 4) DEFAULT 0.0,
    frozen_quantity DECIMAL(18, 4) DEFAULT 0.0,
    avg_cost DECIMAL(18, 4) DEFAULT 0.0,
    current_price DECIMAL(18, 4) DEFAULT 0.0,
    market_value DECIMAL(18, 4) DEFAULT 0.0,
    unrealized_pnl DECIMAL(18, 4) DEFAULT 0.0,
    realized_pnl DECIMAL(18, 4) DEFAULT 0.0,
    total_pnl DECIMAL(18, 4) DEFAULT 0.0,
    pnl_pct DECIMAL(10, 6) DEFAULT 0.0,
    cost_basis DECIMAL(18, 4) DEFAULT 0.0,
    margin_used DECIMAL(18, 4) DEFAULT 0.0,
    leverage DECIMAL(8, 4) DEFAULT 1.0,
    status VARCHAR(32) DEFAULT 'open',
    opened_at TIMESTAMP,
    closed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- 持仓索引
CREATE INDEX IF NOT EXISTS idx_trading_positions_id ON trading_positions(position_id);
CREATE INDEX IF NOT EXISTS idx_trading_positions_account ON trading_positions(account_id);
CREATE INDEX IF NOT EXISTS idx_trading_positions_symbol ON trading_positions(account_id, symbol);
CREATE INDEX IF NOT EXISTS idx_trading_positions_strategy ON trading_positions(strategy_id);
CREATE INDEX IF NOT EXISTS idx_trading_positions_status ON trading_positions(status);

-- 持仓注释
COMMENT ON TABLE trading_positions IS '持仓表，存储当前持仓信息';
COMMENT ON COLUMN trading_positions.position_id IS '持仓唯一标识符';
COMMENT ON COLUMN trading_positions.position_type IS '持仓类型(long/short)';
COMMENT ON COLUMN trading_positions.quantity IS '持仓数量';
COMMENT ON COLUMN trading_positions.avg_cost IS '平均成本';
COMMENT ON COLUMN trading_positions.unrealized_pnl IS '未实现盈亏';

-- ============================================================
-- 5. 资金流水表 (trading_cash_flows)
-- 用于记录资金变动明细
-- ============================================================
CREATE TABLE IF NOT EXISTS trading_cash_flows (
    id SERIAL PRIMARY KEY,
    flow_id VARCHAR(128) NOT NULL UNIQUE,
    account_id VARCHAR(128) NOT NULL,
    flow_type VARCHAR(32) NOT NULL,
    amount DECIMAL(18, 4) NOT NULL,
    balance_before DECIMAL(18, 4),
    balance_after DECIMAL(18, 4),
    currency VARCHAR(16) DEFAULT 'CNY',
    reference_type VARCHAR(32),
    reference_id VARCHAR(128),
    description TEXT,
    status VARCHAR(32) DEFAULT 'completed',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- 资金流水索引
CREATE INDEX IF NOT EXISTS idx_trading_cash_flows_id ON trading_cash_flows(flow_id);
CREATE INDEX IF NOT EXISTS idx_trading_cash_flows_account ON trading_cash_flows(account_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_trading_cash_flows_type ON trading_cash_flows(flow_type);
CREATE INDEX IF NOT EXISTS idx_trading_cash_flows_reference ON trading_cash_flows(reference_type, reference_id);

-- 资金流水注释
COMMENT ON TABLE trading_cash_flows IS '资金流水表，记录资金变动明细';
COMMENT ON COLUMN trading_cash_flows.flow_type IS '流水类型(deposit/withdraw/trade/commission/dividend/transfer)';
COMMENT ON COLUMN trading_cash_flows.reference_type IS '关联类型(order/trade/transfer)';
COMMENT ON COLUMN trading_cash_flows.reference_id IS '关联ID';

-- ============================================================
-- 6. 订单事件表 (trading_order_events)
-- 用于记录订单状态变更历史
-- ============================================================
CREATE TABLE IF NOT EXISTS trading_order_events (
    id SERIAL PRIMARY KEY,
    event_id VARCHAR(128) NOT NULL UNIQUE,
    order_id VARCHAR(128) NOT NULL,
    event_type VARCHAR(32) NOT NULL,
    old_status VARCHAR(32),
    new_status VARCHAR(32),
    event_data JSONB DEFAULT '{}',
    message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 订单事件索引
CREATE INDEX IF NOT EXISTS idx_trading_order_events_order ON trading_order_events(order_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_trading_order_events_type ON trading_order_events(event_type);

-- 订单事件注释
COMMENT ON TABLE trading_order_events IS '订单事件表，记录订单状态变更历史';
COMMENT ON COLUMN trading_order_events.event_type IS '事件类型(created/submitted/filled/cancelled/rejected/expired)';

-- ============================================================
-- 7. 结算记录表 (trading_settlements)
-- 用于存储结算信息
-- ============================================================
CREATE TABLE IF NOT EXISTS trading_settlements (
    id SERIAL PRIMARY KEY,
    settlement_id VARCHAR(128) NOT NULL UNIQUE,
    account_id VARCHAR(128) NOT NULL,
    settlement_date DATE NOT NULL,
    settlement_type VARCHAR(32) DEFAULT 'daily',
    opening_balance DECIMAL(18, 4),
    closing_balance DECIMAL(18, 4),
    total_trades INTEGER DEFAULT 0,
    total_volume DECIMAL(18, 4) DEFAULT 0.0,
    total_turnover DECIMAL(18, 4) DEFAULT 0.0,
    total_commission DECIMAL(18, 4) DEFAULT 0.0,
    total_stamp_duty DECIMAL(18, 4) DEFAULT 0.0,
    total_fees DECIMAL(18, 4) DEFAULT 0.0,
    realized_pnl DECIMAL(18, 4) DEFAULT 0.0,
    status VARCHAR(32) DEFAULT 'pending',
    settled_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- 结算记录索引
CREATE INDEX IF NOT EXISTS idx_trading_settlements_id ON trading_settlements(settlement_id);
CREATE INDEX IF NOT EXISTS idx_trading_settlements_account ON trading_settlements(account_id, settlement_date DESC);
CREATE INDEX IF NOT EXISTS idx_trading_settlements_date ON trading_settlements(settlement_date);
CREATE INDEX IF NOT EXISTS idx_trading_settlements_status ON trading_settlements(status);

-- 结算记录注释
COMMENT ON TABLE trading_settlements IS '结算记录表，存储结算信息';
COMMENT ON COLUMN trading_settlements.settlement_type IS '结算类型(daily/monthly/trade)';
COMMENT ON COLUMN trading_settlements.status IS '结算状态(pending/completed/failed)';

-- ============================================================
-- 8. 风险监控表 (trading_risk_metrics)
-- 用于存储风险指标
-- ============================================================
CREATE TABLE IF NOT EXISTS trading_risk_metrics (
    id SERIAL PRIMARY KEY,
    metric_id VARCHAR(128) NOT NULL UNIQUE,
    account_id VARCHAR(128) NOT NULL,
    metric_type VARCHAR(32) NOT NULL,
    metric_value DECIMAL(18, 6),
    threshold_value DECIMAL(18, 6),
    breach_flag BOOLEAN DEFAULT FALSE,
    calculation_time TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- 风险监控索引
CREATE INDEX IF NOT EXISTS idx_trading_risk_metrics_account ON trading_risk_metrics(account_id, calculation_time DESC);
CREATE INDEX IF NOT EXISTS idx_trading_risk_metrics_type ON trading_risk_metrics(metric_type);
CREATE INDEX IF NOT EXISTS idx_trading_risk_metrics_breach ON trading_risk_metrics(breach_flag);

-- 风险监控注释
COMMENT ON TABLE trading_risk_metrics IS '风险监控表，存储风险指标';
COMMENT ON COLUMN trading_risk_metrics.metric_type IS '指标类型(var/margin_ratio/position_concentration/drawdown)';

-- ============================================================
-- 9. 触发器: 自动更新 updated_at 字段
-- ============================================================
CREATE OR REPLACE FUNCTION update_trading_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- trading_accounts 更新触发器
DROP TRIGGER IF EXISTS update_trading_accounts_updated_at ON trading_accounts;
CREATE TRIGGER update_trading_accounts_updated_at
    BEFORE UPDATE ON trading_accounts
    FOR EACH ROW
    EXECUTE FUNCTION update_trading_updated_at_column();

-- trading_orders 更新触发器
DROP TRIGGER IF EXISTS update_trading_orders_updated_at ON trading_orders;
CREATE TRIGGER update_trading_orders_updated_at
    BEFORE UPDATE ON trading_orders
    FOR EACH ROW
    EXECUTE FUNCTION update_trading_updated_at_column();

-- trading_positions 更新触发器
DROP TRIGGER IF EXISTS update_trading_positions_updated_at ON trading_positions;
CREATE TRIGGER update_trading_positions_updated_at
    BEFORE UPDATE ON trading_positions
    FOR EACH ROW
    EXECUTE FUNCTION update_trading_updated_at_column();

-- ============================================================
-- 10. 触发器: 订单状态变更事件记录
-- ============================================================
CREATE OR REPLACE FUNCTION record_order_status_change()
RETURNS TRIGGER AS $$
BEGIN
    IF OLD.status IS DISTINCT FROM NEW.status THEN
        INSERT INTO trading_order_events (
            event_id, order_id, event_type, old_status, new_status, event_data, created_at
        ) VALUES (
            'evt_' || NEW.order_id || '_' || EXTRACT(EPOCH FROM CURRENT_TIMESTAMP)::TEXT,
            NEW.order_id,
            LOWER(NEW.status),
            OLD.status,
            NEW.status,
            jsonb_build_object(
                'filled_quantity', NEW.filled_quantity,
                'avg_fill_price', NEW.avg_fill_price,
                'updated_at', NEW.updated_at
            ),
            CURRENT_TIMESTAMP
        );
    END IF;
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS trigger_order_status_change ON trading_orders;
CREATE TRIGGER trigger_order_status_change
    AFTER UPDATE ON trading_orders
    FOR EACH ROW
    EXECUTE FUNCTION record_order_status_change();

-- ============================================================
-- 11. 视图: 账户持仓汇总视图
-- ============================================================
CREATE OR REPLACE VIEW v_account_positions_summary AS
SELECT 
    a.account_id,
    a.account_name,
    a.balance,
    a.frozen_balance,
    a.available_balance,
    COUNT(DISTINCT p.symbol) as position_count,
    SUM(p.market_value) as total_position_value,
    SUM(p.unrealized_pnl) as total_unrealized_pnl,
    SUM(p.realized_pnl) as total_realized_pnl,
    a.balance + COALESCE(SUM(p.market_value), 0) as total_value,
    a.updated_at
FROM trading_accounts a
LEFT JOIN trading_positions p ON a.account_id = p.account_id AND p.quantity > 0
WHERE a.status = 'active'
GROUP BY a.account_id, a.account_name, a.balance, a.frozen_balance, a.available_balance, a.updated_at
ORDER BY a.account_id;

COMMENT ON VIEW v_account_positions_summary IS '账户持仓汇总视图，按账户汇总持仓和资金信息';

-- ============================================================
-- 12. 视图: 订单执行统计视图
-- ============================================================
CREATE OR REPLACE VIEW v_order_execution_stats AS
SELECT 
    o.account_id,
    o.symbol,
    o.side,
    o.order_type,
    COUNT(*) as total_orders,
    COUNT(*) FILTER (WHERE o.status = 'filled') as filled_orders,
    COUNT(*) FILTER (WHERE o.status IN ('pending', 'submitted', 'partial')) as active_orders,
    COUNT(*) FILTER (WHERE o.status = 'cancelled') as cancelled_orders,
    COUNT(*) FILTER (WHERE o.status = 'rejected') as rejected_orders,
    SUM(o.quantity) as total_quantity,
    SUM(o.filled_quantity) as total_filled_quantity,
    AVG(o.filled_quantity / NULLIF(o.quantity, 0)) as avg_fill_rate,
    AVG(EXTRACT(EPOCH FROM (o.filled_at - o.created_at))) as avg_fill_time_seconds,
    MAX(o.created_at) as last_order_time
FROM trading_orders o
WHERE o.created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY o.account_id, o.symbol, o.side, o.order_type
ORDER BY total_orders DESC;

COMMENT ON VIEW v_order_execution_stats IS '订单执行统计视图，按账户、标的、方向统计订单执行情况';

-- ============================================================
-- 13. 视图: 交易汇总视图
-- ============================================================
CREATE OR REPLACE VIEW v_trading_summary AS
SELECT 
    t.account_id,
    t.symbol,
    DATE(t.traded_at) as trade_date,
    t.side,
    COUNT(*) as trade_count,
    SUM(t.quantity) as total_quantity,
    SUM(t.amount) as total_amount,
    SUM(t.commission + t.stamp_duty + t.transfer_fee) as total_fees,
    SUM(t.pnl) as total_pnl,
    AVG(t.price) as avg_price
FROM trading_trades t
WHERE t.traded_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY t.account_id, t.symbol, DATE(t.traded_at), t.side
ORDER BY trade_date DESC, total_amount DESC;

COMMENT ON VIEW v_trading_summary IS '交易汇总视图，按日期汇总交易数据';

-- ============================================================
-- 14. 存储过程: 清理过期订单
-- ============================================================
CREATE OR REPLACE PROCEDURE cleanup_expired_orders(
    p_days_old INTEGER DEFAULT 7
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_cleaned_count INTEGER;
BEGIN
    UPDATE trading_orders 
    SET status = 'expired', 
        expired_at = CURRENT_TIMESTAMP,
        updated_at = CURRENT_TIMESTAMP
    WHERE status IN ('pending', 'submitted')
      AND created_at < CURRENT_TIMESTAMP - INTERVAL '1 day' * p_days_old;
    
    GET DIAGNOSTICS v_cleaned_count = ROW_COUNT;
    
    RAISE NOTICE 'Expired % orders older than % days', v_cleaned_count, p_days_old;
    
    COMMIT;
END;
$$;

COMMENT ON PROCEDURE cleanup_expired_orders IS '清理过期订单，将长时间未成交订单标记为expired';

-- ============================================================
-- 15. 存储过程: 计算账户风险指标
-- ============================================================
CREATE OR REPLACE PROCEDURE calculate_account_risk_metrics(
    p_account_id VARCHAR(128)
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_balance DECIMAL(18,4);
    v_position_value DECIMAL(18,4);
    v_margin_used DECIMAL(18,4);
    v_margin_ratio DECIMAL(10,4);
    v_position_concentration DECIMAL(10,4);
BEGIN
    SELECT balance INTO v_balance 
    FROM trading_accounts WHERE account_id = p_account_id;
    
    SELECT COALESCE(SUM(market_value), 0), COALESCE(SUM(margin_used), 0)
    INTO v_position_value, v_margin_used
    FROM trading_positions WHERE account_id = p_account_id AND quantity > 0;
    
    IF v_margin_used > 0 THEN
        v_margin_ratio := (v_balance + v_position_value) / v_margin_used;
    ELSE
        v_margin_ratio := NULL;
    END IF;
    
    SELECT COALESCE(MAX(market_value / NULLIF(v_position_value, 0)), 0)
    INTO v_position_concentration
    FROM trading_positions WHERE account_id = p_account_id AND quantity > 0;
    
    INSERT INTO trading_risk_metrics (
        metric_id, account_id, metric_type, metric_value, calculation_time
    ) VALUES
        ('rm_' || p_account_id || '_margin_' || EXTRACT(EPOCH FROM CURRENT_TIMESTAMP)::TEXT,
         p_account_id, 'margin_ratio', v_margin_ratio, CURRENT_TIMESTAMP),
        ('rm_' || p_account_id || '_concentration_' || EXTRACT(EPOCH FROM CURRENT_TIMESTAMP)::TEXT,
         p_account_id, 'position_concentration', v_position_concentration, CURRENT_TIMESTAMP)
    ON CONFLICT (metric_id) DO UPDATE SET
        metric_value = EXCLUDED.metric_value,
        calculation_time = EXCLUDED.calculation_time;
    
    COMMIT;
END;
$$;

COMMENT ON PROCEDURE calculate_account_risk_metrics IS '计算账户风险指标，包括保证金比例和持仓集中度';

-- ============================================================
-- 16. 存储过程: 日终结算处理
-- ============================================================
CREATE OR REPLACE PROCEDURE process_daily_settlement(
    p_account_id VARCHAR(128),
    p_settlement_date DATE DEFAULT CURRENT_DATE
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_opening_balance DECIMAL(18,4);
    v_closing_balance DECIMAL(18,4);
    v_total_trades INTEGER;
    v_total_turnover DECIMAL(18,4);
    v_total_commission DECIMAL(18,4);
    v_realized_pnl DECIMAL(18,4);
    v_settlement_id VARCHAR(128);
BEGIN
    v_settlement_id := 'settle_' || p_account_id || '_' || TO_CHAR(p_settlement_date, 'YYYYMMDD');
    
    SELECT balance INTO v_opening_balance
    FROM trading_accounts WHERE account_id = p_account_id;
    
    SELECT COUNT(*), COALESCE(SUM(amount), 0), COALESCE(SUM(commission), 0), COALESCE(SUM(pnl), 0)
    INTO v_total_trades, v_total_turnover, v_total_commission, v_realized_pnl
    FROM trading_trades 
    WHERE account_id = p_account_id 
      AND DATE(traded_at) = p_settlement_date;
    
    SELECT balance INTO v_closing_balance
    FROM trading_accounts WHERE account_id = p_account_id;
    
    INSERT INTO trading_settlements (
        settlement_id, account_id, settlement_date, settlement_type,
        opening_balance, closing_balance, total_trades, total_turnover,
        total_commission, realized_pnl, status, settled_at
    ) VALUES (
        v_settlement_id, p_account_id, p_settlement_date, 'daily',
        v_opening_balance, v_closing_balance, v_total_trades, v_total_turnover,
        v_total_commission, v_realized_pnl, 'completed', CURRENT_TIMESTAMP
    )
    ON CONFLICT (settlement_id) DO UPDATE SET
        closing_balance = EXCLUDED.closing_balance,
        total_trades = EXCLUDED.total_trades,
        total_turnover = EXCLUDED.total_turnover,
        total_commission = EXCLUDED.total_commission,
        realized_pnl = EXCLUDED.realized_pnl,
        status = EXCLUDED.status,
        settled_at = EXCLUDED.settled_at;
    
    UPDATE trading_accounts 
    SET last_settlement_at = CURRENT_TIMESTAMP,
        updated_at = CURRENT_TIMESTAMP
    WHERE account_id = p_account_id;
    
    COMMIT;
END;
$$;

COMMENT ON PROCEDURE process_daily_settlement IS '日终结算处理，生成每日结算记录';

-- ============================================================
-- 17. 权限设置 (根据实际环境调整)
-- ============================================================
-- GRANT SELECT, INSERT, UPDATE, DELETE ON trading_accounts TO rqa2025_admin;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON trading_orders TO rqa2025_admin;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON trading_trades TO rqa2025_admin;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON trading_positions TO rqa2025_admin;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON trading_cash_flows TO rqa2025_admin;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON trading_order_events TO rqa2025_admin;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON trading_settlements TO rqa2025_admin;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON trading_risk_metrics TO rqa2025_admin;
-- GRANT EXECUTE ON PROCEDURE cleanup_expired_orders TO rqa2025_admin;
-- GRANT EXECUTE ON PROCEDURE calculate_account_risk_metrics TO rqa2025_admin;
-- GRANT EXECUTE ON PROCEDURE process_daily_settlement TO rqa2025_admin;
