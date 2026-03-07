-- RQA2026 数据库模式定义
-- 生产环境数据库初始化脚本

-- 创建数据库
CREATE DATABASE IF NOT EXISTS rqa2026_prod;
USE rqa2026_prod;

-- 用户表
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- 投资组合表
CREATE TABLE portfolios (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id),
    name VARCHAR(100) NOT NULL,
    description TEXT,
    total_value DECIMAL(15,2) DEFAULT 0,
    risk_level VARCHAR(20) DEFAULT 'medium',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- 资产表
CREATE TABLE assets (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    asset_type VARCHAR(20) NOT NULL, -- stock, bond, crypto, etc.
    current_price DECIMAL(10,4),
    market_cap DECIMAL(20,2),
    sector VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- 持仓表
CREATE TABLE positions (
    id SERIAL PRIMARY KEY,
    portfolio_id INT REFERENCES portfolios(id),
    asset_id INT REFERENCES assets(id),
    quantity DECIMAL(15,6) NOT NULL,
    average_cost DECIMAL(10,4),
    current_value DECIMAL(15,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- 交易记录表
CREATE TABLE transactions (
    id SERIAL PRIMARY KEY,
    portfolio_id INT REFERENCES portfolios(id),
    asset_id INT REFERENCES assets(id),
    transaction_type VARCHAR(10) NOT NULL, -- buy, sell
    quantity DECIMAL(15,6) NOT NULL,
    price DECIMAL(10,4) NOT NULL,
    total_amount DECIMAL(15,2) NOT NULL,
    fee DECIMAL(8,2) DEFAULT 0,
    transaction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'completed'
);

-- 风险评估结果表
CREATE TABLE risk_assessments (
    id SERIAL PRIMARY KEY,
    portfolio_id INT REFERENCES portfolios(id),
    assessment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    risk_score DECIMAL(5,4), -- 0.0000 to 1.0000
    volatility DECIMAL(5,4),
    sharpe_ratio DECIMAL(8,4),
    max_drawdown DECIMAL(5,4),
    var_95 DECIMAL(5,4), -- Value at Risk 95%
    expected_return DECIMAL(5,4),
    recommendation TEXT,
    assessed_by VARCHAR(50) DEFAULT 'quantum_engine'
);

-- AI分析结果表
CREATE TABLE ai_analyses (
    id SERIAL PRIMARY KEY,
    portfolio_id INT REFERENCES portfolios(id),
    analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    sentiment_score DECIMAL(3,2), -- -1.00 to 1.00
    market_trend VARCHAR(20), -- bullish, bearish, neutral
    confidence_level DECIMAL(3,2), -- 0.00 to 1.00
    key_insights TEXT,
    recommendations TEXT,
    analyzed_by VARCHAR(50) DEFAULT 'ai_engine'
);

-- BCI会话表
CREATE TABLE bci_sessions (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id),
    session_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_end TIMESTAMP NULL,
    duration_seconds INT,
    consciousness_level DECIMAL(3,2),
    attention_score DECIMAL(3,2),
    stress_level DECIMAL(3,2),
    decisions_made INT DEFAULT 0,
    accuracy_rate DECIMAL(5,4),
    feedback_quality DECIMAL(3,2)
);

-- 融合引擎决策记录表
CREATE TABLE fusion_decisions (
    id SERIAL PRIMARY KEY,
    portfolio_id INT REFERENCES portfolios(id),
    decision_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    decision_type VARCHAR(50), -- risk_adjustment, portfolio_rebalance, etc.
    confidence DECIMAL(3,2),
    fusion_quality DECIMAL(3,2),
    engines_used TEXT, -- JSON array of engine names
    reasoning TEXT,
    outcome VARCHAR(20) DEFAULT 'pending', -- implemented, rejected, pending
    implemented_at TIMESTAMP NULL
);

-- 系统日志表
CREATE TABLE system_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    level VARCHAR(10), -- INFO, WARNING, ERROR
    component VARCHAR(50), -- quantum_engine, ai_engine, etc.
    message TEXT,
    details TEXT,
    user_id INT REFERENCES users(id) NULL
);

-- 性能指标表
CREATE TABLE performance_metrics (
    id SERIAL PRIMARY KEY,
    metric_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    component VARCHAR(50),
    metric_name VARCHAR(100),
    metric_value DECIMAL(15,6),
    unit VARCHAR(20),
    tags TEXT -- JSON object for additional metadata
);

-- 创建索引以提高查询性能
CREATE INDEX idx_portfolios_user_id ON portfolios(user_id);
CREATE INDEX idx_positions_portfolio_id ON positions(portfolio_id);
CREATE INDEX idx_transactions_portfolio_id ON transactions(portfolio_id);
CREATE INDEX idx_risk_assessments_portfolio_id ON risk_assessments(portfolio_id);
CREATE INDEX idx_ai_analyses_portfolio_id ON ai_analyses(portfolio_id);
CREATE INDEX idx_fusion_decisions_portfolio_id ON fusion_decisions(portfolio_id);
CREATE INDEX idx_system_logs_timestamp ON system_logs(timestamp);
CREATE INDEX idx_system_logs_component ON system_logs(component);
CREATE INDEX idx_performance_metrics_timestamp ON performance_metrics(metric_timestamp);
CREATE INDEX idx_performance_metrics_component ON performance_metrics(component);

-- 插入初始数据
INSERT INTO users (username, email, password_hash, role) VALUES
('admin', 'admin@rqa2026.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/Le1EcMskgbKpQKQG6', 'admin'),
('demo_user', 'demo@rqa2026.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/Le1EcMskgbKpQKQG6', 'user');

INSERT INTO assets (symbol, name, asset_type, current_price, sector) VALUES
('AAPL', 'Apple Inc.', 'stock', 150.25, 'Technology'),
('GOOGL', 'Alphabet Inc.', 'stock', 2800.50, 'Technology'),
('MSFT', 'Microsoft Corporation', 'stock', 305.75, 'Technology'),
('BTC', 'Bitcoin', 'crypto', 35000.00, 'Cryptocurrency'),
('ETH', 'Ethereum', 'crypto', 2200.00, 'Cryptocurrency'),
('SPY', 'SPDR S&P 500 ETF', 'etf', 420.80, 'Index'),
('BND', 'Vanguard Total Bond Market ETF', 'etf', 75.25, 'Bonds');

-- 创建视图用于常用查询
CREATE VIEW portfolio_summary AS
SELECT
    p.id,
    p.name,
    p.total_value,
    p.risk_level,
    u.username as owner,
    COUNT(pos.id) as total_positions,
    SUM(pos.current_value) as calculated_value
FROM portfolios p
LEFT JOIN users u ON p.user_id = u.id
LEFT JOIN positions pos ON p.id = pos.portfolio_id
GROUP BY p.id, p.name, p.total_value, p.risk_level, u.username;

CREATE VIEW recent_decisions AS
SELECT
    fd.id,
    fd.decision_timestamp,
    fd.decision_type,
    fd.confidence,
    fd.fusion_quality,
    p.name as portfolio_name,
    u.username as owner
FROM fusion_decisions fd
JOIN portfolios p ON fd.portfolio_id = p.id
JOIN users u ON p.user_id = u.id
WHERE fd.decision_timestamp >= NOW() - INTERVAL '30 days'
ORDER BY fd.decision_timestamp DESC;

-- 权限设置
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO rqa2026_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO rqa2026_user;

-- 备份策略注释
-- 建议每日备份关键表：users, portfolios, positions, transactions
-- 建议每小时备份：system_logs, performance_metrics
-- 建议实时备份：fusion_decisions (重要决策记录)
