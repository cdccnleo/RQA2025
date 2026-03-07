-- RQA2025 Database Initialization Script
-- This script initializes the database schema for the RQA2025 trading system

-- Create extensions if needed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create schemas for better organization
CREATE SCHEMA IF NOT EXISTS trading;
CREATE SCHEMA IF NOT EXISTS analysis;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Set search path
SET search_path TO public, trading, analysis, monitoring;

-- Create basic tables for trading system
CREATE TABLE IF NOT EXISTS trading.strategies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    strategy_type VARCHAR(100),
    parameters JSONB,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trading.positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(50) NOT NULL,
    quantity DECIMAL(20,8),
    avg_price DECIMAL(20,8),
    current_price DECIMAL(20,8),
    pnl DECIMAL(20,8),
    strategy_id INTEGER REFERENCES trading.strategies(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trading.orders (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(50) NOT NULL,
    order_type VARCHAR(20) NOT NULL, -- 'buy', 'sell'
    quantity DECIMAL(20,8) NOT NULL,
    price DECIMAL(20,8),
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'filled', 'cancelled'
    strategy_id INTEGER REFERENCES trading.strategies(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    filled_at TIMESTAMP
);

-- Create analysis tables
CREATE TABLE IF NOT EXISTS analysis.market_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open_price DECIMAL(20,8),
    high_price DECIMAL(20,8),
    low_price DECIMAL(20,8),
    close_price DECIMAL(20,8),
    volume BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp)
);

CREATE TABLE IF NOT EXISTS analysis.features (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    feature_value DECIMAL(20,8),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp, feature_name)
);

-- Create monitoring tables
CREATE TABLE IF NOT EXISTS monitoring.system_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(20,8),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS monitoring.performance_logs (
    id SERIAL PRIMARY KEY,
    component VARCHAR(100) NOT NULL,
    operation VARCHAR(100) NOT NULL,
    duration_ms DECIMAL(10,2),
    status VARCHAR(20) DEFAULT 'success',
    error_message TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_strategies_name ON trading.strategies(name);
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON trading.positions(symbol);
CREATE INDEX IF NOT EXISTS idx_orders_symbol ON trading.orders(symbol);
CREATE INDEX IF NOT EXISTS idx_orders_status ON trading.orders(status);
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time ON analysis.market_data(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_features_symbol_time ON analysis.features(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_system_metrics_name_time ON monitoring.system_metrics(metric_name, timestamp);
CREATE INDEX IF NOT EXISTS idx_performance_logs_component_time ON monitoring.performance_logs(component, timestamp);

-- Create a default strategy for testing
INSERT INTO trading.strategies (name, description, strategy_type, parameters)
VALUES ('default_strategy', 'Default trading strategy for testing', 'trend_following', '{"period": 20, "threshold": 0.02}')
ON CONFLICT (name) DO NOTHING;

-- Grant permissions
GRANT USAGE ON SCHEMA trading TO rqa2025_user;
GRANT USAGE ON SCHEMA analysis TO rqa2025_user;
GRANT USAGE ON SCHEMA monitoring TO rqa2025_user;

GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA trading TO rqa2025_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA analysis TO rqa2025_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA monitoring TO rqa2025_user;

GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA trading TO rqa2025_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA analysis TO rqa2025_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA monitoring TO rqa2025_user;

-- Create a simple health check function
CREATE OR REPLACE FUNCTION public.health_check()
RETURNS TEXT AS $$
BEGIN
    RETURN 'RQA2025 Database is healthy - ' || now()::text;
END;
$$ LANGUAGE plpgsql;

-- Log initialization completion
DO $$
BEGIN
    RAISE NOTICE 'RQA2025 Database initialization completed successfully';
END $$;










