-- RQA2025 数据库初始化脚本
-- 预投产环境使用

-- 创建数据库（如果不存在）
-- 注意：通过环境变量 POSTGRES_DB 已经创建了 rqa2025 数据库

-- 切换到应用数据库
\c rqa2025;

-- 创建应用用户（如果不存在）
DO $$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'rqa2025_app') THEN
      CREATE USER rqa2025_app WITH PASSWORD 'rqa2025_app_secure_pass';
   END IF;
END
$$;

-- 授予应用用户权限
GRANT CONNECT ON DATABASE rqa2025 TO rqa2025_app;
GRANT USAGE ON SCHEMA public TO rqa2025_app;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO rqa2025_app;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO rqa2025_app;

-- 创建健康检查相关的表
CREATE TABLE IF NOT EXISTS health_checks (
    id SERIAL PRIMARY KEY,
    service_name VARCHAR(255) NOT NULL,
    check_type VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL,
    response_time FLOAT,
    error_message TEXT,
    checked_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- 创建索引以提高查询性能
CREATE INDEX IF NOT EXISTS idx_health_checks_service_name ON health_checks(service_name);
CREATE INDEX IF NOT EXISTS idx_health_checks_status ON health_checks(status);
CREATE INDEX IF NOT EXISTS idx_health_checks_checked_at ON health_checks(checked_at);
CREATE INDEX IF NOT EXISTS idx_health_checks_check_type ON health_checks(check_type);

-- 创建监控指标表
CREATE TABLE IF NOT EXISTS monitoring_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(255) NOT NULL,
    metric_value FLOAT NOT NULL,
    labels JSONB,
    collected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_monitoring_metrics_name ON monitoring_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_monitoring_metrics_collected_at ON monitoring_metrics(collected_at);

-- 创建告警记录表
CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    alert_name VARCHAR(255) NOT NULL,
    severity VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    description TEXT,
    labels JSONB,
    annotations JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at);

-- 创建配置表
CREATE TABLE IF NOT EXISTS configurations (
    id SERIAL PRIMARY KEY,
    config_key VARCHAR(255) UNIQUE NOT NULL,
    config_value JSONB,
    config_type VARCHAR(50) DEFAULT 'json',
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_configurations_key ON configurations(config_key);

-- 插入一些基础配置数据
INSERT INTO configurations (config_key, config_value, description) VALUES
('health_check.interval', '30', '健康检查间隔(秒)'),
('health_check.timeout', '10', '健康检查超时时间(秒)'),
('monitoring.enabled', 'true', '是否启用监控'),
('alerting.enabled', 'true', '是否启用告警'),
('cache.ttl', '3600', '缓存过期时间(秒)')
ON CONFLICT (config_key) DO NOTHING;

-- 创建用户会话表（如果需要）
CREATE TABLE IF NOT EXISTS user_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255),
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_user_sessions_session_id ON user_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_expires_at ON user_sessions(expires_at);

-- 创建API访问日志表
CREATE TABLE IF NOT EXISTS api_access_logs (
    id SERIAL PRIMARY KEY,
    method VARCHAR(10) NOT NULL,
    path VARCHAR(500) NOT NULL,
    status_code INTEGER NOT NULL,
    response_time FLOAT,
    ip_address INET,
    user_agent TEXT,
    request_id VARCHAR(255),
    accessed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_api_access_logs_path ON api_access_logs(path);
CREATE INDEX IF NOT EXISTS idx_api_access_logs_status_code ON api_access_logs(status_code);
CREATE INDEX IF NOT EXISTS idx_api_access_logs_accessed_at ON api_access_logs(accessed_at);

-- 创建性能指标表
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    service_name VARCHAR(255) NOT NULL,
    operation_name VARCHAR(255) NOT NULL,
    duration FLOAT NOT NULL,
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    tags JSONB,
    measured_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_performance_metrics_service ON performance_metrics(service_name);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_operation ON performance_metrics(operation_name);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_measured_at ON performance_metrics(measured_at);

-- 授予应用用户对所有表的权限
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO rqa2025_app;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO rqa2025_app;

-- 设置默认权限（新创建的表自动授予权限）
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO rqa2025_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO rqa2025_app;

-- 创建健康检查视图
CREATE OR REPLACE VIEW health_check_summary AS
SELECT
    service_name,
    check_type,
    status,
    COUNT(*) as total_checks,
    AVG(response_time) as avg_response_time,
    MAX(checked_at) as last_check,
    COUNT(CASE WHEN status = 'healthy' THEN 1 END) as healthy_count,
    COUNT(CASE WHEN status = 'unhealthy' THEN 1 END) as unhealthy_count,
    ROUND(
        COUNT(CASE WHEN status = 'healthy' THEN 1 END)::numeric /
        COUNT(*)::numeric * 100, 2
    ) as health_percentage
FROM health_checks
WHERE checked_at >= CURRENT_TIMESTAMP - INTERVAL '1 hour'
GROUP BY service_name, check_type, status;

-- 授予应用用户视图访问权限
GRANT SELECT ON health_check_summary TO rqa2025_app;

-- 输出初始化完成信息
DO $$
BEGIN
    RAISE NOTICE 'RQA2025 数据库初始化完成';
    RAISE NOTICE '创建的用户: rqa2025_app';
    RAISE NOTICE '创建的表: health_checks, monitoring_metrics, alerts, configurations, user_sessions, api_access_logs, performance_metrics';
    RAISE NOTICE '创建的视图: health_check_summary';
END
$$;

