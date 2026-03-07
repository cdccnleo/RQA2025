-- 基本面数据表结构
-- 用于存储股票基本面数据，与价格数据表分离

CREATE TABLE IF NOT EXISTS akshare_fundamental_data (
    id BIGSERIAL PRIMARY KEY,
    source_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    report_date DATE NOT NULL,
    company_name VARCHAR(100),
    industry VARCHAR(50),
    pe DECIMAL(10, 4),
    pb DECIMAL(10, 4),
    market_cap DECIMAL(20, 2),
    revenue DECIMAL(20, 2),
    net_profit DECIMAL(20, 2),
    roe DECIMAL(10, 4),
    data_source VARCHAR(50) DEFAULT 'akshare',
    collected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    persistence_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_fundamental_record UNIQUE(source_id, symbol, report_date)
);

-- 创建索引优化查询性能
CREATE INDEX IF NOT EXISTS idx_akshare_fundamental_symbol_date ON akshare_fundamental_data(symbol, report_date DESC);
CREATE INDEX IF NOT EXISTS idx_akshare_fundamental_industry ON akshare_fundamental_data(industry);
CREATE INDEX IF NOT EXISTS idx_akshare_fundamental_source_collected ON akshare_fundamental_data(source_id, collected_at DESC);

-- 注释
COMMENT ON TABLE akshare_fundamental_data IS 'AKShare基本面数据存储表';
COMMENT ON COLUMN akshare_fundamental_data.id IS '主键ID';
COMMENT ON COLUMN akshare_fundamental_data.source_id IS '数据源ID';
COMMENT ON COLUMN akshare_fundamental_data.symbol IS '股票代码';
COMMENT ON COLUMN akshare_fundamental_data.report_date IS '报告日期';
COMMENT ON COLUMN akshare_fundamental_data.company_name IS '公司名称';
COMMENT ON COLUMN akshare_fundamental_data.industry IS '所属行业';
COMMENT ON COLUMN akshare_fundamental_data.pe IS '市盈率';
COMMENT ON COLUMN akshare_fundamental_data.pb IS '市净率';
COMMENT ON COLUMN akshare_fundamental_data.market_cap IS '市值';
COMMENT ON COLUMN akshare_fundamental_data.revenue IS '营收';
COMMENT ON COLUMN akshare_fundamental_data.net_profit IS '净利润';
COMMENT ON COLUMN akshare_fundamental_data.roe IS '净资产收益率';
COMMENT ON COLUMN akshare_fundamental_data.data_source IS '数据来源';
COMMENT ON COLUMN akshare_fundamental_data.collected_at IS '采集时间';
COMMENT ON COLUMN akshare_fundamental_data.persistence_timestamp IS '持久化时间';
