-- Bond yield curve table for 国债收益率曲线
CREATE TABLE IF NOT EXISTS akshare_bond_yield (
    id BIGSERIAL PRIMARY KEY,
    source_id VARCHAR(50) NOT NULL,
    curve_name VARCHAR(100) NOT NULL,
    curve_type VARCHAR(50) NOT NULL,  -- 'treasury' / 'commercial_bank' / 'corporate_aaa'
    date DATE NOT NULL,
    yield_3m NUMERIC(10, 6),
    yield_6m NUMERIC(10, 6),
    yield_1y NUMERIC(10, 6),
    yield_3y NUMERIC(10, 6),
    yield_5y NUMERIC(10, 6),
    yield_7y NUMERIC(10, 6),
    yield_10y NUMERIC(10, 6),
    yield_30y NUMERIC(10, 6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_id, curve_name, date)
);

CREATE INDEX IF NOT EXISTS idx_bond_yield_date ON akshare_bond_yield(date DESC);
CREATE INDEX IF NOT EXISTS idx_bond_yield_curve ON akshare_bond_yield(curve_name, date DESC);
