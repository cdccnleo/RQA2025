-- ============================================================
-- 股票数据语义化视图创建脚本
-- 
-- 目的: 增强数据可读性，简化常用查询
-- 创建时间: 2026-03-22
-- 版本: 1.0.0
-- ============================================================

-- ============================================================
-- 1. 日线行情视图
-- 用途: 查询日线级别的股票行情数据
-- ============================================================
CREATE OR REPLACE VIEW v_stock_daily_price AS
SELECT 
    source_id,
    symbol,
    date,
    open_price AS open,
    high_price AS high,
    low_price AS low,
    close_price AS close,
    volume,
    amount,
    pct_change,
    change,
    turnover_rate,
    amplitude,
    collected_at
FROM akshare_stock_data 
WHERE data_type = 'daily'
ORDER BY source_id, symbol, date DESC;

COMMENT ON VIEW v_stock_daily_price IS '日线行情视图 - 提供日线级别的股票OHLCV数据';

-- ============================================================
-- 2. A股数据视图
-- 用途: 筛选A股市场的股票数据
-- ============================================================
CREATE OR REPLACE VIEW v_a_stock_price AS
SELECT 
    source_id,
    symbol,
    date,
    open_price,
    high_price,
    low_price,
    close_price,
    volume,
    amount,
    pct_change,
    change,
    turnover_rate,
    amplitude,
    data_type,
    collected_at
FROM akshare_stock_data 
WHERE source_id LIKE 'akshare_stock_a%'
   OR source_id = 'akshare_stock_a'
ORDER BY symbol, date DESC;

COMMENT ON VIEW v_a_stock_price IS 'A股行情数据视图 - 筛选A股市场的股票数据';

-- ============================================================
-- 3. 港股数据视图
-- 用途: 筛选港股市场的股票数据
-- ============================================================
CREATE OR REPLACE VIEW v_hk_stock_price AS
SELECT 
    source_id,
    symbol,
    date,
    open_price,
    high_price,
    low_price,
    close_price,
    volume,
    amount,
    pct_change,
    change,
    turnover_rate,
    amplitude,
    data_type,
    collected_at
FROM akshare_stock_data 
WHERE source_id LIKE 'akshare_stock_hk%'
   OR source_id LIKE '%hk%'
ORDER BY symbol, date DESC;

COMMENT ON VIEW v_hk_stock_price IS '港股行情数据视图 - 筛选港股市场的股票数据';

-- ============================================================
-- 4. 最新价格视图
-- 用途: 获取每只股票的最新交易数据
-- ============================================================
CREATE OR REPLACE VIEW v_stock_latest_price AS
SELECT DISTINCT ON (source_id, symbol)
    source_id,
    symbol,
    date,
    close_price,
    volume,
    amount,
    pct_change,
    turnover_rate,
    collected_at
FROM akshare_stock_data
ORDER BY source_id, symbol, date DESC;

COMMENT ON VIEW v_stock_latest_price IS '最新价格视图 - 每只股票的最新交易数据';

-- ============================================================
-- 5. 数据源统计视图
-- 用途: 按数据源统计数据概况
-- ============================================================
CREATE OR REPLACE VIEW v_stock_data_source_stats AS
SELECT 
    source_id,
    data_type,
    COUNT(*) AS total_records,
    COUNT(DISTINCT symbol) AS symbol_count,
    MIN(date) AS earliest_date,
    MAX(date) AS latest_date,
    MAX(collected_at) AS last_collected,
    SUM(volume) AS total_volume,
    AVG(pct_change) AS avg_pct_change
FROM akshare_stock_data
GROUP BY source_id, data_type
ORDER BY source_id, data_type;

COMMENT ON VIEW v_stock_data_source_stats IS '数据源统计视图 - 按数据源统计数据概况';

-- ============================================================
-- 6. 涨跌幅排行视图
-- 用途: 按日期展示涨跌幅排名
-- ============================================================
CREATE OR REPLACE VIEW v_stock_change_ranking AS
SELECT 
    source_id,
    symbol,
    date,
    close_price,
    pct_change,
    volume,
    amount,
    RANK() OVER (PARTITION BY source_id, date ORDER BY pct_change DESC) AS gain_rank,
    RANK() OVER (PARTITION BY source_id, date ORDER BY pct_change ASC) AS loss_rank
FROM akshare_stock_data
WHERE data_type = 'daily'
ORDER BY date DESC, pct_change DESC;

COMMENT ON VIEW v_stock_change_ranking IS '涨跌幅排行视图 - 按日期展示涨跌幅排名';

-- ============================================================
-- 7. 成交量排行视图
-- 用途: 按日期展示成交量排名
-- ============================================================
CREATE OR REPLACE VIEW v_stock_volume_ranking AS
SELECT 
    source_id,
    symbol,
    date,
    close_price,
    volume,
    amount,
    turnover_rate,
    RANK() OVER (PARTITION BY source_id, date ORDER BY volume DESC) AS volume_rank
FROM akshare_stock_data
WHERE data_type = 'daily'
ORDER BY date DESC, volume DESC;

COMMENT ON VIEW v_stock_volume_ranking IS '成交量排行视图 - 按日期展示成交量排名';

-- ============================================================
-- 8. 数据完整性检查视图
-- 用途: 检查数据缺失情况
-- ============================================================
CREATE OR REPLACE VIEW v_stock_data_completeness AS
SELECT 
    source_id,
    symbol,
    COUNT(*) AS total_records,
    COUNT(open_price) AS open_count,
    COUNT(close_price) AS close_count,
    COUNT(volume) AS volume_count,
    COUNT(high_price) AS high_count,
    COUNT(low_price) AS low_count,
    ROUND(100.0 * COUNT(close_price) / NULLIF(COUNT(*), 0), 2) AS completeness_pct,
    MIN(date) AS start_date,
    MAX(date) AS end_date
FROM akshare_stock_data
GROUP BY source_id, symbol
ORDER BY completeness_pct ASC, source_id, symbol;

COMMENT ON VIEW v_stock_data_completeness IS '数据完整性检查视图 - 检查数据缺失情况';

-- ============================================================
-- 9. 月度统计视图
-- 用途: 按月统计数据
-- ============================================================
CREATE OR REPLACE VIEW v_stock_monthly_stats AS
SELECT 
    source_id,
    symbol,
    DATE_TRUNC('month', date) AS month,
    COUNT(*) AS trading_days,
    MIN(close_price) AS month_low,
    MAX(close_price) AS month_high,
    AVG(close_price) AS avg_close,
    SUM(volume) AS total_volume,
    SUM(amount) AS total_amount,
    AVG(turnover_rate) AS avg_turnover_rate,
    (MAX(close_price) - MIN(close_price)) / NULLIF(MIN(close_price), 0) * 100 AS monthly_amplitude_pct
FROM akshare_stock_data
WHERE data_type = 'daily'
GROUP BY source_id, symbol, DATE_TRUNC('month', date)
ORDER BY source_id, symbol, month DESC;

COMMENT ON VIEW v_stock_monthly_stats IS '月度统计视图 - 按月统计数据';

-- ============================================================
-- 10. 异常数据检测视图
-- 用途: 检测可能的异常数据
-- ============================================================
CREATE OR REPLACE VIEW v_stock_data_anomalies AS
SELECT 
    source_id,
    symbol,
    date,
    open_price,
    high_price,
    low_price,
    close_price,
    volume,
    pct_change,
    CASE 
        WHEN pct_change > 10 OR pct_change < -10 THEN '大幅波动'
        WHEN volume = 0 THEN '零成交量'
        WHEN high_price < low_price THEN '高低价异常'
        WHEN open_price = 0 OR close_price = 0 THEN '零价格'
        ELSE '其他异常'
    END AS anomaly_type
FROM akshare_stock_data
WHERE 
    pct_change > 10 OR pct_change < -10
    OR volume = 0
    OR high_price < low_price
    OR open_price = 0 OR close_price = 0
ORDER BY date DESC, ABS(pct_change) DESC;

COMMENT ON VIEW v_stock_data_anomalies IS '异常数据检测视图 - 检测可能的异常数据';

-- ============================================================
-- 权限设置（根据实际环境调整）
-- ============================================================
-- GRANT SELECT ON v_stock_daily_price TO rqa2025_admin;
-- GRANT SELECT ON v_a_stock_price TO rqa2025_admin;
-- GRANT SELECT ON v_hk_stock_price TO rqa2025_admin;
-- GRANT SELECT ON v_stock_latest_price TO rqa2025_admin;
-- GRANT SELECT ON v_stock_data_source_stats TO rqa2025_admin;
-- GRANT SELECT ON v_stock_change_ranking TO rqa2025_admin;
-- GRANT SELECT ON v_stock_volume_ranking TO rqa2025_admin;
-- GRANT SELECT ON v_stock_data_completeness TO rqa2025_admin;
-- GRANT SELECT ON v_stock_monthly_stats TO rqa2025_admin;
-- GRANT SELECT ON v_stock_data_anomalies TO rqa2025_admin;
