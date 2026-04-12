#!/usr/bin/env python3
"""
完整修复：PostgreSQL 16个生产数据源 + JSON文件原子写入 + 禁用健康检测覆盖
"""
import psycopg2
import json
import os
import tempfile

conn = psycopg2.connect(
    host='rqa2025-postgres',
    database='rqa2025_prod',
    user='rqa2025_admin',
    password='SecurePass123!'
)
conn.autocommit = True
cur = conn.cursor()

print("=" * 60)
print("完整数据源配置修复")
print("=" * 60)

# ============================================================
# 16 个生产数据源配置
# ============================================================
production_sources = [
    {"id": "akshare_stock_a", "name": "AKShare A股数据", "type": "股票数据", "enabled": True, "url": "https://akshare.akfamily.xyz", "rate_limit": "1次/天", "config": {"description": "AKShare A股实时行情", "akshare_function": "stock_zh_a_spot"}},
    {"id": "akshare_stock_hk", "name": "AKShare 港股数据", "type": "股票数据", "enabled": True, "url": "https://akshare.akfamily.xyz", "rate_limit": "1次/天", "config": {"description": "东方财富港股实时行情", "akshare_function": "stock_hk_spot", "timeout_ms": 90000}},
    {"id": "akshare_index", "name": "AKShare A股指数", "type": "指数数据", "enabled": True, "url": "https://akshare.akfamily.xyz", "rate_limit": "1次/天", "config": {"description": "新浪A股指数实时行情", "akshare_function": "stock_zh_index_spot_sina"}},
    {"id": "akshare_bond", "name": "AKShare 国债收益率", "type": "债券数据", "enabled": True, "url": "https://akshare.akfamily.xyz", "rate_limit": "1次/天", "config": {"description": "中国国债收益率数据", "akshare_function": "bond_china_yield"}},
    {"id": "akshare_forex", "name": "AKShare 外汇牌价", "type": "外汇数据", "enabled": True, "url": "https://akshare.akfamily.xyz", "rate_limit": "1次/天", "config": {"description": "中国银行外汇牌价", "akshare_function": "currency_boc_safe"}},
    {"id": "akshare_macro_china", "name": "AKShare 中国宏观数据", "type": "宏观经济", "enabled": True, "url": "https://akshare.akfamily.xyz", "rate_limit": "1次/天", "config": {"description": "中国GDP等宏观数据", "akshare_function": "macro_china_gdp_yearly"}},
    {"id": "akshare_macro_usa", "name": "AKShare 美国宏观数据", "type": "宏观经济", "enabled": True, "url": "https://akshare.akfamily.xyz", "rate_limit": "1次/天", "config": {"description": "美国CPI/GDP等宏观数据", "akshare_function": "macro_usa_gdp_monthly"}},
    {"id": "akshare_news_js", "name": "AKShare 上海金属期货新闻", "type": "财经新闻", "enabled": True, "url": "https://akshare.akfamily.xyz", "rate_limit": "1次/天", "config": {"description": "上海金属期货新闻", "akshare_function": "futures_news_shmet"}},
    {"id": "akshare_news_eastmoney", "name": "AKShare 东方财富新闻", "type": "财经新闻", "enabled": True, "url": "https://akshare.akfamily.xyz", "rate_limit": "1次/天", "config": {"description": "东方财富财经新闻", "akshare_function": "stock_news_em"}},
    {"id": "akshare_news_all", "name": "AKShare CCTV经济新闻", "type": "财经新闻", "enabled": True, "url": "https://akshare.akfamily.xyz", "rate_limit": "1次/天", "config": {"description": "CCTV经济新闻", "akshare_function": "news_cctv"}},
    {"id": "akshare_commodity_gold", "name": "AKShare 黄金现货", "type": "大宗商品", "enabled": True, "url": "https://akshare.akfamily.xyz", "rate_limit": "1次/天", "config": {"description": "上海黄金交易所黄金现货基准价", "akshare_function": "spot_golden_benchmark_sge"}},
    {"id": "akshare_commodity_energy", "name": "AKShare 国内油价", "type": "大宗商品", "enabled": True, "url": "https://akshare.akfamily.xyz", "rate_limit": "1次/天", "config": {"description": "国内油价历史数据", "akshare_function": "energy_oil_hist"}},
    {"id": "akshare_commodity_crude", "name": "AKShare 美国原油", "type": "大宗商品", "enabled": True, "url": "https://akshare.akfamily.xyz", "rate_limit": "1次/天", "config": {"description": "EIA美国原油库存变化率", "akshare_function": "macro_usa_eia_crude_rate"}},
    {"id": "akshare_commodity_natural_gas", "name": "AKShare 天然气", "type": "大宗商品", "enabled": False, "url": "https://akshare.akfamily.xyz", "rate_limit": "1次/天", "config": {"description": "NYMEX天然气期货（函数已从akshare移除）", "akshare_function": "macro_usa_api_crude_stock", "note": "暂时禁用"}},
    {"id": "akshare_commodity", "name": "AKShare 大宗商品总览", "type": "大宗商品", "enabled": True, "url": "https://akshare.akfamily.xyz", "rate_limit": "1次/天", "config": {"description": "能源油品明细数据", "akshare_function": "energy_oil_detail"}},
    {"id": "baostock_ashare", "name": "BaoStock A股数据", "type": "股票数据", "enabled": True, "url": "https://github.com/baostock/baostock", "rate_limit": "1次/天", "config": {"description": "BaoStock A股历史数据备份", "akshare_function": "baostock_list"}},
]

config_data = {
    "data_sources": production_sources,
    "metadata": {
        "version": "3.0.0",
        "last_updated": "2026-04-13T05:35:00",
        "environment": "production",
        "fix_version": "fix-complete-2026-04-13",
        "description": "RQA2025 量化交易系统 - 完整16个生产数据源"
    }
}

# Step 1: 写入 PostgreSQL
cur.execute("""
    INSERT INTO data_source_configs (config_key, config_data, environment, version, updated_at)
    VALUES ('data_sources_production', %s, 'production', '3.0.0', CURRENT_TIMESTAMP)
    ON CONFLICT (config_key) 
    DO UPDATE SET config_data = EXCLUDED.config_data, environment = EXCLUDED.environment,
                  version = EXCLUDED.version, updated_at = CURRENT_TIMESTAMP
""", (json.dumps(config_data, ensure_ascii=False),))
print(f"✅ PostgreSQL 已写入 16 个数据源")

# Step 2: 原子性写入 JSON
json_path = '/app/data/data_sources_config.json'
with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', 
                                  dir=os.path.dirname(json_path), 
                                  suffix='.tmp', delete=False) as tf:
    temp_path = tf.name
    json.dump(config_data, tf, ensure_ascii=False, indent=2)
    tf.flush()
    os.fsync(tf.fileno())
os.replace(temp_path, json_path)

size = os.path.getsize(json_path)
with open(json_path) as f:
    verify = json.load(f)
print(f"✅ JSON 文件已原子写入: {size} bytes, {len(verify.get('data_sources', []))} 个数据源")

cur.close()
conn.close()

# Step 3: 通知应用重新加载配置
import requests
try:
    r = requests.post('http://localhost:8000/api/v1/data/sources/cache/clear', timeout=5)
    print(f"✅ 配置缓存已清除: {r.status_code}")
except Exception as e:
    print(f"⚠️ 缓存清除失败: {e}")

print("\n✅ 完整修复完成")
