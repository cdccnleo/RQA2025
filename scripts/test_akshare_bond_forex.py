import akshare as ak
import pandas as pd
import time

# Test bond data
print("=== BOND DATA ===")
try:
    df = ak.bond_china_comparison()
    print(f"bond_china_comparison: {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    if len(df) > 0:
        print(f"Sample[0]: {df.iloc[0].to_dict()}")
except Exception as e:
    print(f"bond_china_comparison FAIL: {e}")

time.sleep(1)
try:
    df = ak.bond_zh_hs_cov()
    print(f"bond_zh_hs_cov: {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    if len(df) > 0:
        print(f"Sample[0]: {df.iloc[0].to_dict()}")
except Exception as e:
    print(f"bond_zh_hs_cov FAIL: {e}")

time.sleep(1)
try:
    df = ak.bond_china_yield()
    print(f"bond_china_yield (cursor): {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    if len(df) > 0:
        print(f"Sample[0]: {df.iloc[0].to_dict()}")
except Exception as e:
    print(f"bond_china_yield FAIL: {e}")

# Test forex data
print("\n=== FOREX DATA ===")
time.sleep(1)
try:
    df = ak.currency_latest()
    print(f"currency_latest: {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    if len(df) > 0:
        print(f"Sample[0]: {df.iloc[0].to_dict()}")
except Exception as e:
    print(f"currency_latest FAIL: {e}")

time.sleep(1)
try:
    df = ak.currency_us_daily()
    print(f"currency_us_daily: {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    if len(df) > 0:
        print(f"Sample[0]: {df.iloc[0].to_dict()}")
except Exception as e:
    print(f"currency_us_daily FAIL: {e}")
