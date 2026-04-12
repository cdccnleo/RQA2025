import akshare as ak
import time

# Try more specific forex functions
print("=== FOREX detailed test ===")

# currency_latest needs symbol parameter
try:
    df = ak.currency_latest(symbol="usd")
    print(f"currency_latest(usd): {len(df)} rows, cols={df.columns.tolist()}")
    if len(df) > 0:
        print(f"  Sample: {df.iloc[0].to_dict()}")
except Exception as e:
    print(f"currency_latest(usd): {str(e)[:100]}")

time.sleep(1)
try:
    df = ak.currency_latest(symbol="eur")
    print(f"currency_latest(eur): {len(df)} rows")
    if len(df) > 0:
        print(f"  Sample: {df.iloc[0].to_dict()}")
except Exception as e:
    print(f"currency_latest(eur): {str(e)[:100]}")

time.sleep(1)
# Try currency_history
try:
    df = ak.currency_history(symbol="usd/cny")
    print(f"currency_history(usd/cny): {len(df)} rows, cols={df.columns.tolist()}")
    if len(df) > 0:
        print(f"  Sample: {df.iloc[0].to_dict()}")
except Exception as e:
    print(f"currency_history(usd/cny): {str(e)[:100]}")

time.sleep(1)
# bond_zh_hs_value we already know fails
# Try bond spot data
try:
    df = ak.bond_zh_hs_spot()
    print(f"bond_zh_hs_spot: {len(df)} rows, cols={df.columns.tolist()[:6]}")
    if len(df) > 0:
        print(f"  Sample: {df.iloc[0].to_dict()}")
except Exception as e:
    print(f"bond_zh_hs_spot: {str(e)[:100]}")
