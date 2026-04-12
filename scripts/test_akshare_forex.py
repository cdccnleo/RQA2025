import akshare as ak
import time

# Search for forex functions
print("=== FOREX functions ===")
forex_funcs = [
    'currency_latest',
    'currency_history',
    'currency_symbol',
    'forex_usd_cny',
    'forex_usd',
    'exchange_rate',
    'currency_swap',
]
for f in forex_funcs:
    func = getattr(ak, f, None)
    if func:
        try:
            df = func()
            print(f"{f}: OK {len(df)} rows, cols={df.columns.tolist()[:5]}")
            if len(df) > 0:
                print(f"  Sample: {df.iloc[0].to_dict()}")
        except Exception as e:
            print(f"{f}: {str(e)[:80]}")
    else:
        print(f"{f}: not found")
    time.sleep(0.5)

print("\n=== currency_zh_portfolio ===")
try:
    df = ak.currency_zh_portfolio()
    print(f"currency_zh_portfolio: {len(df)} rows, cols={df.columns.tolist()[:5]}")
    if len(df) > 0:
        print(f"  Sample: {df.iloc[0].to_dict()}")
except Exception as e:
    print(f"FAIL: {e}")
