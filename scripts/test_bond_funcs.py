import akshare as ak
import time

# Test all bond-related functions that might exist
bond_funcs = [
    'bond_china_yield',
    'bond_zh_hs_spot',
    'bond_zh_hs_daily',
    'bond_sse',
    'bond_szse',
    'bond_ai指数',
    'bond_cb',
    'bond_zh_cov',
]

print("=== BOND FUNCTIONS ===")
for f in bond_funcs:
    func = getattr(ak, f, None)
    if func:
        try:
            df = func()
            print(f"{f}: OK {len(df)} rows, cols={df.columns.tolist()[:5]}")
            if len(df) > 0:
                print(f"  Sample: {df.iloc[0].to_dict()}")
        except Exception as e:
            print(f"{f}: FAIL - {str(e)[:60]}")
    else:
        print(f"{f}: NOT FOUND")
    time.sleep(0.5)

# Test forex with correct params
print("\n=== FOREX FUNCTIONS ===")
forex_funcs = [
    'currency_latest',
    'currency_history',
    'forex_currency_pair',
    'forex_sina',
    'forex_china',
]
for f in forex_funcs:
    func = getattr(ak, f, None)
    if func:
        try:
            sig = func.__code__.co_varnames[:func.__code__.co_argcount]
            print(f"{f}: params={sig}")
            if f == 'currency_latest':
                df = func(base='USD')
            elif f == 'currency_history':
                df = func(symbol='USD/CNY')
            else:
                df = func()
            print(f"  -> OK {len(df)} rows")
            if len(df) > 0:
                print(f"  Sample: {df.iloc[0].to_dict()}")
        except Exception as e:
            print(f"{f}: FAIL - {str(e)[:80]}")
    else:
        print(f"{f}: NOT FOUND")
    time.sleep(0.5)
