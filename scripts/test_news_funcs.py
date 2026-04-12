import akshare.news as news_mod
import time

print("Module contents:", dir(news_mod)[:10])

funcs = [
    ('news_cctv', lambda: getattr(news_mod, 'news_cctv')()),
    ('news_baidu', lambda: getattr(news_mod, 'news_baidu')(symbol='财经')),
    ('news_stock', lambda: getattr(news_mod, 'news_stock')()),
]

for name, fn in funcs:
    try:
        df = fn()
        cols = df.columns.tolist()
        print(f"{name}: {len(df)} rows, cols={cols[:5]}")
        if len(df) > 0:
            print(f"  Sample[0]: {df.iloc[0].to_dict()}")
    except Exception as e:
        print(f"{name}: FAIL - {str(e)[:80]}")
    time.sleep(1)
