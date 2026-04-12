import akshare as ak
from datetime import datetime, timedelta

end = datetime.now().strftime('%Y%m%d')
start = (datetime.now() - timedelta(days=60)).strftime('%Y%m%d')
print(f"Requesting: {start} to {end}")
df = ak.bond_china_yield(start_date=start, end_date=end)
print(f"Got {len(df)} rows")
if len(df) > 0:
    print(f"Date range: {df['日期'].min()} to {df['日期'].max()}")
    print(df.head(3).to_string())
