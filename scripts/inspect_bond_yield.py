import akshare as ak
import pandas as pd

df = ak.bond_china_yield()
print(f"Rows: {len(df)}, Columns: {df.columns.tolist()}")
print(f"Date range: {df.iloc[-1]['日期']} to {df.iloc[0]['日期']}")
print()
print("Latest 5 rows:")
print(df.head(5).to_string())
print()
print("Curve names:", df["曲线名称"].unique().tolist())
print()
# Check if latest data is recent (2024+)
latest_date = df.iloc[0]["日期"]
print(f"Latest date: {latest_date}, is recent: {latest_date.year >= 2024}")
