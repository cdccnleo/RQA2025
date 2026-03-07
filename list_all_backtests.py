#!/usr/bin/env python3
import sys
sys.path.insert(0, '/app')
from src.gateway.web.backtest_persistence import list_backtest_results

results = list_backtest_results(limit=10)
print('可用的回测记录:')
for r in results:
    print(f"  ID: {r.get('backtest_id')}")
    print(f"  日期: {r.get('start_date')} 至 {r.get('end_date')}")
    print()
