#!/usr/bin/env python3
import sys, os, logging, time
logging.basicConfig(level=logging.WARNING)
os.environ['POSTGRES_PASSWORD'] = 'rqa2025_prod_password'

start = time.time()
sys.path.insert(0, '/app')

from src.data.collectors.akshare_collector import AKShareCollector
print(f"Import took {time.time()-start:.1f}s")

c = AKShareCollector()
d = c.collect_commodity_natural_gas()
print(f"Collected: {len(d) if d else 0} records")
if d:
    print(f"Sample: {d[0]}")
    ok = c.save_commodity_natural_gas_to_database(d)
    print(f"Save: {'SUCCESS' if ok else 'FAILED'}")
else:
    print("No data collected")
