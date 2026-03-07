#!/usr/bin/env python3

with open('src/data/integration/enhanced_data_integration_modules/utilities.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 修复缩进
lines[1767] = '        try:\n'
lines[1768] = '            # 预热常用股票数据\n'
lines[1769] = "            common_symbols = ['600519.SH', '000001.SZ', '000002.SZ', '600036.SH']\n"
lines[1770] = '            for symbol in common_symbols:\n'
lines[1771] = '                self._preload_stock_data(symbol)\n'
lines[1772] = "                self._cache_warming_status['warmed_items'] += 1\n"
lines[1774] = '            # 预热指数数据\n'

with open('src/data/integration/enhanced_data_integration_modules/utilities.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('修复了缩进问题')
