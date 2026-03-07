#!/usr/bin/env python3

with open('src/data/integration/enhanced_data_integration_modules/utilities.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 修复缩进问题
content = content.replace('            try:', '        try:')
content = content.replace('                # 预热常用股票数据', '            # 预热常用股票数据')
content = content.replace("                common_symbols = ['600519.SH', '000001.SZ', '000002.SZ', '600036.SH']", "            common_symbols = ['600519.SH', '000001.SZ', '000002.SZ', '600036.SH']")
content = content.replace('                for symbol in common_symbols:', '            for symbol in common_symbols:')

with open('src/data/integration/enhanced_data_integration_modules/utilities.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('修复了缩进问题')
