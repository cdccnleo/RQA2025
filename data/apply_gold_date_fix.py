#!/usr/bin/env python3
"""Apply gold date filter fix to Windows filesystem"""
import re, os

file_path = r"C:\PythonProject\RQA2025\src\data\collectors\akshare_collector.py"

with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

print(f"File size: {len(content)} chars, {content.count(chr(10))} lines")
print(f"合法性过滤 already present: {'合法性过滤' in content}")

# Find the exact pattern to replace
search = "                    date_str = str(trading_time)\n                \n                evening_price = row.get('晚盘价')"
replace = "                    date_str = str(trading_time)\n                \n                # 日期合法性过滤：过滤未来日期和超老日期（akshare偶尔返回错误的历史/未来数据）\n                try:\n                    date_obj = datetime.strptime(date_str[:10], '%Y-%m-%d')\n                    max_date = datetime.now() + timedelta(days=1)\n                    min_date = datetime(2000, 1, 1)\n                    if not (min_date <= date_obj <= max_date):\n                        continue\n                except (ValueError, TypeError):\n                    continue\n                \n                evening_price = row.get('晚盘价')"

if search in content:
    new_content = content.replace(search, replace, 1)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f"Fix applied! New size: {len(new_content)} chars, {new_content.count(chr(10))} lines")
    # Verify
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        verify = f.read()
    print(f"Verification - 合法性过滤 present: {'合法性过滤' in verify}")
else:
    print("ERROR: Pattern not found!")
    # Debug: show lines around date_str
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'date_str = str(trading_time)' in line:
            print(f"Found at line {i+1}")
            for j in range(-2, 8):
                if 0 <= i+j < len(lines):
                    print(f"  {i+j+1}: {repr(lines[i+j])}")
            break
