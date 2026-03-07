"""
检查指标计算次数
"""

import json

# 读取 indicator_calculations.json 文件
try:
    with open('data/indicator_calculations.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("指标计算次数:")
    print("-" * 40)
    for indicator, info in data.items():
        print(f"{indicator}: {info['count']} 次")
    
    print("-" * 40)
    print(f"总计: {sum(info['count'] for info in data.values())} 次")
    
except Exception as e:
    print(f"读取文件失败: {e}")
