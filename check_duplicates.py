import json

# 加载分析结果
with open('test_analysis.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print('重复代码检测机会:')
dups = [opp for opp in data['opportunities'] if 'duplicate' in opp.get('title', '').lower()]
print(f'发现 {len(dups)} 个重复代码问题')

for dup in dups[:5]:
    print(f'- {dup["title"]}')
    print(f'  描述: {dup["description"]}')
    print(f'  文件: {dup["file_path"]}')
    print()

























