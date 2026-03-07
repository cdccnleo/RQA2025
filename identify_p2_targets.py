"""识别P2优先级目标模块"""
import json

with open('test_logs/health_coverage_focused.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 分析模块
targets = []
for filepath, filedata in data['files'].items():
    if 'src' in filepath and 'infrastructure' in filepath and 'health' in filepath:
        percent = filedata['summary']['percent_covered']
        missing = filedata['summary']['missing_lines']
        total = filedata['summary']['num_statements']
        filename = filepath.replace('\\', '/').split('/')[-1]
        
        # P2策略：40-65%覆盖率，有提升空间的模块
        if 40 <= percent < 65 and missing > 10 and total > 50:
            roi = missing / (70 - percent) if percent < 70 else 0
            targets.append((filename, percent, missing, total, roi))

targets.sort(key=lambda x: x[4], reverse=True)

print('🎯 第8轮循环 - P2优先级模块识别')
print('=' * 90)
print(f'当前: 59.34% | 目标: 65-70% | 策略: 提升40-65%模块到70%+')
print('=' * 90)
print(f'\n{"序号":<4} {"模块名称":<50} {"当前":<8} {"目标":<8} {"缺失":<8} {"ROI":<6}')
print('-' * 90)

for i, (name, percent, missing, total, roi) in enumerate(targets[:10], 1):
    print(f'{i:<4} {name:<50} {percent:>6.1f}% → 70%+ {missing:>6}行 {roi:>5.1f}')

print(f'\n💡 策略:')
print(f'  选择ROI最高的5个模块')
print(f'  预期新增覆盖: 250-400行')
print(f'  预期提升覆盖率: 2-3%')
print(f'  目标: 达到62-65%覆盖率')

