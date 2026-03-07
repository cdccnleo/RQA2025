"""检查健康管理模块投产达标情况"""
import json
import os
from datetime import datetime

# 查找最新的覆盖率报告
coverage_files = [f for f in os.listdir('test_logs') if f.startswith('health_coverage') and f.endswith('.json')]
if not coverage_files:
    print("❌ 未找到覆盖率报告文件")
    exit(1)

latest_file = max([(os.path.getmtime(f'test_logs/{f}'), f) for f in coverage_files if os.path.exists(f'test_logs/{f}')])[1]

# 读取覆盖率数据
with open(f'test_logs/{latest_file}', 'r', encoding='utf-8') as f:
    coverage_data = json.load(f)

totals = coverage_data['totals']
percent_covered = totals['percent_covered']
covered_lines = totals['covered_lines']
total_lines = totals['num_statements']
missing_lines = totals['missing_lines']

print('=' * 80)
print('📊 健康管理模块投产达标检查')
print('=' * 80)
print(f'\n数据源: {latest_file}')
print(f'检查时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
print(f'覆盖率: {percent_covered:.2f}%')
print(f'已覆盖: {covered_lines}/{total_lines}行')
print(f'总代码: {total_lines}行')
print(f'缺失: {missing_lines}行')

print('\n🎯 投产标准评估:')
print('=' * 50)

# 评估各项标准
standards = [
    ('覆盖率 > 50%', percent_covered > 50, '基础要求'),
    ('覆盖率 > 60%', percent_covered > 60, '良好标准'),
    ('覆盖率 > 70%', percent_covered > 70, '优秀标准'),
    ('覆盖率 > 80%', percent_covered > 80, '卓越标准'),
    ('测试稳定', percent_covered > 40, '稳定性要求'),
    ('代码质量', missing_lines < total_lines * 0.5, '质量要求')
]

for standard, passed, description in standards:
    status = '✅ 达标' if passed else '❌ 不达标'
    print(f'{standard:<15} {status:<10} ({description})')

# 总体评估
if percent_covered >= 70:
    overall_status = '🟢 完全达标 - 优秀'
elif percent_covered >= 60:
    overall_status = '🟡 基本达标 - 良好'
elif percent_covered >= 50:
    overall_status = '🟠 勉强达标 - 基础'
else:
    overall_status = '🔴 不达标 - 需改进'

print(f'\n📋 总体评估: {overall_status}')
print('=' * 80)

# 下一步建议
print('\n🚀 下一步建议:')
if percent_covered < 50:
    print('1. 紧急提升覆盖率到50%以上')
    print('2. 重点测试核心功能模块')
elif percent_covered < 60:
    print('1. 继续提升到60%里程碑')
    print('2. 优化低覆盖率模块')
elif percent_covered < 70:
    print('1. 冲刺70%优秀标准')
    print('2. 完善边缘情况测试')
else:
    print('1. 已达到优秀标准，可考虑投产')
    print('2. 继续优化到80%+卓越标准')

print('=' * 80)