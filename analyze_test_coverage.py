#!/usr/bin/env python3
"""
基础设施层测试覆盖率分析脚本
"""

import os

def analyze_test_coverage():
    print('🔍 基础设施层测试覆盖率现状分析报告')
    print('=' * 80)
    print()

    modules = [
        'constants', 'core', 'interfaces', 'config', 'distributed',
        'versioning', 'resource', 'monitoring', 'logging', 'security',
        'health', 'utils', 'ops', 'optimization'
    ]

    module_analysis = {}

    for module in modules:
        module_path = f'src/infrastructure/{module}'
        test_path = f'tests/infrastructure'

        # 统计源码文件数
        if os.path.exists(module_path):
            src_files = []
            for root, dirs, files in os.walk(module_path):
                for file in files:
                    if file.endswith('.py'):
                        src_files.append(os.path.join(root, file))
            src_count = len(src_files)
        else:
            src_count = 0

        # 统计测试文件数
        test_files = []
        if os.path.exists(test_path):
            for root, dirs, files in os.walk(test_path):
                for file in files:
                    if file.endswith('.py') and module in file:
                        test_files.append(os.path.join(root, file))

        # 检查utils子目录
        if module == 'utils' and os.path.exists(f'{test_path}/utils'):
            for root, dirs, files in os.walk(f'{test_path}/utils'):
                for file in files:
                    if file.endswith('.py'):
                        test_files.append(os.path.join(root, file))

        test_count = len(test_files)

        module_analysis[module] = {
            'src_files': src_count,
            'test_files': test_count,
            'test_ratio': test_count / max(src_count, 1)
        }

    print('📊 各模块源码与测试文件统计:')
    print('-' * 60)
    print('模块名称     源码文件 测试文件 测试比例 状态')
    print('-' * 60)

    for module, data in module_analysis.items():
        ratio = data['test_ratio']
        if ratio >= 1.0:
            status = '✅ 充足'
        elif ratio >= 0.5:
            status = '⚠️  一般'
        elif ratio > 0:
            status = '❌ 不足'
        else:
            status = '🚫 缺失'

        print(f'{module:<12} {data["src_files"]:<8} {data["test_files"]:<8} {ratio:<10.2f} {status}')

    print()
    print('🎯 测试覆盖率提升优先级分析:')
    print('-' * 50)

    # 按测试比例排序，优先处理测试最少的模块
    priority_order = sorted(module_analysis.items(), key=lambda x: x[1]['test_ratio'])

    print('优先级排序 (从最需要测试的模块开始):')
    for i, (module, data) in enumerate(priority_order, 1):
        if data['test_ratio'] == 0:
            status = '🚫 紧急'
        elif data['test_ratio'] < 0.1:
            status = '❌ 重要'
        else:
            status = '⚠️ 一般'
        print(f'{i:2d}. {module:<12} - {status} (测试比例: {data["test_ratio"]:.2f})')

    return module_analysis

if __name__ == '__main__':
    analyze_test_coverage()
