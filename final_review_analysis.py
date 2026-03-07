#!/usr/bin/env python3
"""
最终代码审查分析脚本
"""

import os
import re
from collections import defaultdict, Counter


def analyze_code_organization():
    """分析代码组织结构"""
    print('=== 代码组织结构分析 ===')

    # 统计各模块的文件数量和行数
    modules = {
        'core': 'src/infrastructure/cache/core',
        'interfaces': 'src/infrastructure/cache/interfaces',
        'distributed': 'src/infrastructure/cache/distributed',
        'strategies': 'src/infrastructure/cache/strategies',
        'monitoring': 'src/infrastructure/cache/monitoring',
        'utils': 'src/infrastructure/cache/utils',
        'exceptions': 'src/infrastructure/cache/exceptions'
    }

    stats = {}
    total_files = 0
    total_lines = 0

    for module_name, module_path in modules.items():
        if os.path.exists(module_path):
            files = [f for f in os.listdir(module_path) if f.endswith(
                '.py') and not f.startswith('__')]
            file_count = len(files)
            total_files += file_count

            lines = 0
            for file in files:
                try:
                    with open(os.path.join(module_path, file), 'r', encoding='utf-8') as f:
                        lines += len(f.readlines())
                except:
                    pass

            total_lines += lines
            stats[module_name] = {'files': file_count, 'lines': lines}

    print(f'总文件数: {total_files}')
    print(f'总代码行数: {total_lines}')
    print()

    for module, data in stats.items():
        print(f'{module:12} | {data["files"]:2} 文件 | {data["lines"]:4} 行')

    return stats


def check_class_inheritance():
    """检查类继承关系"""
    print()
    print('=== 类继承关系分析 ===')

    inheritance_patterns = defaultdict(list)

    for root, dirs, files in os.walk('src/infrastructure/cache'):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()

                        # 查找类定义
                        class_matches = re.findall(r'class (\w+)(?:\(([^)]*)\))?:', content)
                        for class_name, bases in class_matches:
                            if bases:
                                base_classes = [b.strip() for b in bases.split(',')]
                                inheritance_patterns[class_name].extend(base_classes)
                except:
                    pass

    # 统计继承模式
    abc_count = 0
    protocol_count = 0
    multi_inheritance = 0

    for class_name, bases in inheritance_patterns.items():
        if 'ABC' in bases or any('abstractmethod' in b for b in bases):
            abc_count += 1
        if 'Protocol' in bases or any('Protocol' in b for b in bases):
            protocol_count += 1
        if len(bases) > 1:
            multi_inheritance += 1

    print(f'ABC继承类数: {abc_count}')
    print(f'Protocol继承类数: {protocol_count}')
    print(f'多重继承类数: {multi_inheritance}')

    # 显示一些重要的继承关系
    print()
    print('关键类继承关系:')
    key_classes = ['UnifiedCacheManager', 'MultiLevelCache', 'CacheComponent', 'BaseCacheComponent']
    for cls in key_classes:
        if cls in inheritance_patterns:
            bases = inheritance_patterns[cls]
            print(f'  {cls} -> {bases}')
        else:
            print(f'  {cls} -> 无继承')


def check_method_duplication():
    """检查方法重复定义"""
    print()
    print('=== 方法重复分析 ===')

    method_locations = defaultdict(list)

    for root, dirs, files in os.walk('src/infrastructure/cache'):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()

                        # 查找方法定义
                        methods = re.findall(r'def (\w+)\s*\(', content)
                        for method in methods:
                            method_locations[method].append(os.path.basename(filepath))
                except:
                    pass

    # 找出重复的方法
    duplicates = {k: v for k, v in method_locations.items() if len(v) > 1}

    if duplicates:
        print(f'发现 {len(duplicates)} 个重复方法名')

        # 统计重复程度
        duplicate_counts = Counter(len(v) for v in duplicates.values())
        print('重复分布:')
        for count, freq in sorted(duplicate_counts.items()):
            print(f'  {count}个文件中重复: {freq}个方法')

        # 显示最严重的重复
        most_duplicated = sorted(duplicates.items(), key=lambda x: len(x[1]), reverse=True)[:10]
        print()
        print('最严重重复的前10个方法:')
        for method, files in most_duplicated:
            print(f'  {method}: {len(files)} 个文件')
    else:
        print('✅ 未发现方法重复')


def check_import_patterns():
    """检查导入模式"""
    print()
    print('=== 导入模式分析 ===')

    absolute_imports = []
    relative_imports = []

    for root, dirs, files in os.walk('src/infrastructure/cache'):
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()

                        # 检查导入
                        lines = content.split('\n')
                        for line in lines:
                            line = line.strip()
                            if line.startswith('from infrastructure.cache'):
                                absolute_imports.append(filepath)
                            elif line.startswith('from .') or line.startswith('from ..'):
                                relative_imports.append(filepath)
                except:
                    pass

    print(f'绝对导入文件数: {len(set(absolute_imports))}')
    print(f'相对导入文件数: {len(set(relative_imports))}')

    if absolute_imports:
        print('⚠️  仍使用绝对导入的文件:')
        for f in list(set(absolute_imports))[:5]:
            print(f'  - {os.path.basename(f)}')
        if len(set(absolute_imports)) > 5:
            print(f'  ... 还有 {len(set(absolute_imports)) - 5} 个文件')


def check_protocol_adoption():
    """检查Protocol模式的采用情况"""
    print()
    print('=== Protocol模式采用分析 ===')

    protocol_files = []
    abc_files = []

    for root, dirs, files in os.walk('src/infrastructure/cache'):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()

                        if 'from typing import Protocol' in content or 'Protocol' in content:
                            if 'class' in content and 'Protocol' in content:
                                protocol_files.append(filepath)

                        if 'from abc import' in content or '@abstractmethod' in content:
                            abc_files.append(filepath)
                except:
                    pass

    print(f'使用Protocol的文件数: {len(protocol_files)}')
    print(f'使用ABC的文件数: {len(abc_files)}')

    # 检查interfaces目录的Protocol采用情况
    interfaces_protocol = [f for f in protocol_files if 'interfaces' in f]
    print(f'interfaces模块Protocol文件数: {len(interfaces_protocol)}')


if __name__ == '__main__':
    # 执行所有分析
    analyze_code_organization()
    check_class_inheritance()
    check_method_duplication()
    check_import_patterns()
    check_protocol_adoption()

    print()
    print('=== 代码组织分析完成 ===')
