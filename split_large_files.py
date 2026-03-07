#!/usr/bin/env python3
"""
基础设施层配置管理大文件拆分工具
"""

import os
from typing import List, Tuple


def extract_class_definitions(content: str) -> List[Tuple[str, int, int]]:
    """提取类定义及其范围"""
    lines = content.split('\n')
    classes = []
    brace_count = 0
    in_class = False
    class_start = 0
    current_class = ""

    for i, line in enumerate(lines):
        stripped = line.strip()

        # 查找类定义开始
        if stripped.startswith('class '):
            if in_class:
                # 结束之前的类
                classes.append((current_class, class_start, i-1))

            current_class = stripped.split('class ')[1].split('(')[0].split(':')[0].strip()
            class_start = i
            in_class = True
            brace_count = 0

        elif in_class:
            # 计算缩进级别
            if stripped and not stripped.startswith(' ') and not stripped.startswith('\t') and stripped != '':
                if brace_count <= 0:  # 缩进结束
                    classes.append((current_class, class_start, i-1))
                    in_class = False
                    current_class = ""
                    class_start = 0

    # 处理最后一个类
    if in_class:
        classes.append((current_class, class_start, len(lines)-1))

    return classes


def split_file_by_classes(file_path: str, output_dir: str):
    """按类拆分文件"""

    print(f'🔄 拆分文件: {os.path.basename(file_path)}')

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        lines = content.split('\n')
        classes = extract_class_definitions(content)

        if len(classes) <= 1:
            print(f'   ⚠️  只有 {len(classes)} 个类，跳过拆分')
            return []

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 提取公共导入和代码
        imports_and_globals = []
        first_class_line = min(start for _, start, _ in classes) if classes else len(lines)

        for i in range(first_class_line):
            line = lines[i]
            stripped = line.strip()
            if (stripped.startswith(('import ', 'from ')) or
                stripped.startswith(('class ', 'def ')) or
                    (stripped and not stripped.startswith('#'))):
                break
            imports_and_globals.append(line)

        # 为每个类创建单独文件
        split_files = []

        for class_name, start_line, end_line in classes:
            # 创建文件名
            file_name = f"{class_name.lower()}.py"
            output_file = os.path.join(output_dir, file_name)

            # 提取类内容
            class_lines = lines[start_line:end_line+1]

            # 合并内容
            new_content = '\n'.join(imports_and_globals + [''] + class_lines)

            # 写入文件
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(new_content)

            split_files.append(output_file)
            print(f'   ✅ 创建: {os.path.basename(output_file)}')

        # 创建__init__.py文件
        init_content = '''"""拆分后的模块初始化文件"""

from infrastructure.config.core.imports import (
    # 导入所有拆分的类
'''

        for class_name, _, _ in classes:
            init_content += f'from .{class_name.lower()} import {class_name}\n'

        init_content += '''

__all__ = [\n'''
        for class_name, _, _ in classes:
            init_content += f'    "{class_name}",\n'
        init_content += ']\n'

        init_file = os.path.join(output_dir, '__init__.py')
        with open(init_file, 'w', encoding='utf-8') as f:
            f.write(init_content)

        print(f'   ✅ 创建: __init__.py')

        return split_files

    except Exception as e:
        print(f'   ❌ 拆分失败: {e}')
        return []


def split_large_files():
    """拆分所有大文件"""

    print('=== 📦 Phase 1.3: 拆分大文件 ===')
    print()

    # 定义拆分策略
    split_strategies = {
        # 存储模块拆分
        'src/infrastructure/config/storage/config_storage.py': 'src/infrastructure/config/storage/types/',

        # 安全模块拆分
        'src/infrastructure/config/security/enhanced_secure_config.py': 'src/infrastructure/config/security/components/',

        # 版本模块拆分
        'src/infrastructure/config/version/config_version_manager.py': 'src/infrastructure/config/version/components/',

        # 监控模块优化（已有拆分，不再拆分）
        # 'src/infrastructure/config/monitoring/performance_monitor_dashboard.py': None,

        # 测试文件保持原样（测试代码可以相对集中）
        # 'src/infrastructure/config/tests/cloud_native_test_platform.py': None,
        # 'src/infrastructure/config/tests/edge_computing_test_platform.py': None,
    }

    total_split = 0
    total_files = 0

    for source_file, target_dir in split_strategies.items():
        if not os.path.exists(source_file):
            print(f'❌ 源文件不存在: {source_file}')
            continue

        print(f'🔄 处理: {os.path.basename(source_file)}')

        if target_dir:
            split_files = split_file_by_classes(source_file, target_dir)
            if split_files:
                total_split += 1
                total_files += len(split_files)
                print(f'   📊 拆分为 {len(split_files)} 个文件')
            else:
                print(f'   ⚠️  拆分失败或不需要拆分')
        else:
            print(f'   ⏭️  跳过拆分（按策略）')

        print()

    # 处理测试文件的优化（不完全拆分，只提取公共部分）
    print('🔄 优化测试文件结构...')

    test_files = [
        'src/infrastructure/config/tests/cloud_native_test_platform.py',
        'src/infrastructure/config/tests/edge_computing_test_platform.py'
    ]

    for test_file in test_files:
        if os.path.exists(test_file):
            # 检查文件大小，如果仍然很大则进行轻量级拆分
            size_kb = os.path.getsize(test_file) / 1024
            if size_kb > 25:
                print(f'   📏 {os.path.basename(test_file)} 仍然较大 ({size_kb:.1f}KB)，建议后续优化')

    print()
    print(f'🎯 大文件拆分完成！')
    print(f'   📊 成功拆分: {total_split} 个文件')
    print(f'   📁 新建文件: {total_files} 个')
    print()

    # 验证拆分结果
    print('✅ 验证拆分结果:')
    large_files_check = [
        'src/infrastructure/config/storage/config_storage.py',
        'src/infrastructure/config/security/enhanced_secure_config.py',
        'src/infrastructure/config/version/config_version_manager.py'
    ]

    remaining_large = 0
    for file_path in large_files_check:
        if os.path.exists(file_path):
            size_kb = os.path.getsize(file_path) / 1024
            if size_kb > 20:
                print(f'   ⚠️  {os.path.basename(file_path)}: {size_kb:.1f} KB (仍需优化)')
                remaining_large += 1
            else:
                print(f'   ✅ {os.path.basename(file_path)}: {size_kb:.1f} KB')

    if remaining_large == 0:
        print('   🎉 所有大文件已成功拆分！')
    else:
        print(f'   📋 还有 {remaining_large} 个文件需要进一步优化')


if __name__ == '__main__':
    split_large_files()
