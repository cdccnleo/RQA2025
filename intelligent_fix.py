#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能修复测试文件中的重复方法定义和缩进问题
"""

import re
import os


def fix_duplicate_methods_and_indentation(file_path):
    """智能修复测试文件中的重复方法定义和缩进问题"""

    try:
        # 读取文件
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')

        fixed_lines = []
        i = 0

        while i < len(lines):
            line = lines[i].strip()
            original_line = lines[i]

            # 处理类定义
            if line.startswith('class '):
                fixed_lines.append(original_line)
                i += 1
                continue

            # 处理方法定义
            method_match = re.match(r'^(\s*)def\s+(\w+)\s*\(', original_line)
            if method_match:
                indent = method_match.group(1)
                method_name = method_match.group(2)

                # 检查是否是重复的方法定义
                # 查找下一个同名方法定义
                duplicate_found = False
                for j in range(i + 1, min(i + 10, len(lines))):  # 检查接下来的几行
                    next_line = lines[j]
                    if re.match(rf'^\s*def\s+{method_name}\s*\(', next_line):
                        # 找到重复的方法定义，跳过这个重复的
                        duplicate_found = True
                        break

                if not duplicate_found:
                    # 检查文档字符串是否在同一行
                    if '"""' in original_line:
                        # 分离方法定义和文档字符串
                        method_def = re.sub(r'\s*""".*?"""\s*$', '', original_line)
                        docstring = re.search(r'"""(.*?)"""', original_line)
                        if docstring:
                            fixed_lines.append(method_def)
                            fixed_lines.append(f"{indent}    \"\"\"{docstring.group(1)}\"\"\"")
                        else:
                            fixed_lines.append(original_line)
                    else:
                        fixed_lines.append(original_line)
                # 如果找到重复，跳过这个重复的定义

            else:
                # 处理方法体内容
                if line and not line.startswith(' ') and not line.startswith('\t') and not line.startswith('#'):
                    # 非空行且不是注释，可能需要检查缩进
                    fixed_lines.append(original_line)
                else:
                    fixed_lines.append(original_line)

            i += 1

        # 第二次处理：修复pytestmark格式
        content = '\n'.join(fixed_lines)

        # 查找并修复pytestmark
        if 'pytestmark = [' in content:
            # 确保pytestmark格式正确
            content = re.sub(r'pytestmark\s*=\s*\[([^\]]*)\]',
                             r'pytestmark = [\1]', content, flags=re.DOTALL)

        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f'Fixed {file_path}')
        return True

    except Exception as e:
        print(f'Error fixing {file_path}: {e}')
        return False


def process_test_files(root_dir, limit=None):
    """处理测试文件"""

    test_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.startswith('test_') and file.endswith('.py'):
                test_files.append(os.path.join(root, file))

    # 按优先级排序
    priority_files = [
        'tests/unit/infrastructure/base/test_base.py',
        'tests/unit/infrastructure/cache/test_unified_cache.py',
        'tests/unit/infrastructure/cache/test_multi_level_cache.py',
        'tests/unit/infrastructure/config/test_config_system.py',
        'tests/unit/infrastructure/config/test_unified_config_manager.py',
    ]

    # 将优先级文件排在前面
    sorted_files = []
    for pf in priority_files:
        if os.path.exists(pf):
            sorted_files.append(pf)

    # 添加其他文件
    for tf in test_files:
        if tf not in sorted_files:
            sorted_files.append(tf)

    # 限制处理数量
    if limit:
        sorted_files = sorted_files[:limit]

    print(f'Processing {len(sorted_files)} test files...')

    fixed_count = 0
    for file_path in sorted_files:
        if fix_duplicate_methods_and_indentation(file_path):
            fixed_count += 1

    print(f'Successfully processed {fixed_count}/{len(sorted_files)} files')


if __name__ == '__main__':
    # 处理基础设施层的测试文件
    process_test_files('tests/unit/infrastructure', limit=10)  # 先处理10个文件测试效果
