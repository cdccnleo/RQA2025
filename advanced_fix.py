#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级测试文件修复工具 - 专门处理缩进和语法错误
"""

import re
import ast
import os


def fix_indentation_issues(content):
    """修复缩进问题"""

    lines = content.split('\n')
    fixed_lines = []
    indent_stack = []  # 跟踪缩进级别

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # 跳过空行
        if not stripped:
            fixed_lines.append(line)
            i += 1
            continue

        # 处理类定义
        if stripped.startswith('class '):
            # 重置缩进栈
            indent_stack = [0]
            fixed_lines.append(line)

        # 处理方法定义
        elif re.match(r'^\s*def\s+\w+\s*\(', line):
            # 方法定义应该在类内部，缩进为4个空格
            if indent_stack and indent_stack[-1] == 0:  # 在类中
                if not line.startswith('    '):
                    line = '    ' + line.lstrip()
            fixed_lines.append(line)

        # 处理控制流语句 (if, for, while, try, etc.)
        elif re.match(r'^\s*(if|for|while|try|except|finally|with|else|elif)\s', line):
            # 这些语句应该在方法内部，缩进为8个空格 (类缩进4 + 方法缩进4)
            if not line.startswith('        '):
                # 计算当前应该的缩进级别
                current_indent = len(line) - len(line.lstrip())
                if current_indent < 8:
                    line = '        ' + line.lstrip()
            fixed_lines.append(line)

        # 处理普通代码行
        elif stripped and not stripped.startswith('#'):
            # 检查是否在控制流块内
            current_indent = len(line) - len(line.lstrip())

            # 如果缩进少于8个空格，可能是方法内的代码
            if current_indent < 8 and current_indent > 0:
                # 检查前一行是否是控制流语句
                if fixed_lines and re.match(r'^\s*(if|for|while|try|with)\s.*:$', fixed_lines[-1]):
                    # 需要增加缩进
                    additional_indent = 4
                    line = '        ' + line.lstrip()
                elif not line.startswith('        '):
                    # 方法内部的普通代码
                    line = '        ' + line.lstrip()

            fixed_lines.append(line)

        else:
            fixed_lines.append(line)

        i += 1

    return '\n'.join(fixed_lines)


def fix_duplicate_methods(content):
    """修复重复的方法定义"""

    lines = content.split('\n')
    fixed_lines = []
    seen_methods = set()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # 检查方法定义
        method_match = re.match(r'^\s*def\s+(\w+)\s*\(', line)
        if method_match:
            method_name = method_match.group(1)

            if method_name in seen_methods:
                # 跳过重复的方法定义
                # 找到下一个方法定义或类定义
                while i < len(lines):
                    next_line = lines[i]
                    if re.match(r'^\s*(def\s+\w+|class\s+\w+)', next_line):
                        break
                    i += 1
                continue
            else:
                seen_methods.add(method_name)
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)

        i += 1

    return '\n'.join(fixed_lines)


def fix_docstring_format(content):
    """修复文档字符串格式"""

    # 将同一行的方法定义和文档字符串分离
    content = re.sub(
        r'(\n\s*def\s+\w+\s*\([^)]*\))\s*:\s*("""[^"]*""")',
        r'\1:\n        \2',
        content
    )

    return content


def comprehensive_fix_file(file_path):
    """综合修复测试文件"""

    try:
        # 读取文件
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        print(f'Processing {file_path}...')

        # 1. 修复重复方法定义
        content = fix_duplicate_methods(content)
        print('  ✓ Fixed duplicate methods')

        # 2. 修复文档字符串格式
        content = fix_docstring_format(content)
        print('  ✓ Fixed docstring format')

        # 3. 修复缩进问题
        content = fix_indentation_issues(content)
        print('  ✓ Fixed indentation issues')

        # 4. 验证语法
        try:
            ast.parse(content)
            print('  ✓ Syntax validation passed')
        except SyntaxError as e:
            print(f'  ⚠️  Syntax error remains: {e}')
            return False

        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f'  ✅ Successfully fixed {file_path}')
        return True

    except Exception as e:
        print(f'  ❌ Error fixing {file_path}: {e}')
        return False


def process_core_files():
    """处理核心文件"""

    core_files = [
        'tests/unit/infrastructure/cache/test_unified_cache.py',
        'tests/unit/infrastructure/cache/test_multi_level_cache.py',
        'tests/unit/infrastructure/config/test_config_system.py',
        'tests/unit/infrastructure/config/test_unified_config_manager.py',
        'tests/unit/infrastructure/cache/test_cache_strategy.py',
        'tests/unit/infrastructure/cache/test_cache_simple_memory_cache.py'
    ]

    results = []
    for file_path in core_files:
        if os.path.exists(file_path):
            success = comprehensive_fix_file(file_path)
            results.append((file_path, success))
        else:
            print(f'⚠️  File not found: {file_path}')
            results.append((file_path, False))

    # 统计结果
    successful = sum(1 for _, success in results if success)
    total = len(results)

    print(f'\n📊 Fix Results: {successful}/{total} files successfully fixed')

    return results


if __name__ == '__main__':
    process_core_files()
