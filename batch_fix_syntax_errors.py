#!/usr/bin/env python3
"""
批量修复基础设施层测试文件的语法错误
主要处理重复方法定义和缩进问题
"""

import ast


def fix_duplicate_methods_and_indent(content):
    """修复重复方法定义和缩进问题"""
    lines = content.split('\n')
    fixed_lines = []
    skip_next = False

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        # 跳过空行
        if not line:
            fixed_lines.append('')
            i += 1
            continue

        # 处理类定义
        if line.startswith('class '):
            fixed_lines.append(line)
            i += 1
            continue

        # 处理方法定义
        if line.startswith('    def '):
            method_name = line.split('(')[0].strip()

            # 检查是否有重复的方法定义
            if i + 1 < len(lines):
                next_line = lines[i + 1].rstrip()
                if (next_line.startswith('    def ') and
                    next_line.split('(')[0].strip() == method_name and
                        '"""' in next_line):
                    # 跳过空的方法定义，保留有文档字符串的版本
                    i += 1
                    continue

            fixed_lines.append(line)
            i += 1
            continue

        # 处理其他行
        fixed_lines.append(line)
        i += 1

    return '\n'.join(fixed_lines)


def fix_indentation_issues(content):
    """修复缩进问题"""
    lines = content.split('\n')
    fixed_lines = []
    indent_stack = []

    for line in lines:
        stripped = line.rstrip()
        if not stripped:
            fixed_lines.append('')
            continue

        current_indent = len(line) - len(line.lstrip())

        # 处理类和函数定义
        if stripped.startswith('class ') or stripped.startswith('def '):
            # 重置缩进栈
            while indent_stack:
                indent_stack.pop()
            indent_stack.append(current_indent)
            fixed_lines.append(line)
        # 处理控制结构
        elif any(stripped.startswith(keyword) for keyword in ['if ', 'for ', 'while ', 'try:', 'except', 'with ', 'elif ', 'else:']):
            indent_stack.append(current_indent)
            fixed_lines.append(line)
        # 处理其他行
        else:
            # 计算期望的缩进
            expected_indent = indent_stack[-1] + 4 if indent_stack else 0

            # 如果缩进不正确，修复它
            if current_indent != expected_indent and indent_stack:
                fixed_line = ' ' * expected_indent + stripped.lstrip()
                fixed_lines.append(fixed_line)
            else:
                fixed_lines.append(line)

            # 处理控制结构的结束
            if stripped.startswith(('return ', 'break', 'continue', 'pass', 'raise ')) or stripped.endswith(':'):
                if indent_stack and current_indent <= indent_stack[-1]:
                    indent_stack.pop()

    return '\n'.join(fixed_lines)


def fix_file(file_path):
    """修复单个文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 首先修复重复方法定义
        content = fix_duplicate_methods_and_indent(content)

        # 然后修复缩进问题
        content = fix_indentation_issues(content)

        # 验证语法
        ast.parse(content)

        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return True, None

    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def main():
    """主函数"""
    print("开始批量修复基础设施层测试文件的语法错误...")

    # 获取所有错误的文件（先处理前5个测试）
    error_files = [
        'tests/unit/infrastructure/cache/test_cache_client_sdk.py',
        'tests/unit/infrastructure/cache/test_cache_client_sdk_simple.py',
        'tests/unit/infrastructure/cache/test_cache_core_components.py',
        'tests/unit/infrastructure/cache/test_cache_dependency.py',
        'tests/unit/infrastructure/cache/test_cache_exceptions.py'
    ]

    print(f"需要修复 {len(error_files)} 个文件")

    fixed_count = 0
    failed_files = []

    for file_path in error_files:
        print(f"修复文件: {file_path}")
        success, error = fix_file(file_path)
        if success:
            fixed_count += 1
            print("  ✅ 修复成功")
        else:
            failed_files.append((file_path, error))
            print(f"  ❌ 修复失败: {error}")

    print("\n批量修复完成!")
    print(f"成功修复: {fixed_count} 个文件")
    if failed_files:
        print(f"修复失败: {len(failed_files)} 个文件")
        for file_path, error in failed_files:
            print(f"  - {file_path}: {error}")


if __name__ == "__main__":
    main()
