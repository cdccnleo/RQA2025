#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量修复所有基础设施层测试文件的缩进错误
"""

import os
import re
from pathlib import Path


def fix_indentation_in_file(file_path: str) -> bool:
    """修复单个文件的缩进问题"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        except UnicodeDecodeError:
            return False

    original_content = content
    lines = content.split('\n')
    fixed_lines = []
    in_class = False
    class_indent = 0

    for i, line in enumerate(lines):
        # 检测类定义
        if line.strip().startswith('class ') and 'Test' in line.strip():
            in_class = True
            class_indent = len(line) - len(line.lstrip())
            fixed_lines.append(line)
        elif in_class:
            # 检查方法定义
            if line.strip().startswith('def ') and not line.startswith('    '):
                # 方法没有正确缩进
                fixed_lines.append('    ' + line.lstrip())
            elif line.strip().startswith('"""') and not line.startswith('        ') and not line.startswith('    '):
                # 文档字符串没有正确缩进
                if i > 0 and lines[i-1].strip().startswith('def '):
                    # 这是方法文档字符串
                    fixed_lines.append('        ' + line.lstrip())
                else:
                    # 这是类文档字符串
                    fixed_lines.append('    ' + line.lstrip())
            elif line.strip() and not line.startswith('    '):
                # 其他类内内容没有正确缩进
                fixed_lines.append('    ' + line.lstrip())
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)

    new_content = '\n'.join(fixed_lines)

    if new_content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True

    return False


def fix_all_indentations(root_path: str) -> dict:
    """修复所有文件的缩进问题"""
    results = {
        'fixed': [],
        'skipped': [],
        'errors': []
    }

    root_dir = Path(root_path)

    for py_file in root_dir.rglob('test_*.py'):
        try:
            if fix_indentation_in_file(str(py_file)):
                results['fixed'].append(py_file.name)
                print(f"Fixed: {py_file.name}")
            else:
                results['skipped'].append(py_file.name)
        except Exception as e:
            results['errors'].append(f"{py_file.name}: {e}")

    return results


def verify_syntax(root_path: str) -> dict:
    """验证语法正确性"""
    import subprocess
    import sys

    results = {
        'valid': [],
        'invalid': [],
        'errors': []
    }

    root_dir = Path(root_path)

    for py_file in root_dir.rglob('test_*.py'):
        try:
            result = subprocess.run([
                sys.executable, '-m', 'py_compile', str(py_file)
            ], capture_output=True, timeout=10)

            if result.returncode == 0:
                results['valid'].append(py_file.name)
            else:
                results['invalid'].append(py_file.name)

        except subprocess.TimeoutExpired:
            results['errors'].append(f"{py_file.name}: timeout")
        except Exception as e:
            results['errors'].append(f"{py_file.name}: {e}")

    return results


def main():
    """主函数"""
    print("开始修复基础设施层测试文件的缩进问题...")

    # 修复缩进问题
    results = fix_all_indentations("tests/unit/infrastructure")

    print("\n修复结果:")
    print(f"  - 成功修复: {len(results['fixed'])} 个文件")
    print(f"  - 无需修复: {len(results['skipped'])} 个文件")
    print(f"  - 修复失败: {len(results['errors'])} 个文件")

    if results['errors']:
        print("\n修复失败的文件:")
        for error in results['errors']:
            print(f"  - {error}")

    # 验证语法
    print("\n验证语法正确性...")
    syntax_results = verify_syntax("tests/unit/infrastructure")

    print(f"语法验证结果:")
    print(f"  - 语法正确: {len(syntax_results['valid'])} 个文件")
    print(f"  - 语法错误: {len(syntax_results['invalid'])} 个文件")
    print(f"  - 验证失败: {len(syntax_results['errors'])} 个文件")

    if syntax_results['invalid']:
        print("\n仍存在语法错误的文件的文件:")
        for file_name in syntax_results['invalid'][:10]:  # 只显示前10个
            print(f"  - {file_name}")

    total_files = len(syntax_results['valid']) + len(syntax_results['invalid'])
    if total_files > 0:
        success_rate = len(syntax_results['valid']) / total_files * 100
        print(".1f"
    print("\n建议:")
    if success_rate > 80:
        print("- 大部分文件已修复，可以重新运行测试收集")
    else:
        print("- 仍有很多文件需要手动修复，请检查具体错误")
    print("- 考虑使用专业的代码格式化工具如black或autopep8")


if __name__ == "__main__":
    main()
