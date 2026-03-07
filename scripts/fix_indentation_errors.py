#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复缩进错误和其他语法问题
"""

from pathlib import Path


def fix_indentation_issues(root_path: str) -> int:
    """修复缩进问题"""
    fixed_count = 0
    root_dir = Path(root_path)

    for py_file in root_dir.rglob('test_*.py'):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(py_file, 'r', encoding='latin-1') as f:
                    content = f.read()
            except UnicodeDecodeError:
                continue

        original_content = content
        content = fix_class_indentation(content)

        if content != original_content:
            try:
                with open(py_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                fixed_count += 1
                print(f"Fixed indentation: {py_file.name}")
            except Exception as e:
                print(f"Failed to write {py_file}: {e}")

    return fixed_count


def fix_class_indentation(content: str) -> str:
    """修复类缩进问题"""
    lines = content.split('\n')
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # 检查装饰器后是否紧跟类定义
        if line.strip().startswith('@pytest.mark.timeout('):
            result.append(line)
            i += 1

            # 检查下一行
            if i < len(lines):
                next_line = lines[i]
                # 如果下一行是空行，跳过
                if next_line.strip() == '':
                    result.append(next_line)
                    i += 1
                    if i < len(lines):
                        next_line = lines[i]

                # 如果下一行是类定义，确保它有正确的缩进
                if next_line.strip().startswith('class ') and 'Test' in next_line.strip():
                    # 检查当前缩进
                    current_indent = len(line) - len(line.lstrip())
                    class_indent = len(next_line) - len(next_line.lstrip())

                    # 如果类定义缩进不正确，修复它
                    if class_indent != current_indent + 4:  # 期望4个空格的缩进
                        expected_indent = ' ' * (current_indent + 4)
                        fixed_class_line = expected_indent + next_line.strip()
                        result.append(fixed_class_line)
                    else:
                        result.append(next_line)
                else:
                    result.append(next_line)
            i += 1
        else:
            result.append(line)
            i += 1

    return '\n'.join(result)


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
            ], capture_output=True, timeout=5)

            if result.returncode == 0:
                results['valid'].append(py_file.name)
            else:
                results['invalid'].append(py_file.name)
                print(f"Invalid syntax in {py_file.name}:")
                if result.stderr:
                    print(result.stderr.decode('utf-8', errors='ignore'))

        except subprocess.TimeoutExpired:
            results['errors'].append(f"{py_file.name}: timeout")
        except Exception as e:
            results['errors'].append(f"{py_file.name}: {e}")

    return results


def main():
    """主函数"""
    print("开始修复基础设施层测试文件的缩进问题...")

    # 修复缩进问题
    fixed_count = fix_indentation_issues("tests/unit/infrastructure")
    print(f"修复了 {fixed_count} 个文件的缩进问题")

    # 验证语法
    print("\n验证语法正确性...")
    syntax_results = verify_syntax("tests/unit/infrastructure")

    print(f"语法验证结果:")
    print(f"  - 语法正确: {len(syntax_results['valid'])} 个文件")
    print(f"  - 语法错误: {len(syntax_results['invalid'])} 个文件")
    print(f"  - 验证失败: {len(syntax_results['errors'])} 个文件")

    if syntax_results['invalid']:
        print("\n语法有问题的文件:")
        for file_name in syntax_results['invalid'][:10]:  # 只显示前10个
            print(f"  - {file_name}")

    print("\n建议:")
    print("- 重新运行测试收集以验证修复效果")
    print("- 如果仍有语法错误，可能需要手动检查")


if __name__ == "__main__":
    main()
