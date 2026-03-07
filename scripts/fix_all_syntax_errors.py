#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复所有测试文件的语法错误
专门处理pytest装饰器的位置错误
"""

from pathlib import Path


def fix_syntax_errors(root_path: str) -> dict:
    """修复所有语法错误"""
    results = {
        'fixed': [],
        'errors': [],
        'skipped': []
    }

    root_dir = Path(root_path)

    # 查找所有测试文件
    for py_file in root_dir.rglob('test_*.py'):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(py_file, 'r', encoding='latin-1') as f:
                    content = f.read()
            except UnicodeDecodeError:
                results['errors'].append(f"{py_file.name}: Unicode decode error")
                continue

        # 检查是否有语法错误
        original_content = content
        modified = False

        # 修复装饰器位置错误
        content = fix_decorator_position(content)
        if content != original_content:
            modified = True

        # 修复其他常见的语法错误
        content = fix_common_syntax_errors(content)
        if content != original_content and not modified:
            modified = True

        if modified:
            try:
                with open(py_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                results['fixed'].append(py_file.name)
                print(f"Fixed: {py_file.name}")
            except Exception as e:
                results['errors'].append(f"{py_file.name}: Write error - {e}")
        else:
            results['skipped'].append(py_file.name)

    return results


def fix_decorator_position(content: str) -> str:
    """修复装饰器位置错误"""
    lines = content.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # 查找类定义
        if line.startswith('class ') and 'Test' in line:
            # 检查下一行是否是装饰器
            if i + 1 < len(lines) and '@pytest.mark.timeout(' in lines[i + 1]:
                # 找到错误位置，修复它
                timeout_line = lines[i + 1]
                # 移除错误的装饰器行
                del lines[i + 1]
                # 在类定义前插入装饰器
                lines.insert(i, timeout_line)
                lines.insert(i + 1, '')  # 添加空行
                i += 2  # 跳过新添加的行
                continue

        i += 1

    return '\n'.join(lines)


def fix_common_syntax_errors(content: str) -> str:
    """修复其他常见的语法错误"""
    # 这里可以添加其他语法错误修复逻辑
    return content


def verify_fixes(results: dict) -> dict:
    """验证修复结果"""
    verification_results = {
        'successful': [],
        'still_broken': [],
        'unable_to_verify': []
    }

    import subprocess
    import sys

    for file_name in results['fixed']:
        try:
            # 尝试导入文件来验证语法
            file_path = f"tests/unit/infrastructure/{file_name}"
            result = subprocess.run([
                sys.executable, '-c', f"import ast; ast.parse(open('{file_path}').read())"
            ], capture_output=True, timeout=10)

            if result.returncode == 0:
                verification_results['successful'].append(file_name)
            else:
                verification_results['still_broken'].append(file_name)

        except subprocess.TimeoutExpired:
            verification_results['unable_to_verify'].append(file_name)
        except Exception as e:
            verification_results['unable_to_verify'].append(file_name)

    return verification_results


def main():
    """主函数"""
    print("开始修复基础设施层测试文件的语法错误...")

    # 修复语法错误
    results = fix_syntax_errors("tests/unit/infrastructure")

    print("\n修复结果:")
    print(f"  - 成功修复: {len(results['fixed'])} 个文件")
    print(f"  - 修复失败: {len(results['errors'])} 个文件")
    print(f"  - 无需修复: {len(results['skipped'])} 个文件")

    if results['errors']:
        print("\n修复失败的文件:")
        for error in results['errors']:
            print(f"  - {error}")

    # 验证修复结果
    if results['fixed']:
        print("\n验证修复结果...")
        verification = verify_fixes(results)
        print(f"  - 验证成功: {len(verification['successful'])} 个文件")
        print(f"  - 仍存在问题: {len(verification['still_broken'])} 个文件")

        if verification['still_broken']:
            print("  - 仍存在问题的文件:")
            for file_name in verification['still_broken']:
                print(f"    - {file_name}")

    print("\n建议:")
    print("- 重新运行测试收集以验证修复效果")
    print("- 如果仍有问题，可能需要手动检查和修复")


if __name__ == "__main__":
    main()
