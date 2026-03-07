#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速修复基础设施层测试文件的缩进问题
"""

from pathlib import Path


def quick_fix_file(file_path):
    """快速修复单个文件的缩进问题"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        except UnicodeDecodeError:
            return False

    lines = content.split('\n')
    fixed_lines = []
    in_class = False

    for line in lines:
        stripped = line.lstrip()

        # 检测类定义
        if stripped.startswith('class ') and 'Test' in stripped:
            in_class = True
            fixed_lines.append(line)  # 保持类定义不变
        elif in_class and stripped and not stripped.startswith('#'):
            # 类内的非空行且不是注释
            if stripped.startswith('def '):
                # 方法定义
                if not line.startswith('    '):
                    fixed_lines.append('    ' + stripped)
                else:
                    fixed_lines.append(line)
            elif stripped.startswith('"""') or stripped.startswith("'''"):
                # 文档字符串
                if not line.startswith('    ') and not line.startswith('        '):
                    fixed_lines.append('        ' + stripped)
                else:
                    fixed_lines.append(line)
            else:
                # 其他代码行
                if not line.startswith('    ') and not line.startswith('        '):
                    fixed_lines.append('        ' + stripped)
                else:
                    fixed_lines.append(line)
        else:
            fixed_lines.append(line)

    new_content = '\n'.join(fixed_lines)

    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True

    return False


def fix_all_files(root_path):
    """修复所有文件"""
    root_dir = Path(root_path)
    fixed_count = 0

    for py_file in root_dir.rglob('test_*.py'):
        try:
            if quick_fix_file(str(py_file)):
                fixed_count += 1
                print(f"Fixed: {py_file.name}")
        except Exception as e:
            print(f"Error fixing {py_file.name}: {e}")

    return fixed_count


def validate_files(root_path):
    """验证文件语法"""
    import subprocess
    import sys

    root_dir = Path(root_path)
    valid_count = 0
    total_count = 0

    for py_file in root_dir.rglob('test_*.py'):
        total_count += 1
        try:
            result = subprocess.run([
                sys.executable, '-m', 'py_compile', str(py_file)
            ], capture_output=True, timeout=5)

            if result.returncode == 0:
                valid_count += 1
            else:
                print(f"Invalid: {py_file.name}")

        except subprocess.TimeoutExpired:
            print(f"Timeout: {py_file.name}")
        except Exception as e:
            print(f"Error: {py_file.name} - {e}")

    return valid_count, total_count


def main():
    """主函数"""
    print("开始快速修复基础设施层测试文件的缩进问题...")

    # 修复文件
    fixed_count = fix_all_files("tests/unit/infrastructure")
    print(f"\n修复了 {fixed_count} 个文件的缩进问题")

    # 验证结果
    print("\n验证修复结果...")
    valid_count, total_count = validate_files("tests/unit/infrastructure")

    print(f"验证结果: {valid_count}/{total_count} 个文件语法正确")
    success_rate = valid_count / total_count * 100 if total_count > 0 else 0
    print(".1f")
    if success_rate > 90:
        print("\n✅ 大部分文件已修复！可以重新运行测试收集。")
    else:
        print("\n⚠️ 仍有一些文件需要手动修复。")


if __name__ == "__main__":
    main()
