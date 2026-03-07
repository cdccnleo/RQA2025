#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复测试文件中的缩进问题
"""

import os


def fix_indentation(file_path):
    """修复单个文件中的缩进问题"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        fixed_lines = []
        in_class = False
        in_method = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                # 空行保持原样
                fixed_lines.append(line)
                continue

            # 检查是否是类定义
            if stripped.startswith('class '):
                in_class = True
                in_method = False
                fixed_lines.append(line)
            # 检查是否是方法定义
            elif stripped.startswith('def ') and in_class:
                in_method = True
                # 方法定义应该有4个空格缩进
                if not line.startswith('    '):
                    line = '    ' + line.lstrip()
                fixed_lines.append(line)
            # 检查是否是其他代码行
            elif in_class and in_method and not stripped.startswith('#') and not stripped.startswith('@'):
                # 方法内的代码应该有8个空格缩进
                if not line.startswith('        ') and line.startswith('    '):
                    line = '    ' + line
                elif not line.startswith('    ') and not line.startswith('\t'):
                    line = '        ' + line.lstrip()
                fixed_lines.append(line)
            else:
                fixed_lines.append(line)

        # 写入修复后的内容
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(fixed_lines)

        print(f'✅ 修复了文件缩进: {file_path}')
        return True

    except Exception as e:
        print(f'❌ 处理文件 {file_path} 时出错: {e}')
        return False


def main():
    """主函数"""
    files_to_fix = [
        'tests/unit/infrastructure/utils/test_dynamic_executor.py',
        'tests/unit/infrastructure/utils/test_file_system.py',
        'tests/unit/infrastructure/utils/test_utils.py'
    ]

    fixed_count = 0
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            if fix_indentation(file_path):
                fixed_count += 1
        else:
            print(f'⚠️  文件不存在: {file_path}')

    print(f"\n修复了 {fixed_count} 个文件的缩进")

    # 验证修复结果
    print("\n验证修复结果:")
    import ast
    for file_path in files_to_fix:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            ast.parse(source)
            print(f'✅ {file_path}: 语法正确')
        except SyntaxError as e:
            print(f'❌ {file_path}: 仍存在语法错误')
        except Exception as e:
            print(f'❌ {file_path}: 其他错误 - {e}')


if __name__ == "__main__":
    main()
