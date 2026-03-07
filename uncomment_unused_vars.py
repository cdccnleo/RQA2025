#!/usr/bin/env python3
"""
批量取消所有未使用变量注释的脚本
"""

import os
import re


def uncomment_unused_vars():
    """取消所有未使用变量注释"""

    # 查找所有 Python 文件
    python_files = []
    for root, dirs, files in os.walk('src'):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))

    total_files = len(python_files)
    processed_files = 0
    total_changes = 0

    print(f"找到 {total_files} 个 Python 文件")

    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content

            # 匹配并取消注释未使用的变量
            # 模式：# variable_name = ...  # 未使用的变量
            pattern = r'#\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+?)\s+# 未使用的变量'
            content = re.sub(pattern, r'\1 = \2', content)

            # 匹配并取消注释未使用的变量（多行情况）
            # 处理包含括号的复杂表达式
            pattern_complex = r'#\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(\{.*?\}|\[.*?\]|.*?)\s+# 未使用的变量'
            content = re.sub(pattern_complex, r'\1 = \2', content, flags=re.DOTALL)

            # 处理简单的变量赋值
            pattern_simple = r'#\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([^#\n]+)\s+# 未使用的变量'
            content = re.sub(pattern_simple, r'\1 = \2', content)

            # 处理函数调用等复杂表达式
            pattern_func = r'#\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([a-zA-Z_][a-zA-Z0-9_]*\(.*?\))\s+# 未使用的变量'
            content = re.sub(pattern_func, r'\1 = \2', content)

            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                changes_in_file = len(re.findall(r'# 未使用的变量', original_content))
                total_changes += changes_in_file
                processed_files += 1

                print(f"处理文件: {file_path} ({changes_in_file} 处修改)")

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")

    print("\n处理完成:")
    print(f"- 处理文件数: {processed_files}")
    print(f"- 总修改数: {total_changes}")


if __name__ == "__main__":
    uncomment_unused_vars()
