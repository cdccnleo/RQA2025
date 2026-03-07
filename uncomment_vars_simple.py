#!/usr/bin/env python3
"""
简单的批量取消未使用变量注释脚本
"""

import os
import re


def process_file(filepath):
    """处理单个文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # 取消注释未使用的变量
        pattern = r'#\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+?)\s*# 未使用的变量'
        content = re.sub(pattern, r'\1 = \2', content)

        # 处理多行字典/列表等复杂表达式
        pattern_complex = r'#\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(\{.*?\}|\[.*?\])\s*# 未使用的变量'
        content = re.sub(pattern_complex, r'\1 = \2', content, flags=re.DOTALL)

        # 处理函数调用
        pattern_func = r'#\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([a-zA-Z_][a-zA-Z0-9_]*\(.*?\))\s*# 未使用的变量'
        content = re.sub(pattern_func, r'\1 = \2', content)

        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True

    except Exception as e:
        print(f"处理文件 {filepath} 时出错: {e}")

    return False


def main():
    """主函数"""
    src_dir = 'src'

    if not os.path.exists(src_dir):
        print(f"目录 {src_dir} 不存在")
        return

    total_files = 0
    processed_files = 0
    total_changes = 0

    # 遍历所有 Python 文件
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                total_files += 1

                print(f"处理文件: {filepath}")
                if process_file(filepath):
                    processed_files += 1
                    print("  已修改")
    print(f"\n处理完成:")
    print(f"- 总文件数: {total_files}")
    print(f"- 修改文件数: {processed_files}")

    # 检查是否还有剩余的未使用变量注释
    remaining_count = 0
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        remaining_count += content.count('# 未使用的变量')
                except:
                    pass

    if remaining_count > 0:
        print(f"- 剩余未处理的未使用变量注释: {remaining_count}")
    else:
        print("- 所有未使用变量注释已成功取消!")


if __name__ == "__main__":
    main()
