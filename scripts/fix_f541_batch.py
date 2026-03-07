#!/usr/bin/env python3
"""
批量修复F541 f-string问题的脚本
"""

import os
import re
from pathlib import Path


def fix_f541_in_file(file_path: Path) -> int:
    """修复单个文件中的F541问题"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # 查找所有f字符串并修复
        def replace_fstring(match):
            fstring_content = match.group(1)
            # 检查是否包含变量占位符
            if '{' in fstring_content and '}' in fstring_content:
                # 检查大括号是否匹配
                if fstring_content.count('{') == fstring_content.count('}'):
                    # 包含变量且语法正确，保持f前缀
                    return match.group(0)
                else:
                    # 语法错误，移除f前缀
                    return f'"{fstring_content}"'
            else:
                # 不包含变量，移除f前缀
                return f'"{fstring_content}"'

        # 处理双引号f字符串
        content = re.sub(r'f"([^"]*)"', replace_fstring, content)

        # 处理单引号f字符串
        def replace_fstring_single(match):
            fstring_content = match.group(1)
            if '{' in fstring_content and '}' in fstring_content:
                if fstring_content.count('{') == fstring_content.count('}'):
                    return match.group(0)
                else:
                    return f"'{fstring_content}'"
            else:
                return f"'{fstring_content}'"

        content = re.sub(r"f'([^']*)'", replace_fstring_single, content)

        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return 1

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")

    return 0


def find_files_with_f541():
    """查找包含F541问题的文件"""
    test_dir = Path(__file__).parent.parent / "tests"
    fixed_count = 0

    print("🔧 开始批量修复F541 f-string问题...")

    # 递归遍历tests目录
    for file_path in test_dir.rglob("*.py"):
        if fix_f541_in_file(file_path):
            fixed_count += 1
            print(f"✅ 已修复: {file_path}")

    print(f"\n📊 修复完成! 共修复了 {fixed_count} 个文件")
    return fixed_count


if __name__ == "__main__":
    find_files_with_f541()



