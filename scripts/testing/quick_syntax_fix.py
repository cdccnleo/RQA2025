#!/usr/bin/env python3
"""
快速语法错误修复工具

专门修复基础设施层Python文件中的缩进错误
"""

import os
import re
from pathlib import Path


def fix_return_dict_pattern(content):
    """修复return语句后的字典字面量"""
    # 模式1: return {} 后面直接跟字典键值对
    pattern1 = r'return \{\}\s*\n(\s*"[^"]*":\s*[^,]*,\s*\n)*(\s*"[^"]*":\s*[^}]*\s*\n)\s*\}'

    def replace_func(match):
        lines = match.group(0).split('\n')
        result_lines = []

        for i, line in enumerate(lines):
            if i == 0:  # return {} 行
                result_lines.append(line)
            elif line.strip() and not line.strip().startswith('}'):
                # 字典内容需要缩进
                stripped = line.strip()
                if stripped.startswith('"') and ':' in stripped:
                    result_lines.append('        ' + stripped)
                else:
                    result_lines.append(line)
            else:
                result_lines.append(line)

        return '\n'.join(result_lines)

    return re.sub(pattern1, replace_func, content, flags=re.MULTILINE)


def fix_file(file_path):
    """修复单个文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # 修复return {} 后面直接跟字典内容的模式
        content = fix_return_dict_pattern(content)

        # 如果内容有变化，写入文件
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True

        return False

    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False


def fix_all_infrastructure_files():
    """修复所有基础设施层Python文件的语法错误"""
    infrastructure_path = Path("src/infrastructure")
    fixed_count = 0

    if not infrastructure_path.exists():
        print("Infrastructure path not found")
        return

    # 遍历所有Python文件
    for root, dirs, files in os.walk(infrastructure_path):
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                if fix_file(file_path):
                    print(f"Fixed: {file_path}")
                    fixed_count += 1

    print(f"\nFixed {fixed_count} files")


if __name__ == "__main__":
    fix_all_infrastructure_files()
