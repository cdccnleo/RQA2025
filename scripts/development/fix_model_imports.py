#!/usr/bin/env python3
"""
快速修复模型层测试导入路径
"""

import re
from pathlib import Path


def fix_imports_in_file(file_path):
    """修复单个文件的导入路径"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 修复导入路径
    replacements = [
        (r'from models\.', 'from src.models.'),
        (r'from src\.model\.', 'from src.models.'),
    ]

    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"已修复: {file_path}")


def main():
    """主函数"""
    test_dir = Path("tests/unit/model")

    if not test_dir.exists():
        print("测试目录不存在")
        return

    # 修复所有模型测试文件
    for test_file in test_dir.glob("test_*.py"):
        fix_imports_in_file(test_file)

    print("模型层导入路径修复完成")


if __name__ == "__main__":
    main()
