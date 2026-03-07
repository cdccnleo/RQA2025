#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修正测试文件中的 m_logging 导入路径
将所有 src.infrastructure.m_logging 改为 src.infrastructure.logging
"""

import re
from pathlib import Path


def fix_imports_in_file(file_path):
    """修正单个文件中的导入路径"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 修正导入路径
        original_content = content
        content = re.sub(
            r'from src\.infrastructure\.m_logging\.',
            'from src.infrastructure.logging.',
            content
        )
        content = re.sub(
            r'import src\.infrastructure\.m_logging\.',
            'import src.infrastructure.logging.',
            content
        )
        content = re.sub(
            r'@patch\(\'src\.infrastructure\.m_logging\.',
            '@patch(\'src.infrastructure.logging.',
            content
        )

        # 如果内容有变化，写回文件
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"已修正: {file_path}")
            return True
        return False
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return False


def find_and_fix_files():
    """查找并修正所有包含 m_logging 导入的文件"""
    project_root = Path(".")
    test_dir = project_root / "tests"

    if not test_dir.exists():
        print("未找到 tests 目录")
        return

    fixed_count = 0
    total_files = 0

    # 查找所有 Python 文件
    for py_file in test_dir.rglob("*.py"):
        total_files += 1
        if fix_imports_in_file(py_file):
            fixed_count += 1

    print(f"\n处理完成:")
    print(f"总文件数: {total_files}")
    print(f"修正文件数: {fixed_count}")


if __name__ == "__main__":
    find_and_fix_files()
