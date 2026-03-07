#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全面修正所有 m_logging 导入路径
"""

import re
from pathlib import Path


def fix_imports_in_file(file_path):
    """修正单个文件中的导入路径"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # 修正各种导入模式
        patterns = [
            (r'from src\.infrastructure\.m_logging\.', 'from src.infrastructure.logging.'),
            (r'import src\.infrastructure\.m_logging\.', 'import src.infrastructure.logging.'),
            (r'@patch\(\'src\.infrastructure\.m_logging\.', '@patch(\'src.infrastructure.logging.'),
            (r'from infrastructure\.m_logging\.', 'from src.infrastructure.logging.'),
            (r'import infrastructure\.m_logging\.', 'import src.infrastructure.logging.'),
            (r'@patch\(\'infrastructure\.m_logging\.', '@patch(\'src.infrastructure.logging.'),
            (r'from m_logging\.', 'from src.infrastructure.logging.'),
            (r'import m_logging\.', 'import src.infrastructure.logging.'),
            (r'@patch\(\'m_logging\.', '@patch(\'src.infrastructure.logging.'),
        ]

        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)

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

    # 查找所有 Python 文件
    python_files = []
    for py_file in project_root.rglob("*.py"):
        python_files.append(py_file)

    fixed_count = 0
    total_files = len(python_files)

    for py_file in python_files:
        if fix_imports_in_file(py_file):
            fixed_count += 1

    print(f"\n处理完成:")
    print(f"总文件数: {total_files}")
    print(f"修正文件数: {fixed_count}")


if __name__ == "__main__":
    find_and_fix_files()
