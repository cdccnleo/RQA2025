#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 大文件查找脚本

查找项目中的大文件
"""

from pathlib import Path


def find_large_files():
    """查找大文件"""
    print("=== RQA2025 大文件查找 ===")
    print()

    project_root = Path(__file__).parent.parent
    threshold = 1000  # 1000行阈值

    print(f"查找大于 {threshold} 行的Python文件...")
    print()

    large_files = []

    for py_file in project_root.rglob("*.py"):
        # 排除venv和__pycache__目录
        if "venv" in str(py_file) or "__pycache__" in str(py_file):
            continue

        try:
            with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                line_count = sum(1 for line in f)

            if line_count >= threshold:
                large_files.append((py_file.relative_to(project_root), line_count))

        except Exception as e:
            print(f"Error reading {py_file}: {e}")

    # 按行数排序
    large_files.sort(key=lambda x: x[1], reverse=True)

    if large_files:
        print("发现的大文件:")
        print("-" * 50)
        for i, (file_path, line_count) in enumerate(large_files, 1):
            print("2d")
        print("-" * 50)
        print(f"总计: {len(large_files)} 个大文件")
        return True
    else:
        print("✅ 没有发现大文件")
        return False


if __name__ == "__main__":
    needs_refactoring = find_large_files()
    exit(0 if not needs_refactoring else 1)
