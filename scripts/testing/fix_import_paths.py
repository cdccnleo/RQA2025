#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 基础设施层测试导入路径修复脚本
批量修复测试文件中的导入路径问题
"""

import re
from pathlib import Path
from typing import List, Tuple


def find_test_files() -> List[Path]:
    """查找所有基础设施层测试文件"""
    project_root = Path(__file__).parent.parent.parent
    test_dir = project_root / "tests" / "unit" / "infrastructure"

    test_files = []
    for file_path in test_dir.rglob("*.py"):
        if file_path.name.startswith("test_") or file_path.name.startswith("conftest"):
            test_files.append(file_path)

    return test_files


def fix_import_paths(file_path: Path) -> Tuple[bool, List[str]]:
    """修复单个文件中的导入路径"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        changes = []

        # 修复 from infrastructure. 开头的导入
        pattern1 = r'from infrastructure\.(.*?) import (.*)'
        matches1 = re.findall(pattern1, content)

        for match in matches1:
            old_import = f"from infrastructure.{match[0]} import {match[1]}"
            new_import = f"from src.infrastructure.{match[0]} import {match[1]}"
            content = content.replace(old_import, new_import)
            changes.append(f"  {old_import} → {new_import}")

        # 修复 import infrastructure. 开头的导入
        pattern2 = r'import infrastructure\.(.*)'
        matches2 = re.findall(pattern2, content)

        for match in matches2:
            old_import = f"import infrastructure.{match}"
            new_import = f"import src.infrastructure.{match}"
            content = content.replace(old_import, new_import)
            changes.append(f"  {old_import} → {new_import}")

        # 如果有变化，写回文件
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, changes

        return False, []

    except Exception as e:
        return False, [f"错误: {e}"]


def main():
    """主函数"""
    print("🔧 开始修复基础设施层测试导入路径...")
    print("=" * 60)

    # 查找测试文件
    test_files = find_test_files()
    print(f"找到 {len(test_files)} 个测试文件")

    # 统计修复结果
    fixed_files = 0
    total_changes = 0

    # 修复每个文件
    for file_path in test_files:
        print(f"\n📄 处理文件: {file_path.relative_to(Path(__file__).parent.parent.parent)}")

        fixed, changes = fix_import_paths(file_path)

        if fixed:
            fixed_files += 1
            total_changes += len(changes)
            print(f"  ✅ 已修复 {len(changes)} 个导入路径:")
            for change in changes:
                print(change)
        else:
            if changes:  # 有错误
                print(f"  ❌ 修复失败:")
                for change in changes:
                    print(change)
            else:
                print("  ℹ️  无需修复")

    # 输出总结
    print("\n" + "=" * 60)
    print("📊 修复完成总结:")
    print(f"  总文件数: {len(test_files)}")
    print(f"  修复文件数: {fixed_files}")
    print(f"  总修复数: {total_changes}")

    if fixed_files > 0:
        print(f"\n✅ 成功修复了 {fixed_files} 个文件中的 {total_changes} 个导入路径")
        print("现在可以重新运行基础设施层测试了")
    else:
        print("\nℹ️  没有发现需要修复的导入路径")


if __name__ == "__main__":
    main()
