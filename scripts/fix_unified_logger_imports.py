#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复数据层中unified_logger直接导入问题的脚本
"""

import re
from pathlib import Path


def fix_unified_logger_imports():
    """修复unified_logger直接导入问题"""

    data_src_path = Path("src/data")

    # 需要修复的导入模式
    old_patterns = [
        r'from src\.infrastructure\.logging\.unified_logger import UnifiedLogger',
        r'from src\.infrastructure\.logging\.unified_logger import get_logger',
        r'from src\.infrastructure\.logging\.unified_logger import UnifiedLogger, get_logger'
    ]

    new_imports = [
        'from src.infrastructure.logging import UnifiedLogger',
        'from src.infrastructure.logging import get_logger',
        'from src.infrastructure.logging import UnifiedLogger, get_logger'
    ]

    fixed_files = 0

    # 遍历所有Python文件
    for py_file in data_src_path.rglob("*.py"):
        content = py_file.read_text(encoding='utf-8')
        original_content = content

        # 修复每种导入模式
        for i, old_pattern in enumerate(old_patterns):
            if re.search(old_pattern, content):
                content = re.sub(old_pattern, new_imports[i], content)

        # 如果内容有变化，写回文件
        if content != original_content:
            py_file.write_text(content, encoding='utf-8')
            fixed_files += 1
            print(f"✅ 修复文件: {py_file}")

    return fixed_files


def main():
    """主函数"""
    print("🔧 开始修复unified_logger直接导入问题...")
    print("=" * 60)

    # 修复导入问题
    fixed_count = fix_unified_logger_imports()

    print("\n📊 修复统计:")
    print(f"✅ 修复了 {fixed_count} 个文件")
    print("\n🎯 unified_logger导入问题修复完成!")
    print("\n📝 修复说明:")
    print("   - 将直接导入unified_logger改为通过__init__.py导入")
    print("   - 保持了原有的功能接口不变")
    print("   - 提高了模块的导入稳定性")


if __name__ == "__main__":
    main()
