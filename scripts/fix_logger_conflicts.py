#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复数据层日志冲突问题的脚本
"""

import re
from pathlib import Path


def fix_logger_conflicts():
    """修复日志冲突问题"""

    data_src_path = Path("src/data")

    # 需要移除的重复函数定义模式
    conflict_patterns = [
        # 移除重复的get_infrastructure_logger函数定义
        (r'def get_infrastructure_logger\(name\):\s*\n\s*"""[^"]*"""\s*\n\s*try:\s*\n\s*from src\.infrastructure\.logging import get_infrastructure_logger\s*\n\s*return get_infrastructure_logger\(name\)\s*\n\s*except ImportError:\s*\n\s*import logging\s*\n\s*return logging\.getLogger\(name\)',
         ''),  # 完全移除这个重复函数

        # 简化重复的导入和使用
        (r'try:\s*\n\s*from src\.infrastructure\.logging import get_infrastructure_logger\s*\n\s*logger = get_infrastructure_logger\([^)]+\)\s*\n\s*except ImportError:\s*\n\s*import logging\s*\n\s*logger = logging\.getLogger\([^)]+\)',
         ''),  # 移除整个try-except块，后面会重新添加正确的导入

        # 修复简单的重复定义
        (r'def get_infrastructure_logger\(name\):\s*\n\s*import logging\s*\n\s*return logging\.getLogger\(name\)',
         ''),  # 移除简单的重复定义
    ]

    fixed_files = 0

    # 遍历所有Python文件
    for py_file in data_src_path.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue  # 跳过__init__.py文件

        content = py_file.read_text(encoding='utf-8')
        original_content = content

        # 查找文件中是否已经有正确的导入
        has_correct_import = False
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'from src.infrastructure.logging import get_infrastructure_logger' in line:
                # 检查下一行是否是logger定义
                if i + 1 < len(lines) and 'logger = get_infrastructure_logger(' in lines[i + 1]:
                    has_correct_import = True
                    break

        # 如果没有正确的导入，添加一个
        if not has_correct_import:
            # 查找现有的logger定义
            logger_pattern = r'logger = get_logger\(__name__\)'
            if re.search(logger_pattern, content):
                # 替换为基础设施logger
                content = re.sub(
                    logger_pattern,
                    'from src.infrastructure.logging import get_infrastructure_logger\nlogger = get_infrastructure_logger(__name__)',
                    content
                )

        # 移除重复的函数定义
        for pattern, replacement in conflict_patterns:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)

        # 如果内容有变化，写回文件
        if content != original_content:
            py_file.write_text(content, encoding='utf-8')
            fixed_files += 1
            print(f"✅ 修复文件: {py_file}")

    return fixed_files


def create_fallback_logger():
    """创建降级日志器以防基础设施日志不可用"""

    fallback_code = '''
# 降级日志器，当基础设施日志不可用时使用
def get_fallback_logger(name: str):
    """获取降级日志器"""
    import logging
    return logging.getLogger(name)
'''

    # 检查是否需要添加到__init__.py
    init_file = Path("src/data/__init__.py")
    if init_file.exists():
        content = init_file.read_text(encoding='utf-8')
        if "get_fallback_logger" not in content:
            # 在文件末尾添加降级日志器
            content += "\n" + fallback_code
            init_file.write_text(content, encoding='utf-8')
            print("✅ 添加降级日志器到data/__init__.py")


def main():
    """主函数"""
    print("🔧 开始修复数据层日志冲突问题...")
    print("=" * 60)

    # 创建降级日志器
    create_fallback_logger()

    # 修复日志冲突
    fixed_count = fix_logger_conflicts()

    print("\n📊 修复统计:")
    print(f"✅ 修复了 {fixed_count} 个文件")
    print("\n🎯 日志冲突修复完成!")
    print("\n📝 修复说明:")
    print("   - 移除了重复的get_infrastructure_logger函数定义")
    print("   - 统一使用基础设施日志系统的导入路径")
    print("   - 添加了降级日志器以提高兼容性")
    print("\n🔄 现在数据层将使用统一的基础设施日志系统")


if __name__ == "__main__":
    main()
