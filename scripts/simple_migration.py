#!/usr/bin/env python3
"""
简单的统一基础设施集成迁移测试
"""

import re
import os


def migrate_single_file():
    """迁移单个文件进行测试"""

    # 选择一个测试文件
    test_file = 'src/data/data_manager.py'

    if not os.path.exists(test_file):
        print(f"文件不存在: {test_file}")
        return

    print(f"迁移文件: {test_file}")

    # 读取文件内容
    with open(test_file, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # 定义迁移映射
    import_map = {
        r'from src\.infrastructure\.cache\.unified_cache import UnifiedCacheManager': 'from src.core.integration import get_data_adapter',
        r'from src\.infrastructure\.logging\.unified_logger import UnifiedLogger': 'from src.core.integration import get_data_adapter',
        r'from src\.infrastructure\.config\.unified_config import UnifiedConfigManager': 'from src.core.integration import get_data_adapter'
    }

    # 应用迁移
    for old_import, new_import in import_map.items():
        if re.search(old_import, content):
            print(f"匹配到: {old_import}")
            content = re.sub(old_import, new_import, content)
            print(f"替换为: {new_import}")

    # 如果内容有变化，写入文件
    if content != original_content:
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print("文件已更新")
    else:
        print("文件无需更新")

    # 显示变更
    print("\n变更内容:")
    print("-" * 50)
    for line in content.split('\n'):
        if 'from src.core.integration' in line:
            print(f"✓ {line}")


if __name__ == "__main__":
    migrate_single_file()
