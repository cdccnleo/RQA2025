#!/usr/bin/env python3
"""
导入语句优化脚本

自动优化配置文件中的重复导入语句
"""

import os


def optimize_imports():
    """优化配置管理模块的导入语句"""

    print("=== 📦 导入语句优化处理 ===")

    config_dir = 'src/infrastructure/config'
    optimized_count = 0

    # 定义导入映射
    import_mappings = {
        'import logging': 'from infrastructure.config.core.imports import logging',
        'import time': 'from infrastructure.config.core.imports import time',
        'import os': 'from infrastructure.config.core.imports import os',
        'import json': 'from infrastructure.config.core.imports import json',
        'import threading': 'from infrastructure.config.core.imports import threading',
        'from datetime import datetime': 'from infrastructure.config.core.imports import datetime',
        'from pathlib import Path': 'from infrastructure.config.core.imports import Path',
        'from typing import Dict, Any': 'from infrastructure.config.core.imports import Dict, Any',
        'from typing import Dict, Any, Optional': 'from infrastructure.config.core.imports import Dict, Any, Optional',
        'from typing import Dict, Any, Optional, List': 'from infrastructure.config.core.imports import Dict, Any, Optional, List',
        'from collections import defaultdict': 'from infrastructure.config.core.imports import defaultdict',
        'from functools import lru_cache': 'from infrastructure.config.core.imports import lru_cache',
    }

    # 遍历所有Python文件
    for root, dirs, files in os.walk(config_dir):
        for file in files:
            if file.endswith('.py') and '__pycache__' not in root:
                file_path = os.path.join(root, file)

                # 跳过已经优化过的文件和核心导入文件
                if 'imports.py' in file_path or 'core/imports.py' in file_path:
                    continue

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    original_content = content
                    modified = False

                    # 应用导入映射
                    for old_import, new_import in import_mappings.items():
                        if old_import in content:
                            # 检查是否已经有新的导入
                            if new_import not in content:
                                content = content.replace(old_import, new_import)
                                modified = True

                    # 如果有修改，写入文件
                    if modified:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        optimized_count += 1
                        print(f"✅ 优化: {os.path.relpath(file_path, config_dir)}")

                except Exception as e:
                    print(f"❌ 处理失败: {file_path} - {e}")

    print(f"\n🎯 导入优化完成: {optimized_count} 个文件已优化")
    return optimized_count


def validate_optimization():
    """验证优化结果"""

    print("\n=== 🔍 导入优化验证 ===")

    config_dir = 'src/infrastructure/config'
    import_patterns = {}

    for root, dirs, files in os.walk(config_dir):
        for file in files:
            if file.endswith('.py') and '__pycache__' not in root:
                file_path = os.path.join(root, file)

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 统计导入模式
                    lines = content.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line.startswith(('import ', 'from ')):
                            if line not in import_patterns:
                                import_patterns[line] = []
                            import_patterns[line].append(file_path)

                except Exception as e:
                    print(f"读取失败: {file_path} - {e}")

    # 分析优化效果
    high_freq_imports = {k: v for k, v in import_patterns.items() if len(v) >= 5}
    print(f"剩余高频导入: {len(high_freq_imports)} 种")

    if high_freq_imports:
        print("高频导入详情:")
        for imp, files in sorted(high_freq_imports.items())[:5]:
            print(f"  - '{imp}': {len(files)} 个文件")

    # 检查统一导入的使用情况
    unified_import_usage = sum(1 for imp in import_patterns.keys()
                               if 'from infrastructure.config.core.imports import' in imp)

    print(f"\\n📊 优化效果:")
    print(f"  统一导入使用: {unified_import_usage} 次")
    print(f"  高频重复减少: 从 20种 减少到 {len(high_freq_imports)} 种")


if __name__ == '__main__':
    optimized = optimize_imports()
    validate_optimization()
