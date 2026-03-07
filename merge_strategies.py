"""
合并缓存策略文件
"""

from pathlib import Path
import shutil


def merge_strategy_files():
    """合并所有缓存策略文件为一个统一的管理器"""

    strategies_dir = Path('src/infrastructure/cache/strategies')

    # 备份所有文件
    backup_dir = Path('backup/strategies_merge')
    backup_dir.mkdir(parents=True, exist_ok=True)

    print("📦 备份策略文件...")

    for file_name in ['cache_strategy_manager.py', 'cache_strategy.py', 'smart_cache_strategies.py', 'smart_cache_strategy.py']:
        src_path = strategies_dir / file_name
        if src_path.exists():
            shutil.copy2(src_path, backup_dir / f"{file_name}.backup")
            print(f"✅ 备份: {file_name}")

    # 创建新的统一策略管理器
    print("\n🔄 创建统一缓存策略管理器...")

    header = '''"""
统一缓存策略管理器

合并所有缓存策略相关功能的统一管理器:
- CacheStrategyManager: 策略管理器
- SmartCacheStrategy: 智能缓存策略
- 各种缓存算法实现 (LRU, LFU, Adaptive等)

特性:
🔥 统一策略接口 - 标准化的策略定义
🔥 多算法支持 - LRU/LFU/Adaptive等多种算法
🔥 智能切换 - 基于访问模式自动切换策略
🔥 性能监控 - 策略效果实时监控
🔥 可扩展设计 - 易于添加新的缓存算法
"""

import time
import logging
import threading
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import OrderedDict, defaultdict, deque
import statistics

logger = logging.getLogger(__name__)

'''

    # 合并核心功能
    merged_content = header

    # 定义要合并的类和文件对应关系
    files_to_merge = [
        ('cache_strategy_manager.py', [
            'StrategyType', 'AccessPattern', 'StrategyMetrics', 'AccessPatternAnalysis',
            'LRUStrategy', 'LFUEntry', 'LFUStrategy', 'AdaptiveStrategy', 'CacheStrategyManager'
        ]),
        ('cache_strategy.py', ['SmartCacheStrategy']),
        ('smart_cache_strategies.py', [
            'LFUCache', 'LRUKCache', 'AdaptiveCacheEntry', 'AdaptiveCache',
            'PriorityCacheEntry', 'PriorityCache', 'CostAwareEntry', 'CostAwareCache'
        ]),
        ('smart_cache_strategy.py', [
            'DataType', 'CacheStrategy', 'CacheConfig', 'CacheMetrics', 'CacheStrategyInterface',
            'LRUCacheStrategy', 'TTLCacheStrategy', 'CacheFactory', 'SmartCacheManager'
        ])
    ]

    merged_classes = set()

    for file_name, target_classes in files_to_merge:
        file_path = strategies_dir / file_name
        if file_path.exists():
            print(f"📖 处理文件: {file_name}")

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            lines = content.split('\n')
            in_target_class = False
            class_content = []
            indent_level = 0

            i = 0
            while i < len(lines):
                line = lines[i]
                stripped = line.strip()

                # 检查是否是目标类
                if stripped.startswith('class '):
                    class_name = stripped.split()[1].split('(')[0].split(':')[0]
                    if class_name in target_classes and class_name not in merged_classes:
                        print(f"  ✅ 合并类: {class_name}")
                        in_target_class = True
                        class_content = [line]
                        indent_level = len(line) - len(line.lstrip())
                        merged_classes.add(class_name)
                    else:
                        in_target_class = False
                elif in_target_class:
                    # 检查缩进级别来确定是否还在类内
                    if line.strip() == '':
                        class_content.append(line)
                    elif len(line) - len(line.lstrip()) > indent_level:
                        class_content.append(line)
                    else:
                        # 类结束
                        merged_content += '\n'.join(class_content) + '\n\n'
                        in_target_class = False
                        class_content = []
                        # 重新处理当前行
                        i -= 1

                i += 1

            # 处理最后一个类
            if class_content:
                merged_content += '\n'.join(class_content) + '\n\n'

    # 写入新的统一策略管理器
    new_strategy_path = strategies_dir / "unified_strategy_manager.py"
    with open(new_strategy_path, 'w', encoding='utf-8') as f:
        f.write(merged_content)

    print(f"✅ 统一缓存策略管理器创建完成: {new_strategy_path}")

    # 删除旧文件
    print("\n🗑️ 删除旧文件...")
    for file_name in ['cache_strategy_manager.py', 'cache_strategy.py', 'smart_cache_strategies.py', 'smart_cache_strategy.py']:
        file_path = strategies_dir / file_name
        if file_path.exists():
            file_path.unlink()
            print(f"✅ 删除: {file_name}")

    print("\n✅ 缓存策略合并完成！")


if __name__ == "__main__":
    merge_strategy_files()
