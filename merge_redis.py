"""
合并Redis适配器文件
"""

from pathlib import Path
import shutil


def merge_redis_adapters():
    """合并所有Redis适配器为一个统一文件"""

    redis_dir = Path('src/infrastructure/cache/storage')

    # 备份所有文件
    backup_dir = Path('backup/redis_merge')
    backup_dir.mkdir(parents=True, exist_ok=True)

    print("📦 备份Redis文件...")

    for file_name in ['redis_adapter_unified.py', 'redis_cache.py', 'redis_storage.py', 'redis.py']:
        src_path = redis_dir / file_name
        if src_path.exists():
            shutil.copy2(src_path, backup_dir / f"{file_name}.backup")
            print(f"✅ 备份: {file_name}")

    # 创建新的统一适配器
    print("\n🔄 创建统一Redis适配器...")

    header = '''"""
统一Redis适配器

合并所有Redis相关功能的统一适配器:
- UnifiedRedisAdapter: 高级Redis适配器功能
- RedisCache: Redis缓存实现
- RedisStorage: Redis存储适配器
- RedisAdapter: 基础Redis适配器
- RedisClusterAdapter: 集群适配器
- AShareRedisAdapter: A股专用适配器

特性:
🔥 统一接口设计 - 消除接口不一致问题
🔥 多模式支持 - 单机/集群模式无缝切换
🔥 智能连接管理 - 连接池和故障恢复
🔥 数据压缩优化 - 自动压缩大对象
🔥 性能监控 - 实时指标收集和告警
🔥 生产就绪 - 健康检查和优雅关闭
🔥 缓存功能 - 完整的缓存操作支持
🔥 存储功能 - 配置和数据存储支持
"""

import json
import zlib
import pickle
import time
import logging
import threading
import hashlib
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from contextlib import contextmanager
import weakref

try:
    import redis
    from redis import Redis, ConnectionPool
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    Redis = None
    ConnectionPool = None

logger = logging.getLogger(__name__)

'''

    # 合并核心功能
    merged_content = header

    # 从各个文件中提取重要的类和功能
    files_to_merge = [
        ('redis_adapter_unified.py', ['RedisMode', 'CompressionLevel',
         'RedisConfig', 'RedisMetrics', 'CircuitBreaker', 'UnifiedRedisAdapter']),
        ('redis_cache.py', ['RedisCache']),
        ('redis_storage.py', ['RedisStorage']),
        ('redis.py', ['RedisAdapter', 'RedisClusterAdapter', 'AShareRedisAdapter'])
    ]

    merged_classes = set()

    for file_name, target_classes in files_to_merge:
        file_path = redis_dir / file_name
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

    # 写入新的统一适配器
    new_adapter_path = redis_dir / "redis_adapter.py"
    with open(new_adapter_path, 'w', encoding='utf-8') as f:
        f.write(merged_content)

    print(f"✅ 统一Redis适配器创建完成: {new_adapter_path}")

    # 删除旧文件
    print("\n🗑️ 删除旧文件...")
    for file_name in ['redis_adapter_unified.py', 'redis_cache.py', 'redis_storage.py', 'redis.py']:
        file_path = redis_dir / file_name
        if file_path.exists():
            file_path.unlink()
            print(f"✅ 删除: {file_name}")

    print("\n✅ Redis适配器合并完成！")


if __name__ == "__main__":
    merge_redis_adapters()
