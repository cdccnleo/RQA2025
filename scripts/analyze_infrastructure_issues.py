#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 基础设施层问题分析脚本

分析基础设施层测试失败的具体原因
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def analyze_infrastructure_issues():
    """分析基础设施层问题"""

    print("🔍 分析基础设施层测试问题...")

    # 1. 分析缓存模块问题
    print("\n📊 分析缓存模块问题:")
    cache_test_files = [
        'tests/unit/infrastructure/cache/test_cache_basic.py',
        'tests/unit/infrastructure/cache/test_cache_core.py',
        'tests/unit/infrastructure/cache/test_base_cache_manager.py'
    ]

    for test_file in cache_test_files:
        if os.path.exists(test_file):
            print(f"\n📁 检查 {test_file}:")
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 查找导入语句
                lines = content.split('\n')
                imports = [line for line in lines if line.startswith(
                    'from') or line.startswith('import')]

                print("  导入语句:")
                for imp in imports[:5]:  # 只显示前5个
                    print(f"    {imp}")

                # 查找测试类
                test_classes = [line for line in lines if 'class Test' in line]
                print("  测试类:")
                for cls in test_classes:
                    print(f"    {cls}")

            except Exception as e:
                print(f"  ❌ 读取文件失败: {e}")
        else:
            print(f"  ❌ 文件不存在: {test_file}")

    # 2. 分析安全模块问题
    print("\n🔒 分析安全模块问题:")
    security_test_file = 'tests/unit/infrastructure/security/test_base_security.py'

    if os.path.exists(security_test_file):
        print(f"\n📁 检查 {security_test_file}:")
        try:
            with open(security_test_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 查找导入语句
            lines = content.split('\n')
            imports = [line for line in lines if line.startswith(
                'from') or line.startswith('import')]

            print("  导入语句:")
            for imp in imports[:10]:  # 显示前10个
                print(f"    {imp}")

        except Exception as e:
            print(f"  ❌ 读取文件失败: {e}")

    # 3. 分析资源模块问题
    print("\n📦 分析资源模块问题:")
    resource_test_file = 'tests/unit/infrastructure/resource/test_resource_manager.py'

    if os.path.exists(resource_test_file):
        print(f"\n📁 检查 {resource_test_file}:")
        try:
            with open(resource_test_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 查找导入语句
            lines = content.split('\n')
            imports = [line for line in lines if line.startswith(
                'from') or line.startswith('import')]

            print("  导入语句:")
            for imp in imports[:5]:  # 显示前5个
                print(f"    {imp}")

        except Exception as e:
            print(f"  ❌ 读取文件失败: {e}")

    # 4. 生成缺失模块列表
    print("\n📝 生成缺失模块列表:")

    missing_modules = [
        'src.infrastructure.resource.quota_manager',
        'src.infrastructure.resource.resource_manager',
        'src.infrastructure.core.monitoring',
        'src.infrastructure.config.security',
        'src.infrastructure.core.resource_manager',
        'src.infrastructure.utils.helpers.date_utils',
        'src.adapters.miniqmt.miniqmt_data_adapter',
        'src.infrastructure.extensions',
        'src.infrastructure.config.security.security_manager',
        'src.infrastructure.config.security.encryption_service'
    ]

    print("  需要创建的模块:")
    for module in missing_modules:
        print(f"    • {module}")

    print("\n💡 建议行动计划:")
    print("  1. 优先创建核心缓存模块")
    print("  2. 创建基础资源管理模块")
    print("  3. 创建安全模块")
    print("  4. 修复类型错误问题")
    print("  5. 测试单个模块的断言失败")


def create_core_cache_modules():
    """创建核心缓存模块"""

    print("\n🏗️ 创建核心缓存模块...")

    # 1. 创建基础缓存管理器
    base_cache_manager_content = '''
"""
RQA2025 Base Cache Manager

基础缓存管理器实现
"""

from typing import Any, Dict, List, Optional
import logging
import time
import threading
from src.infrastructure.cache.global_interfaces import ICacheStrategy, CacheEvictionStrategy

logger = logging.getLogger(__name__)

class BaseCacheManager:
    """基础缓存管理器"""

    def __init__(self, max_size: int = 1000, ttl: int = 3600, strategy: Optional[ICacheStrategy] = None):
        self.max_size = max_size
        self.ttl = ttl
        self.strategy = strategy or CacheEvictionStrategy.LRU
        self.cache = {}
        self.access_times = {}
        self.creation_times = {}
        self.lock = threading.RLock()

        logger.info(f"初始化基础缓存管理器: max_size={max_size}, ttl={ttl}")

    def get(self, key: str) -> Any:
        """获取缓存项"""
        with self.lock:
            if key in self.cache:
                current_time = time.time()
                creation_time = self.creation_times.get(key, 0)

                # 检查是否过期
                if current_time - creation_time > self.ttl:
                    self._remove_item(key)
                    return None

                # 更新访问时间
                self.access_times[key] = current_time
                return self.cache[key]

            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存项"""
        with self.lock:
            try:
                current_time = time.time()
                item_ttl = ttl or self.ttl

                # 检查缓存是否已满
                if len(self.cache) >= self.max_size and key not in self.cache:
                    self._evict_items()

                # 设置缓存项
                self.cache[key] = value
                self.creation_times[key] = current_time
                self.access_times[key] = current_time

                logger.debug(f"缓存项设置成功: {key}")
                return True

            except Exception as e:
                logger.error(f"设置缓存项失败 {key}: {e}")
                return False

    def delete(self, key: str) -> bool:
        """删除缓存项"""
        with self.lock:
            if key in self.cache:
                self._remove_item(key)
                return True
            return False

    def clear(self) -> bool:
        """清空缓存"""
        with self.lock:
            try:
                self.cache.clear()
                self.access_times.clear()
                self.creation_times.clear()
                logger.info("缓存已清空")
                return True
            except Exception as e:
                logger.error(f"清空缓存失败: {e}")
                return False

    def size(self) -> int:
        """获取缓存大小"""
        return len(self.cache)

    def keys(self) -> List[str]:
        """获取所有键"""
        return list(self.cache.keys())

    def _remove_item(self, key: str):
        """移除缓存项"""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_times:
            del self.access_times[key]
        if key in self.creation_times:
            del self.creation_times[key]

    def _evict_items(self):
        """驱逐缓存项"""
        if not self.cache:
            return

        # 简单的LRU策略：移除最少访问的项
        if len(self.access_times) > 0:
            # 找到最少访问的键
            lru_key = min(self.access_times, key=self.access_times.get)
            self._remove_item(lru_key)
            logger.debug(f"LRU驱逐缓存项: {lru_key}")

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'ttl': self.ttl,
            'hit_rate': 0.0,  # 需要实现命中率统计
            'strategy': str(self.strategy)
        }
'''
    # 创建基础缓存管理器
    os.makedirs(project_root / 'src' / 'infrastructure' / 'cache', exist_ok=True)
    with open(project_root / 'src' / 'infrastructure' / 'cache' / 'base_cache_manager.py', 'w', encoding='utf-8') as f:
        f.write(base_cache_manager_content)
    print("✅ 创建了基础缓存管理器")

    # 2. 创建缓存服务
    cache_service_content = '''
"""
RQA2025 Cache Service

缓存服务实现
"""

from typing import Any, Dict, List, Optional
import logging
from .base_cache_manager import BaseCacheManager

logger = logging.getLogger(__name__)

class CacheService:
    """缓存服务"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.cache_manager = BaseCacheManager(
            max_size=self.config.get('max_size', 1000),
            ttl=self.config.get('ttl', 3600)
        )
        self.initialized = False

    def initialize(self) -> bool:
        """初始化服务"""
        try:
            self.initialized = True
            logger.info("缓存服务初始化成功")
            return True
        except Exception as e:
            logger.error(f"缓存服务初始化失败: {e}")
            return False

    def shutdown(self) -> bool:
        """关闭服务"""
        try:
            self.cache_manager.clear()
            self.initialized = False
            logger.info("缓存服务已关闭")
            return True
        except Exception as e:
            logger.error(f"缓存服务关闭失败: {e}")
            return False

    def get(self, key: str) -> Any:
        """获取缓存项"""
        if not self.initialized:
            raise RuntimeError("缓存服务未初始化")
        return self.cache_manager.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存项"""
        if not self.initialized:
            raise RuntimeError("缓存服务未初始化")
        return self.cache_manager.set(key, value, ttl)

    def delete(self, key: str) -> bool:
        """删除缓存项"""
        if not self.initialized:
            raise RuntimeError("缓存服务未初始化")
        return self.cache_manager.delete(key)

    def clear(self) -> bool:
        """清空缓存"""
        if not self.initialized:
            raise RuntimeError("缓存服务未初始化")
        return self.cache_manager.clear()

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            'service': 'cache_service',
            'status': 'healthy' if self.initialized else 'uninitialized',
            'cache_stats': self.cache_manager.get_stats()
        }
'''
    with open(project_root / 'src' / 'infrastructure' / 'cache' / 'cache_service.py', 'w', encoding='utf-8') as f:
        f.write(cache_service_content)
    print("✅ 创建了缓存服务")

    # 3. 创建缓存工厂
    cache_factory_content = '''
"""
RQA2025 Cache Factory

缓存工厂实现
"""

from typing import Any, Dict, Optional
import logging
from .base_cache_manager import BaseCacheManager
from .cache_service import CacheService

logger = logging.getLogger(__name__)

class CacheFactory:
    """缓存工厂"""

    _instances = {}

    @classmethod
    def create_cache_manager(cls, cache_type: str = 'memory', config: Optional[Dict[str, Any]] = None) -> BaseCacheManager:
        """创建缓存管理器"""
        config = config or {}

        if cache_type == 'memory':
            return BaseCacheManager(
                max_size=config.get('max_size', 1000),
                ttl=config.get('ttl', 3600)
            )
        else:
            # 默认使用内存缓存
            logger.warning(f"不支持的缓存类型 {cache_type}，使用内存缓存")
            return BaseCacheManager(
                max_size=config.get('max_size', 1000),
                ttl=config.get('ttl', 3600)
            )

    @classmethod
    def create_cache_service(cls, config: Optional[Dict[str, Any]] = None) -> CacheService:
        """创建缓存服务"""
        service = CacheService(config)
        service.initialize()
        return service

    @classmethod
    def get_cache_service(cls, service_name: str = 'default', config: Optional[Dict[str, Any]] = None) -> CacheService:
        """获取缓存服务实例（单例模式）"""
        if service_name not in cls._instances:
            cls._instances[service_name] = cls.create_cache_service(config)

        return cls._instances[service_name]
'''
    with open(project_root / 'src' / 'infrastructure' / 'cache' / 'cache_factory.py', 'w', encoding='utf-8') as f:
        f.write(cache_factory_content)
    print("✅ 创建了缓存工厂")

    # 4. 更新缓存模块的__init__.py
    cache_init_content = '''
"""
RQA2025 Infrastructure Cache Module

缓存相关组件
"""

from .base_cache_manager import BaseCacheManager
from .cache_service import CacheService
from .cache_factory import CacheFactory
from .global_interfaces import *

__all__ = [
    'BaseCacheManager',
    'CacheService',
    'CacheFactory',
    'ICacheStrategy',
    'CacheEvictionStrategy',
    'PartitionStrategy',
    'RepairStrategy'
]
'''
    with open(project_root / 'src' / 'infrastructure' / 'cache' / '__init__.py', 'w', encoding='utf-8') as f:
        f.write(cache_init_content)
    print("✅ 更新了缓存模块初始化文件")


def main():
    """主函数"""
    try:
        analyze_infrastructure_issues()
        create_core_cache_modules()

        print(f"\n{'=' * 60}")
        print("🎉 基础设施层核心模块创建完成！")
        print("=" * 60)
        print("现在可以开始测试缓存模块的具体断言失败了。")

        return 0
    except Exception as e:
        print(f"❌ 处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
