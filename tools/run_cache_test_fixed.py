#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
运行修复后缓存测试的脚本
"""

import sys
import os
import gc
import psutil
from unittest.mock import Mock

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def run_cache_test_fixed():
    """运行修复后的缓存测试"""
    print("开始运行修复后的缓存测试...")

    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024
    print(f"初始内存: {initial_memory:.2f} MB")

    try:
        # 直接导入缓存相关模块，避免导入整个infrastructure包
        from src.infrastructure.cache.enhanced_cache_manager import (
            EnhancedCacheManager,
            CacheConfig
        )

        print("模块导入成功")

        # 创建配置
        config = CacheConfig(
            max_size=100,
            ttl=3600,
            enable_stats=True,
            auto_cleanup=True
        )
        print("配置创建成功")

        # 创建模拟缓存
        mock_cache = Mock()
        mock_cache.get.side_effect = Exception("Test error")
        mock_cache.set.return_value = True
        mock_cache.delete.return_value = True
        mock_cache.exists.return_value = False
        mock_cache.clear.return_value = True
        mock_cache.list_keys.return_value = []
        mock_cache.health_check.return_value = {"status": "healthy"}

        # 创建缓存管理器
        cache_manager = EnhancedCacheManager(
            config=config,
            l1_cache=mock_cache
        )
        print("缓存管理器创建成功")

        # 测试异常处理
        value = cache_manager.get("test_key")
        print(f"异常处理测试结果: {value}")

        # 验证错误计数增加
        stats = cache_manager.stats()
        print(f"统计信息: {stats}")

        # 验证测试通过
        assert value is None
        assert stats["overall"]["error_count"] >= 1

        # 清理
        del cache_manager
        del mock_cache
        gc.collect()

        # 检查内存
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        print(f"最终内存: {final_memory:.2f} MB, 增长: {memory_increase:.2f} MB")

        if memory_increase < 50:
            print("修复后的缓存测试通过!")
            return True
        else:
            print(f"内存增长过大: {memory_increase:.2f} MB")
            return False

    except Exception as e:
        print(f"测试异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_cache_test_fixed()
    sys.exit(0 if success else 1)
