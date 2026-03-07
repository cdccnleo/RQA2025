#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
缓存内存分析脚本
逐步分析内存占用情况
"""

import sys
import os
import gc
import psutil

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def analyze_memory_step_by_step():
    """逐步分析内存占用"""
    print("开始逐步内存分析...")

    process = psutil.Process(os.getpid())

    # 步骤1: 基础内存
    initial_memory = process.memory_info().rss / 1024 / 1024
    print(f"步骤1 - 基础内存: {initial_memory:.2f} MB")

    # 步骤2: 导入基础模块
    import time

    step2_memory = process.memory_info().rss / 1024 / 1024
    print(f"步骤2 - 导入基础模块: {step2_memory:.2f} MB (增长: {step2_memory - initial_memory:.2f} MB)")

    # 步骤3: 导入缓存相关模块
    try:
        from src.infrastructure.cache.enhanced_cache_manager import (
            EnhancedCacheManager,
            CacheConfig
        )
        step3_memory = process.memory_info().rss / 1024 / 1024
        print(f"步骤3 - 导入缓存模块: {step3_memory:.2f} MB (增长: {step3_memory - step2_memory:.2f} MB)")
    except Exception as e:
        print(f"步骤3失败: {str(e)}")
        return False

    # 步骤4: 创建配置
    try:
        config = CacheConfig(max_size=5, ttl=30)
        step4_memory = process.memory_info().rss / 1024 / 1024
        print(f"步骤4 - 创建配置: {step4_memory:.2f} MB (增长: {step4_memory - step3_memory:.2f} MB)")
    except Exception as e:
        print(f"步骤4失败: {str(e)}")
        return False

    # 步骤5: 创建缓存管理器
    try:
        cache_manager = EnhancedCacheManager(config=config)
        step5_memory = process.memory_info().rss / 1024 / 1024
        print(f"步骤5 - 创建缓存管理器: {step5_memory:.2f} MB (增长: {step5_memory - step4_memory:.2f} MB)")
    except Exception as e:
        print(f"步骤5失败: {str(e)}")
        return False

    # 步骤6: 执行基本操作
    try:
        cache_manager.set("test_key", "test_value")
        value = cache_manager.get("test_key")
        stats = cache_manager.stats()
        step6_memory = process.memory_info().rss / 1024 / 1024
        print(f"步骤6 - 执行基本操作: {step6_memory:.2f} MB (增长: {step6_memory - step5_memory:.2f} MB)")
    except Exception as e:
        print(f"步骤6失败: {str(e)}")
        return False

    # 步骤7: 清理资源
    try:
        del cache_manager
        gc.collect()
        time.sleep(0.5)  # 等待垃圾回收
        step7_memory = process.memory_info().rss / 1024 / 1024
        print(f"步骤7 - 清理资源: {step7_memory:.2f} MB (增长: {step7_memory - initial_memory:.2f} MB)")
    except Exception as e:
        print(f"步骤7失败: {str(e)}")
        return False

    # 总结
    total_increase = step7_memory - initial_memory
    print(f"\n总结:")
    print(f"总内存增长: {total_increase:.2f} MB")

    if total_increase > 50:
        print(f"警告: 内存增长过大 ({total_increase:.2f} MB)")
        return False
    else:
        print("内存分析通过!")
        return True


def test_memory_cache_manager_only():
    """只测试MemoryCacheManager的内存占用"""
    print("\n开始MemoryCacheManager内存测试...")

    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024

    try:
        from src.infrastructure.cache.memory_cache_manager import MemoryCacheManager

        # 创建多个MemoryCacheManager实例
        managers = []
        for i in range(10):
            manager = MemoryCacheManager(max_size=100, ttl=3600)
            managers.append(manager)

            # 添加一些数据
            for j in range(10):
                manager.set(f"key_{i}_{j}", f"value_{i}_{j}")

        current_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = current_memory - initial_memory
        print(f"创建10个MemoryCacheManager后内存: {current_memory:.2f} MB (增长: {memory_increase:.2f} MB)")

        # 清理
        for manager in managers:
            del manager
        gc.collect()

        final_memory = process.memory_info().rss / 1024 / 1024
        final_increase = final_memory - initial_memory
        print(f"清理后内存: {final_memory:.2f} MB (增长: {final_increase:.2f} MB)")

        if final_increase < 20:
            print("MemoryCacheManager内存测试通过!")
            return True
        else:
            print(f"MemoryCacheManager内存增长过大: {final_increase:.2f} MB")
            return False

    except Exception as e:
        print(f"MemoryCacheManager测试异常: {str(e)}")
        return False


def main():
    """主函数"""
    print("=" * 50)
    print("缓存内存分析")
    print("=" * 50)

    # 测试1: 逐步分析
    test1_passed = analyze_memory_step_by_step()

    # 测试2: MemoryCacheManager测试
    test2_passed = test_memory_cache_manager_only()

    # 总结
    print("\n" + "=" * 50)
    print("分析结果总结:")
    print(f"逐步内存分析: {'通过' if test1_passed else '失败'}")
    print(f"MemoryCacheManager测试: {'通过' if test2_passed else '失败'}")

    if test1_passed and test2_passed:
        print("所有测试通过!")
        return 0
    else:
        print("部分测试失败!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
