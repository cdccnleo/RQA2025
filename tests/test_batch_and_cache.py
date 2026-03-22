#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量处理和任务缓存测试脚本

用于验证以下改进项：
1. P1-2: 实现批量处理器
2. P1-3: 实现任务缓存机制
"""

import asyncio
import sys
import time
from typing import Dict, Any

# 添加项目路径
sys.path.insert(0, r'c:\PythonProject\RQA2025')


async def test_batch_processor():
    """测试批量处理器"""
    print("=" * 60)
    print("测试 P1-2: 批量处理器")
    print("=" * 60)
    
    try:
        from src.core.orchestration.scheduler import get_unified_scheduler
        
        # 测试1: 创建带批量处理的调度器
        print("\n1. 测试批量处理器初始化...")
        scheduler = get_unified_scheduler(
            max_workers=4,
            enable_batch_processing=True,
            batch_config={
                "strategy": "hybrid",
                "max_batch_size": 10,
                "max_wait_time_ms": 2000,
                "min_batch_size": 3
            }
        )
        print("✅ 调度器创建成功（启用批量处理）")
        
        # 检查批量处理器状态
        if scheduler._batch_processor:
            print("✅ 批量处理器已初始化")
        else:
            print("⚠️ 批量处理器未初始化")
            return False
        
        # 测试2: 获取批量处理统计
        print("\n2. 测试获取批量处理统计...")
        stats = scheduler.get_batch_processor_stats()
        print(f"✅ 批量处理统计获取成功")
        print(f"   - 启用状态: {stats.get('enabled', False)}")
        if stats.get('enabled'):
            print(f"   - 配置: {stats.get('statistics', {}).get('config', {})}")
        
        # 测试3: 提交批量任务
        print("\n3. 测试提交批量任务...")
        task_ids = []
        for i in range(5):
            task_id = await scheduler.submit_batch_task(
                task_type='test_task',
                payload={'index': i, 'data': f'test_{i}'},
                priority=5
            )
            if task_id:
                task_ids.append(task_id)
                print(f"   ✅ 任务提交成功: {task_id}")
        
        print(f"✅ 共提交 {len(task_ids)} 个批量任务")
        
        # 等待批量处理
        await asyncio.sleep(3)
        
        # 测试4: 强制刷新批量处理器
        print("\n4. 测试强制刷新批量处理器...")
        flushed_count = await scheduler.flush_batch_processor()
        print(f"✅ 批量处理器已刷新，处理 {flushed_count} 个任务")
        
        print("\n🎉 P1-2 测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_task_cache():
    """测试任务缓存机制"""
    print("\n" + "=" * 60)
    print("测试 P1-3: 任务缓存机制")
    print("=" * 60)
    
    try:
        from src.core.orchestration.scheduler import get_unified_scheduler
        
        # 测试1: 创建带缓存的调度器
        print("\n1. 测试任务缓存初始化...")
        scheduler = get_unified_scheduler(
            max_workers=4,
            enable_task_cache=True,
            cache_config={
                "max_size": 100,
                "default_ttl_seconds": 60,
                "enable_prefetch": True
            }
        )
        print("✅ 调度器创建成功（启用任务缓存）")
        
        # 检查缓存状态
        if scheduler._task_cache:
            print("✅ 任务缓存已初始化")
        else:
            print("⚠️ 任务缓存未初始化")
            return False
        
        # 测试2: 设置缓存
        print("\n2. 测试设置缓存...")
        test_payload = {'symbol': '000001', 'indicators': ['sma', 'rsi']}
        test_result = {'features': {'sma': 10.5, 'rsi': 65.3}}
        
        success = await scheduler.set_cached_task_result(
            task_type='feature_extraction',
            payload=test_payload,
            result=test_result,
            ttl_seconds=300
        )
        if success:
            print("✅ 缓存设置成功")
        else:
            print("❌ 缓存设置失败")
            return False
        
        # 测试3: 获取缓存
        print("\n3. 测试获取缓存...")
        cached_result = await scheduler.get_cached_task_result(
            task_type='feature_extraction',
            payload=test_payload
        )
        
        if cached_result:
            print("✅ 缓存命中")
            print(f"   - 结果: {cached_result}")
        else:
            print("❌ 缓存未命中")
            return False
        
        # 测试4: 获取缓存统计
        print("\n4. 测试获取缓存统计...")
        stats = scheduler.get_task_cache_stats()
        print(f"✅ 缓存统计获取成功")
        print(f"   - 启用状态: {stats.get('enabled', False)}")
        if stats.get('enabled'):
            cache_stats = stats.get('statistics', {})
            print(f"   - 缓存大小: {cache_stats.get('size', 0)}")
            print(f"   - 命中率: {cache_stats.get('hit_rate', 0):.2%}")
            print(f"   - 请求总数: {cache_stats.get('total_requests', 0)}")
        
        # 测试5: 清空缓存
        print("\n5. 测试清空缓存...")
        cleared_count = await scheduler.clear_task_cache()
        print(f"✅ 缓存已清空，共 {cleared_count} 条")
        
        # 验证缓存已清空
        cached_result = await scheduler.get_cached_task_result(
            task_type='feature_extraction',
            payload=test_payload
        )
        if cached_result is None:
            print("✅ 缓存已正确清空")
        else:
            print("⚠️ 缓存未完全清空")
        
        print("\n🎉 P1-3 测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """运行所有测试"""
    print("\n" + "🧪 " * 20)
    print("批量处理和任务缓存验证测试")
    print("🧪 " * 20 + "\n")
    
    results = []
    
    # 测试 P1-2
    result1 = await test_batch_processor()
    results.append(("P1-2: 批量处理器", result1))
    
    # 测试 P1-3
    result2 = await test_task_cache()
    results.append(("P1-3: 任务缓存机制", result2))
    
    # 打印测试总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} - {name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"\n总计: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！架构改进实施成功。")
        return 0
    else:
        print(f"\n⚠️ {total - passed} 项测试失败，请检查实现。")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)
