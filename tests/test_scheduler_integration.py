#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一调度器集成测试脚本

用于验证以下改进项：
1. P0-1: 统一调度器实现
2. P1-1: Prometheus指标监控完善
"""

import asyncio
import sys
import time
from typing import Dict, Any

# 添加项目路径
sys.path.insert(0, r'c:\PythonProject\RQA2025')


async def test_unified_scheduler_integration():
    """测试统一调度器集成"""
    print("=" * 60)
    print("测试 P0-1: 统一调度器集成")
    print("=" * 60)
    
    try:
        from src.features.core.engine import FeatureEngine
        
        # 创建FeatureEngine实例
        engine = FeatureEngine()
        print("✅ FeatureEngine 创建成功")
        
        # 检查统一调度器是否初始化
        if engine._unified_scheduler:
            print("✅ 统一调度器已初始化")
        else:
            print("⚠️ 统一调度器未初始化（可能依赖未安装）")
            return False
        
        # 测试启动调度器
        start_result = await engine.start_scheduler()
        if start_result:
            print("✅ 统一调度器启动成功")
        else:
            print("⚠️ 统一调度器启动失败")
            return False
        
        # 获取调度器状态
        status = engine.get_scheduler_status()
        print(f"📊 调度器状态: {status}")
        
        # 测试创建任务
        task_config = {
            'stock_code': '000001',
            'indicators': ['sma', 'rsi', 'macd'],
            'timeframes': ['1d']
        }
        
        task = await engine.create_task('technical', task_config)
        print(f"✅ 任务创建成功: {task['task_id']}")
        print(f"   - 任务类型: {task['task_type']}")
        print(f"   - 调度器: {task.get('scheduler', 'unknown')}")
        print(f"   - 状态: {task['status']}")
        
        # 测试获取任务状态
        task_status = await engine.get_task_status(task['task_id'])
        print(f"✅ 任务状态查询成功: {task_status}")
        
        # 测试停止任务
        stop_result = await engine.stop_task(task['task_id'])
        print(f"✅ 任务停止成功: {stop_result}")
        
        # 测试删除任务
        delete_result = await engine.delete_task(task['task_id'])
        print(f"✅ 任务删除成功: {delete_result}")
        
        # 停止调度器
        stop_result = await engine.stop_scheduler()
        if stop_result:
            print("✅ 统一调度器停止成功")
        
        print("\n🎉 P0-1 测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prometheus_metrics():
    """测试Prometheus指标监控"""
    print("\n" + "=" * 60)
    print("测试 P1-1: Prometheus指标监控")
    print("=" * 60)
    
    try:
        from src.features.core.engine import FeatureEngine
        
        # 创建FeatureEngine实例
        engine = FeatureEngine()
        print("✅ FeatureEngine 创建成功")
        
        # 检查指标监控是否初始化
        if engine._metrics:
            print("✅ Prometheus指标监控已初始化")
        else:
            print("⚠️ Prometheus指标监控未初始化")
            return False
        
        # 测试记录指标
        engine.record_metric('feature_engine_test_counter_total', 1, {'test': 'true'})
        print("✅ 指标记录成功")
        
        # 测试获取指标
        metrics = engine.get_metrics()
        if metrics:
            print("✅ Prometheus指标生成成功")
            print(f"📊 指标长度: {len(metrics)} 字符")
            # 打印部分指标内容
            lines = metrics.split('\n')[:10]
            print("📊 指标示例:")
            for line in lines:
                if line.strip():
                    print(f"   {line}")
        else:
            print("⚠️ Prometheus指标生成失败")
            return False
        
        # 测试获取指标摘要
        summary = engine.get_metrics_summary()
        print("✅ 指标摘要获取成功")
        print(f"📊 特征数量: {summary.get('feature_engine', {}).get('features_count', 0)}")
        print(f"📊 指标数量: {summary.get('feature_engine', {}).get('indicators_count', 0)}")
        print(f"📊 任务数量: {summary.get('feature_engine', {}).get('tasks_count', 0)}")
        
        print("\n🎉 P1-1 测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """运行所有测试"""
    print("\n" + "🧪 " * 20)
    print("统一调度器架构改进验证测试")
    print("🧪 " * 20 + "\n")
    
    results = []
    
    # 测试 P0-1
    result1 = await test_unified_scheduler_integration()
    results.append(("P0-1: 统一调度器集成", result1))
    
    # 测试 P1-1
    result2 = test_prometheus_metrics()
    results.append(("P1-1: Prometheus指标监控", result2))
    
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
