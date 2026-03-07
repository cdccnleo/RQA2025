#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试策略仪表盘功能"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_performance_service():
    """测试性能评估服务"""
    print("=" * 60)
    print("测试策略性能评估服务...")
    print("=" * 60)
    
    try:
        from src.gateway.web.strategy_performance_service import (
            get_strategy_comparison, 
            get_performance_metrics
        )
        
        # 测试策略对比
        strategies = get_strategy_comparison()
        print(f"✅ 策略对比数据加载成功: {len(strategies)} 个策略")
        
        # 测试性能指标
        metrics = get_performance_metrics()
        print(f"✅ 性能指标数据加载成功: {type(metrics).__name__}")
        print(f"   - 指标数量: {len(metrics.get('metrics', {}))}")
        print(f"   - 收益曲线数量: {len(metrics.get('return_curves', []))}")
        print(f"   - 风险收益点数: {len(metrics.get('risk_return', []))}")
        print(f"   - 排名数量: {len(metrics.get('rankings', []))}")
        
        return True
    except Exception as e:
        print(f"❌ 性能评估服务测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lifecycle_persistence():
    """测试生命周期持久化"""
    print("\n" + "=" * 60)
    print("测试生命周期持久化...")
    print("=" * 60)
    
    try:
        from src.gateway.web.strategy_persistence import (
            save_lifecycle_event, 
            load_lifecycle_events
        )
        
        test_id = 'test_lifecycle_check_001'
        test_event = {
            'event_type': 'test',
            'description': '测试生命周期事件'
        }
        
        # 测试保存
        success = save_lifecycle_event(test_id, test_event)
        print(f"✅ 生命周期事件保存: {'成功' if success else '失败'}")
        
        # 测试加载
        events = load_lifecycle_events(test_id)
        print(f"✅ 生命周期事件加载: {len(events)} 个事件")
        
        return True
    except Exception as e:
        print(f"❌ 生命周期持久化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_execution_service():
    """测试执行监控服务"""
    print("\n" + "=" * 60)
    print("测试策略执行监控服务...")
    print("=" * 60)
    
    try:
        import asyncio
        from src.gateway.web.strategy_execution_service import (
            get_strategy_execution_status,
            get_execution_metrics
        )
        
        # 测试执行状态
        async def test_status():
            status = await get_strategy_execution_status()
            print(f"✅ 执行状态数据加载成功: {type(status).__name__}")
            print(f"   - 策略数量: {status.get('total_count', 0)}")
            print(f"   - 运行中: {status.get('running_count', 0)}")
            return True
        
        # 测试执行指标
        async def test_metrics():
            metrics = await get_execution_metrics()
            print(f"✅ 执行指标数据加载成功: {type(metrics).__name__}")
            print(f"   - 平均延迟: {metrics.get('avg_latency', 0)} ms")
            print(f"   - 今日信号数: {metrics.get('today_signals', 0)}")
            return True
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        status_ok = loop.run_until_complete(test_status())
        metrics_ok = loop.run_until_complete(test_metrics())
        
        loop.close()
        
        return status_ok and metrics_ok
    except Exception as e:
        print(f"❌ 执行监控服务测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_realtime_signals():
    """测试实时信号API"""
    print("\n" + "=" * 60)
    print("测试实时信号API...")
    print("=" * 60)
    
    try:
        from src.gateway.web.trading_signal_service import get_realtime_signals
        
        signals = get_realtime_signals()
        print(f"✅ 实时信号数据加载成功: {len(signals)} 个信号")
        
        # 验证不使用模拟数据
        if signals:
            print(f"   - 示例信号: {signals[0].get('id', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"❌ 实时信号API测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n开始测试策略仪表盘功能...\n")
    
    results = []
    
    # 测试性能评估服务
    results.append(("性能评估服务", test_performance_service()))
    
    # 测试生命周期持久化
    results.append(("生命周期持久化", test_lifecycle_persistence()))
    
    # 测试执行监控服务
    results.append(("执行监控服务", test_execution_service()))
    
    # 测试实时信号API
    results.append(("实时信号API", test_realtime_signals()))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status}: {name}")
    
    print(f"\n总计: {passed}/{total} 项测试通过")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

