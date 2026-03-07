#!/usr/bin/env python3
"""
测试AI智能筛选和市场适应性功能
"""

import sys
sys.path.append('.')

import asyncio
import time
from typing import Dict, Any

def test_ai_smart_filter():
    """测试AI智能筛选功能"""
    print("🧪 测试AI智能筛选功能")

    try:
        from src.infrastructure.ai.smart_stock_filter import get_smart_stock_filter, MarketState
        from datetime import datetime

        smart_filter = get_smart_stock_filter()
        print("✅ AI智能筛选器初始化成功")

        # 测试股票数据
        test_stocks = [
            {
                'code': '000001',
                'price': 15.5,
                'volume': 1000000,
                'turnover': 15500000,
                'volatility': 0.02,
                'market_cap': 5000000000,
                'pe_ratio': 12.5,
                'pb_ratio': 1.2,
                'turnover_rate': 0.015,
                'amplitude': 0.025
            },
            {
                'code': '000002',
                'price': 25.8,
                'volume': 500000,
                'turnover': 12900000,
                'volatility': 0.03,
                'market_cap': 3000000000,
                'pe_ratio': 18.5,
                'pb_ratio': 1.8,
                'turnover_rate': 0.008,
                'amplitude': 0.035
            }
        ]

        # 测试重要性预测
        print("1. 测试股票重要性预测...")
        importance_scores = smart_filter.predict_stock_importance(test_stocks)
        print(f"   重要性评分结果: {importance_scores}")

        # 测试流动性预测
        print("2. 测试股票流动性预测...")
        liquidity_scores = smart_filter.predict_stock_liquidity(test_stocks)
        print(f"   流动性评分结果: {liquidity_scores}")

        # 测试智能选择（简化版）
        print("3. 测试智能股票选择...")
        all_stocks = ['000001', '000002', '000003']
        strategy_config = {
            'strategy_id': 'hf_trading',
            'pool_size': 10,
            'liquidity_threshold': 10000
        }

        selected_stocks = smart_filter.select_optimal_stocks(
            all_stocks, strategy_config, None, target_size=2
        )
        print(f"   智能选择结果: {selected_stocks}")

        return True

    except Exception as e:
        print(f"❌ AI智能筛选测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_market_adaptive_monitor():
    """测试市场适应性监控功能"""
    print("🧪 测试市场适应性监控功能")

    try:
        from src.infrastructure.monitoring.services.market_adaptive_monitor import get_market_adaptive_monitor

        monitor = get_market_adaptive_monitor()
        print("✅ 市场适应性监控器初始化成功")

        # 测试获取适应性配置
        print("1. 测试适应性配置...")
        config = monitor.get_current_adaptive_config()
        print(f"   适应性配置: {config}")

        # 测试市场状态摘要（简化）
        print("2. 测试市场状态摘要...")
        summary = monitor.get_market_state_summary()
        print(f"   市场状态摘要: 数据点={summary.get('data_points', 0)}")

        return True

    except Exception as e:
        print(f"❌ 市场适应性监控测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_scheduler_adaptive_adjustment():
    """测试调度器适应性调整功能"""
    print("🧪 测试调度器适应性调整功能")

    try:
        from src.core.orchestration.business_process.service_scheduler import get_data_collection_scheduler

        scheduler = get_data_collection_scheduler()
        if not scheduler:
            print("❌ 无法获取调度器实例")
            return False

        print("✅ 调度器实例获取成功")

        # 测试参数调整
        print("1. 测试参数调整...")
        original_params = scheduler.get_current_parameters()
        print(f"   原始参数: {original_params}")

        # 调整参数
        scheduler.adjust_parameters(
            batch_size=30,
            interval_seconds=45.0,
            priority_multipliers={'high': 1.5, 'medium': 1.2, 'low': 0.8}
        )

        adjusted_params = scheduler.get_current_parameters()
        print(f"   调整后参数: {adjusted_params}")

        # 重置参数
        print("2. 测试参数重置...")
        scheduler.reset_to_defaults()
        reset_params = scheduler.get_current_parameters()
        print(f"   重置后参数: {reset_params}")

        return True

    except Exception as e:
        print(f"❌ 调度器适应性调整测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_async_tests():
    """运行异步测试"""
    print("\n" + "="*50)
    print("🚀 开始AI和市场适应性功能测试")
    print("="*50)

    results = []

    # 测试AI智能筛选
    result1 = test_ai_smart_filter()
    results.append(("AI智能筛选", result1))

    # 测试市场适应性监控
    result2 = test_market_adaptive_monitor()
    results.append(("市场适应性监控", result2))

    # 测试调度器适应性调整
    result3 = await test_scheduler_adaptive_adjustment()
    results.append(("调度器适应性调整", result3))

    # 输出测试结果
    print("\n" + "="*50)
    print("📊 测试结果总结")
    print("="*50)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\n总体结果: {passed}/{total} 个测试通过")

    if passed == total:
        print("🎉 所有测试通过！AI和市场适应性功能工作正常")
    else:
        print("⚠️ 部分测试失败，请检查相关功能")

    return passed == total


def main():
    """主函数"""
    # 运行异步测试
    result = asyncio.run(run_async_tests())

    # 输出最终结果
    print(f"\n测试完成，最终结果: {'成功' if result else '失败'}")
    return 0 if result else 1


if __name__ == "__main__":
    exit(main())