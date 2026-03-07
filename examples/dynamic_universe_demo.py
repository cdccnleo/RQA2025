#!/usr/bin/env python3
"""
动态股票池管理系统演示脚本

展示DynamicUniverseManager、IntelligentUniverseUpdater和DynamicWeightAdjuster的协同工作
"""

from src.trading.universe.dynamic_weight_adjuster import DynamicWeightAdjuster
from src.trading.universe.intelligent_updater import IntelligentUniverseUpdater
from src.trading.universe.dynamic_universe_manager import DynamicUniverseManager
from datetime import datetime
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_sample_market_data():
    """创建示例市场数据"""
    np.random.seed(42)  # 确保结果可重现

    stocks = ['000001.SZ', '000002.SZ', '000858.SZ', '600036.SH', '600519.SH',
              '000858.SZ', '002415.SZ', '300059.SZ', '600000.SH', '000001.SZ']

    data = {
        'volatility': np.random.uniform(0.1, 0.8, len(stocks)),
        'turnover_rate': np.random.uniform(0.01, 0.1, len(stocks)),
        'volume': np.random.uniform(1000000, 10000000, len(stocks)),
        'market_cap': np.random.uniform(100000000, 10000000000, len(stocks)),
        'beta': np.random.uniform(0.5, 2.5, len(stocks))
    }

    return pd.DataFrame(data, index=stocks)


def main():
    """主演示函数"""
    print("=" * 60)
    print("动态股票池管理系统演示")
    print("=" * 60)

    # 1. 创建配置
    universe_config = {
        'max_universe_size': 10,
        'beta_threshold': 2.0,
        'max_volatility': 0.5,
        'min_liquidity': 0.02,
        'composite_weights': {
            'liquidity': 0.3,
            'volatility': 0.2,
            'fundamental': 0.3,
            'technical': 0.2
        }
    }

    updater_config = {
        'update_frequency': 'daily',
        'max_update_interval': 24,
        'performance_threshold': 0.05,
        'volatility_threshold': 0.3,
        'liquidity_threshold': 0.02,
        'market_state_threshold': 0.1,
        'max_history_length': 100
    }

    weight_config = {
        'base_weights': {
            'fundamental': 0.3,
            'liquidity': 0.25,
            'technical': 0.25,
            'sentiment': 0.1,
            'volatility': 0.1
        },
        'market_state_weights': {
            'bull': {
                'fundamental': 0.35,
                'liquidity': 0.2,
                'technical': 0.25,
                'sentiment': 0.1,
                'volatility': 0.1
            },
            'bear': {
                'fundamental': 0.25,
                'liquidity': 0.3,
                'technical': 0.2,
                'sentiment': 0.05,
                'volatility': 0.2
            }
        },
        'adjustment_sensitivity': 1.0,
        'max_adjustment': 0.3,
        'min_weight': 0.05,
        'max_weight': 0.5
    }

    # 2. 初始化组件
    print("\n1. 初始化组件...")
    universe_manager = DynamicUniverseManager(universe_config)
    intelligent_updater = IntelligentUniverseUpdater(updater_config)
    weight_adjuster = DynamicWeightAdjuster(weight_config)

    print("✓ DynamicUniverseManager 已初始化")
    print("✓ IntelligentUniverseUpdater 已初始化")
    print("✓ DynamicWeightAdjuster 已初始化")

    # 3. 创建市场数据
    print("\n2. 生成市场数据...")
    market_data = create_sample_market_data()
    print(f"✓ 生成了 {len(market_data)} 只股票的市场数据")
    print(f"数据预览:\n{market_data.head()}")

    # 4. 初始股票池更新
    print("\n3. 执行初始股票池更新...")
    initial_universe = universe_manager.update_universe(market_data)
    print(f"✓ 初始股票池包含 {len(initial_universe)} 只股票")
    print(f"活跃股票池: {list(initial_universe.keys())}")

    # 5. 智能更新检查
    print("\n4. 执行智能更新检查...")
    update_decision = intelligent_updater.should_update_universe(
        current_time=datetime.now(),
        current_market_state="bull",
        market_data=market_data
    )

    print(f"✓ 更新决策: {'需要更新' if update_decision.should_update else '无需更新'}")
    if update_decision.should_update:
        print(f"  触发原因: {update_decision.reason}")
        print(f"  紧急程度: {update_decision.urgency_level}")
        print(f"  预期影响: {update_decision.estimated_impact:.2f}")

        # 记录更新
        intelligent_updater.record_update(update_decision.trigger, update_decision.reason)

    # 6. 动态权重调整
    print("\n5. 执行动态权重调整...")
    weight_result = weight_adjuster.adjust_weights(
        market_state="bull",
        performance_metrics={
            'liquidity': 0.3,
            'volatility': 0.2,
            'fundamental': 0.4,
            'technical': 0.1
        },
        risk_metrics={'overall_risk': 0.6},
        market_data=market_data
    )

    print(f"✓ 权重调整完成")
    print(f"  原始权重: {weight_result.original_weights}")
    print(f"  调整后权重: {weight_result.adjusted_weights}")
    print(f"  调整因子: {weight_result.adjustment_factors}")
    print(f"  调整原因: {weight_result.reason}")

    # 7. 综合工作流程演示
    print("\n6. 综合工作流程演示...")

    # 模拟市场状态变化
    print("  - 检测到市场状态变化: bull -> bear")
    intelligent_updater._check_market_state_change("bear")

    # 再次检查更新需求
    update_decision2 = intelligent_updater.should_update_universe(
        current_time=datetime.now(),
        current_market_state="bear",
        market_data=market_data
    )

    print(f"  - 更新决策: {'需要更新' if update_decision2.should_update else '无需更新'}")

    # 调整权重以适应熊市
    weight_result2 = weight_adjuster.adjust_weights(
        market_state="bear",
        performance_metrics={
            'liquidity': 0.2,
            'volatility': 0.4,
            'fundamental': 0.3,
            'technical': 0.1
        }
    )

    print(f"  - 熊市权重调整: {weight_result2.adjusted_weights}")

    # 8. 统计信息展示
    print("\n7. 统计信息展示...")

    # 宇宙管理器统计
    universe_stats = universe_manager.get_universe_statistics()
    print(f"  - 宇宙管理器统计:")
    print(f"    总更新次数: {universe_stats['update_count']}")
    print(f"    当前活跃股票数: {universe_stats['active_universe_size']}")

    # 智能更新器统计
    updater_stats = intelligent_updater.get_update_statistics()
    print(f"  - 智能更新器统计:")
    print(f"    总更新次数: {updater_stats['update_statistics']['total_updates']}")
    print(f"    市场状态变化次数: {updater_stats['update_statistics']['market_state_updates']}")

    # 权重调整器统计
    weight_stats = weight_adjuster.get_adjustment_statistics()
    print(f"  - 权重调整器统计:")
    print(f"    总调整次数: {weight_stats['total_adjustments']}")
    print(f"    当前权重: {weight_adjuster.get_current_weights()}")

    # 9. 性能基准测试
    print("\n8. 性能基准测试...")
    import time

    start_time = time.time()

    # 执行100次更新操作
    for i in range(100):
        universe_manager.update_universe(market_data)
        weight_adjuster.adjust_weights(market_state="bull" if i % 2 == 0 else "bear")
        intelligent_updater.should_update_universe(
            current_time=datetime.now(),
            current_market_state="bull" if i % 2 == 0 else "bear",
            market_data=market_data
        )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"✓ 100次操作耗时: {elapsed_time:.3f} 秒")
    print(f"  平均每次操作: {elapsed_time/100*1000:.2f} 毫秒")

    print("\n" + "=" * 60)
    print("演示完成！动态股票池管理系统运行正常。")
    print("=" * 60)


if __name__ == "__main__":
    main()
