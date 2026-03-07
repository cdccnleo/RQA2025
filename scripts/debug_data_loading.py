#!/usr/bin/env python3
"""
数据加载调试脚本
"""

import asyncio
from src.data.data_manager import DataManager
from src.data.optimization.data_optimizer import DataOptimizer, OptimizationConfig


async def debug_data_loading():
    """调试数据加载功能"""
    print("🔍 开始调试数据加载功能...")

    try:
        # 1. 测试数据管理器
        print("\n1. 测试数据管理器...")
        data_manager = DataManager()

        # 2. 测试股票数据加载器
        print("\n2. 测试股票数据加载器...")
        from src.data.loader.stock_loader import StockDataLoader

        # 创建股票数据加载器
        stock_loader = StockDataLoader(save_path="data/stock")

        # 测试单个股票数据加载
        print("   测试加载单个股票数据...")
        try:
            symbol_data = stock_loader.load_data(
                symbol="600519",
                start_date="2024-01-01",
                end_date="2024-01-07",
                adjust="hfq"
            )
            print(f"   ✅ 成功加载数据，形状: {symbol_data.shape}")
            print(f"   📊 列名: {symbol_data.columns.tolist()}")
        except Exception as e:
            print(f"   ❌ 加载失败: {e}")
            return

        # 3. 测试数据优化器
        print("\n3. 测试数据优化器...")
        config = OptimizationConfig(
            max_workers=2,
            enable_parallel_loading=True,
            enable_cache=True,
            enable_quality_monitor=True,
            enable_performance_monitor=True
        )

        optimizer = DataOptimizer(config)

        # 测试优化数据加载
        print("   测试优化数据加载...")
        try:
            result = await optimizer.optimize_data_loading(
                data_type='stock',
                start_date='2024-01-01',
                end_date='2024-01-07',
                frequency='1d',
                symbols=['600519.SH']
            )

            print(f"   ✅ 优化加载成功: {result.success}")
            print(f"   📊 缓存命中: {result.cache_hit}")
            print(f"   ⏱️  加载时间: {result.load_time_ms:.2f}ms")

            if result.data_model:
                print(f"   📈 数据形状: {result.data_model.data.shape}")
            else:
                print("   ❌ 数据模型为空")

        except Exception as e:
            print(f"   ❌ 优化加载失败: {e}")
            import traceback
            traceback.print_exc()

        print("\n✅ 调试完成")

    except Exception as e:
        print(f"❌ 调试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_data_loading())
