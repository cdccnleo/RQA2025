#!/usr/bin/env python3
"""
调试DataFrame布尔判断问题
"""

import asyncio
from src.data.data_manager import DataManager


async def debug_dataframe_issue():
    """调试DataFrame布尔判断问题"""
    print("🔍 开始调试DataFrame布尔判断问题...")

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
            print(f"   ✅ 数据加载成功，形状: {symbol_data.shape}")
            print(f"   ✅ 数据类型: {type(symbol_data)}")
            print(f"   ✅ 是否为空: {symbol_data.empty}")

            # 测试DataFrame布尔判断
            print("\n   测试DataFrame布尔判断...")
            if symbol_data is not None and not symbol_data.empty:
                print("   ✅ DataFrame布尔判断正确")
            else:
                print("   ❌ DataFrame布尔判断有问题")

        except Exception as e:
            print(f"   ❌ 数据加载失败: {e}")
            return

        # 3. 测试数据管理器加载
        print("\n3. 测试数据管理器加载...")
        try:
            data_model = await data_manager.load_data(
                'stock',
                '2024-01-01',
                '2024-01-07',
                '1d',
                symbols=['600519']
            )
            print(f"   ✅ 数据管理器加载成功")
            print(f"   ✅ 数据模型类型: {type(data_model)}")
            print(f"   ✅ 数据形状: {data_model.data.shape}")

        except Exception as e:
            print(f"   ❌ 数据管理器加载失败: {e}")
            return

        print("\n🎉 DataFrame布尔判断问题调试完成！")

    except Exception as e:
        print(f"❌ 调试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_dataframe_issue())
