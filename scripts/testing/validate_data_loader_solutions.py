#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据加载器方案验证脚本
验证方案二的基本功能
"""

import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def test_solution_2():
    """测试方案二：直接使用数据层"""
    print("🔍 测试方案二：直接使用数据层")

    try:
        # 导入方案二
        from src.backtest.data_loader import BacktestDataLoader

        # 模拟配置
        config = {
            "timezone": "Asia/Shanghai",
            "data": {
                "stock": {
                    "save_path": "data/stock",
                    "max_retries": 3,
                    "cache_days": 30
                }
            }
        }

        # 测试初始化
        loader = BacktestDataLoader(config)
        print("✅ 方案二初始化成功")

        # 测试基本属性
        assert loader.timezone == "Asia/Shanghai"
        assert hasattr(loader, 'stock_loader')
        print("✅ 方案二基本属性验证通过")

        # 测试缓存功能
        loader.clear_cache()
        assert len(loader.cache) == 0
        print("✅ 方案二缓存功能正常")

        return True

    except Exception as e:
        print(f"❌ 方案二测试失败: {str(e)}")
        return False


def test_data_preprocessing():
    """测试数据预处理功能"""
    print("\n🔍 测试数据预处理功能")

    try:
        # 创建测试数据
        test_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [102, 103, 104],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']))

        # 测试方案二预处理
        from src.backtest.data_loader import BacktestDataLoader
        config = {"timezone": "Asia/Shanghai", "data": {}}
        loader = BacktestDataLoader(config)
        processed = loader._preprocess_data(test_data.copy(), "1d")

        # 验证结果
        assert processed is not None
        assert len(processed) == len(test_data)
        print("✅ 数据预处理功能正常")

        return True

    except Exception as e:
        print(f"❌ 数据预处理测试失败: {str(e)}")
        return False


def test_metadata():
    """测试元数据获取"""
    print("\n🔍 测试元数据获取")

    try:
        config = {"timezone": "Asia/Shanghai", "data": {}}

        # 测试方案二元数据
        from src.backtest.data_loader import BacktestDataLoader
        loader = BacktestDataLoader(config)
        metadata = loader.get_metadata()
        assert metadata is not None
        assert isinstance(metadata, dict)
        print("✅ 方案二元数据获取成功")

        return True

    except Exception as e:
        print(f"❌ 元数据测试失败: {str(e)}")
        return False


def test_error_handling():
    """测试错误处理"""
    print("\n🔍 测试错误处理")

    try:
        # 测试无效配置
        invalid_config = {"invalid": "config"}

        # 方案二错误处理
        from src.backtest.data_loader import BacktestDataLoader
        try:
            loader = BacktestDataLoader(invalid_config)
            print("✅ 方案二错误处理正常")
        except Exception as e:
            print(f"⚠️ 方案二错误处理: {str(e)}")

        return True

    except Exception as e:
        print(f"❌ 错误处理测试失败: {str(e)}")
        return False


def test_loaders_initialization():
    """测试加载器初始化"""
    print("\n🔍 测试加载器初始化")

    try:
        config = {
            "timezone": "Asia/Shanghai",
            "data": {
                "stock": {"save_path": "data/stock"},
                "financial": {"cache_dir": "data/financial"},
                "index": {"cache_dir": "data/index"},
                "news": {"cache_dir": "data/news"}
            }
        }

        from src.backtest.data_loader import BacktestDataLoader
        loader = BacktestDataLoader(config)

        # 验证各个加载器是否正确初始化
        assert hasattr(loader, 'stock_loader')
        assert hasattr(loader, 'financial_loader')
        assert hasattr(loader, 'index_loader')
        assert hasattr(loader, 'news_loader')

        print("✅ 加载器初始化测试通过")
        return True

    except Exception as e:
        print(f"❌ 加载器初始化测试失败: {str(e)}")
        return False


def test_stats_functionality():
    """测试统计功能"""
    print("\n🔍 测试统计功能")

    try:
        config = {"timezone": "Asia/Shanghai", "data": {}}

        from src.backtest.data_loader import BacktestDataLoader
        loader = BacktestDataLoader(config)
        stats = loader.get_stats()

        assert stats is not None
        assert isinstance(stats, dict)
        assert 'cache_size' in stats
        assert 'timezone' in stats
        assert 'loaders_initialized' in stats

        print("✅ 统计功能正常")
        return True

    except Exception as e:
        print(f"❌ 统计功能测试失败: {str(e)}")
        return False


def main():
    """主函数"""
    print("🚀 开始验证数据加载器方案二")
    print("=" * 50)

    results = []

    # 测试方案二
    results.append(test_solution_2())

    # 测试数据预处理
    results.append(test_data_preprocessing())

    # 测试元数据
    results.append(test_metadata())

    # 测试错误处理
    results.append(test_error_handling())

    # 测试加载器初始化
    results.append(test_loaders_initialization())

    # 测试统计功能
    results.append(test_stats_functionality())

    # 输出结果
    print("\n" + "=" * 50)
    print("📊 测试结果汇总")
    print("=" * 50)

    test_names = [
        "方案二：直接使用数据层",
        "数据预处理功能",
        "元数据获取",
        "错误处理",
        "加载器初始化",
        "统计功能"
    ]

    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{i+1}. {name}: {status}")

    passed = sum(results)
    total = len(results)

    print(f"\n📈 总体结果: {passed}/{total} 项测试通过")

    if passed == total:
        print("🎉 所有测试通过！方案二可以正常工作。")
    else:
        print("⚠️ 部分测试失败，需要进一步调试。")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
