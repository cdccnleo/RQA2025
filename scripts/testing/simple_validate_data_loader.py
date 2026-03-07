#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BacktestDataLoader 简化验证脚本
避免环境问题，只测试基本功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def simple_validate():
    """简化验证BacktestDataLoader基本功能"""
    print("🚀 开始简化验证 BacktestDataLoader")
    print("=" * 50)

    results = []

    # 1. 测试导入
    try:
        from src.backtest.data_loader import BacktestDataLoader
        print("✅ 导入成功")
        results.append(True)
    except Exception as e:
        print(f"❌ 导入失败: {str(e)}")
        results.append(False)
        return False

    # 2. 测试初始化
    try:
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

        loader = BacktestDataLoader(config)
        assert loader is not None
        assert loader.timezone == "Asia/Shanghai"
        print("✅ 初始化成功")
        results.append(True)
    except Exception as e:
        print(f"❌ 初始化失败: {str(e)}")
        results.append(False)

    # 3. 测试元数据获取
    try:
        metadata = loader.get_metadata()
        assert metadata is not None
        assert isinstance(metadata, dict)
        print("✅ 元数据获取成功")
        results.append(True)
    except Exception as e:
        print(f"❌ 元数据获取失败: {str(e)}")
        results.append(False)

    # 4. 测试统计信息获取
    try:
        stats = loader.get_stats()
        assert stats is not None
        assert isinstance(stats, dict)
        print("✅ 统计信息获取成功")
        results.append(True)
    except Exception as e:
        print(f"❌ 统计信息获取失败: {str(e)}")
        results.append(False)

    # 5. 测试缓存功能
    try:
        loader.clear_cache()
        assert len(loader.cache) == 0
        print("✅ 缓存功能正常")
        results.append(True)
    except Exception as e:
        print(f"❌ 缓存功能失败: {str(e)}")
        results.append(False)

    # 6. 测试未实现的功能
    try:
        # 测试财务数据加载（未实现）
        fundamental_result = loader.load_fundamental("000001", "2023-01-01", "2023-01-31")
        assert fundamental_result is not None

        # 测试新闻数据加载（未实现）
        news_result = loader.load_news("000001", "2023-01-01", "2023-01-31")
        assert news_result is not None

        # 测试指数数据加载（未实现）
        index_result = loader.load_index("000300", "2023-01-01", "2023-01-31")
        assert index_result is not None

        print("✅ 未实现功能测试成功")
        results.append(True)
    except Exception as e:
        print(f"❌ 未实现功能测试失败: {str(e)}")
        results.append(False)

    # 7. 测试错误处理
    try:
        # 测试无股票加载器时的错误处理
        empty_config = {"timezone": "Asia/Shanghai", "data": {}}
        empty_loader = BacktestDataLoader(empty_config)

        try:
            empty_loader.load_ohlcv("000001", "2023-01-01", "2023-01-10")
            print("⚠️ 错误处理测试：应该抛出异常但没有")
            results.append(False)
        except RuntimeError as e:
            if "股票数据加载器未初始化" in str(e):
                print("✅ 错误处理正常")
                results.append(True)
            else:
                print(f"❌ 错误处理异常类型不正确: {str(e)}")
                results.append(False)
    except Exception as e:
        print(f"❌ 错误处理测试失败: {str(e)}")
        results.append(False)

    # 8. 测试股票池加载
    try:
        universe = loader.load_universe("2023-01-01")
        assert universe is not None
        assert isinstance(universe, list)
        assert len(universe) > 0
        print("✅ 股票池加载成功")
        results.append(True)
    except Exception as e:
        print(f"❌ 股票池加载失败: {str(e)}")
        results.append(False)

    # 9. 测试时区处理
    try:
        utc_config = {"timezone": "UTC", "data": {}}
        utc_loader = BacktestDataLoader(utc_config)
        assert utc_loader.timezone == "UTC"
        print("✅ 时区处理正常")
        results.append(True)
    except Exception as e:
        print(f"❌ 时区处理失败: {str(e)}")
        results.append(False)

    # 输出结果
    print("\n" + "=" * 50)
    print("📊 简化验证结果汇总")
    print("=" * 50)

    test_names = [
        "导入",
        "初始化",
        "元数据获取",
        "统计信息获取",
        "缓存功能",
        "未实现功能",
        "错误处理",
        "股票池加载",
        "时区处理"
    ]

    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{i+1}. {name}: {status}")

    passed = sum(results)
    total = len(results)

    print(f"\n📈 总体结果: {passed}/{total} 项测试通过")

    if passed == total:
        print("🎉 所有测试通过！BacktestDataLoader 基本功能正常。")
        return True
    else:
        print("⚠️ 部分测试失败，需要进一步调试。")
        return False


def main():
    """主函数"""
    print("🚀 BacktestDataLoader 简化验证")
    print("=" * 60)

    success = simple_validate()

    print("\n" + "=" * 60)
    print("🎯 最终验证结果")
    print("=" * 60)

    if success:
        print("🎉 简化验证通过！BacktestDataLoader 可以正常使用。")
        return True
    else:
        print("⚠️ 简化验证失败，需要进一步调试。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
