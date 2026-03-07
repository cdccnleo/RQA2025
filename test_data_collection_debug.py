#!/usr/bin/env python3
"""
调试数据采集问题的测试脚本

专门用于诊断为什么数据采集返回0条记录的问题
"""

import sys
import os
import asyncio

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_data_collection_detailed():
    """详细测试数据采集过程"""
    print("🔍 详细数据采集调试")
    print("=" * 60)

    try:
        # 1. 测试配置解析
        print("\n1. 配置解析测试:")
        from src.gateway.web.data_collectors import get_akshare_function_config, get_supported_data_types

        supported_types = get_supported_data_types()
        print(f"支持的数据类型: {supported_types}")

        # 2. 测试函数映射
        print("\n2. 函数映射测试:")
        test_mappings = {
            'daily': 'stock_zh_a_hist',
            '1min': 'stock_zh_a_hist_min_em',
            'realtime': 'stock_zh_a_spot_em'
        }

        for data_type, expected_func in test_mappings.items():
            config = get_akshare_function_config(data_type)
            if config and config['function'] == expected_func:
                print(f"✅ {data_type} -> {expected_func}")
            else:
                print(f"❌ {data_type} -> 期望 {expected_func}, 实际 {config['function'] if config else 'None'}")

        # 3. 测试数据源配置
        print("\n3. 数据源配置测试:")
        with open('data/data_sources_config.json', 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        akshare_config = None
        for source in config_data:
            if source.get('id') == 'akshare_stock_a':
                akshare_config = source
                break

        if akshare_config:
            source_config = akshare_config.get('config', {})
            print(f"数据源ID: {akshare_config.get('id')}")
            print(f"数据源类型: {akshare_config.get('type')}")

            # 检查新的配置结构
            if 'data_type_configs' in source_config:
                dt_configs = source_config['data_type_configs']
                enabled_types = [dt for dt, config in dt_configs.items() if config.get('enabled', False)]
                print(f"启用的数据类型: {enabled_types}")
                print(f"配置的股票: {source_config.get('custom_stocks', [])}")
            else:
                print("❌ 未找到 data_type_configs 配置")
        else:
            print("❌ 未找到 akshare_stock_a 数据源")

        # 4. 模拟数据采集调用
        print("\n4. 模拟数据采集调用:")
        try:
            from src.gateway.web.data_collectors import collect_from_akshare_adapter

            # 模拟配置
            test_config = {
                "id": "akshare_stock_a",
                "type": "股票数据",
                "config": {
                    "akshare_category": "A股",
                    "custom_stocks": ["000001"],  # 使用更常见的股票代码测试
                    "data_type_configs": {
                        "daily": {"enabled": True, "description": "日线数据"}
                    }
                }
            }

            print("调用数据采集函数...")
            result = await collect_from_akshare_adapter(test_config, None, None)

            print("采集结果:")
            print(f"  返回类型: {type(result)}")
            if isinstance(result, list):
                print(f"  记录数量: {len(result)}")
                if result:
                    print(f"  示例记录: {result[0]}")
            elif isinstance(result, dict):
                print(f"  字典结构: {result.keys()}")
                print(f"  数据条数: {len(result.get('data', []))}")

        except Exception as e:
            print(f"❌ 数据采集调用失败: {e}")
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"❌ 测试过程出错: {e}")
        import traceback
        traceback.print_exc()

async def test_akshare_direct():
    """直接测试AKShare API调用"""
    print("\n5. 直接测试AKShare API:")
    print("=" * 40)

    try:
        import akshare as ak
        print(f"AKShare版本: {ak.__version__}")

        # 测试股票代码
        test_symbols = ["000001", "000002"]  # 使用更常见的股票代码

        for symbol in test_symbols:
            print(f"\n测试股票: {symbol}")

            try:
                # 测试日线数据
                print("  测试日线数据...")
                df_daily = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date="20241201",
                    end_date="20241210",
                    adjust="hfq"
                )
                print(f"    日线数据: {len(df_daily) if df_daily is not None else 0} 条")

                # 测试实时数据
                print("  测试实时数据...")
                df_realtime = ak.stock_zh_a_spot_em()
                if df_realtime is not None and not df_realtime.empty:
                    filtered = df_realtime[df_realtime['代码'] == symbol]
                    print(f"    实时数据: {len(filtered)} 条 (全市场: {len(df_realtime)} 条)")
                else:
                    print("    实时数据: 获取失败"

            except Exception as e:
                print(f"    ❌ API调用失败: {e}")

    except ImportError:
        print("❌ AKShare未安装")
    except Exception as e:
        print(f"❌ AKShare测试失败: {e}")

def analyze_config_file():
    """分析配置文件"""
    print("\n6. 配置文件分析:")
    print("=" * 30)

    try:
        with open('data/data_sources_config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)

        print(f"配置文件包含 {len(config)} 个数据源")

        for source in config:
            if source.get('id') == 'akshare_stock_a':
                print(f"\nAKShare数据源详情:")
                print(f"  ID: {source.get('id')}")
                print(f"  名称: {source.get('name')}")
                print(f"  类型: {source.get('type')}")
                print(f"  状态: {source.get('status')}")

                source_config = source.get('config', {})
                print(f"  股票池类型: {source_config.get('stock_pool_type')}")
                print(f"  自定义股票: {source_config.get('custom_stocks', [])}")

                if 'data_type_configs' in source_config:
                    dt_configs = source_config['data_type_configs']
                    enabled = [dt for dt, cfg in dt_configs.items() if cfg.get('enabled')]
                    print(f"  启用的数据类型: {enabled}")
                else:
                    print("  ❌ 缺少 data_type_configs 配置")

                break

    except Exception as e:
        print(f"❌ 配置文件分析失败: {e}")

async def main():
    """主函数"""
    print("🐛 数据采集问题深度调试")
    print("=" * 60)
    print("逐步诊断数据采集返回0条记录的根本原因")
    print("=" * 60)

    # 执行各项测试
    await test_data_collection_detailed()
    await test_akshare_direct()
    analyze_config_file()

    print("\n" + "=" * 60)
    print("🔍 诊断建议:")
    print("1. 检查股票代码是否正确（AKShare需要6位数字代码）")
    print("2. 确认日期范围是否合理（不能是未来日期）")
    print("3. 验证AKShare API是否可以正常访问")
    print("4. 检查网络连接和API频率限制")
    print("5. 确认数据类型配置是否正确启用")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())