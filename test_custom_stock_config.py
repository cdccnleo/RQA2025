#!/usr/bin/env python3
"""
测试自选股数据采集配置
"""

import sys
sys.path.append('.')

def test_custom_stock_config():
    """测试自选股配置"""
    print("🧪 测试自选股数据采集配置")

    try:
        # 1. 测试股票池选择逻辑
        from src.gateway.web.data_collectors import get_core_universe_symbols, get_extended_universe_symbols

        # 模拟股票列表（使用真实的股票代码格式）
        all_symbols = [f"{i:06d}" for i in range(1, 100)]  # 000001到000099

        print("1. 测试核心股票池选择...")
        core_symbols = get_core_universe_symbols(all_symbols, 10)
        print(f"   核心池选择结果: {len(core_symbols)}只股票")
        print(f"   示例股票: {core_symbols[:3] if core_symbols else '无'}")

        print("2. 测试扩展股票池选择...")
        extended_symbols = get_extended_universe_symbols(all_symbols, 15)
        print(f"   扩展池选择结果: {len(extended_symbols)}只股票")
        print(f"   示例股票: {extended_symbols[:3] if extended_symbols else '无'}")

        # 2. 测试配置验证
        print("3. 测试配置验证...")
        try:
            from src.gateway.web.datasource_routes import validate_stock_config

            # 测试有效的自选股配置
            valid_config = {
                "stock_pool_type": "custom",
                "custom_stocks": ["000001", "000002", "600036"],
                "data_types": ["daily", "realtime"],
                "batch_size": 20
            }

            result = validate_stock_config(valid_config)
            print(f"   有效配置验证结果: {result['valid']}")
            print(f"   错误数量: {len(result['errors'])}")
            print(f"   警告数量: {len(result['warnings'])}")

            # 测试无效配置
            invalid_config = {
                "stock_pool_type": "custom",
                "custom_stocks": ["invalid", "000001"],
                "batch_size": 200
            }

            invalid_result = validate_stock_config(invalid_config)
            print(f"   无效配置验证结果: {invalid_result['valid']}")
            print(f"   错误: {invalid_result['errors']}")

        except Exception as e:
            print(f"   配置验证测试失败: {e}")

        print("✅ 自选股配置测试完成")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = test_custom_stock_config()
    print(f"\n测试结果: {'通过' if result else '失败'}")