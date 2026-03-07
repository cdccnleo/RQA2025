#!/usr/bin/env python3
"""
验证数据采集修复是否有效
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_import():
    """测试模块导入"""
    try:
        from src.gateway.web.data_collectors import collect_from_akshare_adapter
        print("✅ 模块导入成功")
        return True
    except Exception as e:
        print(f"❌ 模块导入失败: {e}")
        return False

def test_function_signature():
    """测试函数签名"""
    try:
        import inspect
        from src.gateway.web.data_collectors import collect_from_akshare_adapter

        sig = inspect.signature(collect_from_akshare_adapter)
        params = list(sig.parameters.keys())

        expected_params = ['source_config', 'request_data', 'existing_dates_by_type']
        if all(param in params for param in expected_params):
            print("✅ 函数签名正确")
            return True
        else:
            print(f"❌ 函数签名不完整，期望参数: {expected_params}，实际参数: {params}")
            return False

    except Exception as e:
        print(f"❌ 函数签名检查失败: {e}")
        return False

def main():
    """主函数"""
    print("🔧 验证数据采集修复...")
    print("=" * 50)

    tests = [
        ("模块导入", test_import),
        ("函数签名", test_function_signature),
    ]

    results = {}
    for test_name, test_func in tests:
        print(f"\n测试: {test_name}")
        results[test_name] = test_func()

    print("\n" + "=" * 50)
    print("📊 验证结果:")

    all_passed = True
    for test_name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {test_name}: {status}")
        if not success:
            all_passed = False

    if all_passed:
        print("\n🎉 所有验证通过！修复成功。")
        print("现在可以重新启动容器测试数据采集功能。")
    else:
        print("\n⚠️  部分验证失败，需要进一步修复。")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())