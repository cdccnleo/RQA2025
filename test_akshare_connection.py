#!/usr/bin/env python3
"""
测试AKShare连接性
用于诊断数据采集错误
"""

import sys
import os
import traceback

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_akshare_import():
    """测试AKShare导入"""
    try:
        import akshare
        print(f"✅ AKShare导入成功，版本: {akshare.__version__}")
        return True
    except ImportError as e:
        print(f"❌ AKShare导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ AKShare导入异常: {e}")
        traceback.print_exc()
        return False

def test_akshare_connection():
    """测试AKShare网络连接"""
    try:
        import akshare
        print("🔄 正在测试AKShare连接...")

        # 测试A股实时数据（最基本的测试）
        try:
            df = akshare.stock_zh_a_spot_em()
            print(f"✅ A股实时数据获取成功，数据行数: {len(df)}")
            return True
        except Exception as e:
            print(f"❌ A股实时数据获取失败: {e}")
            return False

    except Exception as e:
        print(f"❌ AKShare连接测试异常: {e}")
        traceback.print_exc()
        return False

def test_network_connectivity():
    """测试网络连通性"""
    try:
        import requests
        print("🔄 正在测试网络连通性...")

        # 测试akfamily.xyz连接
        try:
            response = requests.get("https://akshare.akfamily.xyz", timeout=10)
            print(f"✅ AKShare官网连接成功，状态码: {response.status_code}")
            return True
        except Exception as e:
            print(f"❌ AKShare官网连接失败: {e}")
            return False

    except Exception as e:
        print(f"❌ 网络连通性测试异常: {e}")
        return False

def main():
    """主函数"""
    print("🚀 开始AKShare连接诊断...")
    print("=" * 50)

    results = {}

    # 1. 测试导入
    print("\n1. 测试AKShare导入:")
    results['import'] = test_akshare_import()

    # 2. 测试网络连通性
    print("\n2. 测试网络连通性:")
    results['network'] = test_network_connectivity()

    # 3. 测试AKShare连接（只有在导入成功时才测试）
    if results['import']:
        print("\n3. 测试AKShare数据获取:")
        results['connection'] = test_akshare_connection()
    else:
        results['connection'] = False
        print("\n3. AKShare数据获取测试跳过（导入失败）")

    print("\n" + "=" * 50)
    print("📊 诊断结果汇总:")

    for test_name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {test_name}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 所有测试通过！数据采集应该正常工作。")
        return 0
    else:
        print("\n⚠️  部分测试失败，数据采集可能存在问题。")
        print("建议检查:")
        if not results['network']:
            print("  - 网络连接问题")
        if not results['import']:
            print("  - AKShare库安装问题")
        if not results['connection']:
            print("  - AKShare API访问问题")
        return 1

if __name__ == "__main__":
    sys.exit(main())