#!/usr/bin/env python3
"""
CI/CD路由验证脚本
完整的路由验证流程，包括健康检查和功能测试
"""

import sys
import os
import subprocess

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def run_command(cmd: list, description: str) -> bool:
    """运行命令并返回是否成功"""
    print(f"\n{'=' * 80}")
    print(f"  {description}")
    print("=" * 80)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        print(result.stdout)
        if result.stderr:
            print("错误输出:", result.stderr)
        
        if result.returncode != 0:
            print(f"❌ {description} 失败 (退出码: {result.returncode})")
            return False
        else:
            print(f"✅ {description} 成功")
            return True
    except subprocess.TimeoutError:
        print(f"❌ {description} 超时")
        return False
    except Exception as e:
        print(f"❌ {description} 执行失败: {e}")
        return False


def main():
    """主函数"""
    print("=" * 80)
    print("  CI/CD 路由验证")
    print("=" * 80)
    print()
    
    results = {}
    
    # 1. 路由健康检查
    results["健康检查"] = run_command(
        [sys.executable, "scripts/check_routes_health.py"],
        "路由健康检查"
    )
    
    # 2. API端点测试
    results["API端点测试"] = run_command(
        [sys.executable, "-m", "pytest", 
         "tests/dashboard_verification/test_api_endpoints.py", "-v"],
        "API端点测试"
    )
    
    # 3. WebSocket测试
    results["WebSocket测试"] = run_command(
        [sys.executable, "-m", "pytest",
         "tests/dashboard_verification/test_websocket_connections.py", "-v"],
        "WebSocket连接测试"
    )
    
    # 4. 业务流程数据流测试
    results["业务流程测试"] = run_command(
        [sys.executable, "-m", "pytest",
         "tests/dashboard_verification/test_business_process_flow.py", "-v"],
        "业务流程数据流测试"
    )
    
    # 5. 页面加载测试
    results["页面加载测试"] = run_command(
        [sys.executable, "-m", "pytest",
         "tests/dashboard_verification/test_dashboard_pages.py", "-v"],
        "页面加载测试"
    )
    
    # 汇总结果
    print("\n" + "=" * 80)
    print("  验证结果汇总")
    print("=" * 80)
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed
    
    print(f"总计: {total}")
    print(f"通过: {passed} ✅")
    print(f"失败: {failed} ❌")
    print(f"通过率: {passed/total*100:.1f}%")
    print()
    
    print("详细结果:")
    for name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name}: {status}")
    
    print("=" * 80)
    print()
    
    if failed == 0:
        print("✅ 所有验证通过！")
        return 0
    else:
        print(f"❌ {failed} 项验证失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())

