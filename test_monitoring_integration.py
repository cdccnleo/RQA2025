#!/usr/bin/env python3
"""
测试数据采集监控集成功能

验证监控仪表板集成的各项功能是否正常工作
"""

import sys
import os
import asyncio
import json
from typing import Dict, Any

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_monitoring_apis():
    """测试监控API的基本连通性"""
    import requests

    base_url = "http://localhost:8000"
    apis_to_test = [
        "/api/v1/monitoring/data-collection/health",
        "/api/v1/monitoring/data-collection/metrics",
        "/api/v1/monitoring/cache/stats",
        "/api/v1/monitoring/data-collection/alerts?resolved=false",
        "/api/v1/monitoring/data-collection/alerts?resolved=true"
    ]

    print("🔍 测试监控API连通性")
    print("=" * 50)

    results = {}

    for api in apis_to_test:
        try:
            print(f"测试: {api}")
            response = requests.get(f"{base_url}{api}", timeout=10)

            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"  ✅ 成功 (状态码: {response.status_code})")
                    results[api] = {"status": "success", "data": data}
                except json.JSONDecodeError:
                    print(f"  ⚠️ 响应不是有效的JSON")
                    results[api] = {"status": "invalid_json", "response": response.text[:200]}
            else:
                print(f"  ❌ 失败 (状态码: {response.status_code})")
                results[api] = {"status": "error", "status_code": response.status_code}

        except requests.exceptions.RequestException as e:
            print(f"  ❌ 连接失败: {e}")
            results[api] = {"status": "connection_error", "error": str(e)}
        except Exception as e:
            print(f"  ❌ 未知错误: {e}")
            results[api] = {"status": "unknown_error", "error": str(e)}

        print()

    return results

def test_cache_operations():
    """测试缓存操作API"""
    import requests

    base_url = "http://localhost:8000"
    operations = [
        ("GET", "/api/v1/monitoring/cache/stats"),
        ("POST", "/api/v1/monitoring/cache/clear"),
        ("POST", "/api/v1/monitoring/cache/clear?api_name=stock_zh_a_daily")
    ]

    print("🔧 测试缓存操作API")
    print("=" * 50)

    results = {}

    for method, api in operations:
        try:
            print(f"测试 {method}: {api}")

            if method == "GET":
                response = requests.get(f"{base_url}{api}", timeout=10)
            else:
                response = requests.post(f"{base_url}{api}", timeout=10)

            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"  ✅ 成功 (状态码: {response.status_code})")
                    results[f"{method} {api}"] = {"status": "success", "data": data}
                except json.JSONDecodeError:
                    print(f"  ⚠️ 响应不是有效的JSON")
                    results[f"{method} {api}"] = {"status": "invalid_json", "response": response.text[:200]}
            else:
                print(f"  ❌ 失败 (状态码: {response.status_code})")
                results[f"{method} {api}"] = {"status": "error", "status_code": response.status_code, "response": response.text[:200]}

        except requests.exceptions.RequestException as e:
            print(f"  ❌ 连接失败: {e}")
            results[f"{method} {api}"] = {"status": "connection_error", "error": str(e)}
        except Exception as e:
            print(f"  ❌ 未知错误: {e}")
            results[f"{method} {api}"] = {"status": "unknown_error", "error": str(e)}

        print()

    return results

def analyze_results(monitoring_results: Dict[str, Any], cache_results: Dict[str, Any]):
    """分析测试结果"""
    print("📊 测试结果分析")
    print("=" * 50)

    # 分析监控API
    monitoring_success = 0
    monitoring_total = len(monitoring_results)

    for api, result in monitoring_results.items():
        if result["status"] == "success":
            monitoring_success += 1

    # 分析缓存API
    cache_success = 0
    cache_total = len(cache_results)

    for operation, result in cache_results.items():
        if result["status"] == "success":
            cache_success += 1

    print(f"监控API: {monitoring_success}/{monitoring_total} 通过")
    print(f"缓存API: {cache_success}/{cache_total} 通过")
    print(f"总体成功率: {(monitoring_success + cache_success) / (monitoring_total + cache_total) * 100:.1f}%")

    # 详细分析
    print("\n📋 详细结果:")

    if monitoring_success < monitoring_total:
        print("⚠️  监控API问题:")
        for api, result in monitoring_results.items():
            if result["status"] != "success":
                print(f"  • {api}: {result['status']}")

    if cache_success < cache_total:
        print("⚠️  缓存API问题:")
        for operation, result in cache_results.items():
            if result["status"] != "success":
                print(f"  • {operation}: {result['status']}")

    # 检查数据结构
    if "/api/v1/monitoring/data-collection/health" in monitoring_results:
        health_result = monitoring_results["/api/v1/monitoring/data-collection/health"]
        if health_result["status"] == "success":
            data = health_result["data"]
            expected_fields = ["overall_health_score", "total_sources", "healthy_sources", "overall_success_rate"]
            missing_fields = [field for field in expected_fields if field not in data]
            if missing_fields:
                print(f"⚠️  健康检查API缺少字段: {missing_fields}")
            else:
                print("✅ 健康检查API数据结构正确")

    return (monitoring_success + cache_success) == (monitoring_total + cache_total)

async def main():
    """主函数"""
    print("🧪 数据采集监控集成测试")
    print("=" * 60)
    print("此测试验证监控仪表板集成的各项功能")
    print("=" * 60)

    # 检查服务器是否运行
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            print("❌ 服务器未正常运行，请先启动服务器")
            return 1
    except:
        print("❌ 无法连接到服务器，请确保服务器正在运行在 http://localhost:8000")
        return 1

    # 1. 测试监控API
    print("\n" + "=" * 30)
    monitoring_results = test_monitoring_apis()

    # 2. 测试缓存操作API
    print("\n" + "=" * 30)
    cache_results = test_cache_operations()

    # 3. 分析结果
    print("\n" + "=" * 30)
    all_passed = analyze_results(monitoring_results, cache_results)

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有监控集成测试通过！")
        print("监控仪表板已成功集成到数据源配置页面。")
        print("\n📋 验证步骤:")
        print("1. 访问: http://localhost:8080/data-sources-config.html")
        print("2. 滚动到'数据采集监控仪表板'部分")
        print("3. 检查健康状态、详细指标、缓存统计和告警信息")
        print("4. 测试缓存清理功能")
    else:
        print("⚠️  部分测试失败，请检查上述问题")
        print("\n🔧 故障排除:")
        print("1. 确保所有监控相关的Python模块已正确安装")
        print("2. 检查API端点是否正确实现")
        print("3. 查看服务器日志获取详细错误信息")

    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)