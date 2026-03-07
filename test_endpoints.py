#!/usr/bin/env python3
"""
测试RQA2025后端API端点连通性
"""

import asyncio
import aiohttp
import json
from typing import Dict, Any

async def test_endpoint(url: str, expected_status: int = 200) -> Dict[str, Any]:
    """测试单个端点"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                result = {
                    "url": url,
                    "status": response.status,
                    "success": response.status == expected_status,
                    "headers": dict(response.headers),
                }

                if response.status == expected_status:
                    try:
                        data = await response.json()
                        result["data"] = data
                    except:
                        try:
                            text = await response.text()
                            result["data"] = text[:200]  # 限制响应长度
                        except:
                            result["data"] = "无法解析响应"
                else:
                    result["error"] = f"HTTP {response.status}"

                return result
    except Exception as e:
        return {
            "url": url,
            "status": None,
            "success": False,
            "error": str(e)
        }

async def test_all_endpoints(base_url: str = "http://localhost:8000"):
    """测试所有关键端点"""
    endpoints = [
        ("/health", 200),
        ("/ready", 200),
        ("/", 200),
        ("/docs", 200),
        ("/api/v1/status", 200),
    ]

    print(f"🧪 测试后端API端点连通性: {base_url}")
    print("=" * 60)

    results = []
    for endpoint, expected_status in endpoints:
        url = f"{base_url}{endpoint}"
        print(f"测试端点: {endpoint}")
        result = await test_endpoint(url, expected_status)
        results.append(result)

        if result["success"]:
            print(f"  ✅ 成功 (HTTP {result['status']})")
            if "data" in result and isinstance(result["data"], dict):
                status = result["data"].get("status", "unknown")
                print(f"     状态: {status}")
        else:
            print(f"  ❌ 失败: {result.get('error', '未知错误')}")

        print()

    # 汇总结果
    successful = sum(1 for r in results if r["success"])
    total = len(results)

    print("=" * 60)
    print(f"测试结果: {successful}/{total} 个端点正常")
    print()

    if successful == total:
        print("🎉 所有端点测试通过！后端服务运行正常。")
    else:
        print("⚠️  部分端点测试失败，请检查后端服务配置。")

    return results

async def test_frontend_proxy(base_url: str = "http://localhost:8080"):
    """测试前端nginx代理"""
    print(f"🧪 测试前端nginx代理: {base_url}")
    print("=" * 60)

    # 测试前端页面
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/", timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    print("✅ 前端页面可访问")
                    content = await response.text()
                    if "RQA2025" in content:
                        print("✅ 前端页面内容正确")
                    else:
                        print("⚠️  前端页面内容可能有问题")
                else:
                    print(f"❌ 前端页面访问失败: HTTP {response.status}")
    except Exception as e:
        print(f"❌ 前端页面访问异常: {e}")

    print()

if __name__ == "__main__":
    async def main():
        # 测试后端API
        await test_all_endpoints()

        # 测试前端nginx（如果运行的话）
        try:
            await test_frontend_proxy()
        except:
            print("前端nginx测试跳过（可能未运行）")

    asyncio.run(main())