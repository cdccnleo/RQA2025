#!/usr/bin/env python3
"""
nginx配置修复测试脚本

测试nginx配置修复后，历史数据采集监控API是否正常工作
"""

import requests
import json
import time
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_api_endpoints():
    """测试API端点"""
    base_url = "http://localhost"
    endpoints = [
        "/api/v1/monitoring/historical-collection/status",
        "/api/v1/monitoring/historical-collection/scheduler/status",
        "/api/v1/monitoring/data-collection/alerts?resolved=true",  # 对比测试
    ]

    print("🔍 测试API端点..."    print("=" * 60)

    results = {}
    for endpoint in endpoints:
        try:
            print(f"测试端点: {endpoint}")
            response = requests.get(f"{base_url}{endpoint}", timeout=10)

            if response.status_code == 200:
                print(f"  ✅ 状态码: {response.status_code} - 正常")
                results[endpoint] = "SUCCESS"
            else:
                print(f"  ❌ 状态码: {response.status_code} - 失败")
                results[endpoint] = f"FAILED_{response.status_code}"

        except requests.exceptions.RequestException as e:
            print(f"  ❌ 连接失败: {e}")
            results[endpoint] = f"ERROR_{str(e)}"
        except Exception as e:
            print(f"  ❌ 其他错误: {e}")
            results[endpoint] = f"UNKNOWN_{str(e)}"

        print()

    return results

def test_websocket_connection():
    """测试WebSocket连接"""
    print("🔍 测试WebSocket连接...")
    print("=" * 60)

    try:
        import websocket
        ws_url = "ws://localhost/ws/historical-collection"

        print(f"连接到: {ws_url}")

        # 创建WebSocket连接
        ws = websocket.create_connection(ws_url, timeout=10)

        # 发送订阅消息
        subscribe_msg = {
            "type": "subscribe",
            "topics": ["historical_collection_status"]
        }
        ws.send(json.dumps(subscribe_msg))
        print("  ✅ 发送订阅消息成功")

        # 接收响应
        response = ws.recv()
        data = json.loads(response)
        if data.get("type") == "subscribed":
            print("  ✅ 收到订阅确认消息")
            result = "SUCCESS"
        else:
            print(f"  ⚠️ 收到意外消息: {data}")
            result = "UNEXPECTED_RESPONSE"

        ws.close()
        print("  ✅ WebSocket连接关闭")

    except ImportError:
        print("  ⚠️ websocket-client库未安装，跳过WebSocket测试")
        print("    安装命令: pip install websocket-client")
        result = "LIBRARY_MISSING"
    except Exception as e:
        print(f"  ❌ WebSocket测试失败: {e}")
        result = f"ERROR_{str(e)}"

    print()
    return result

def test_nginx_health():
    """测试nginx健康状态"""
    print("🔍 测试nginx健康状态...")
    print("=" * 60)

    try:
        response = requests.get("http://localhost/health", timeout=5)

        if response.status_code == 200:
            content = response.text.strip()
            if "healthy" in content.lower():
                print("  ✅ nginx健康检查通过")
                return "SUCCESS"
            else:
                print(f"  ⚠️ nginx响应内容异常: {content}")
                return "UNEXPECTED_CONTENT"
        else:
            print(f"  ❌ nginx健康检查失败，状态码: {response.status_code}")
            return f"FAILED_{response.status_code}"

    except Exception as e:
        print(f"  ❌ nginx健康检查连接失败: {e}")
        return f"ERROR_{str(e)}"

def main():
    """主函数"""
    print("🚀 RQA2025 Nginx配置修复测试")
    print("=" * 60)
    print("此脚本测试nginx配置修复后，历史数据采集监控API是否正常工作")
    print()

    # 测试nginx健康状态
    nginx_health = test_nginx_health()

    if "SUCCESS" not in nginx_health:
        print("❌ nginx未正常运行，请先确保RQA2025系统已启动")
        print("启动命令: docker-compose -f docker-compose.prod.yml up -d")
        return 1

    # 测试API端点
    api_results = test_api_endpoints()

    # 测试WebSocket
    ws_result = test_websocket_connection()

    # 汇总结果
    print("=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)

    all_success = True

    # 检查API结果
    print("\n🔗 API端点测试:")
    for endpoint, result in api_results.items():
        status = "✅" if "SUCCESS" in result else "❌"
        print(f"  {status} {endpoint}: {result}")
        if "SUCCESS" not in result:
            all_success = False

    # 检查WebSocket结果
    print("
🔌 WebSocket测试:"    ws_status = "✅" if ws_result in ["SUCCESS", "LIBRARY_MISSING"] else "❌"
    print(f"  {ws_status} WebSocket连接: {ws_result}")

    if ws_result == "LIBRARY_MISSING":
        all_success = False
        print("    💡 请安装websocket-client库: pip install websocket-client")

    # 最终结果
    print("
🎯 最终结果:"    if all_success:
        print("  ✅ 所有测试通过！nginx配置修复成功")
        print("  🎉 历史数据采集监控功能现已正常工作")
        print("  📱 访问监控页面: http://localhost/data-collection-monitor.html")
        return 0
    else:
        print("  ❌ 部分测试失败，请检查nginx配置和容器状态")
        print("  🔧 重启nginx: docker restart rqa2025-nginx")
        print("  📋 查看日志: docker logs rqa2025-nginx")
        return 1

if __name__ == "__main__":
    exit_code = main()
    print(f"\n程序退出码: {exit_code}")
    sys.exit(exit_code)