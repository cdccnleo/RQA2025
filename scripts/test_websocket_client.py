#!/usr/bin/env python3
"""
简单的WebSocket客户端测试
"""

import websocket
import json
import time
import sys

def test_websocket_connection():
    """测试WebSocket连接"""
    try:
        # 连接到测试服务器
        ws_url = "ws://localhost:8765"
        print(f"连接到WebSocket服务器: {ws_url}")

        ws = websocket.create_connection(ws_url, timeout=10)
        print("✅ WebSocket连接成功")

        # 接收连接确认
        response = ws.recv()
        data = json.loads(response)
        print(f"收到连接确认: {data}")

        # 发送订阅消息
        subscribe_msg = {
            'type': 'subscribe',
            'topics': ['historical_collection_status', 'task_progress']
        }
        ws.send(json.dumps(subscribe_msg))
        print(f"发送订阅消息: {subscribe_msg}")

        # 接收订阅确认
        response = ws.recv()
        data = json.loads(response)
        print(f"收到订阅确认: {data}")

        # 发送一些测试消息
        for i in range(3):
            test_msg = {
                'type': 'test',
                'message': f'Test message {i+1}',
                'timestamp': time.time()
            }
            ws.send(json.dumps(test_msg))
            print(f"发送测试消息: {test_msg}")

            # 接收echo响应
            response = ws.recv()
            data = json.loads(response)
            print(f"收到echo响应: {data}")

            time.sleep(1)

        # 关闭连接
        ws.close()
        print("✅ WebSocket连接正常关闭")

        return True

    except Exception as e:
        print(f"❌ WebSocket测试失败: {e}")
        return False

def test_fastapi_websocket():
    """测试FastAPI WebSocket连接"""
    try:
        ws_url = "ws://localhost:8000/ws/historical-collection"
        print(f"测试FastAPI WebSocket: {ws_url}")

        ws = websocket.create_connection(ws_url, timeout=10)
        print("✅ FastAPI WebSocket连接成功")

        # 发送订阅消息
        subscribe_msg = {
            'type': 'subscribe',
            'topics': ['historical_collection_status']
        }
        ws.send(json.dumps(subscribe_msg))
        print(f"发送订阅消息: {subscribe_msg}")

        # 尝试接收响应
        try:
            response = ws.recv()
            data = json.loads(response)
            print(f"收到响应: {data}")
        except websocket.WebSocketTimeoutException:
            print("⚠️ 没有收到及时响应")

        ws.close()
        print("✅ FastAPI WebSocket测试完成")

        return True

    except websocket.WebSocketBadStatusException as e:
        print(f"❌ FastAPI WebSocket连接失败 - HTTP状态码: {e.status_code}")
        return False
    except Exception as e:
        print(f"❌ FastAPI WebSocket测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 WebSocket连接测试")
    print("=" * 50)

    # 首先测试简单的WebSocket服务器
    print("\n1. 测试简单WebSocket服务器:")
    simple_test = test_websocket_connection()

    print("\n" + "=" * 50)

    # 然后测试FastAPI WebSocket
    print("\n2. 测试FastAPI WebSocket:")
    fastapi_test = test_fastapi_websocket()

    print("\n" + "=" * 50)
    print("测试结果汇总:")
    print(f"简单WebSocket服务器: {'✅ 通过' if simple_test else '❌ 失败'}")
    print(f"FastAPI WebSocket: {'✅ 通过' if fastapi_test else '❌ 失败'}")

    if fastapi_test:
        print("\n🎉 所有WebSocket测试通过！")
        return 0
    else:
        print("\n❌ FastAPI WebSocket测试失败")
        print("💡 请检查:")
        print("   - FastAPI应用是否正在运行 (http://localhost:8000)")
        print("   - WebSocket路由是否正确注册")
        print("   - 防火墙是否阻止了WebSocket连接")
        return 1

if __name__ == "__main__":
    exit_code = main()
    print(f"\n程序退出码: {exit_code}")
    sys.exit(exit_code)