#!/usr/bin/env python3
"""
测试网络连接脚本
"""

import socket
import time

def test_connection(host, port, timeout=5):
    """测试TCP连接"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        start_time = time.time()
        result = sock.connect_ex((host, port))
        end_time = time.time()

        if result == 0:
            print(f"✅ 连接成功: {host}:{port} (耗时: {end_time - start_time:.3f}秒)")
            return True
        else:
            print(f"❌ 连接失败: {host}:{port} (错误码: {result})")
            return False

    except Exception as e:
        print(f"💥 连接异常: {host}:{port} - {e}")
        return False
    finally:
        try:
            sock.close()
        except:
            pass

def main():
    print("🔍 测试网络连接")
    print("=" * 40)

    # 测试本地8080端口
    print("\n测试本地8080端口 (nginx前端)...")
    test_connection("127.0.0.1", 8080)

    # 测试本地8000端口
    print("\n测试本地8000端口 (FastAPI后端)...")
    test_connection("127.0.0.1", 8000)

    print("\n" + "=" * 40)
    print("测试完成")

if __name__ == "__main__":
    main()