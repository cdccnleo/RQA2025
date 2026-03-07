#!/usr/bin/env python3
"""
检查后端服务状态
"""

import socket
import sys
import requests
from pathlib import Path

def check_port(host='localhost', port=8000):
    """检查端口是否开放"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        print(f"检查端口时出错: {e}")
        return False

def check_http_endpoint(url='http://localhost:8000'):
    """检查HTTP端点是否响应"""
    try:
        response = requests.get(f"{url}/", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except requests.exceptions.ConnectionError:
        return False, None
    except Exception as e:
        print(f"检查HTTP端点时出错: {e}")
        return False, None

def check_health_endpoint(url='http://localhost:8000'):
    """检查健康检查端点"""
    try:
        response = requests.get(f"{url}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except requests.exceptions.ConnectionError:
        return False, None
    except Exception as e:
        print(f"检查健康端点时出错: {e}")
        return False, None

def main():
    print("=" * 60)
    print("后端服务状态检查")
    print("=" * 60)
    
    # 检查端口
    print("\n1. 检查端口8000...")
    port_open = check_port('localhost', 8000)
    if port_open:
        print("   ✅ 端口8000已开放")
    else:
        print("   ❌ 端口8000未开放 - 后端服务可能未启动")
        print("\n   启动后端服务:")
        print("   python scripts/start_api_server.py")
        return
    
    # 检查HTTP端点
    print("\n2. 检查HTTP端点...")
    http_ok, http_data = check_http_endpoint()
    if http_ok:
        print("   ✅ HTTP端点响应正常")
        if http_data:
            print(f"   📄 响应: {http_data}")
    else:
        print("   ❌ HTTP端点无响应")
        return
    
    # 检查健康检查端点
    print("\n3. 检查健康检查端点...")
    health_ok, health_data = check_health_endpoint()
    if health_ok:
        print("   ✅ 健康检查端点响应正常")
        if health_data:
            print(f"   📄 响应: {health_data}")
    else:
        print("   ⚠️  健康检查端点无响应（可能未实现）")
    
    print("\n" + "=" * 60)
    print("✅ 后端服务运行正常")
    print("=" * 60)
    print("\n后端服务地址: http://localhost:8000")
    print("API文档地址: http://localhost:8000/docs")

if __name__ == "__main__":
    try:
        import requests
    except ImportError:
        print("❌ 缺少 requests 库，请安装: pip install requests")
        sys.exit(1)
    
    main()
