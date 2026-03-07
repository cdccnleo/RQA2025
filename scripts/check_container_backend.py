#!/usr/bin/env python3
"""
检查容器中的后端服务状态

用于诊断容器内后端服务无法访问的问题
"""

import sys
import os
import socket
import requests
import time
from pathlib import Path

def check_port_listening(host='0.0.0.0', port=8000):
    """检查端口是否在监听"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host if host != '0.0.0.0' else '127.0.0.1', port))
        sock.close()
        return result == 0
    except Exception as e:
        print(f"检查端口时出错: {e}")
        return False

def check_http_endpoint(url='http://localhost:8000/health', timeout=5):
    """检查HTTP端点"""
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except requests.exceptions.ConnectionError:
        return False, None
    except Exception as e:
        print(f"检查HTTP端点时出错: {e}")
        return False, None

def check_application_import():
    """检查应用是否可以导入"""
    try:
        sys.path.insert(0, '/app')
        sys.path.insert(0, '/app/src')
        from src.gateway.web.app_factory import create_app
        app = create_app()
        return True, len(app.routes)
    except Exception as e:
        return False, str(e)

def main():
    print("=" * 60)
    print("容器后端服务诊断")
    print("=" * 60)
    
    # 检查环境
    print("\n1. 环境检查...")
    print(f"   Python版本: {sys.version}")
    print(f"   工作目录: {os.getcwd()}")
    print(f"   PYTHONPATH: {os.environ.get('PYTHONPATH', '未设置')}")
    print(f"   RQA_ENV: {os.environ.get('RQA_ENV', '未设置')}")
    
    # 检查应用导入
    print("\n2. 检查应用导入...")
    can_import, result = check_application_import()
    if can_import:
        print(f"   ✅ 应用导入成功，路由数: {result}")
    else:
        print(f"   ❌ 应用导入失败: {result}")
        return
    
    # 检查端口
    print("\n3. 检查端口8000...")
    if check_port_listening('127.0.0.1', 8000):
        print("   ✅ 端口8000正在监听")
    else:
        print("   ❌ 端口8000未监听")
        print("   提示: 服务可能未启动或绑定失败")
    
    # 检查HTTP端点
    print("\n4. 检查HTTP端点...")
    http_ok, http_data = check_http_endpoint()
    if http_ok:
        print("   ✅ HTTP端点响应正常")
        if http_data:
            print(f"   📄 响应: {http_data}")
    else:
        print("   ❌ HTTP端点无响应")
        print("   可能原因:")
        print("   - 服务未启动")
        print("   - 服务绑定到错误的地址")
        print("   - 服务启动但未完全初始化")
    
    # 检查常见问题
    print("\n5. 常见问题检查...")
    
    # 检查启动脚本
    startup_script = Path('/app/scripts/start_api_server.py')
    if startup_script.exists():
        print(f"   ✅ 启动脚本存在: {startup_script}")
    else:
        print(f"   ❌ 启动脚本不存在: {startup_script}")
    
    # 检查配置文件
    config_file = Path('/app/data/data_sources_config.json')
    if config_file.exists():
        print(f"   ✅ 配置文件存在: {config_file}")
    else:
        print(f"   ⚠️  配置文件不存在: {config_file}")
    
    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)

if __name__ == "__main__":
    try:
        import requests
    except ImportError:
        print("❌ 缺少 requests 库，请安装: pip install requests")
        sys.exit(1)
    
    main()
