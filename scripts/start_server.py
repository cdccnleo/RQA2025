#!/usr/bin/env python3
"""
RQA2025 后端服务统一启动脚本

功能：
- 检查端口8000是否被占用
- 使用 app_factory.create_app() 创建应用
- 启动 uvicorn 服务器
- 启动后验证服务是否响应
- 提供清晰的错误信息

使用方法：
    python scripts/start_server.py
"""

import sys
import os
import socket
import time
import requests
from pathlib import Path

# 设置Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))


def check_port(host='localhost', port=8000):
    """检查端口是否被占用"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        print(f"检查端口时出错: {e}")
        return False


def verify_server_ready(url='http://localhost:8000', max_attempts=20, interval=0.5):
    """
    验证服务器是否已就绪
    
    Args:
        url: 服务器URL
        max_attempts: 最大尝试次数
        interval: 每次尝试间隔（秒）
    
    Returns:
        bool: 服务器是否已就绪
    """
    print(f"\n🔍 验证服务器是否就绪（最多尝试{max_attempts}次）...")
    
    for i in range(max_attempts):
        try:
            response = requests.get(f"{url}/health", timeout=2)
            if response.status_code == 200:
                print(f"✅ 服务器验证成功（第{i+1}次尝试）")
                try:
                    data = response.json()
                    print(f"   📄 健康检查响应: {data}")
                except:
                    pass
                return True
        except requests.exceptions.ConnectionError:
            if i < max_attempts - 1:
                time.sleep(interval)
            else:
                print(f"❌ 服务器验证失败：无法连接到 {url}")
                return False
        except Exception as e:
            if i < max_attempts - 1:
                time.sleep(interval)
            else:
                print(f"❌ 服务器验证失败: {e}")
                return False
    
    print(f"⚠️  服务器验证超时（尝试{max_attempts}次），但服务可能已启动")
    return False


def main():
    """主函数"""
    print("=" * 60)
    print("启动 RQA2025 后端服务（统一启动脚本）")
    print("=" * 60)
    
    # 检查端口是否被占用
    print("\n1. 检查端口8000...")
    if check_port('localhost', 8000):
        print("   ⚠️  警告: 端口8000已被占用")
        print("   可能的原因：")
        print("   - 后端服务已在运行")
        print("   - 其他程序占用了该端口")
        print("\n   解决方案：")
        print("   - 停止已运行的服务")
        print("   - 或使用其他端口（修改 PORT 环境变量）")
        
        response = input("\n   是否继续启动? (y/n): ")
        if response.lower() != 'y':
            print("已取消启动")
            return
        print("   继续启动...")
    else:
        print("   ✅ 端口8000可用")
    
    # 导入应用
    print("\n2. 导入应用模块...")
    try:
        from src.gateway.web.app_factory import create_app
        app = create_app()
        print(f"   ✅ 应用创建成功，路由数: {len(app.routes)}")
    except ImportError as e:
        print(f"   ❌ 导入错误: {e}")
        print("   请确保所有依赖都已安装: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"   ❌ 应用创建失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 启动服务器
    print("\n3. 启动uvicorn服务器...")
    print("   📍 服务地址: http://localhost:8000")
    print("   📚 API文档: http://localhost:8000/docs")
    print("   ❤️  健康检查: http://localhost:8000/health")
    print("   ⏹️  按 Ctrl+C 停止服务\n")
    
    try:
        import uvicorn
        
        # 在后台启动验证任务
        import threading
        verification_done = threading.Event()
        
        def verify_after_start():
            """在服务器启动后验证"""
            time.sleep(3)  # 等待服务器启动
            if verify_server_ready():
                verification_done.set()
        
        verification_thread = threading.Thread(target=verify_after_start, daemon=True)
        verification_thread.start()
        
        # 启动服务器
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info",
            access_log=True
        )
        
    except KeyboardInterrupt:
        print("\n\n✅ 服务已停止")
    except ImportError as e:
        print(f"\n❌ 导入错误: {e}")
        print("请确保已安装 uvicorn: pip install uvicorn")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    try:
        import requests
    except ImportError:
        print("❌ 缺少 requests 库，请安装: pip install requests")
        sys.exit(1)
    
    main()
