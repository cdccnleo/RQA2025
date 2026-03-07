#!/usr/bin/env python3
"""
启动后端服务
这是一个简化的启动脚本，用于快速启动后端API服务
"""

import sys
import os
from pathlib import Path

# 设置Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

def main():
    """启动后端服务"""
    print("=" * 60)
    print("启动 RQA2025 后端服务")
    print("=" * 60)
    
    try:
        # 检查端口是否被占用
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 8000))
        sock.close()
        
        if result == 0:
            print("⚠️  警告: 端口8000已被占用")
            print("   请先停止占用端口的进程，或使用其他端口")
            response = input("   是否继续启动? (y/n): ")
            if response.lower() != 'y':
                print("已取消启动")
                return
        
        print("\n📦 导入应用模块...")
        from src.gateway.web.app_factory import create_app
        
        print("✅ 应用模块导入成功")
        app = create_app()
        print(f"✅ 应用创建成功，路由数: {len(app.routes)}")
        
        print("\n🚀 启动uvicorn服务器...")
        print("   📍 服务地址: http://localhost:8000")
        print("   📚 API文档: http://localhost:8000/docs")
        print("   ❤️  健康检查: http://localhost:8000/health")
        print("   ⏹️  按 Ctrl+C 停止服务\n")
        
        import uvicorn
        import threading
        import time
        
        # 启动后验证服务器（在后台线程中）
        def verify_after_start():
            """在服务器启动后验证"""
            time.sleep(3)  # 等待服务器启动
            try:
                import requests
                response = requests.get("http://localhost:8000/health", timeout=5)
                if response.status_code == 200:
                    print(f"\n✅ 服务器验证成功: {response.json()}")
                else:
                    print(f"\n⚠️  服务器响应异常: HTTP {response.status_code}")
            except ImportError:
                print("\n⚠️  无法验证服务器（缺少 requests 库）")
            except Exception as e:
                print(f"\n⚠️  服务器验证失败（但服务可能已启动）: {e}")
        
        verification_thread = threading.Thread(target=verify_after_start, daemon=True)
        verification_thread.start()
        
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
        print("请确保所有依赖都已安装: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
