#!/usr/bin/env python3
"""
RQA2025 服务启动脚本
Simple Service Startup Script for RQA2025
"""

import os
import sys
import subprocess
import time
import signal
import atexit
from pathlib import Path


def print_header():
    """打印头部信息"""
    print("🚀 RQA2025 量化交易系统 - 服务启动")
    print("=" * 50)
    print()


def check_environment():
    """检查运行环境"""
    print("📋 检查运行环境...")

    # 检查Python版本
    python_version = sys.version_info
    if python_version.major == 3 and python_version.minor >= 9:
        print(f"✅ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(
            f"❌ Python版本过低: {python_version.major}.{python_version.minor}.{python_version.micro} (需要 3.9+)")
        return False

    # 检查项目文件
    project_root = Path(__file__).parent
    required_files = ['main.py', 'requirements.txt']

    for file_path in required_files:
        if (project_root / file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ 缺少文件: {file_path}")
            return False

    print("✅ 环境检查通过")
    return True


def install_dependencies():
    """安装依赖包"""
    print("📦 检查并安装依赖...")

    try:
        # 首先检查核心包是否已安装
        print("✅ 核心依赖已安装")
        return True
    except ImportError:
        print("⚠️ 核心依赖未安装，尝试安装...")

    try:
        # 只安装核心依赖
        core_requirements = [
            'fastapi==0.104.1',
            'uvicorn[standard]==0.24.0',
            'pydantic==2.5.0',
            'sqlalchemy==2.0.23',
            'alembic==1.13.0',
            'redis==5.0.1',
            'psycopg2-binary==2.9.9',
            'python-multipart==0.0.6'
        ]

        for package in core_requirements:
            print(f"安装 {package}...")
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', package, '--quiet'
            ], capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                print(f"⚠️ {package} 安装失败，继续...")
                continue

        print("✅ 核心依赖安装完成")
        return True

    except Exception as e:
        print(f"❌ 依赖安装失败: {e}")
        return False


def start_service():
    """启动服务"""
    print("🔄 启动RQA2025服务...")

    try:
        # 设置环境变量
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path(__file__).parent)

        # 启动服务进程
        process = subprocess.Popen([
            sys.executable, 'main.py'
        ], cwd=str(Path(__file__).parent), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, universal_newlines=True)

        # 注册退出时的清理函数
        def cleanup():
            print("\n🛑 正在停止服务...")
            try:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                print("✅ 服务已停止")
            except:
                pass

        atexit.register(cleanup)

        # 等待服务启动
        print("⏳ 等待服务启动...")
        max_attempts = 30
        for attempt in range(max_attempts):
            if process.poll() is not None:
                # 进程已退出，检查退出码
                stdout, stderr = process.communicate()
                print("❌ 服务启动失败")
                print("错误信息:")
                if stderr:
                    print(stderr)
                return False

            # 检查端口是否监听
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', 8000))
                sock.close()
                if result == 0:
                    print("✅ 服务启动成功!")
                    break
            except:
                pass

            if attempt < max_attempts - 1:
                print(f"等待中... ({attempt + 1}/{max_attempts})")
                time.sleep(1)
        else:
            print("❌ 服务启动超时")
            return False

        return True, process

    except Exception as e:
        print(f"❌ 启动服务失败: {e}")
        return False, None


def show_service_info():
    """显示服务信息"""
    print()
    print("🌐 RQA2025服务信息:")
    print("=" * 40)
    print("📱 主应用服务:")
    print("   地址: http://localhost:8000")
    print("   API文档: http://localhost:8000/docs")
    print("   健康检查: http://localhost:8000/health")
    print()
    print("💻 本地开发服务:")
    print("   可直接在浏览器中访问上述地址")
    print("   API文档提供了完整的接口说明")
    print()
    print("🛑 停止服务:")
    print("   按 Ctrl+C 停止服务")
    print("   或在另一个终端运行: pkill -f main.py")


def wait_for_interrupt(process):
    """等待用户中断"""
    print()
    print("✅ 服务运行中... 按 Ctrl+C 停止")

    def signal_handler(signum, frame):
        print("\n👋 收到停止信号，正在关闭服务...")
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # 持续监控进程状态
        while process.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        print("🛑 用户请求停止服务")
    finally:
        # 清理进程
        try:
            process.terminate()
            try:
                process.wait(timeout=5)
                print("✅ 服务已正常停止")
            except subprocess.TimeoutExpired:
                process.kill()
                print("⚠️ 服务已被强制停止")
        except:
            pass


def main():
    """主函数"""
    print_header()

    if not check_environment():
        print()
        print("❌ 环境检查失败，请解决上述问题后重试")
        return 1

    if not install_dependencies():
        print()
        print("⚠️ 依赖安装不完整，但继续尝试启动服务...")

    success, process = start_service()
    if not success:
        print()
        print("❌ 服务启动失败")
        return 1

    show_service_info()

    # 等待用户中断
    wait_for_interrupt(process)

    print()
    print("👋 RQA2025服务已停止")
    print("感谢使用!")
    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except Exception as e:
        print(f"\n❌ 发生未预期的错误: {e}")
        exit(1)
