#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 本地启动脚本

在离线环境中启动RQA2025服务
"""

import os
import sys
import time
import subprocess
from pathlib import Path


def start_services():
    """启动RQA2025服务"""
    project_root = Path(__file__).parent

    print("🚀 启动RQA2025服务...")

    # 激活虚拟环境
    venv_path = project_root / "venv"
    if os.name == 'nt':  # Windows
        python_path = venv_path / "Scripts" / "python.exe"
        activate_script = venv_path / "Scripts" / "activate.bat"
    else:  # Linux/Mac
        python_path = venv_path / "bin" / "python"
        activate_script = venv_path / "bin" / "activate"

    if not python_path.exists():
        print("❌ 虚拟环境未找到，请先运行离线环境设置脚本")
        return False

    try:
        # 设置环境变量
        env = os.environ.copy()
        env['PYTHONPATH'] = str(project_root / "src")
        env['RQA2025_ENV'] = 'development'

        # 启动主服务
        print("📊 启动主服务...")
        main_service = subprocess.Popen([
            str(python_path), "scripts/run_distributed_system.py"
        ], cwd=project_root, env=env)

        print("✅ 主服务已启动")

        # 等待服务启动
        time.sleep(5)

        print("\n🎉 RQA2025服务启动成功!")
        print("\n📋 服务信息:")
        print("  - 主服务PID:", main_service.pid)
        print("  - 工作目录:", project_root)
        print("  - 环境: 离线开发环境")

        print("\n📝 停止服务请按 Ctrl+C")

        try:
            main_service.wait()
        except KeyboardInterrupt:
            print("\n🛑 正在停止服务...")
            main_service.terminate()
            main_service.wait()
            print("✅ 服务已停止")

    except Exception as e:
        print(f"❌ 启动服务失败: {e}")
        return False

    return True


def check_service_health():
    """检查服务健康"""
    try:
        import requests
        response = requests.get("http://localhost:8080/health", timeout=5)
        if response.status_code == 200:
            print("✅ 服务健康检查通过")
            return True
        else:
            print("⚠️  服务健康检查失败")
            return False
    except Exception as e:
        print(f"⚠️  无法连接到服务: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "health":
        check_service_health()
    else:
        start_services()
