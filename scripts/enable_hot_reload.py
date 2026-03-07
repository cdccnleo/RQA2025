#!/usr/bin/env python3
"""
RQA2025 热重载配置工具
为生产环境启用热重载功能（仅用于开发调试）
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def enable_hot_reload():
    """启用热重载模式"""
    print("🔥 启用RQA2025热重载模式...")

    # 检查当前环境
    env = os.getenv("RQA_ENV", "production")
    if env == "production":
        print("⚠️  警告：正在生产环境启用热重载模式！")
        confirm = input("确认继续？(yes/no): ")
        if confirm.lower() != "yes":
            print("已取消")
            return

    # 查找运行中的容器
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=rqa2025", "--format", "{{.Names}}"],
            capture_output=True, text=True, check=True
        )
        containers = result.stdout.strip().split('\n') if result.stdout.strip() else []

        if not containers:
            print("❌ 未找到运行中的RQA2025容器")
            return

        print(f"找到容器: {containers}")

        for container in containers:
            if container.strip():
                print(f"重启容器: {container}")
                subprocess.run(["docker", "restart", container.strip()], check=True)

        print("✅ 热重载已启用")

    except subprocess.CalledProcessError as e:
        print(f"❌ 启用热重载失败: {e}")
        return

    print("\n📝 使用说明:")
    print("1. 修改代码后，容器会自动重载")
    print("2. 查看日志: docker logs -f <container_name>")
    print("3. 停止热重载: python disable_hot_reload.py")

def disable_hot_reload():
    """禁用热重载模式"""
    print("🛑 禁用RQA2025热重载模式...")

    try:
        # 查找运行中的容器
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=rqa2025", "--format", "{{.Names}}"],
            capture_output=True, text=True, check=True
        )
        containers = result.stdout.strip().split('\n') if result.stdout.strip() else []

        for container in containers:
            if container.strip():
                print(f"重启容器以禁用热重载: {container}")
                subprocess.run(["docker", "restart", container.strip()], check=True)

        print("✅ 热重载已禁用")

    except subprocess.CalledProcessError as e:
        print(f"❌ 禁用热重载失败: {e}")

def check_reload_status():
    """检查热重载状态"""
    print("🔍 检查热重载状态...")

    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=rqa2025", "--format", "{{.Names}}"],
            capture_output=True, text=True, check=True
        )
        containers = result.stdout.strip().split('\n') if result.stdout.strip() else []

        if not containers:
            print("❌ 未找到运行中的RQA2025容器")
            return

        for container in containers:
            if container.strip():
                # 检查容器是否启用了reload
                inspect_result = subprocess.run(
                    ["docker", "inspect", container.strip()],
                    capture_output=True, text=True
                )

                if "--reload" in inspect_result.stdout:
                    print(f"✅ {container}: 热重载已启用")
                else:
                    print(f"❌ {container}: 热重载未启用")

    except subprocess.CalledProcessError as e:
        print(f"❌ 检查状态失败: {e}")

def main():
    parser = argparse.ArgumentParser(description="RQA2025 热重载配置工具")
    parser.add_argument("action", choices=["enable", "disable", "status"],
                       help="执行操作")

    args = parser.parse_args()

    if args.action == "enable":
        enable_hot_reload()
    elif args.action == "disable":
        disable_hot_reload()
    elif args.action == "status":
        check_reload_status()

if __name__ == "__main__":
    main()