#!/usr/bin/env python3
"""
RQA2025快速启动脚本
一键启动完整的容器化环境
"""

import subprocess
import sys
import time
from pathlib import Path


def print_banner():
    """打印启动横幅"""
    print("=" * 80)
    print("🚀 RQA2025量化交易系统 - 快速启动")
    print("=" * 80)
    print("🏗️  即将启动以下服务:")
    print("   • RQA2025应用服务 (端口8000)")
    print("   • PostgreSQL数据库 (端口5432)")
    print("   • Redis缓存 (端口6379)")
    print("   • Prometheus监控 (端口9090)")
    print("   • Grafana可视化 (端口3000)")
    print("   • Nginx反向代理 (端口80)")
    print("=" * 80)


def check_docker():
    """检查Docker是否安装"""
    try:
        result = subprocess.run(["docker", "--version"],
                                capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Docker已安装:", result.stdout.strip())
            return True
    except FileNotFoundError:
        pass

    print("❌ Docker未安装或未运行")
    print("请先安装Docker: https://docs.docker.com/get-docker/")
    return False


def check_docker_compose():
    """检查Docker Compose是否可用"""
    try:
        result = subprocess.run(["docker-compose", "--version"],
                                capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Docker Compose可用:", result.stdout.strip())
            return True
    except FileNotFoundError:
        pass

    # 尝试docker compose (新版本)
    try:
        result = subprocess.run(["docker", "compose", "version"],
                                capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Docker Compose V2可用:", result.stdout.strip())
            return True
    except FileNotFoundError:
        pass

    print("❌ Docker Compose不可用")
    return False


def build_images():
    """构建Docker镜像"""
    print("\n🏗️  构建Docker镜像...")
    try:
        # 检查是否使用Docker Compose V2
        try:
            result = subprocess.run(["docker", "compose", "build"],
                                    capture_output=True, text=True, cwd=".")
            if result.returncode == 0:
                print("✅ Docker镜像构建成功 (使用Docker Compose V2)")
                return True
        except FileNotFoundError:
            pass

        # 使用传统docker-compose
        result = subprocess.run(["docker-compose", "build"],
                                capture_output=True, text=True, cwd=".")
        if result.returncode == 0:
            print("✅ Docker镜像构建成功")
            return True
        else:
            print("❌ Docker镜像构建失败:")
            print(result.stderr)
            return False

    except Exception as e:
        print(f"❌ 构建过程中出错: {e}")
        return False


def start_services():
    """启动所有服务"""
    print("\n🚀 启动服务...")
    try:
        # 检查是否使用Docker Compose V2
        try:
            result = subprocess.run(["docker", "compose", "up", "-d"],
                                    capture_output=True, text=True, cwd=".")
            if result.returncode == 0:
                print("✅ 服务启动成功 (使用Docker Compose V2)")
                return True
        except FileNotFoundError:
            pass

        # 使用传统docker-compose
        result = subprocess.run(["docker-compose", "up", "-d"],
                                capture_output=True, text=True, cwd=".")
        if result.returncode == 0:
            print("✅ 服务启动成功")
            return True
        else:
            print("❌ 服务启动失败:")
            print(result.stderr)
            return False

    except Exception as e:
        print(f"❌ 启动过程中出错: {e}")
        return False


def wait_for_services():
    """等待服务启动完成"""
    print("\n⏳ 等待服务启动...")
    time.sleep(10)  # 等待10秒让服务完全启动

    # 检查关键服务状态
    services_to_check = [
        ("RQA应用", "http://localhost:8000/health"),
        ("Prometheus", "http://localhost:9090"),
        ("Grafana", "http://localhost:3000")
    ]

    for service_name, url in services_to_check:
        try:
            import requests
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {service_name}服务正常 ({url})")
            else:
                print(f"⚠️  {service_name}服务响应异常 ({response.status_code})")
        except Exception:
            print(f"⚠️  {service_name}服务连接失败 ({url})")


def show_service_info():
    """显示服务信息"""
    print("\n📋 服务访问信息:")
    print("=" * 50)
    print("🌐 Web界面:")
    print("   • RQA2025应用: http://localhost:8000")
    print("   • Grafana监控: http://localhost:3000 (admin/admin)")
    print("   • Prometheus:   http://localhost:9090")
    print("")
    print("🔌 数据库连接:")
    print("   • PostgreSQL: localhost:5432 (rqa/rqa_password)")
    print("   • Redis:       localhost:6379")
    print("")
    print("📊 监控指标:")
    print("   • 应用指标: http://localhost:8000/metrics")
    print("   • 系统监控: http://localhost:9090/targets")
    print("=" * 50)


def show_commands():
    """显示常用命令"""
    print("\n🛠️  常用管理命令:")
    print("=" * 50)
    print("# 查看服务状态")
    print("docker-compose ps")
    print("")
    print("# 查看服务日志")
    print("docker-compose logs -f rqa-app")
    print("")
    print("# 停止所有服务")
    print("docker-compose down")
    print("")
    print("# 重启特定服务")
    print("docker-compose restart rqa-app")
    print("")
    print("# 清理所有数据")
    print("docker-compose down -v --remove-orphans")
    print("=" * 50)


def main():
    """主函数"""
    print_banner()

    # 检查环境
    if not check_docker():
        sys.exit(1)

    if not check_docker_compose():
        sys.exit(1)

    # 检查必要的文件
    required_files = ["Dockerfile", "docker-compose.yml", "requirements.txt"]
    missing_files = [f for f in required_files if not Path(f).exists()]

    if missing_files:
        print(f"❌ 缺少必要的文件: {', '.join(missing_files)}")
        sys.exit(1)

    # 构建镜像
    if not build_images():
        sys.exit(1)

    # 启动服务
    if not start_services():
        sys.exit(1)

    # 等待服务启动
    wait_for_services()

    # 显示信息
    show_service_info()
    show_commands()

    print("\n🎉 RQA2025系统启动完成！")
    print("💡 提示: 首次启动可能需要1-2分钟让所有服务完全就绪")


if __name__ == "__main__":
    main()
