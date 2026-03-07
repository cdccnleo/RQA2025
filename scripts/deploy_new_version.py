#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 新版本部署脚本
用于将代码更新发布到Docker容器
"""

import os
import sys
import subprocess
import time
import datetime
from pathlib import Path
import json

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent


def get_version():
    """获取当前版本号"""
    version_file = PROJECT_ROOT / "VERSION"
    if version_file.exists():
        with open(version_file, 'r') as f:
            return f.read().strip()
    
    # 如果没有版本文件，使用时间戳作为版本号
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")


def update_version():
    """更新版本号"""
    version = get_version()
    version_file = PROJECT_ROOT / "VERSION"
    
    # 如果版本文件存在，增加版本号
    if version_file.exists():
        try:
            # 尝试解析版本号并增加
            parts = version.split('.')
            if len(parts) == 3:
                parts[2] = str(int(parts[2]) + 1)
                version = '.'.join(parts)
            else:
                # 使用时间戳
                version = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        except:
            version = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    
    # 写入版本文件
    with open(version_file, 'w') as f:
        f.write(version)
    
    return version


def log_info(message):
    """输出信息日志"""
    print(f"ℹ️  {message}")


def log_success(message):
    """输出成功日志"""
    print(f"✅ {message}")


def log_error(message):
    """输出错误日志"""
    print(f"❌ {message}", file=sys.stderr)


def log_warning(message):
    """输出警告日志"""
    print(f"⚠️  {message}")


def run_command(cmd, cwd=None, check=True, capture_output=False):
    """执行命令"""
    try:
        if isinstance(cmd, str):
            cmd = cmd.split()
        
        result = subprocess.run(
            cmd,
            cwd=cwd or PROJECT_ROOT,
            check=check,
            capture_output=capture_output,
            text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        log_error(f"命令执行失败: {' '.join(cmd)}")
        if e.stderr:
            log_error(f"错误信息: {e.stderr}")
        raise


def check_docker():
    """检查Docker环境"""
    log_info("检查Docker环境...")
    
    try:
        # 检查Docker是否安装
        run_command(["docker", "--version"], capture_output=True)
        log_success("Docker已安装")
        
        # 检查Docker Compose是否安装
        run_command(["docker-compose", "--version"], capture_output=True)
        log_success("Docker Compose已安装")
        
        # 检查Docker服务是否运行
        run_command(["docker", "ps"], capture_output=True)
        log_success("Docker服务运行正常")
        
        return True
    except Exception as e:
        log_error(f"Docker环境检查失败: {e}")
        return False


def build_image(version):
    """构建Docker镜像"""
    log_info(f"构建Docker镜像 (版本: {version})...")
    
    image_name = f"rqa2025-app:{version}"
    latest_image = "rqa2025-app:latest"
    
    try:
        # 构建新版本镜像
        log_info("开始构建镜像...")
        run_command([
            "docker", "build",
            "-t", image_name,
            "-t", latest_image,
            "-f", "Dockerfile",
            "."
        ])
        
        log_success(f"镜像构建成功: {image_name}")
        return True
    except Exception as e:
        log_error(f"镜像构建失败: {e}")
        return False


def stop_old_containers():
    """停止旧容器"""
    log_info("停止旧容器...")
    
    try:
        # 使用docker-compose停止服务
        run_command(["docker-compose", "down"], capture_output=True)
        log_success("旧容器已停止")
        return True
    except Exception as e:
        log_warning(f"停止旧容器时出现警告: {e}")
        # 尝试直接停止容器
        try:
            run_command(["docker-compose", "stop"], capture_output=True)
            log_success("旧容器已停止")
            return True
        except:
            log_warning("无法停止旧容器，继续部署...")
            return False


def start_new_containers():
    """启动新容器"""
    log_info("启动新容器...")
    
    try:
        # 启动服务
        run_command(["docker-compose", "up", "-d"])
        log_success("新容器已启动")
        return True
    except Exception as e:
        log_error(f"启动新容器失败: {e}")
        return False


def wait_for_services(timeout=60):
    """等待服务启动"""
    log_info(f"等待服务启动 (超时: {timeout}秒)...")
    
    start_time = time.time()
    max_retries = timeout // 5
    
    for i in range(max_retries):
        try:
            # 检查容器状态
            result = run_command(
                ["docker-compose", "ps"],
                capture_output=True
            )
            
            # 检查是否有容器在运行
            if "Up" in result.stdout:
                elapsed = time.time() - start_time
                log_success(f"服务已启动 (耗时: {elapsed:.1f}秒)")
                return True
            
            time.sleep(5)
            log_info(f"等待中... ({i+1}/{max_retries})")
        except Exception as e:
            log_warning(f"检查服务状态时出现错误: {e}")
            time.sleep(5)
    
    log_warning("服务启动超时，但继续检查...")
    return False


def check_service_health():
    """检查服务健康状态"""
    log_info("检查服务健康状态...")
    
    services = [
        ("rqa2025-app", "http://localhost:8000/health"),
        ("rqa2025-web", "http://localhost:8080/health")
    ]
    
    all_healthy = True
    
    for service_name, health_url in services:
        try:
            # 检查容器是否运行
            result = run_command(
                ["docker", "ps", "--filter", f"name={service_name}", "--format", "{{.Status}}"],
                capture_output=True
            )
            
            if "Up" in result.stdout:
                log_success(f"{service_name} 容器运行正常")
            else:
                log_warning(f"{service_name} 容器状态异常")
                all_healthy = False
        except Exception as e:
            log_warning(f"检查 {service_name} 健康状态失败: {e}")
            all_healthy = False
    
    return all_healthy


def show_service_info(version):
    """显示服务信息"""
    log_success("部署完成!")
    print("\n" + "="*60)
    print("📋 服务访问信息")
    print("="*60)
    print(f"版本号: {version}")
    print(f"部署时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n🌐 服务地址:")
    print("  - Web界面: http://localhost:8080")
    print("  - API服务: http://localhost:8000")
    print("  - API文档: http://localhost:8000/docs")
    print("\n📊 监控服务:")
    print("  - Prometheus: http://localhost:9090")
    print("  - Grafana: http://localhost:3000")
    print("\n🔧 管理命令:")
    print("  - 查看日志: docker-compose logs -f")
    print("  - 查看状态: docker-compose ps")
    print("  - 停止服务: docker-compose down")
    print("="*60)


def deploy_new_version():
    """部署新版本"""
    print("🚀 RQA2025 新版本部署开始")
    print("="*60)
    
    # 1. 检查Docker环境
    if not check_docker():
        log_error("Docker环境检查失败，无法继续部署")
        return False
    
    # 2. 更新版本号
    version = update_version()
    log_info(f"当前版本: {version}")
    
    # 3. 构建新镜像
    if not build_image(version):
        log_error("镜像构建失败，部署中止")
        return False
    
    # 4. 停止旧容器
    stop_old_containers()
    
    # 5. 启动新容器
    if not start_new_containers():
        log_error("启动新容器失败")
        return False
    
    # 6. 等待服务启动
    wait_for_services()
    
    # 7. 检查服务健康
    if not check_service_health():
        log_warning("部分服务健康检查未通过，请手动检查")
    
    # 8. 显示服务信息
    show_service_info(version)
    
    return True


if __name__ == "__main__":
    try:
        success = deploy_new_version()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        log_warning("部署被用户中断")
        sys.exit(1)
    except Exception as e:
        log_error(f"部署过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

