#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查前端和后端应用更新状态
"""

import subprocess
import os
from pathlib import Path
from datetime import datetime


def run_cmd(cmd, capture=True):
    """执行命令"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=capture,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        return result.stdout.strip() if capture else result.returncode == 0
    except Exception as e:
        return f"错误: {e}" if capture else False


def check_backend_update():
    """检查后端应用更新状态"""
    print("=" * 60)
    print("📦 检查后端应用更新状态")
    print("=" * 60)
    
    project_root = Path(__file__).parent.parent
    
    # 检查镜像构建时间
    print("\n1. Docker镜像信息:")
    image_info = run_cmd('docker images rqa2025-app:latest --format "{{.CreatedAt}}"')
    if image_info:
        print(f"   镜像创建时间: {image_info}")
    
    # 检查容器创建时间
    print("\n2. 容器信息:")
    container_created = run_cmd('docker inspect rqa2025-rqa2025-app-1 --format="{{.Created}}"')
    if container_created:
        print(f"   容器创建时间: {container_created}")
    
    # 检查关键后端文件
    print("\n3. 关键后端文件对比:")
    backend_files = [
        "src/gateway/web/api.py",
        "src/core/app.py",
        "scripts/start_api_server.py"
    ]
    
    for file_path in backend_files:
        local_file = project_root / file_path
        if local_file.exists():
            local_time = datetime.fromtimestamp(local_file.stat().st_mtime)
            local_size = local_file.stat().st_size
            
            # 获取容器中的文件信息
            container_file = f"/app/{file_path}"
            container_stat = run_cmd(f'docker exec rqa2025-rqa2025-app-1 stat -c "%y %s" {container_file} 2>/dev/null')
            
            if container_stat and not container_stat.startswith("错误"):
                parts = container_stat.split()
                if len(parts) >= 2:
                    container_time_str = " ".join(parts[:-1])
                    container_size = parts[-1]
                    try:
                        # 解析时间（格式: 2025-01-06 23:25:37.524716700 +0800）
                        container_time = datetime.strptime(container_time_str.split('.')[0], "%Y-%m-%d %H:%M:%S")
                        
                        status = "✅ 已更新" if local_time <= container_time else "⚠️  需要更新"
                        print(f"\n   {file_path}:")
                        print(f"     本地: {local_time.strftime('%Y-%m-%d %H:%M:%S')} ({local_size} bytes)")
                        print(f"     容器: {container_time.strftime('%Y-%m-%d %H:%M:%S')} ({container_size} bytes)")
                        print(f"     状态: {status}")
                    except:
                        print(f"\n   {file_path}:")
                        print(f"     本地: {local_time.strftime('%Y-%m-%d %H:%M:%S')} ({local_size} bytes)")
                        print(f"     容器: {container_stat}")
            else:
                print(f"\n   {file_path}:")
                print(f"     本地: {local_time.strftime('%Y-%m-%d %H:%M:%S')} ({local_size} bytes)")
                print(f"     容器: ⚠️  文件不存在或无法访问")
        else:
            print(f"\n   {file_path}: ⚠️  本地文件不存在")


def check_frontend_update():
    """检查前端应用更新状态"""
    print("\n" + "=" * 60)
    print("🌐 检查前端应用更新状态")
    print("=" * 60)
    
    project_root = Path(__file__).parent.parent
    
    # 检查Web容器
    print("\n1. Web容器信息:")
    container_created = run_cmd('docker inspect rqa2025-rqa2025-web-1 --format="{{.Created}}"')
    if container_created:
        print(f"   容器创建时间: {container_created}")
    
    # 检查关键前端文件
    print("\n2. 关键前端文件对比:")
    frontend_files = [
        "web-static/dashboard.html",
        "web-static/index.html",
        "web-static/data-sources-config.html",
        "web-static/nginx.conf"
    ]
    
    for file_path in frontend_files:
        local_file = project_root / file_path
        if local_file.exists():
            local_time = datetime.fromtimestamp(local_file.stat().st_mtime)
            local_size = local_file.stat().st_size
            
            # 获取容器中的文件名
            container_filename = file_path.split("/")[-1]
            container_file = f"/usr/share/nginx/html/{container_filename}"
            
            # 检查容器中的文件
            container_stat = run_cmd(f'docker exec rqa2025-rqa2025-web-1 stat -c "%y %s" {container_file} 2>/dev/null')
            
            if container_stat and not container_stat.startswith("错误"):
                parts = container_stat.split()
                if len(parts) >= 2:
                    container_time_str = " ".join(parts[:-1])
                    container_size = parts[-1]
                    try:
                        container_time = datetime.strptime(container_time_str.split('.')[0], "%Y-%m-%d %H:%M:%S")
                        
                        # 前端文件通过volume挂载，时间可能不同，比较大小更准确
                        status = "✅ 已挂载" if abs(local_size - int(container_size)) < 10 else "⚠️  需要检查"
                        print(f"\n   {container_filename}:")
                        print(f"     本地: {local_time.strftime('%Y-%m-%d %H:%M:%S')} ({local_size} bytes)")
                        print(f"     容器: {container_time.strftime('%Y-%m-%d %H:%M:%S')} ({container_size} bytes)")
                        print(f"     状态: {status} (通过volume挂载)")
                    except:
                        print(f"\n   {container_filename}:")
                        print(f"     本地: {local_time.strftime('%Y-%m-%d %H:%M:%S')} ({local_size} bytes)")
                        print(f"     容器: {container_stat}")
            else:
                print(f"\n   {container_filename}:")
                print(f"     本地: {local_time.strftime('%Y-%m-%d %H:%M:%S')} ({local_size} bytes)")
                print(f"     容器: ⚠️  文件不存在或无法访问")
        else:
            print(f"\n   {file_path}: ⚠️  本地文件不存在")


def check_container_status():
    """检查容器运行状态"""
    print("\n" + "=" * 60)
    print("🔍 检查容器运行状态")
    print("=" * 60)
    
    containers = ["rqa2025-rqa2025-app-1", "rqa2025-rqa2025-web-1"]
    
    for container in containers:
        status = run_cmd(f'docker inspect {container} --format="{{{{.State.Status}}}}"')
        health = run_cmd(f'docker inspect {container} --format="{{{{.State.Health.Status}}}}"')
        
        print(f"\n{container}:")
        print(f"   状态: {status}")
        if health and health != "<no value>":
            print(f"   健康: {health}")


def main():
    """主函数"""
    print("🔍 RQA2025 应用更新状态检查")
    print("=" * 60)
    
    check_container_status()
    check_backend_update()
    check_frontend_update()
    
    print("\n" + "=" * 60)
    print("✅ 检查完成")
    print("=" * 60)
    print("\n💡 提示:")
    print("   - 后端应用需要重新构建镜像并重启容器才能更新")
    print("   - 前端文件通过volume挂载，修改本地文件后刷新浏览器即可")
    print("   - 如果后端需要更新，请运行: python scripts/deploy_containers.py")


if __name__ == "__main__":
    main()
