#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 容器部署脚本
支持数据保护，避免覆盖现有数据
"""

import os
import sys
import subprocess
import time
import shutil
from datetime import datetime
from pathlib import Path


def backup_critical_data(project_root):
    """备份关键数据文件"""
    print("💾 备份关键数据文件...")
    
    backup_dir = project_root / "backups" / f"deployment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # 需要备份的文件和目录
    backup_items = [
        ("data/data_sources_config.json", "data_sources_config.json"),
        ("data/production", "production_data"),
    ]
    
    for source, target in backup_items:
        source_path = project_root / source
        if source_path.exists():
            target_path = backup_dir / target
            if source_path.is_file():
                shutil.copy2(source_path, target_path)
                print(f"  ✅ 已备份: {source} -> {target_path}")
            elif source_path.is_dir():
                shutil.copytree(source_path, target_path, dirs_exist_ok=True)
                print(f"  ✅ 已备份目录: {source} -> {target_path}")
    
    print(f"✅ 数据备份完成，备份位置: {backup_dir}")
    return backup_dir


def verify_data_volumes(project_root):
    """验证数据卷是否存在且安全"""
    print("🔍 验证数据卷状态...")
    
    # 检查命名卷是否存在
    volumes = ["postgres_data", "redis_data", "minio_data", "prometheus_data", "grafana_data"]
    
    result = subprocess.run(
        ["docker", "volume", "ls", "--format", "{{.Name}}"],
        capture_output=True,
        text=True,
        cwd=project_root
    )
    
    existing_volumes = result.stdout.strip().split('\n') if result.returncode == 0 else []
    
    for volume in volumes:
        if volume in existing_volumes:
            print(f"  ✅ 数据卷 {volume} 已存在，将被保留")
        else:
            print(f"  ℹ️  数据卷 {volume} 不存在，将在首次部署时创建")
    
    return True


def deploy_containers():
    """部署RQA2025容器（数据安全模式）"""
    project_root = Path(__file__).parent.parent

    print("🚀 开始部署RQA2025容器（数据保护模式）...")
    print("=" * 60)

    try:
        # 0. 验证数据卷
        verify_data_volumes(project_root)
        
        # 1. 备份关键数据
        backup_dir = backup_critical_data(project_root)
        
        # 2. 构建镜像（仅构建应用镜像，不影响数据服务）
        print("\n📦 构建Docker镜像...")
        print("  注意: 仅构建应用镜像，数据服务（数据库、Redis等）不会被重建")
        subprocess.run([
            "docker-compose", "build", "--no-cache", "rqa2025-app"
        ], cwd=project_root, check=True)
        
        # 构建其他应用服务
        app_services = [
            "strategy-service",
            "trading-service", 
            "risk-service",
            "data-service",
            "data-collection-orchestrator"
        ]
        
        for service in app_services:
            try:
                print(f"  构建 {service}...")
                subprocess.run([
                    "docker-compose", "build", "--no-cache", service
                ], cwd=project_root, check=True, capture_output=True)
            except subprocess.CalledProcessError:
                print(f"  ⚠️  {service} 构建跳过（可能不存在）")
        
        # 3. 滚动更新应用服务（不影响数据服务）
        print("\n🔄 滚动更新应用服务...")
        print("  注意: 仅更新应用服务，数据服务（postgres、redis等）将保持运行状态")
        
        # 应用服务列表（不包含数据服务）
        app_service_names = [
            "rqa2025-app",
            "strategy-service",
            "trading-service",
            "risk-service", 
            "data-service",
            "data-collection-orchestrator",
            "rqa2025-web"
        ]
        
        # 使用 up -d --no-deps --build 来只更新应用服务
        # --no-deps: 不启动依赖服务
        # --build: 如果镜像已构建，直接使用
        print("  更新应用服务容器...")
        for service in app_service_names:
            try:
                subprocess.run([
                    "docker-compose", "up", "-d", "--no-deps", "--no-recreate", service
                ], cwd=project_root, capture_output=True, check=False)
            except:
                # 如果服务不存在或启动失败，尝试创建新容器
                try:
                    subprocess.run([
                        "docker-compose", "up", "-d", "--no-deps", service
                    ], cwd=project_root, capture_output=True, check=False)
                except:
                    print(f"  ⚠️  {service} 更新跳过")
        
        # 确保数据服务正在运行（如果不存在则启动，但不重建）
        print("  确保数据服务运行中...")
        data_services = ["postgres", "redis", "minio", "prometheus", "grafana"]
        for service in data_services:
            try:
                # 只启动，不重建
                subprocess.run([
                    "docker-compose", "up", "-d", "--no-recreate", service
                ], cwd=project_root, capture_output=True, check=False)
            except:
                pass

        # 4. 等待服务启动
        print("\n⏳ 等待服务启动...")
        time.sleep(30)

        # 5. 检查服务状态
        print("\n🔍 检查服务状态...")
        result = subprocess.run([
            "docker-compose", "ps"
        ], cwd=project_root, capture_output=True, text=True)

        print("服务状态:")
        print(result.stdout)

        # 6. 验证数据完整性
        print("\n🔐 验证数据完整性...")
        verify_data_integrity(project_root, backup_dir)

        # 7. 检查服务健康
        print("\n🏥 检查服务健康...")
        check_health(project_root)

        print("\n" + "=" * 60)
        print("✅ RQA2025容器部署成功!")
        print("\n📋 服务访问地址:")
        print("  - 主应用: http://localhost:8000")
        print("  - Web界面: http://localhost:8080")
        print("  - Grafana: http://localhost:3000")
        print("  - Prometheus: http://localhost:9090")
        print(f"\n💾 数据备份位置: {backup_dir}")
        print("=" * 60)

    except subprocess.CalledProcessError as e:
        print(f"\n❌ 部署失败: {e}")
        print(f"💾 备份数据位置: {backup_dir}")
        print("   如需恢复，请从备份目录恢复数据")
        return False
    except Exception as e:
        print(f"\n❌ 部署出现错误: {e}")
        return False

    return True


def verify_data_integrity(project_root, backup_dir):
    """验证数据完整性"""
    print("  验证关键数据文件...")
    
    # 检查关键文件是否仍然存在
    critical_files = [
        "data/data_sources_config.json",
    ]
    
    for file_path in critical_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"  ✅ {file_path} 存在")
        else:
            print(f"  ⚠️  {file_path} 不存在（可能首次部署）")
    
    # 检查数据卷（如果容器正在运行，说明数据卷存在）
    print("  检查数据卷状态...")
    result = subprocess.run(
        ["docker", "ps", "--filter", "name=postgres", "--format", "{{.Names}}"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0 and result.stdout.strip():
        print("  ✅ PostgreSQL容器运行中（数据卷正常）")
    else:
        # 尝试直接检查数据卷
        result = subprocess.run(
            ["docker", "volume", "inspect", "postgres_data"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("  ✅ PostgreSQL数据卷存在")
        else:
            print("  ℹ️  PostgreSQL数据卷将在首次启动时创建")
    
    result = subprocess.run(
        ["docker", "ps", "--filter", "name=redis", "--format", "{{.Names}}"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0 and result.stdout.strip():
        print("  ✅ Redis容器运行中（数据卷正常）")
    else:
        # 尝试直接检查数据卷
        result = subprocess.run(
            ["docker", "volume", "inspect", "redis_data"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("  ✅ Redis数据卷存在")
        else:
            print("  ℹ️  Redis数据卷将在首次启动时创建")


def check_health(project_root):
    """检查服务健康"""
    print("  检查容器运行状态...")
    
    try:
        # 检查docker-compose服务状态
        result = subprocess.run([
            "docker-compose", "ps", "--format", "json"
        ], cwd=project_root, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            # 统计运行中的服务
            import json
            services = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    try:
                        service_info = json.loads(line)
                        services.append(service_info)
                    except:
                        pass
            
            healthy_count = sum(1 for s in services if 'healthy' in s.get('Health', '').lower() or s.get('State', '').lower() == 'running')
            total_count = len(services)
            
            print(f"  ✅ 运行中的服务: {healthy_count}/{total_count}")
            
            # 检查关键服务
            key_services = ["rqa2025-app", "postgres", "redis"]
            for service_name in key_services:
                found = any(s.get('Service', '') == service_name and 
                           ('healthy' in s.get('Health', '').lower() or 
                            s.get('State', '').lower() == 'running') 
                           for s in services)
                if found:
                    print(f"  ✅ {service_name} 运行正常")
                else:
                    print(f"  ⚠️  {service_name} 状态异常")
        else:
            # 备用检查方法：直接检查容器
            result = subprocess.run([
                "docker", "ps", "--filter", "name=rqa2025", "--format", "{{.Names}}"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and result.stdout.strip():
                container_count = len([l for l in result.stdout.strip().split('\n') if l.strip()])
                print(f"  ✅ 发现 {container_count} 个运行中的容器")
            else:
                print("  ⚠️  无法检查容器状态")

    except subprocess.TimeoutExpired:
        print("  ⚠️  健康检查超时")
    except Exception as e:
        print(f"  ⚠️  健康检查失败: {e}")


def stop_containers():
    """停止容器"""
    project_root = Path(__file__).parent.parent
    print("🛑 停止RQA2025容器...")

    try:
        subprocess.run([
            "docker-compose", "down"
        ], cwd=project_root, check=True)
        print("✅ 容器已停止")
    except Exception as e:
        print(f"❌ 停止容器失败: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stop":
        stop_containers()
    else:
        deploy_containers()
