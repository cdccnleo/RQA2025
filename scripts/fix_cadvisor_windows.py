#!/usr/bin/env python3
"""
修复cAdvisor在Windows环境下的兼容性问题

在Windows环境下禁用cAdvisor服务，因为它依赖Linux cgroup文件系统
"""

import subprocess
import time
import requests
import sys
import platform

def check_platform():
    """检查运行平台"""
    print("🔍 检查运行平台...")

    system = platform.system().lower()
    print(f"当前平台: {system}")

    if system == "windows":
        print("⚠️ 检测到Windows平台，cAdvisor将自动禁用")
        return True
    else:
        print("✅ Linux平台，cAdvisor可以正常运行")
        return False

def disable_cadvisor_in_compose():
    """在docker-compose中禁用cAdvisor"""
    print("🔧 在docker-compose中禁用cAdvisor...")

    compose_file = "docker-compose.prod.yml"

    try:
        with open(compose_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 检查是否已经禁用
        if "# cadvisor:" in content and "#   image:" in content:
            print("✅ cAdvisor已经禁用")
            return True

        # 如果没有禁用，则需要手动禁用
        print("⚠️ cAdvisor配置可能未正确禁用")
        print("请确保docker-compose.prod.yml中的cAdvisor服务已被注释掉")
        return False

    except Exception as e:
        print(f"❌ 检查配置文件失败: {e}")
        return False

def restart_containers_without_cadvisor():
    """重启容器（不包含cAdvisor）"""
    print("🔄 重启RQA2025容器（跳过cAdvisor）...")

    try:
        # 停止所有容器
        print("停止所有容器...")
        result = subprocess.run(
            ['docker-compose', '-f', 'docker-compose.prod.yml', 'down'],
            capture_output=True, text=True, timeout=60
        )

        if result.returncode != 0:
            print(f"停止容器失败: {result.stderr}")
            return False

        # 启动除cAdvisor外的所有容器
        print("启动容器（跳过cAdvisor）...")
        result = subprocess.run(
            ['docker-compose', '-f', 'docker-compose.prod.yml', 'up', '-d'],
            capture_output=True, text=True, timeout=120
        )

        if result.returncode == 0:
            print("✅ 容器启动命令执行完成")
            return True
        else:
            print(f"❌ 容器启动失败: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ 重启容器失败: {e}")
        return False

def check_container_status():
    """检查容器状态"""
    print("📊 检查容器状态...")

    try:
        result = subprocess.run(
            ['docker-compose', '-f', 'docker-compose.prod.yml', 'ps'],
            capture_output=True, text=True, timeout=30
        )

        if result.returncode == 0:
            output = result.stdout
            print("容器状态:")
            print(output)

            # 检查是否有失败的容器（排除cAdvisor）
            lines = output.split('\n')
            failed_containers = []

            for line in lines:
                if line.strip() and ('Exit' in line or 'ERROR' in line):
                    # 检查是否是cAdvisor（已禁用）
                    if 'cadvisor' not in line.lower():
                        failed_containers.append(line.strip())

            if failed_containers:
                print("❌ 发现容器启动失败:")
                for container in failed_containers:
                    print(f"  {container}")
                return False
            else:
                print("✅ 所有活跃容器状态正常")
                return True
        else:
            print(f"检查容器状态失败: {result.stderr}")
            return False

    except Exception as e:
        print(f"检查容器状态时出错: {e}")
        return False

def wait_for_services():
    """等待服务启动"""
    print("⏳ 等待服务启动...")

    services_to_check = [
        ('nginx', 'http://localhost/health', 200),
        ('app', 'http://localhost:8000/health', 200),
        ('prometheus', 'http://localhost:9090/-/healthy', 200),
        ('grafana', 'http://localhost:3000/api/health', 200),
    ]

    max_attempts = 60  # 最多等待2分钟
    for attempt in range(max_attempts):
        all_ready = True

        for service_name, url, expected_code in services_to_check:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code != expected_code:
                    all_ready = False
                    print(f"  {service_name}: 状态码 {response.status_code} (期望 {expected_code})")
                else:
                    print(f"  ✅ {service_name}: 正常")
            except Exception as e:
                all_ready = False
                print(f"  {service_name}: 连接失败 - {e}")

        if all_ready:
            print("🎉 所有服务启动完成！")
            return True

        if attempt < max_attempts - 1:
            print(f"等待中... ({attempt + 1}/{max_attempts})")
            time.sleep(2)

    print("❌ 服务启动超时")
    return False

def test_monitoring_apis():
    """测试监控相关API"""
    print("🧪 测试监控API...")

    test_cases = [
        ('http://localhost/health', 'nginx健康检查'),
        ('http://localhost:8000/health', '应用健康检查'),
        ('http://localhost/api/v1/monitoring/data-collection/health', '数据采集API'),
        ('http://localhost/api/v1/monitoring/historical-collection/status', '历史数据采集API'),
    ]

    results = {}
    for url, description in test_cases:
        try:
            response = requests.get(url, timeout=10)
            status = "✅" if response.status_code == 200 else "❌"
            print(f"{status} {description}: {response.status_code}")
            results[url] = response.status_code == 200
        except Exception as e:
            print(f"❌ {description}: 连接失败 - {e}")
            results[url] = False

    return results

def show_monitoring_access():
    """显示监控系统访问信息"""
    print("\n📊 监控系统访问信息:")
    print("=" * 50)
    print("🎯 核心应用:")
    print("  📱 数据采集监控: http://localhost/data-collection-monitor.html")
    print("  🔧 应用API文档: http://localhost:8000/docs")
    print("")
    print("📈 可观测性监控:")
    print("  📊 Grafana: http://localhost:3000 (admin/admin123)")
    print("  📈 Prometheus: http://localhost:9090")
    print("  📝 Loki日志: http://localhost:3100")
    print("")
    print("⚠️ 注意:")
    print("  - cAdvisor已在Windows环境下禁用")
    print("  - 容器监控功能受限，但不影响核心业务")
    print("  - 如需完整监控，请在Linux环境下运行")

def main():
    """主函数"""
    print("🔧 修复RQA2025 cAdvisor Windows兼容性问题")
    print("=" * 60)

    # 检查平台
    is_windows = check_platform()

    if not is_windows:
        print("✅ 非Windows平台，cAdvisor应该可以正常工作")
        print("💡 如果仍有问题，请检查Linux cgroup配置")
        return 0

    # 检查cAdvisor配置
    if not disable_cadvisor_in_compose():
        print("❌ cAdvisor配置检查失败")
        return 1

    # 重启容器
    if not restart_containers_without_cadvisor():
        print("❌ 容器重启失败")
        return 1

    # 检查容器状态
    time.sleep(5)  # 等待容器初始化
    if not check_container_status():
        print("❌ 容器状态异常")
        print("\n🔍 故障排除:")
        print("1. 检查Docker日志: docker-compose -f docker-compose.prod.yml logs")
        print("2. 确认cAdvisor已正确禁用")
        print("3. 检查端口冲突")
        return 1

    # 等待服务启动
    if not wait_for_services():
        print("❌ 服务启动失败")
        return 1

    # 测试API
    results = test_monitoring_apis()

    # 检查结果
    critical_apis = [
        'http://localhost/health',
        'http://localhost:8000/health',
        'http://localhost/api/v1/monitoring/data-collection/health'
    ]

    critical_working = all(results.get(url, False) for url in critical_apis)
    historical_working = results.get('http://localhost/api/v1/monitoring/historical-collection/status', False)

    print("\n" + "=" * 60)
    print("📊 启动结果汇总:")

    if critical_working:
        print("✅ 核心服务启动成功")
        if historical_working:
            print("✅ 历史数据采集API正常")
            print("🎉 系统在Windows环境下成功启动！")
            show_monitoring_access()
            return 0
        else:
            print("⚠️  历史数据采集API仍有问题")
            print("💡 请检查应用日志: docker logs rqa2025-app")
            return 0  # 核心功能正常
    else:
        print("❌ 核心服务启动失败")
        print("🔍 请检查:")
        print("- Docker容器日志: docker-compose -f docker-compose.prod.yml logs")
        print("- 系统资源: df -h && free -h")
        print("- 网络连接")
        return 1

if __name__ == "__main__":
    exit_code = main()
    print(f"\n脚本退出码: {exit_code}")
    sys.exit(exit_code)