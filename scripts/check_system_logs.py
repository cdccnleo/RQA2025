#!/usr/bin/env python3
"""
RQA2025系统日志查看和分析工具
提供完整的系统运行状态检查功能
"""

import subprocess
import json
import time
from pathlib import Path

def run_command(cmd):
    """运行shell命令并返回结果"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return False, "", str(e)

def check_container_status():
    """检查所有容器的运行状态"""
    print("🐳 容器运行状态检查:")
    print("-" * 50)

    success, output, error = run_command("docker ps --format \"{{.Names}}\\t{{.Status}}\\t{{.Ports}}\"")
    if success:
        lines = output.split('\n')
        if len(lines) > 0:
            print("NAMES                STATUS                          PORTS")
            print("-" * 60)
            for line in lines:
                if line.strip():
                    print(line)
        else:
            print("❌ 没有运行中的容器")
    else:
        print(f"❌ 检查失败: {error}")

    print()

def check_service_logs(service_name, container_name):
    """检查特定服务的日志"""
    print(f"📋 {service_name} 日志检查:")
    print("-" * 50)

    # 检查容器是否存在
    success, output, error = run_command(f"docker ps -q -f name={container_name}")
    if not success or not output.strip():
        print(f"❌ 容器 {container_name} 未运行")
        print()
        return

    # 获取最新日志
    success, output, error = run_command(f"docker logs {container_name} --tail 5")
    if success:
        if output.strip():
            print("最新日志:")
            for line in output.split('\n'):
                if line.strip():
                    print(f"  {line}")
        else:
            print("暂无日志输出")
    else:
        print(f"❌ 获取日志失败: {error}")

    print()

def check_api_health():
    """检查API健康状态"""
    print("🔍 API健康检查:")
    print("-" * 50)

    # 检查主应用健康状态
    success, output, error = run_command("curl -s http://localhost:8000/health")
    if success and output.strip():
        try:
            data = json.loads(output)
            print("✅ RQA2025主应用健康检查:")
            print(f"  状态: {data.get('status', 'unknown')}")
            print(f"  服务: {data.get('service', 'unknown')}")
            print(f"  环境: {data.get('environment', 'unknown')}")
            print(f"  容器化: {data.get('container', 'unknown')}")
            print(f"  时间戳: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(data.get('timestamp', 0)))}")
        except json.JSONDecodeError:
            print(f"❌ 响应格式错误: {output}")
    else:
        print("❌ RQA2025主应用健康检查失败")

    # 检查Grafana健康状态
    success, output, error = run_command("curl -s http://localhost:3000/api/health")
    if success and output.strip():
        try:
            data = json.loads(output)
            print("✅ Grafana健康检查:")
            print(f"  数据库: {data.get('database', 'unknown')}")
        except json.JSONDecodeError:
            print(f"❌ Grafana响应格式错误: {output}")
    else:
        print("❌ Grafana健康检查失败")

    print()

def check_database_connection():
    """检查数据库连接状态"""
    print("🗄️ 数据库连接检查:")
    print("-" * 50)

    # 检查PostgreSQL连接
    success, output, error = run_command("docker exec rqa2025-postgres pg_isready -U rqa2025 -d rqa2025")
    if success:
        print("✅ PostgreSQL: 连接正常")
    else:
        print("❌ PostgreSQL: 连接失败")

    # 检查Redis连接
    success, output, error = run_command("docker exec rqa2025-redis redis-cli ping")
    if success and "PONG" in output.upper():
        print("✅ Redis: 连接正常")
    else:
        print("❌ Redis: 连接失败")

    print()

def analyze_system_health():
    """分析整体系统健康状态"""
    print("📊 系统健康状态分析:")
    print("-" * 50)

    issues = []

    # 检查容器数量
    success, output, error = run_command("docker ps -q | wc -l")
    if success:
        container_count = int(output.strip())
        if container_count < 5:
            issues.append(f"运行容器数量不足 (当前: {container_count}, 期望: 5+)")
        else:
            print(f"✅ 容器数量正常: {container_count}个")

    # 检查关键服务
    critical_services = [
        ("rqa2025-app-main", "RQA2025主应用"),
        ("rqa2025-postgres", "PostgreSQL数据库"),
        ("rqa2025-redis", "Redis缓存")
    ]

    for container, service in critical_services:
        success, output, error = run_command(f"docker ps -q -f name={container}")
        if not success or not output.strip():
            issues.append(f"{service} 未运行")
        else:
            print(f"✅ {service}: 运行正常")

    # 检查API响应
    success, output, error = run_command("curl -f -s http://localhost:8000/health > /dev/null")
    if not success:
        issues.append("RQA2025主应用API无响应")
    else:
        print("✅ RQA2025 API: 响应正常")

    # 输出问题总结
    if issues:
        print("⚠️ 发现问题:")
        for issue in issues:
            print(f"  - {issue}")
        print("\n💡 建议检查:")
        print("  1. 运行 'docker-compose logs' 查看详细错误")
        print("  2. 运行 'docker-compose restart' 重启服务")
        print("  3. 检查网络连接和端口占用")
    else:
        print("🎉 系统运行正常，无异常问题！")

    print()

def show_log_commands():
    """显示常用的日志查看命令"""
    print("🛠️ 常用日志查看命令:")
    print("-" * 50)
    print("# 查看所有容器状态")
    print("docker ps --format \"table {{.Names}}\\t{{.Status}}\\t{{.Ports}}\"")
    print()
    print("# 查看特定容器日志")
    print("docker logs <container_name>              # 完整日志")
    print("docker logs <container_name> --tail 20    # 最新20行")
    print("docker logs <container_name> -f           # 实时监控")
    print()
    print("# 查看容器资源使用")
    print("docker stats")
    print()
    print("# 进入容器调试")
    print("docker exec -it <container_name> bash")
    print()
    print("# 重启服务")
    print("docker restart <container_name>")
    print("docker-compose restart")
    print()

def main():
    """主函数"""
    print("🔍 RQA2025系统运行日志分析工具")
    print("=" * 60)
    print(f"检查时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 1. 检查容器状态
    check_container_status()

    # 2. 检查各服务日志
    check_service_logs("RQA2025主应用", "rqa2025-app-main")
    check_service_logs("PostgreSQL数据库", "rqa2025-postgres")
    check_service_logs("Redis缓存", "rqa2025-redis")
    check_service_logs("Prometheus监控", "rqa2025-prometheus")
    check_service_logs("Grafana可视化", "rqa2025-grafana")

    # 3. 检查API健康状态
    check_api_health()

    # 4. 检查数据库连接
    check_database_connection()

    # 5. 分析系统整体健康状态
    analyze_system_health()

    # 6. 显示常用命令
    show_log_commands()

    print("=" * 60)
    print("✅ 日志分析完成！")

if __name__ == "__main__":
    main()
