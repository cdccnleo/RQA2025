#!/usr/bin/env python3
"""
分析容器启动日志中的异常情况
"""

import subprocess
import sys
import re
from pathlib import Path
from datetime import datetime

def run_command(cmd):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)

def analyze_container_health():
    """分析容器健康状态"""
    print("🔍 分析容器健康状态...")
    issues = []

    # 检查容器状态
    returncode, stdout, stderr = run_command('docker inspect rqa2025-rqa2025-app-1 --format "{{.State.Health.Status}}"')
    if returncode == 0:
        health_status = stdout.strip()
        print(f"容器健康状态: {health_status}")
        if health_status != "healthy":
            issues.append(f"容器健康状态异常: {health_status}")
    else:
        issues.append(f"无法检查容器健康状态: {stderr}")

    # 检查容器是否在运行
    returncode, stdout, stderr = run_command('docker ps --filter "name=rqa2025-rqa2025-app-1" --format "{{.Status}}"')
    if returncode == 0 and stdout.strip():
        status = stdout.strip()
        print(f"容器运行状态: {status}")
        if "unhealthy" in status.lower():
            issues.append("容器状态为unhealthy")
    else:
        issues.append("容器可能未在运行")

    return issues

def analyze_startup_logs():
    """分析启动日志"""
    print("\n📄 分析启动日志...")
    issues = []

    # 获取容器日志
    returncode, stdout, stderr = run_command('docker logs rqa2025-rqa2025-app-1 2>&1')
    if returncode != 0:
        issues.append(f"无法获取容器日志: {stderr}")
        return issues

    logs = stdout

    # 检查关键启动信息
    startup_indicators = {
        "uvicorn": "Uvicorn服务器启动",
        "application startup": "应用启动",
        "listening on": "服务器监听端口",
        "error": "错误信息",
        "exception": "异常信息",
        "failed": "启动失败",
        "infrastructure.*initialized": "基础设施初始化"
    }

    found_indicators = {}
    error_lines = []

    for line in logs.split('\n'):
        line_lower = line.lower()

        # 检查错误
        if any(keyword in line_lower for keyword in ['error', 'exception', 'failed', 'critical']):
            error_lines.append(line)

        # 检查启动指标
        for indicator, description in startup_indicators.items():
            if indicator in line_lower:
                if indicator not in found_indicators:
                    found_indicators[indicator] = []
                found_indicators[indicator].append(line)

    # 分析结果
    print("启动指标检查:")
    for indicator, description in startup_indicators.items():
        count = len(found_indicators.get(indicator, []))
        status = "✅" if count > 0 else "❌"
        print(f"  {status} {description}: {count} 次")

    if error_lines:
        print(f"\n发现 {len(error_lines)} 个错误:")
        for i, error in enumerate(error_lines[:5], 1):  # 只显示前5个错误
            print(f"  {i}. {error}")
        if len(error_lines) > 5:
            print(f"  ... 还有 {len(error_lines) - 5} 个错误")

        issues.extend([f"启动错误: {error}" for error in error_lines])

    # 检查基础设施初始化次数
    infra_init_count = len(found_indicators.get('infrastructure.*initialized', []))
    if infra_init_count > 1:
        issues.append(f"基础设施重复初始化 {infra_init_count} 次")

    return issues

def analyze_health_check():
    """分析健康检查"""
    print("\n❤️ 分析健康检查...")
    issues = []

    # 检查健康检查配置
    returncode, stdout, stderr = run_command('docker inspect rqa2025-rqa2025-app-1 --format "{{.Config.Healthcheck}}"')
    if returncode == 0:
        healthcheck_config = stdout.strip()
        print(f"健康检查配置: {healthcheck_config}")
        if not healthcheck_config or healthcheck_config == "<nil>":
            issues.append("容器没有配置健康检查")
    else:
        issues.append(f"无法获取健康检查配置: {stderr}")

    # 尝试手动健康检查
    print("尝试手动健康检查...")
    returncode, stdout, stderr = run_command('timeout 10 docker exec rqa2025-rqa2025-app-1 python -c "import urllib.request; urllib.request.urlopen(\'http://localhost:8000/health\').read()"')
    if returncode == 0:
        print("✅ 手动健康检查通过")
    else:
        print("❌ 手动健康检查失败")
        issues.append("健康检查端点无法访问")

    return issues

def analyze_resource_usage():
    """分析资源使用情况"""
    print("\n📊 分析资源使用情况...")
    issues = []

    # 检查容器资源使用
    returncode, stdout, stderr = run_command('docker stats rqa2025-rqa2025-app-1 --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"')
    if returncode == 0:
        print("容器资源使用:")
        print(stdout)
    else:
        issues.append(f"无法获取资源使用信息: {stderr}")

    return issues

def generate_report():
    """生成分析报告"""
    print("\n" + "="*60)
    print("🔍 容器启动日志分析报告")
    print("="*60)

    all_issues = []

    # 1. 容器健康状态分析
    all_issues.extend(analyze_container_health())

    # 2. 启动日志分析
    all_issues.extend(analyze_startup_logs())

    # 3. 健康检查分析
    all_issues.extend(analyze_health_check())

    # 4. 资源使用分析
    all_issues.extend(analyze_resource_usage())

    # 生成总结
    print("\n" + "="*60)
    print("📋 问题总结")
    print("="*60)

    if all_issues:
        print(f"发现 {len(all_issues)} 个问题:")
        for i, issue in enumerate(all_issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("✅ 未发现明显问题")

    # 提供建议
    print("\n💡 建议解决方案:")
    if any("unhealthy" in issue.lower() for issue in all_issues):
        print("- 检查应用健康检查端点是否正常响应")
        print("- 确认应用是否正确绑定到0.0.0.0:8000")
        print("- 检查应用内部是否有异常导致无法响应健康检查")

    if any("重复初始化" in issue for issue in all_issues):
        print("- 基础设施层存在重复初始化问题，已在之前修复")

    if any("FPGA" in issue.lower() for issue in all_issues):
        print("- FPGA相关功能异常，已在之前修复")

    if any("error" in issue.lower() or "exception" in issue.lower() for issue in all_issues):
        print("- 检查应用日志中的错误信息")
        print("- 确认依赖项是否正确安装")

    print("- 重启容器测试修复效果")
    print("- 检查Docker网络配置")

if __name__ == "__main__":
    generate_report()