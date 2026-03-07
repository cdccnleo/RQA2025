#!/usr/bin/env python3
"""
检查容器后端应用启动状态

分析启动日志中的异常情况
"""

import sys
import os
import time
import requests
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

def check_logs_for_issues():
    """检查日志中的问题"""
    print("🔍 分析启动日志中的问题...")

    log_files = [
        project_root / 'logs' / 'app.log.2025-07-28',
        project_root / 'debug_logs' / 'app.log.1'
    ]

    issues_found = []

    for log_file in log_files:
        if log_file.exists():
            print(f"\n📄 检查日志文件: {log_file.name}")
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                    # 检查FPGA错误
                    if 'FPGAManager' in content and 'get_accelerator' in content:
                        issues_found.append({
                            'type': 'FPGA_ERROR',
                            'description': 'FPGAManager对象缺少get_accelerator属性',
                            'file': log_file.name,
                            'count': content.count('FPGAManager')
                        })

                    # 检查基础设施重复初始化
                    infra_init_count = content.count('Infrastructure layer initialized')
                    if infra_init_count > 5:
                        issues_found.append({
                            'type': 'INFRA_REDUNDANT_INIT',
                            'description': f'基础设施层重复初始化 {infra_init_count} 次',
                            'file': log_file.name,
                            'count': infra_init_count
                        })

                    # 检查调度器启动信息
                    if '数据采集调度器' in content:
                        scheduler_logs = [line for line in content.split('\n') if '数据采集调度器' in line]
                        if scheduler_logs:
                            issues_found.append({
                                'type': 'SCHEDULER_INFO',
                                'description': f'找到 {len(scheduler_logs)} 条调度器相关日志',
                                'logs': scheduler_logs[:3]  # 只显示前3条
                            })

                    # 检查事件总线相关日志
                    if '事件总线' in content or 'EventBus' in content or '监听器' in content:
                        event_logs = [line for line in content.split('\n')
                                    if any(keyword in line for keyword in ['事件总线', 'EventBus', '监听器'])]
                        if event_logs:
                            issues_found.append({
                                'type': 'EVENT_BUS_INFO',
                                'description': f'找到 {len(event_logs)} 条事件总线相关日志',
                                'logs': event_logs[:3]
                            })

            except Exception as e:
                issues_found.append({
                    'type': 'LOG_READ_ERROR',
                    'description': f'无法读取日志文件 {log_file.name}: {e}',
                    'file': log_file.name
                })

    return issues_found

def check_service_status():
    """检查服务运行状态"""
    print("\n🌐 检查服务运行状态...")

    try:
        # 检查本地服务
        response = requests.get('http://localhost:8000/health', timeout=5)
        if response.status_code == 200:
            print("✅ 本地服务正常运行")
            return True
        else:
            print(f"⚠️ 本地服务响应异常: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ 本地服务连接失败: {e}")
        print("   可能原因:")
        print("   - 服务未启动")
        print("   - 端口8000被占用")
        print("   - 防火墙阻止连接")
        return False

def check_scheduler_status():
    """检查调度器状态"""
    print("\n⚙️ 检查调度器状态...")

    try:
        from src.core.orchestration.business_process.service_scheduler import get_data_collection_scheduler

        scheduler = get_data_collection_scheduler()
        is_running = scheduler.is_running()
        status = scheduler.get_status()

        print(f"调度器运行状态: {'✅ 运行中' if is_running else '❌ 未运行'}")
        print(f"启动路径: {status.get('startup_path', '未知')}")
        print(f"启动时间: {status.get('startup_time', '未知')}")
        print(f"启用的数据源数量: {status.get('enabled_sources_count', 0)}")

        return is_running, status

    except Exception as e:
        print(f"❌ 无法检查调度器状态: {e}")
        return False, {}

def check_event_bus_status():
    """检查事件总线状态"""
    print("\n🔄 检查事件总线状态...")

    try:
        from src.core.event_bus import get_event_bus
        from src.core.event_bus.types import EventType

        event_bus = get_event_bus()
        subscriber_count = event_bus.get_subscriber_count(EventType.APPLICATION_STARTUP_COMPLETE)

        print(f"事件总线实例ID: {id(event_bus)}")
        print(f"APPLICATION_STARTUP_COMPLETE订阅者数量: {subscriber_count}")

        if subscriber_count > 0:
            print("✅ 事件订阅正常")
        else:
            print("❌ 缺少事件订阅者")

        return subscriber_count > 0

    except Exception as e:
        print(f"❌ 无法检查事件总线状态: {e}")
        return False

def main():
    """主检查函数"""
    print("🚀 容器后端应用启动状态检查")
    print("=" * 50)

    # 1. 检查日志中的问题
    issues = check_logs_for_issues()

    # 2. 检查服务状态
    service_running = check_service_status()

    # 3. 检查调度器状态
    scheduler_running, scheduler_status = check_scheduler_status()

    # 4. 检查事件总线状态
    event_bus_ok = check_event_bus_status()

    # 输出总结报告
    print("\n" + "=" * 50)
    print("📊 启动状态总结报告")
    print("=" * 50)

    print(f"服务运行状态: {'✅ 正常' if service_running else '❌ 异常'}")
    print(f"调度器运行状态: {'✅ 正常' if scheduler_running else '❌ 未启动'}")
    print(f"事件总线状态: {'✅ 正常' if event_bus_ok else '❌ 异常'}")

    print(f"\n发现的问题数量: {len(issues)}")

    for i, issue in enumerate(issues, 1):
        print(f"\n{i}. {issue['type']}: {issue['description']}")
        if 'file' in issue:
            print(f"   文件: {issue['file']}")
        if 'count' in issue:
            print(f"   出现次数: {issue['count']}")
        if 'logs' in issue:
            print("   相关日志:")
            for log in issue['logs']:
                print(f"   - {log.strip()}")

    # 诊断建议
    print(f"\n🔍 诊断建议:")
    if not service_running:
        print("- 服务未启动，请检查启动脚本和端口占用")
    if not scheduler_running:
        print("- 调度器未启动，可能的主启动流程存在问题")
        if scheduler_status.get('startup_path') == 'fallback_startup_mechanism':
            print("- 调度器通过备用机制启动，说明主启动流程失败")
    if not event_bus_ok:
        print("- 事件总线配置异常，可能导致启动监听器无法工作")

    if issues:
        print("- 请查看上述问题并修复相关代码")

    # 总体状态
    overall_status = service_running and scheduler_running and event_bus_ok
    print(f"\n🎯 总体状态: {'✅ 正常启动' if overall_status else '❌ 存在异常'}")

    return overall_status

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)