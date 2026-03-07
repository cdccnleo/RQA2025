"""ContinuousMonitoringSystem 运行时辅助函数。"""

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional


def start_monitoring(system: "ContinuousMonitoringSystem") -> bool:
    print("🚀 启动RQA2025基础设施层连续监控和优化系统...")

    if system.monitoring_active:
        print("⚠️ 监控系统已在运行中")
        return False

    system.monitoring_active = True
    system.monitoring_thread = threading.Thread(target=system._monitoring_loop)
    system.monitoring_thread.daemon = True
    system.monitoring_thread.start()

    print("✅ 连续监控系统已启动")
    print(f"📊 监控间隔: {system.monitoring_config['interval_seconds']}秒")
    print("🎯 监控内容: 测试覆盖率、性能指标、资源使用、健康状态")

    return True


def stop_monitoring(system: "ContinuousMonitoringSystem") -> bool:
    print("🛑 停止连续监控系统...")
    system.monitoring_active = False

    if system.monitoring_thread:
        system.monitoring_thread.join(timeout=10)

    print("✅ 连续监控系统已停止")
    return True


def monitoring_loop(system: "ContinuousMonitoringSystem") -> None:
    while system.monitoring_active:
        try:
            system._perform_monitoring_cycle()
            time.sleep(system.monitoring_config['interval_seconds'])
        except Exception as exc:  # noqa: BLE001 - 捕获所有异常用于日志输出
            print(f"❌ 监控循环异常: {exc}")
            time.sleep(60)


def perform_monitoring_cycle(system: "ContinuousMonitoringSystem") -> None:
    timestamp = datetime.now()

    print(f"\n📊 执行监控周期 - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

    monitoring_data = system._collect_monitoring_data()
    system._process_alerts(monitoring_data)
    system._process_optimization_suggestions(monitoring_data)
    system._persist_monitoring_results(timestamp, monitoring_data)

    print("✅ 监控周期完成")


def collect_test_coverage(system: "ContinuousMonitoringSystem") -> Dict[str, Any]:
    print("📈 收集测试覆盖率数据...")

    try:
        result = subprocess.run(
            [
                sys.executable,
                '-m', 'pytest',
                '--cov=src',
                '--cov-report=json:coverage_temp.json',
                '--cov-report=term-missing',
                'tests/business_process/test_simple_validation.py',
                '-q',
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        coverage_data = {
            'timestamp': datetime.now(),
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'coverage_percent': 0.0,
        }

        if os.path.exists('coverage_temp.json'):
            with open('coverage_temp.json', 'r', encoding='utf-8') as fp:
                coverage_json = json.load(fp)
                coverage_data['coverage_percent'] = (
                    coverage_json.get('totals', {}).get('percent_covered', 0.0)
                )

            os.remove('coverage_temp.json')

        return coverage_data

    except Exception as exc:  # noqa: BLE001 - 捕获所有异常用于降级
        print(f"❌ 收集覆盖率数据失败: {exc}")
        return {
            'timestamp': datetime.now(),
            'success': False,
            'error': str(exc),
            'coverage_percent': 0.0,
        }


def export_monitoring_report(system: "ContinuousMonitoringSystem", filename: Optional[str] = None) -> str:
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'monitoring_report_{timestamp}.json'

    report_data = {
        'report_title': 'RQA2025 基础设施层连续监控报告',
        'generated_at': datetime.now().isoformat(),
        'monitoring_system': system.get_monitoring_report(),
        'phase': 'Phase 7: 连续监控和优化',
    }

    with open(filename, 'w', encoding='utf-8') as fp:
        json.dump(report_data, fp, ensure_ascii=False, indent=2, default=str)

    print(f"✅ 监控报告已导出到: {filename}")
    return filename
