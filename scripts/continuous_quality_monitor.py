#!/usr/bin/env python3
"""
持续质量监控脚本

定期检查代码质量、测试覆盖率和性能指标
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

class ContinuousQualityMonitor:
    """持续质量监控器"""

    def __init__(self, project_root=None):
        self.project_root = Path(project_root or Path(__file__).parent.parent)
        self.reports_dir = self.project_root / "test_logs"
        self.monitoring_log = self.reports_dir / "quality_monitoring.json"

        # 监控阈值
        self.thresholds = {
            "coverage": {"warning": 75.0, "critical": 70.0},
            "test_failure_rate": {"warning": 0.05, "critical": 0.10},
            "performance_degradation": {"warning": 0.10, "critical": 0.20},
            "code_quality_violations": {"warning": 20, "critical": 50}
        }

    def run_monitoring_cycle(self):
        """运行监控周期"""
        print("🔍 执行质量监控周期...")

        results = {
            "timestamp": datetime.now().isoformat(),
            "coverage": self.check_coverage(),
            "test_health": self.check_test_health(),
            "code_quality": self.check_code_quality(),
            "performance": self.check_performance(),
            "alerts": []
        }

        # 生成告警
        results["alerts"] = self.generate_alerts(results)

        # 保存监控结果
        self.save_monitoring_results(results)

        # 打印摘要
        self.print_monitoring_summary(results)

        return results

    def check_coverage(self):
        """检查覆盖率"""
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", "--cov=src", "--cov-report=json"
            ], capture_output=True, text=True, cwd=self.project_root, timeout=300)

            # 读取覆盖率数据
            coverage_file = self.project_root / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)

                total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0)
                return {
                    "total": total_coverage,
                    "by_file": coverage_data.get("files", {}),
                    "status": "success"
                }

        except Exception as e:
            return {"error": str(e), "status": "failed"}

    def check_test_health(self):
        """检查测试健康状态"""
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", "--tb=no", "-q"
            ], capture_output=True, text=True, cwd=self.project_root, timeout=300)

            # 解析测试结果
            output_lines = result.stdout.strip().split("\n")
            summary_line = None
            for line in reversed(output_lines):
                if any(k in line for k in ["passed", "failed", "skipped"]):
                    summary_line = line
                    break

            return {
                "exit_code": result.returncode,
                "summary": summary_line,
                "status": "success" if result.returncode == 0 else "failed"
            }

        except Exception as e:
            return {"error": str(e), "status": "failed"}

    def check_code_quality(self):
        """检查代码质量"""
        try:
            # 运行flake8
            result = subprocess.run([
                sys.executable, "-m", "flake8", "src/", "--max-line-length=100", "--extend-ignore=E203,W503"
            ], capture_output=True, text=True, cwd=self.project_root)

            violation_count = len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0

            return {
                "flake8_violations": violation_count,
                "status": "success"
            }

        except Exception as e:
            return {"error": str(e), "status": "failed"}

    def check_performance(self):
        """检查性能指标"""
        # 这里可以集成性能基准测试
        return {
            "baseline_comparison": "需要实现性能基准比较",
            "status": "pending"
        }

    def generate_alerts(self, results):
        """生成告警"""
        alerts = []

        # 覆盖率告警
        if "total" in results.get("coverage", {}):
            coverage = results["coverage"]["total"]
            if coverage < self.thresholds["coverage"]["critical"]:
                alerts.append({
                    "level": "critical",
                    "type": "coverage",
                    "message": f"覆盖率严重不足: {coverage:.1f}% < {self.thresholds['coverage']['critical']}%"
                })
            elif coverage < self.thresholds["coverage"]["warning"]:
                alerts.append({
                    "level": "warning",
                    "type": "coverage",
                    "message": f"覆盖率偏低: {coverage:.1f}% < {self.thresholds['coverage']['warning']}%"
                })

        # 测试失败告警
        if results.get("test_health", {}).get("exit_code", 0) != 0:
            alerts.append({
                "level": "critical",
                "type": "test_failures",
                "message": "测试执行失败，请检查测试套件"
            })

        # 代码质量告警
        if "flake8_violations" in results.get("code_quality", {}):
            violations = results["code_quality"]["flake8_violations"]
            if violations > self.thresholds["code_quality_violations"]["critical"]:
                alerts.append({
                    "level": "critical",
                    "type": "code_quality",
                    "message": f"代码质量问题严重: {violations}个违规 > {self.thresholds['code_quality_violations']['critical']}"
                })

        return alerts

    def save_monitoring_results(self, results):
        """保存监控结果"""
        try:
            if self.monitoring_log.exists():
                with open(self.monitoring_log, 'r') as f:
                    history = json.load(f)
            else:
                history = []

            history.append(results)

            # 保留最近30天的记录
            cutoff_date = datetime.now() - timedelta(days=30)
            history = [
                r for r in history
                if datetime.fromisoformat(r["timestamp"]) > cutoff_date
            ]

            with open(self.monitoring_log, 'w') as f:
                json.dump(history, f, indent=2)

        except Exception as e:
            print(f"保存监控结果失败: {e}")

    def print_monitoring_summary(self, results):
        """打印监控摘要"""
        print("\n📊 质量监控摘要:")

        if "total" in results.get("coverage", {}):
            coverage = results["coverage"]["total"]
            print(f"🎯 覆盖率: {coverage:.1f}%")

        test_status = results.get("test_health", {}).get("status", "unknown")
        print(f"🧪 测试状态: {test_status}")

        if "flake8_violations" in results.get("code_quality", {}):
            violations = results["code_quality"]["flake8_violations"]
            print(f"📝 代码质量: {violations}个违规")

        if results.get("alerts"):
            print(f"🚨 告警数量: {len(results['alerts'])}")
            for alert in results["alerts"]:
                print(f"   {alert['level'].upper()}: {alert['message']}")
        else:
            print("✅ 无告警")

def main():
    """主函数"""
    monitor = ContinuousQualityMonitor()

    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        # 持续监控模式
        print("启动持续质量监控模式...")
        while True:
            monitor.run_monitoring_cycle()
            time.sleep(3600)  # 每小时检查一次
    else:
        # 单次检查
        monitor.run_monitoring_cycle()

if __name__ == "__main__":
    main()
