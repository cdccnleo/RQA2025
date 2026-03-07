#!/usr/bin/env python3
"""
架构自动化脚本

建立完全自动化的架构检查和优化系统
"""

import re
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import subprocess
import schedule


class ArchitectureAutomation:
    """架构自动化系统"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.infrastructure_dir = self.project_root / "src" / "infrastructure"
        self.reports_dir = self.project_root / "reports"
        self.scripts_dir = self.project_root / "scripts"

        # 自动化配置
        self.config = {
            "check_interval": 3600,  # 每小时检查一次
            "optimize_interval": 86400,  # 每天优化一次
            "report_interval": 86400,  # 每天生成报告
            "auto_fix": True,  # 是否自动修复
            "notification": True,  # 是否发送通知
            "thresholds": {
                "critical": 20,  # 严重问题阈值
                "warning": 50,   # 警告问题阈值
                "info": 80       # 信息问题阈值
            }
        }

        # 自动化状态
        self.status = {
            "is_running": False,
            "last_check": None,
            "last_optimize": None,
            "last_report": None,
            "check_count": 0,
            "optimize_count": 0,
            "report_count": 0
        }

        # 问题历史
        self.issue_history = []

    def start_automation(self) -> Dict[str, Any]:
        """启动架构自动化系统"""
        print("🚀 启动架构自动化系统...")

        if self.status["is_running"]:
            return {"success": False, "message": "自动化系统已在运行中"}

        self.status["is_running"] = True
        self.status["start_time"] = datetime.now()

        # 设置定时任务
        self._setup_scheduled_tasks()

        print("✅ 架构自动化系统已启动")
        print(f"📅 检查间隔: {self.config['check_interval']}秒")
        print(f"🔧 优化间隔: {self.config['optimize_interval']}秒")
        print(f"📊 报告间隔: {self.config['report_interval']}秒")

        return {
            "success": True,
            "message": "架构自动化系统已启动",
            "config": self.config
        }

    def stop_automation(self) -> Dict[str, Any]:
        """停止架构自动化系统"""
        print("🛑 停止架构自动化系统...")

        if not self.status["is_running"]:
            return {"success": False, "message": "自动化系统未在运行"}

        self.status["is_running"] = False
        self.status["stop_time"] = datetime.now()

        # 清除定时任务
        schedule.clear()

        print("✅ 架构自动化系统已停止")
        return {
            "success": True,
            "message": "架构自动化系统已停止",
            "runtime": str(self.status["stop_time"] - self.status["start_time"])
        }

    def _setup_scheduled_tasks(self):
        """设置定时任务"""
        # 每小时执行检查
        schedule.every(self.config["check_interval"]).seconds.do(
            self._scheduled_check
        )

        # 每天执行优化
        schedule.every(self.config["optimize_interval"]).seconds.do(
            self._scheduled_optimize
        )

        # 每天生成报告
        schedule.every(self.config["report_interval"]).seconds.do(
            self._scheduled_report
        )

    def _scheduled_check(self):
        """定时检查任务"""
        try:
            print("🔍 执行定时架构检查...")
            result = self.run_automated_check()
            self.status["check_count"] += 1
            self.status["last_check"] = datetime.now()

            if result["issues_count"] > self.config["thresholds"]["critical"]:
                self._send_notification("critical", f"发现{result['issues_count']}个严重问题")

            print(f"✅ 检查完成，发现{result['issues_count']}个问题")

        except Exception as e:
            print(f"❌ 定时检查失败: {e}")

    def _scheduled_optimize(self):
        """定时优化任务"""
        try:
            print("🔧 执行定时架构优化...")
            result = self.run_automated_optimize()
            self.status["optimize_count"] += 1
            self.status["last_optimize"] = datetime.now()

            print(f"✅ 优化完成，修复了{result['fixed_count']}个问题")

        except Exception as e:
            print(f"❌ 定时优化失败: {e}")

    def _scheduled_report(self):
        """定时报告任务"""
        try:
            print("📊 生成定时架构报告...")
            result = self.generate_automated_report()
            self.status["report_count"] += 1
            self.status["last_report"] = datetime.now()

            print(f"✅ 报告生成完成: {result['report_path']}")

        except Exception as e:
            print(f"❌ 定时报告生成失败: {e}")

    def run_automated_check(self) -> Dict[str, Any]:
        """执行自动化检查"""
        print("🔍 开始自动化架构检查...")

        # 运行复核脚本
        review_result = self._run_infrastructure_review()
        if review_result["success"]:
            issues_count = review_result["issues_count"]
            severity = self._classify_issue_severity(issues_count)

            # 记录到历史
            self.issue_history.append({
                "timestamp": datetime.now(),
                "type": "check",
                "issues_count": issues_count,
                "severity": severity,
                "details": review_result["details"]
            })

            return {
                "success": True,
                "issues_count": issues_count,
                "severity": severity,
                "details": review_result["details"]
            }
        else:
            return {
                "success": False,
                "error": review_result["error"]
            }

    def run_automated_optimize(self) -> Dict[str, Any]:
        """执行自动化优化"""
        print("🔧 开始自动化架构优化...")

        fixed_count = 0
        error_count = 0

        # 运行各种优化脚本
        optimization_scripts = [
            "cleanup_empty_dirs.py",
            "fix_remaining_interfaces.py",
            "optimize_cross_layer_imports.py",
            "enhance_responsibility_boundaries.py",
            "enhance_interface_documentation.py"
        ]

        for script in optimization_scripts:
            script_path = self.scripts_dir / script
            if script_path.exists():
                try:
                    result = self._run_optimization_script(script_path)
                    if result["success"]:
                        fixed_count += result["fixed_count"]
                    else:
                        error_count += 1
                except Exception as e:
                    print(f"❌ 运行优化脚本 {script} 失败: {e}")
                    error_count += 1

        # 记录到历史
        self.issue_history.append({
            "timestamp": datetime.now(),
            "type": "optimize",
            "fixed_count": fixed_count,
            "error_count": error_count
        })

        return {
            "success": True,
            "fixed_count": fixed_count,
            "error_count": error_count
        }

    def generate_automated_report(self) -> Dict[str, Any]:
        """生成自动化报告"""
        print("📊 开始生成自动化架构报告...")

        # 生成详细报告
        report_data = {
            "timestamp": datetime.now(),
            "status": self.status.copy(),
            "issue_history": self.issue_history[-100:],  # 最近100条记录
            "config": self.config.copy(),
            "summary": self._generate_report_summary()
        }

        # 保存报告
        report_path = self.reports_dir / \
            f"architecture_automation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)

        return {
            "success": True,
            "report_path": str(report_path),
            "data": report_data
        }

    def _run_infrastructure_review(self) -> Dict[str, Any]:
        """运行基础设施复核"""
        try:
            # 这里应该调用实际的复核脚本
            # 由于脚本可能有复杂的参数，我们使用子进程调用
            result = subprocess.run([
                "python", str(self.scripts_dir / "infrastructure_review.py"),
                "--project", str(self.project_root)
            ], capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                # 解析输出，提取问题数量
                output = result.stdout
                issues_match = re.search(r"发现 (\d+) 个问题", output)
                issues_count = int(issues_match.group(1)) if issues_match else 0

                return {
                    "success": True,
                    "issues_count": issues_count,
                    "details": output
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _run_optimization_script(self, script_path: Path) -> Dict[str, Any]:
        """运行优化脚本"""
        try:
            result = subprocess.run([
                "python", str(script_path),
                "--project", str(self.project_root)
            ], capture_output=True, text=True, timeout=600)

            if result.returncode == 0:
                # 解析输出，提取修复数量
                output = result.stdout
                fixed_match = re.search(r"修复了 (\d+) 个", output)
                fixed_count = int(fixed_match.group(1)) if fixed_match else 0

                return {
                    "success": True,
                    "fixed_count": fixed_count,
                    "details": output
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _classify_issue_severity(self, issues_count: int) -> str:
        """分类问题严重程度"""
        if issues_count >= self.config["thresholds"]["critical"]:
            return "critical"
        elif issues_count >= self.config["thresholds"]["warning"]:
            return "warning"
        elif issues_count >= self.config["thresholds"]["info"]:
            return "info"
        else:
            return "good"

    def _send_notification(self, level: str, message: str):
        """发送通知"""
        if not self.config["notification"]:
            return

        print(f"📢 [{level.upper()}] {message}")

        # 这里可以集成邮件、Slack、钉钉等通知服务
        # 例如：
        # self._send_email_notification(level, message)
        # self._send_slack_notification(level, message)

    def _generate_report_summary(self) -> Dict[str, Any]:
        """生成报告摘要"""
        recent_history = self.issue_history[-24:]  # 最近24条记录

        summary = {
            "total_checks": self.status["check_count"],
            "total_optimizations": self.status["optimize_count"],
            "total_reports": self.status["report_count"],
            "average_issues": 0,
            "trend": "stable",  # stable, improving, worsening
            "health_score": 0
        }

        if recent_history:
            # 计算平均问题数
            check_records = [h for h in recent_history if h["type"] == "check"]
            if check_records:
                summary["average_issues"] = sum(r["issues_count"]
                                                for r in check_records) / len(check_records)

            # 计算趋势
            if len(check_records) >= 2:
                recent = sum(r["issues_count"]
                             for r in check_records[-5:]) / min(5, len(check_records))
                earlier = sum(r["issues_count"]
                              for r in check_records[:-5]) / max(1, len(check_records) - 5)

                if recent < earlier * 0.8:
                    summary["trend"] = "improving"
                elif recent > earlier * 1.2:
                    summary["trend"] = "worsening"

            # 计算健康分数
            summary["health_score"] = max(0, 100 - (summary["average_issues"] * 2))

        return summary

    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "is_running": self.status["is_running"],
            "config": self.config,
            "status": self.status,
            "recent_issues": self.issue_history[-10:],  # 最近10条记录
            "summary": self._generate_report_summary()
        }

    def update_config(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """更新配置"""
        self.config.update(new_config)

        # 如果系统在运行，需要重新设置定时任务
        if self.status["is_running"]:
            schedule.clear()
            self._setup_scheduled_tasks()

        return {
            "success": True,
            "message": "配置已更新",
            "config": self.config
        }

    def generate_final_report(self) -> Dict[str, Any]:
        """生成最终报告"""
        return self.generate_automated_report()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='架构自动化系统')
    parser.add_argument('--project', default='.', help='项目根目录')
    parser.add_argument('--start', action='store_true', help='启动自动化系统')
    parser.add_argument('--stop', action='store_true', help='停止自动化系统')
    parser.add_argument('--status', action='store_true', help='查看系统状态')
    parser.add_argument('--check', action='store_true', help='执行一次检查')
    parser.add_argument('--optimize', action='store_true', help='执行一次优化')
    parser.add_argument('--report', action='store_true', help='生成一次报告')
    parser.add_argument('--config', help='更新配置文件(JSON格式)')

    args = parser.parse_args()

    automation = ArchitectureAutomation(args.project)

    if args.start:
        result = automation.start_automation()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.stop:
        result = automation.stop_automation()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.status:
        result = automation.get_status()
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    elif args.check:
        result = automation.run_automated_check()
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    elif args.optimize:
        result = automation.run_automated_optimize()
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    elif args.report:
        result = automation.generate_automated_report()
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    elif args.config:
        try:
            new_config = json.loads(args.config)
            result = automation.update_config(new_config)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        except json.JSONDecodeError as e:
            print(f"❌ 配置格式错误: {e}")

    else:
        # 默认启动系统
        result = automation.start_automation()
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
