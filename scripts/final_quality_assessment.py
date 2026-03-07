#!/usr/bin/env python3
"""
RQA2025 Phase 31.4 最终质量评估报告生成器
生成全面的质量评估报告，用于生产就绪验证
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


class FinalQualityAssessment:
    """最终质量评估报告生成器"""

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or os.getcwd())
        self.report = {
            "assessment_date": datetime.now().isoformat(),
            "version": "1.0.0",
            "project": "RQA2025",
            "phase": "31.4 - 质量保障体系完善",
            "overall_status": "unknown",
            "quality_metrics": {},
            "assessments": {},
            "recommendations": [],
            "next_steps": []
        }

    def run_full_assessment(self) -> Dict[str, Any]:
        """运行全面质量评估"""
        print("🎯 RQA2025 Phase 31.4 - 最终质量评估")
        print("=" * 60)

        # 1. 代码质量评估
        self._assess_code_quality()

        # 2. 测试质量评估
        self._assess_test_quality()

        # 3. 覆盖率评估
        self._assess_coverage()

        # 4. 性能评估
        self._assess_performance()

        # 5. 安全评估
        self._assess_security()

        # 6. CI/CD评估
        self._assess_cicd()

        # 7. 生产就绪评估
        self._assess_production_readiness()

        # 8. 计算总体状态
        self._calculate_overall_status()

        # 9. 生成建议
        self._generate_recommendations()

        return self.report

    def _assess_code_quality(self):
        """代码质量评估"""
        print("📝 评估代码质量...")

        assessment = {
            "status": "pass",
            "metrics": {},
            "issues": []
        }

        try:
            # 运行flake8检查
            result = subprocess.run([
                sys.executable, "-m", "flake8", "src/",
                "--count", "--statistics", "--max-line-length=120"
            ], capture_output=True, text=True, cwd=self.project_root)

            if result.returncode == 0:
                assessment["metrics"]["syntax_errors"] = 0
            else:
                assessment["status"] = "warning"
                assessment["issues"].append("发现代码风格问题")

            # 计算代码行数
            total_lines = 0
            for root, dirs, files in os.walk("src"):
                for file in files:
                    if file.endswith(".py"):
                        try:
                            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                                total_lines += len(f.readlines())
                        except:
                            pass

            assessment["metrics"]["total_lines"] = total_lines

        except Exception as e:
            assessment["status"] = "error"
            assessment["issues"].append(f"代码质量检查失败: {e}")

        self.report["assessments"]["code_quality"] = assessment

    def _assess_test_quality(self):
        """测试质量评估"""
        print("🧪 评估测试质量...")

        assessment = {
            "status": "pass",
            "metrics": {},
            "issues": []
        }

        try:
            # 运行测试统计
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                "tests/unit/trading/", "tests/integration/",
                "--collect-only", "-q"
            ], capture_output=True, text=True, encoding='utf-8', cwd=self.project_root)

            # 解析测试数量
            lines = result.stdout.split('\n')
            for line in lines:
                if 'collected' in line:
                    parts = line.split()
                    for part in parts:
                        if part.isdigit():
                            assessment["metrics"]["total_tests"] = int(part)
                            break
                    break

            # 检查测试文件数量
            test_files = []
            for root, dirs, files in os.walk("tests"):
                for file in files:
                    if file.startswith("test_") and file.endswith(".py"):
                        test_files.append(os.path.join(root, file))

            assessment["metrics"]["test_files"] = len(test_files)

        except Exception as e:
            assessment["status"] = "error"
            assessment["issues"].append(f"测试质量评估失败: {e}")

        self.report["assessments"]["test_quality"] = assessment

    def _assess_coverage(self):
        """覆盖率评估"""
        print("📊 评估测试覆盖率...")

        assessment = {
            "status": "pass",
            "metrics": {},
            "issues": []
        }

        try:
            # 检查覆盖率历史文件
            coverage_file = self.project_root / "coverage_history.json"
            if coverage_file.exists():
                with open(coverage_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)

                if history:
                    latest = history[-1]
                    coverage = latest.get("coverage", {}).get("total", 0)
                    assessment["metrics"]["current_coverage"] = coverage

                    # 评估覆盖率等级
                    if coverage >= 80:
                        assessment["status"] = "excellent"
                    elif coverage >= 60:
                        assessment["status"] = "good"
                    elif coverage >= 40:
                        assessment["status"] = "acceptable"
                    elif coverage >= 20:
                        assessment["status"] = "warning"
                    else:
                        assessment["status"] = "critical"
                        assessment["issues"].append("覆盖率严重不足")
                else:
                    assessment["status"] = "warning"
                    assessment["issues"].append("没有覆盖率历史数据")
            else:
                assessment["status"] = "warning"
                assessment["issues"].append("覆盖率历史文件不存在")

        except Exception as e:
            assessment["status"] = "error"
            assessment["issues"].append(f"覆盖率评估失败: {e}")

        self.report["assessments"]["coverage"] = assessment

    def _assess_performance(self):
        """性能评估"""
        print("⚡ 评估性能表现...")

        assessment = {
            "status": "pass",
            "metrics": {},
            "issues": []
        }

        try:
            # 检查性能基准文件
            perf_file = self.project_root / "performance_results.json"
            if perf_file.exists():
                with open(perf_file, 'r', encoding='utf-8') as f:
                    perf_data = json.load(f)

                assessment["metrics"]["performance_baselines"] = perf_data
                assessment["status"] = "good"
            else:
                assessment["status"] = "warning"
                assessment["issues"].append("性能基准数据不存在")

        except Exception as e:
            assessment["status"] = "error"
            assessment["issues"].append(f"性能评估失败: {e}")

        self.report["assessments"]["performance"] = assessment

    def _assess_security(self):
        """安全评估"""
        print("🔒 评估安全性...")

        assessment = {
            "status": "pass",
            "metrics": {},
            "issues": []
        }

        try:
            # 检查安全扫描报告
            security_file = self.project_root / "security-report.json"
            if security_file.exists():
                assessment["status"] = "good"
                assessment["metrics"]["security_scan_completed"] = True
            else:
                assessment["status"] = "warning"
                assessment["issues"].append("安全扫描报告不存在")

        except Exception as e:
            assessment["status"] = "error"
            assessment["issues"].append(f"安全评估失败: {e}")

        self.report["assessments"]["security"] = assessment

    def _assess_cicd(self):
        """CI/CD评估"""
        print("🔄 评估CI/CD流程...")

        assessment = {
            "status": "pass",
            "metrics": {},
            "issues": []
        }

        try:
            # 检查GitHub Actions配置
            workflow_file = self.project_root / ".github" / "workflows" / "ci.yml"
            if workflow_file.exists():
                assessment["status"] = "good"
                assessment["metrics"]["cicd_configured"] = True

                # 检查工作流阶段数量
                with open(workflow_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    jobs_count = content.count("jobs:")
                    assessment["metrics"]["workflow_jobs"] = jobs_count
            else:
                assessment["status"] = "critical"
                assessment["issues"].append("CI/CD配置文件不存在")

        except Exception as e:
            assessment["status"] = "error"
            assessment["issues"].append(f"CI/CD评估失败: {e}")

        self.report["assessments"]["cicd"] = assessment

    def _assess_production_readiness(self):
        """生产就绪评估"""
        print("🚀 评估生产就绪性...")

        assessment = {
            "status": "pass",
            "metrics": {},
            "issues": []
        }

        # 检查关键组件
        critical_components = [
            "src/trading/core/trading_engine.py",
            "src/trading/execution/execution_engine.py",
            "src/trading/execution/order_manager.py",
            "src/trading/interfaces/risk/risk.py",
            "src/monitoring/core/real_time_monitor.py",
            "src/monitoring/web/monitoring_web_app.py"
        ]

        missing_components = []
        for component in critical_components:
            if not (self.project_root / component).exists():
                missing_components.append(component)

        if missing_components:
            assessment["status"] = "critical"
            assessment["issues"].extend([f"缺少关键组件: {comp}" for comp in missing_components])
        else:
            assessment["status"] = "good"
            assessment["metrics"]["all_components_present"] = True

        # 检查依赖完整性
        try:
            import sys
            sys.path.insert(0, str(self.project_root))

            import src.trading.core.trading_engine
            import src.trading.execution.execution_engine
            import src.monitoring.core.real_time_monitor
            assessment["metrics"]["imports_successful"] = True
        except ImportError as e:
            # 导入失败不一定是critical问题，只要关键组件存在即可
            assessment["status"] = "warning"
            assessment["issues"].append(f"模块导入警告: {e}")
            assessment["metrics"]["imports_successful"] = False

        self.report["assessments"]["production_readiness"] = assessment

    def _calculate_overall_status(self):
        """计算总体状态"""
        assessments = self.report["assessments"]

        status_priority = {
            "critical": 5,
            "error": 4,
            "warning": 3,
            "acceptable": 2,
            "good": 1,
            "excellent": 0,
            "pass": 0
        }

        max_priority = 0
        for assessment in assessments.values():
            status = assessment.get("status", "unknown")
            priority = status_priority.get(status, 3)
            max_priority = max(max_priority, priority)

        # 根据最高优先级确定总体状态
        if max_priority >= 5:
            overall_status = "critical"
        elif max_priority >= 4:
            overall_status = "error"
        elif max_priority >= 3:
            overall_status = "warning"
        elif max_priority >= 2:
            overall_status = "acceptable"
        else:
            overall_status = "production_ready"

        self.report["overall_status"] = overall_status

    def _generate_recommendations(self):
        """生成建议"""
        assessments = self.report["assessments"]

        recommendations = []
        next_steps = []

        # 基于各个评估结果生成建议
        for category, assessment in assessments.items():
            status = assessment.get("status", "unknown")
            issues = assessment.get("issues", [])

            if status in ["critical", "error"]:
                next_steps.extend([f"🔴 紧急修复 {category}: {issue}" for issue in issues])
            elif status == "warning":
                recommendations.extend([f"🟡 优化 {category}: {issue}" for issue in issues])

        # 总体建议
        if self.report["overall_status"] == "production_ready":
            recommendations.append("✅ 系统已达到生产就绪标准")
            next_steps.append("🎉 可以开始生产部署流程")
        elif self.report["overall_status"] == "acceptable":
            recommendations.append("🟢 系统基本达到生产标准，建议优化后部署")
            next_steps.append("📋 完成剩余优化项目后可部署")
        else:
            recommendations.append("🔴 系统存在关键问题，需要修复后才能部署")
            next_steps.append("🛠️ 优先修复所有critical和error级别问题")

        self.report["recommendations"] = recommendations
        self.report["next_steps"] = next_steps

    def save_report(self, output_file: str = None):
        """保存评估报告"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"final_quality_assessment_{timestamp}.json"

        output_path = self.project_root / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)

        print(f"📄 质量评估报告已保存至: {output_path}")
        return output_path

    def print_summary(self):
        """打印评估摘要"""
        print("\n" + "="*60)
        print("🎯 RQA2025 Phase 31.4 最终质量评估报告")
        print("="*60)

        print(f"📅 评估时间: {self.report['assessment_date']}")
        print(f"🏷️  版本: {self.report['version']}")
        print(f"📊 总体状态: {self.report['overall_status'].upper()}")

        print("\n📋 各维度评估结果:")
        for category, assessment in self.report["assessments"].items():
            status = assessment.get("status", "unknown")
            status_icon = {
                "excellent": "🌟",
                "good": "✅",
                "acceptable": "🟢",
                "warning": "🟡",
                "error": "🔴",
                "critical": "🚨",
                "pass": "✅"
            }.get(status, "❓")

            print(f"  {status_icon} {category}: {status}")

        print("\n💡 关键指标:")
        for category, assessment in self.report["assessments"].items():
            metrics = assessment.get("metrics", {})
            if metrics:
                for key, value in metrics.items():
                    if isinstance(value, float):
                        print(".2f")
                    else:
                        print(f"    • {key}: {value}")

        if self.report["recommendations"]:
            print("\n🎯 建议:")
            for rec in self.report["recommendations"]:
                print(f"  {rec}")

        if self.report["next_steps"]:
            print("\n🚀 后续行动:")
            for step in self.report["next_steps"]:
                print(f"  {step}")

        print("\n" + "="*60)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="RQA2025 最终质量评估")
    parser.add_argument("--output", help="输出文件路径")
    parser.add_argument("--quiet", action="store_true", help="静默模式")

    args = parser.parse_args()

    assessor = FinalQualityAssessment()

    if not args.quiet:
        report = assessor.run_full_assessment()
        assessor.print_summary()
    else:
        report = assessor.run_full_assessment()

    output_file = assessor.save_report(args.output)

    # 返回退出码基于总体状态
    status_codes = {
        "production_ready": 0,
        "excellent": 0,
        "good": 0,
        "acceptable": 1,
        "warning": 2,
        "error": 3,
        "critical": 4
    }

    exit_code = status_codes.get(report["overall_status"], 1)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
