#!/usr/bin/env python3
"""
RQA2025 Phase 31.7 最终质量评估报告生成器
生成全面的质量评估报告，包含CI/CD和部署验证
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import argparse
import glob


class Phase31_7_QualityAssessment:
    """Phase 31.7 最终质量评估报告生成器"""

    def __init__(self, input_dir: str = None, output_file: str = "quality_assessment_report.md"):
        self.input_dir = Path(input_dir or ".")
        self.output_file = output_file
        self.report = {
            "assessment_date": datetime.now().isoformat(),
            "version": "1.0.0",
            "project": "RQA2025",
            "phase": "31.7 - CI/CD完善和部署验证",
            "overall_status": "unknown",
            "quality_metrics": {},
            "assessments": {},
            "recommendations": [],
            "next_steps": []
        }

    def run_full_assessment(self) -> Dict[str, Any]:
        """运行全面质量评估"""
        print("🎯 RQA2025 Phase 31.7 - CI/CD和部署质量评估")
        print("=" * 60)

        # 1. 代码质量评估
        self._assess_code_quality()

        # 2. 测试质量评估
        self._assess_test_quality()

        # 3. 覆盖率评估
        self._assess_coverage()

        # 4. 性能评估
        self._assess_performance()

        # 5. CI/CD评估
        self._assess_ci_cd()

        # 6. 部署就绪评估
        self._assess_deployment_readiness()

        # 7. 安全评估
        self._assess_security()

        # 8. 生成综合建议
        self._generate_recommendations()

        # 9. 计算总体状态
        self._calculate_overall_status()

        return self.report

    def _assess_code_quality(self):
        """代码质量评估"""
        print("📝 评估代码质量...")

        quality_metrics = {
            "total_files": 0,
            "python_files": 0,
            "lines_of_code": 0,
            "code_complexity": "unknown",
            "linting_errors": 0,
            "type_check_errors": 0
        }

        # 查找Python文件
        python_files = list(self.input_dir.rglob("*.py"))
        quality_metrics["python_files"] = len(python_files)

        # 计算代码行数
        total_lines = 0
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    total_lines += len(lines)
            except:
                pass
        quality_metrics["lines_of_code"] = total_lines

        # 检查linting结果
        flake8_reports = list(self.input_dir.glob("**/flake8_report.txt"))
        if flake8_reports:
            # 解析flake8报告
            quality_metrics["linting_errors"] = self._parse_linting_report(flake8_reports[0])

        # 检查mypy结果
        mypy_reports = list(self.input_dir.glob("**/mypy_report.txt"))
        if mypy_reports:
            quality_metrics["type_check_errors"] = self._parse_mypy_report(mypy_reports[0])

        self.report["assessments"]["code_quality"] = {
            "status": "pass" if quality_metrics["linting_errors"] < 50 else "fail",
            "metrics": quality_metrics,
            "details": f"代码库包含{quality_metrics['python_files']}个Python文件，{quality_metrics['lines_of_code']}行代码"
        }

    def _assess_test_quality(self):
        """测试质量评估"""
        print("🧪 评估测试质量...")

        test_metrics = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "skipped_tests": 0,
            "test_execution_time": 0,
            "test_files": 0
        }

        # 查找测试文件
        test_files = list(self.input_dir.glob("**/test_*.py"))
        test_metrics["test_files"] = len(test_files)

        # 解析pytest结果
        junit_files = list(self.input_dir.glob("**/test-results-*.xml"))
        if junit_files:
            test_metrics.update(self._parse_junit_reports(junit_files))

        # 计算测试通过率
        total_tests = test_metrics["passed_tests"] + test_metrics["failed_tests"]
        pass_rate = test_metrics["passed_tests"] / total_tests if total_tests > 0 else 0

        self.report["assessments"]["test_quality"] = {
            "status": "pass" if pass_rate > 0.95 else "fail",
            "metrics": test_metrics,
            "details": ".1f"
        }

    def _assess_coverage(self):
        """覆盖率评估"""
        print("📊 评估测试覆盖率...")

        coverage_metrics = {
            "overall_coverage": 0.0,
            "line_coverage": 0.0,
            "branch_coverage": 0.0,
            "missing_lines": 0,
            "covered_lines": 0
        }

        # 查找覆盖率报告
        coverage_files = list(self.input_dir.glob("**/coverage.xml"))
        if coverage_files:
            coverage_metrics.update(self._parse_coverage_report(coverage_files[0]))

        # 查找HTML覆盖率报告
        html_coverage = list(self.input_dir.glob("**/htmlcov/index.html"))
        if html_coverage:
            coverage_metrics["html_report_available"] = True

        coverage_status = "pass" if coverage_metrics["overall_coverage"] >= 30.0 else "fail"

        self.report["assessments"]["coverage"] = {
            "status": coverage_status,
            "metrics": coverage_metrics,
            "details": ".1f"
        }

    def _assess_performance(self):
        """性能评估"""
        print("⚡ 评估性能表现...")

        performance_metrics = {
            "baseline_tests_run": 0,
            "stress_tests_run": 0,
            "concurrency_tests_run": 0,
            "performance_regression": False,
            "memory_usage_mb": 0,
            "response_time_ms": 0
        }

        # 查找性能测试结果
        perf_reports = list(self.input_dir.glob("**/performance_results.json"))
        if perf_reports:
            with open(perf_reports[0], 'r') as f:
                perf_data = json.load(f)
                performance_metrics.update(perf_data.get("summary", {}))

        # 检查Phase 31.6性能基准测试结果
        phase31_6_results = list(self.input_dir.glob("**/test-results-performance*.xml"))
        if phase31_6_results:
            performance_metrics["baseline_tests_run"] = len(phase31_6_results)

        performance_status = "pass" if not performance_metrics.get("performance_regression", True) else "fail"

        self.report["assessments"]["performance"] = {
            "status": performance_status,
            "metrics": performance_metrics,
            "details": f"运行了{performance_metrics['baseline_tests_run']}个性能基准测试"
        }

    def _assess_ci_cd(self):
        """CI/CD评估"""
        print("🔄 评估CI/CD流程...")

        ci_cd_metrics = {
            "workflows_present": False,
            "required_jobs": [],
            "security_scans": False,
            "deployment_jobs": False,
            "rollback_mechanism": False,
            "quality_gates": False
        }

        # 检查GitHub Actions工作流
        workflow_files = list(self.input_dir.glob(".github/workflows/*.yml"))
        ci_cd_metrics["workflows_present"] = len(workflow_files) > 0

        if workflow_files:
            ci_cd_metrics["required_jobs"] = self._analyze_ci_cd_workflows(workflow_files)

        # 检查部署相关配置
        deploy_configs = list(self.input_dir.glob("**/deploy*.yml")) + list(self.input_dir.glob("**/docker*.yml"))
        ci_cd_metrics["deployment_jobs"] = len(deploy_configs) > 0

                    # 检查回滚机制（通过workflow文件内容判断）
        ci_cd_metrics["rollback_mechanism"] = any("rollback" in str(workflow.read_text(encoding='utf-8')).lower()
                                                 for workflow in workflow_files)

        # 检查质量门禁
        ci_cd_metrics["quality_gates"] = any("quality-gate" in str(workflow.read_text(encoding='utf-8')).lower()
                                           for workflow in workflow_files)

        ci_cd_status = "pass" if all([ci_cd_metrics["workflows_present"],
                                    ci_cd_metrics["quality_gates"],
                                    len(ci_cd_metrics["required_jobs"]) >= 3]) else "fail"

        self.report["assessments"]["ci_cd"] = {
            "status": ci_cd_status,
            "metrics": ci_cd_metrics,
            "details": f"检测到{len(workflow_files)}个CI/CD工作流，包含{len(ci_cd_metrics['required_jobs'])}个必要作业"
        }

    def _assess_deployment_readiness(self):
        """部署就绪评估"""
        print("🚀 评估部署就绪状态...")

        deployment_metrics = {
            "dockerfile_present": False,
            "requirements_locked": False,
            "environment_configs": [],
            "deployment_scripts": False,
            "health_checks": False,
            "monitoring_setup": False
        }

        # 检查Docker配置
        dockerfiles = list(self.input_dir.glob("**/Dockerfile*"))
        deployment_metrics["dockerfile_present"] = len(dockerfiles) > 0

        # 检查依赖锁定
        lockfiles = list(self.input_dir.glob("**/requirements*.txt")) + list(self.input_dir.glob("**/Pipfile.lock"))
        deployment_metrics["requirements_locked"] = len(lockfiles) > 0

        # 检查环境配置
        env_files = list(self.input_dir.glob("**/.env*")) + list(self.input_dir.glob("**/config*.yml"))
        deployment_metrics["environment_configs"] = [f.name for f in env_files]

        # 检查部署脚本
        deploy_scripts = list(self.input_dir.glob("**/deploy*.sh")) + list(self.input_dir.glob("**/deploy*.py"))
        deployment_metrics["deployment_scripts"] = len(deploy_scripts) > 0

        # 检查健康检查
        health_files = list(self.input_dir.glob("**/health*.py")) + list(self.input_dir.glob("**/health*.sh"))
        deployment_metrics["health_checks"] = len(health_files) > 0

        # 检查监控配置
        monitoring_files = list(self.input_dir.glob("**/monitoring*.yml")) + list(self.input_dir.glob("**/prometheus*.yml"))
        deployment_metrics["monitoring_setup"] = len(monitoring_files) > 0

        # 计算部署就绪分数
        readiness_score = sum([
            deployment_metrics["dockerfile_present"],
            deployment_metrics["requirements_locked"],
            deployment_metrics["deployment_scripts"],
            deployment_metrics["health_checks"],
            deployment_metrics["monitoring_setup"]
        ]) / 5.0 * 100

        deployment_status = "pass" if readiness_score >= 80 else "fail"

        self.report["assessments"]["deployment_readiness"] = {
            "status": deployment_status,
            "metrics": deployment_metrics,
            "details": ".1f"
        }

    def _assess_security(self):
        """安全评估"""
        print("🔒 评估安全状况...")

        security_metrics = {
            "vulnerability_scan_run": False,
            "high_severity_issues": 0,
            "medium_severity_issues": 0,
            "low_severity_issues": 0,
            "secrets_detected": False,
            "security_headers": False
        }

        # 检查安全扫描报告
        security_reports = list(self.input_dir.glob("**/security-report.json"))
        if security_reports:
            security_metrics["vulnerability_scan_run"] = True
            security_metrics.update(self._parse_security_report(security_reports[0]))

        # 检查机密信息泄露
        secret_patterns = ["password", "secret", "key", "token"]
        secret_files = []

        for pattern in secret_patterns:
            found = list(self.input_dir.glob(f"**/*{pattern}*"))
            if found:
                secret_files.extend([f.name for f in found])

        security_metrics["secrets_detected"] = len(secret_files) > 0
        security_metrics["potential_secret_files"] = secret_files

        security_status = "pass" if (security_metrics["vulnerability_scan_run"] and
                                   security_metrics["high_severity_issues"] == 0 and
                                   not security_metrics["secrets_detected"]) else "fail"

        self.report["assessments"]["security"] = {
            "status": security_status,
            "metrics": security_metrics,
            "details": f"发现{security_metrics['high_severity_issues']}个高危安全问题"
        }

    def _generate_recommendations(self):
        """生成综合建议"""
        print("💡 生成优化建议...")

        recommendations = []

        # 基于各个评估结果生成建议
        assessments = self.report["assessments"]

        # 代码质量建议
        if assessments.get("code_quality", {}).get("status") == "fail":
            recommendations.append("修复代码质量问题：减少linting错误，提高类型检查覆盖率")

        # 测试质量建议
        test_assessment = assessments.get("test_quality", {})
        if test_assessment.get("status") == "fail":
            total_tests = (test_assessment.get("metrics", {}).get("passed_tests", 0) +
                          test_assessment.get("metrics", {}).get("failed_tests", 0))
            pass_rate = test_assessment.get("metrics", {}).get("passed_tests", 0) / total_tests * 100 if total_tests > 0 else 0
            recommendations.append(f"提高测试通过率：当前通过率 {pass_rate:.1f}%，目标>95%")
        # 覆盖率建议
        coverage_assessment = assessments.get("coverage", {})
        if coverage_assessment.get("status") == "fail":
            coverage = coverage_assessment.get("metrics", {}).get("overall_coverage", 0)
            recommendations.append(f"提高测试覆盖率：当前覆盖率 {coverage:.1f}%，目标>30%")
        # CI/CD建议
        if assessments.get("ci_cd", {}).get("status") == "fail":
            recommendations.append("完善CI/CD流程：确保包含质量门禁、部署验证和回滚机制")

        # 部署建议
        if assessments.get("deployment_readiness", {}).get("status") == "fail":
            recommendations.append("提高部署就绪度：添加容器化配置、健康检查和监控设置")

        # 安全建议
        if assessments.get("security", {}).get("status") == "fail":
            recommendations.append("解决安全问题：修复高危漏洞，移除泄露的机密信息")

        # 通用建议
        recommendations.extend([
            "建立定期代码审查流程",
            "实施自动化性能回归测试",
            "配置生产环境监控和告警",
            "建立灾难恢复和业务连续性计划"
        ])

        self.report["recommendations"] = recommendations

        # 生成下一步行动
        self.report["next_steps"] = [
            "修复所有高优先级质量问题",
            "完善CI/CD部署流程",
            "建立生产环境监控体系",
            "进行生产环境试运行",
            "制定上线和回滚计划"
        ]

    def _calculate_overall_status(self):
        """计算总体状态"""
        assessments = self.report["assessments"]

        # 计算各评估的权重
        weights = {
            "code_quality": 0.15,
            "test_quality": 0.20,
            "coverage": 0.15,
            "performance": 0.15,
            "ci_cd": 0.10,
            "deployment_readiness": 0.10,
            "security": 0.15
        }

        # 计算加权分数
        total_score = 0
        total_weight = 0

        for assessment_name, weight in weights.items():
            assessment = assessments.get(assessment_name, {})
            status = assessment.get("status", "unknown")

            if status == "pass":
                score = 100
            elif status == "fail":
                score = 0
            else:
                score = 50  # unknown状态给50分

            total_score += score * weight
            total_weight += weight

        final_score = total_score / total_weight if total_weight > 0 else 0

        # 确定总体状态
        if final_score >= 85:
            overall_status = "production_ready"
        elif final_score >= 70:
            overall_status = "staging_ready"
        elif final_score >= 50:
            overall_status = "development_ready"
        else:
            overall_status = "needs_improvement"

        self.report["overall_status"] = overall_status
        self.report["quality_score"] = final_score

    def _parse_linting_report(self, report_file: Path) -> int:
        """解析linting报告"""
        try:
            with open(report_file, 'r') as f:
                content = f.read()
                # 简单的错误计数（可以根据实际格式调整）
                return content.count("error") + content.count("Error")
        except:
            return 0

    def _parse_mypy_report(self, report_file: Path) -> int:
        """解析mypy报告"""
        try:
            with open(report_file, 'r') as f:
                content = f.read()
                return content.count("error") + content.count("Error")
        except:
            return 0

    def _parse_junit_reports(self, junit_files: List[Path]) -> Dict[str, int]:
        """解析JUnit测试报告"""
        metrics = {"passed_tests": 0, "failed_tests": 0, "skipped_tests": 0}

        for junit_file in junit_files:
            try:
                with open(junit_file, 'r') as f:
                    content = f.read()
                    # 简单的XML解析（可以根据实际格式调整）
                    metrics["passed_tests"] += content.count('testsuite')
                    metrics["failed_tests"] += content.count('failure') + content.count('error')
                    metrics["skipped_tests"] += content.count('skipped')
            except:
                pass

        return metrics

    def _parse_coverage_report(self, coverage_file: Path) -> Dict[str, float]:
        """解析覆盖率报告"""
        metrics = {"overall_coverage": 0.0, "line_coverage": 0.0}

        try:
            with open(coverage_file, 'r') as f:
                content = f.read()
                # 查找覆盖率百分比（可以根据实际格式调整）
                import re
                coverage_match = re.search(r'line-rate="([0-9.]+)"', content)
                if coverage_match:
                    metrics["overall_coverage"] = float(coverage_match.group(1)) * 100
                    metrics["line_coverage"] = metrics["overall_coverage"]
        except:
            pass

        return metrics

    def _parse_security_report(self, security_file: Path) -> Dict[str, int]:
        """解析安全报告"""
        metrics = {"high_severity_issues": 0, "medium_severity_issues": 0, "low_severity_issues": 0}

        try:
            with open(security_file, 'r') as f:
                security_data = json.load(f)
                # 根据bandit输出格式解析
                for issue in security_data.get("results", []):
                    severity = issue.get("issue_severity", "low").lower()
                    if severity == "high":
                        metrics["high_severity_issues"] += 1
                    elif severity == "medium":
                        metrics["medium_severity_issues"] += 1
                    else:
                        metrics["low_severity_issues"] += 1
        except:
            pass

        return metrics

    def _analyze_ci_cd_workflows(self, workflow_files: List[Path]) -> List[str]:
        """分析CI/CD工作流"""
        required_jobs = []

        for workflow_file in workflow_files:
            try:
                with open(workflow_file, 'r') as f:
                    content = f.read().lower()

                    # 检查常见作业类型
                    if 'lint' in content or 'flake8' in content:
                        if 'lint' not in required_jobs:
                            required_jobs.append('lint')

                    if 'test' in content or 'pytest' in content:
                        if 'test' not in required_jobs:
                            required_jobs.append('test')

                    if 'build' in content or 'docker' in content:
                        if 'build' not in required_jobs:
                            required_jobs.append('build')

                    if 'deploy' in content:
                        if 'deploy' not in required_jobs:
                            required_jobs.append('deploy')

                    if 'security' in content or 'bandit' in content:
                        if 'security' not in required_jobs:
                            required_jobs.append('security')

            except:
                pass

        return required_jobs

    def generate_markdown_report(self, output_file: str = None):
        """生成Markdown格式的报告"""
        output_file = output_file or self.output_file

        report_content = f"""# RQA2025 Phase 31.7 最终质量评估报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**版本**: {self.report['version']}
**阶段**: {self.report['phase']}
**总体状态**: {self.report['overall_status'].upper()}
**质量评分**: {self.report.get('quality_score', 0):.1f}/100

## 📊 总体概览

"""

        # 各评估状态
        assessments = self.report["assessments"]
        for assessment_name, assessment in assessments.items():
            status_emoji = "✅" if assessment["status"] == "pass" else "❌" if assessment["status"] == "fail" else "⚠️"
            report_content += f"- **{assessment_name.replace('_', ' ').title()}**: {status_emoji} {assessment['status'].upper()}\n"
            report_content += f"  - {assessment['details']}\n"

        report_content += "\n## 📈 详细评估结果\n\n"

        # 详细结果
        for assessment_name, assessment in assessments.items():
            report_content += f"### {assessment_name.replace('_', ' ').title()}\n\n"
            report_content += f"**状态**: {assessment['status'].upper()}\n\n"
            report_content += f"**详情**: {assessment['details']}\n\n"

            if 'metrics' in assessment:
                report_content += "**关键指标**:\n"
                for key, value in assessment['metrics'].items():
                    if isinstance(value, float):
                        report_content += f"- {key}: {value:.2f}\n"
                    else:
                        report_content += f"- {key}: {value}\n"
            report_content += "\n"

        # 建议
        if self.report["recommendations"]:
            report_content += "## 💡 优化建议\n\n"
            for rec in self.report["recommendations"]:
                report_content += f"- {rec}\n"
            report_content += "\n"

        # 下一步行动
        if self.report["next_steps"]:
            report_content += "## 🎯 后续行动\n\n"
            for step in self.report["next_steps"]:
                report_content += f"- {step}\n"
            report_content += "\n"

        report_content += "---\n"
        report_content += "*此报告由Phase 31.7质量评估系统自动生成*\n"

        # 保存报告
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"📄 Markdown报告已保存到: {output_file}")

    def generate_json_report(self, output_file: str = "quality_assessment_report.json"):
        """生成JSON格式的报告"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)

        print(f"📄 JSON报告已保存到: {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Phase 31.7 最终质量评估")
    parser.add_argument("--input-dir", default=".", help="输入目录路径")
    parser.add_argument("--output", default="quality_assessment_report.md", help="输出文件路径")

    args = parser.parse_args()

    # 创建评估器
    assessor = Phase31_7_QualityAssessment(args.input_dir, args.output)

    # 运行全面评估
    report = assessor.run_full_assessment()

    # 生成报告
    assessor.generate_markdown_report()
    assessor.generate_json_report()

    # 输出总体状态
    status = report["overall_status"]
    score = report.get("quality_score", 0)

    print("\n" + "="*60)
    print("🎯 最终评估结果")
    print(f"   总体状态: {status.upper()}")
    print(".1f")
    print("="*60)

    if status == "production_ready":
        print("🎉 恭喜！系统已达到生产就绪标准")
    elif status == "staging_ready":
        print("✅ 系统可部署到暂存环境")
    elif status == "development_ready":
        print("⚠️ 系统基本可用，建议进一步优化")
    else:
        print("❌ 系统需要重大改进")


if __name__ == "__main__":
    main()
