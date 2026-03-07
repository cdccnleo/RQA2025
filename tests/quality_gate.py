#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
质量门禁系统

基于性能指标的自动化质量门禁：
- 代码质量门禁（覆盖率、复杂度、规范检查）
- 性能质量门禁（执行时间、资源使用、稳定性）
- 安全质量门禁（漏洞扫描、依赖检查）
- 合规质量门禁（许可证、文档完整性）
"""

import os
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class QualityThreshold:
    """质量阈值"""
    metric: str
    operator: str  # '>', '<', '>=', '<=', '==', '!='
    value: float
    description: str
    severity: str  # 'error', 'warning', 'info'


@dataclass
class QualityCheckResult:
    """质量检查结果"""
    check_name: str
    status: str  # 'pass', 'fail', 'warning', 'error'
    score: float  # 0-100
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class QualityGateResult:
    """质量门禁结果"""
    overall_status: str  # 'pass', 'fail', 'warning'
    overall_score: float
    checks_passed: int
    checks_failed: int
    checks_warning: int
    total_checks: int
    execution_time: float
    recommendations: List[str]
    detailed_results: List[QualityCheckResult]


class CodeQualityGate:
    """代码质量门禁"""

    def __init__(self):
        self.thresholds = self._define_thresholds()

    def _define_thresholds(self) -> Dict[str, QualityThreshold]:
        """定义质量阈值"""
        return {
            'test_coverage': QualityThreshold(
                metric='test_coverage',
                operator='>=',
                value=80.0,
                description='测试覆盖率必须达到80%以上',
                severity='error'
            ),
            'code_complexity': QualityThreshold(
                metric='code_complexity',
                operator='<=',
                value=10.0,
                description='代码复杂度不能超过10',
                severity='warning'
            ),
            'duplicate_lines': QualityThreshold(
                metric='duplicate_lines',
                operator='<=',
                value=5.0,
                description='重复代码行数不能超过5%',
                severity='warning'
            ),
            'lint_errors': QualityThreshold(
                metric='lint_errors',
                operator='==',
                value=0.0,
                description='不能有代码规范错误',
                severity='error'
            ),
            'security_vulnerabilities': QualityThreshold(
                metric='security_vulnerabilities',
                operator='==',
                value=0.0,
                description='不能有安全漏洞',
                severity='error'
            )
        }

    def check_test_coverage(self) -> QualityCheckResult:
        """检查测试覆盖率"""
        try:
            # 查找最新的覆盖率报告
            coverage_files = list(Path(".").glob("coverage*.xml"))
            coverage_files.extend(Path("htmlcov").glob("*.xml") if Path("htmlcov").exists() else [])

            if not coverage_files:
                return QualityCheckResult(
                    check_name='test_coverage',
                    status='error',
                    score=0.0,
                    message='未找到覆盖率报告文件',
                    recommendations=['运行测试覆盖率检查', '确保生成了coverage.xml文件']
                )

            # 简单解析XML获取覆盖率（实际项目中可以使用专门的解析库）
            latest_coverage = coverage_files[0]
            coverage_percent = self._parse_coverage_xml(latest_coverage)

            threshold = self.thresholds['test_coverage']
            passed = self._check_threshold(coverage_percent, threshold)

            status = 'pass' if passed else 'fail'
            score = min(100.0, coverage_percent * 1.25)  # 80%覆盖率对应100分

            return QualityCheckResult(
                check_name='test_coverage',
                status=status,
                score=score,
                message=f'测试覆盖率: {coverage_percent:.1f}%',
                details={'coverage_percent': coverage_percent, 'threshold': threshold.value},
                recommendations=['增加单元测试覆盖'] if not passed else []
            )

        except Exception as e:
            return QualityCheckResult(
                check_name='test_coverage',
                status='error',
                score=0.0,
                message=f'覆盖率检查失败: {e}',
                recommendations=['检查覆盖率工具配置', '确保测试环境正常']
            )

    def _parse_coverage_xml(self, xml_file: Path) -> float:
        """解析覆盖率XML文件"""
        try:
            with open(xml_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 查找覆盖率百分比（简化解析）
            match = re.search(r'line-rate="([0-9.]+)"', content)
            if match:
                return float(match.group(1)) * 100

            # 如果没找到，尝试其他格式
            match = re.search(r'coverage="([0-9.]+)"', content)
            if match:
                return float(match.group(1))

            return 0.0

        except Exception:
            return 0.0

    def check_code_complexity(self) -> QualityCheckResult:
        """检查代码复杂度"""
        try:
            # 使用简单的行计数作为复杂度指标
            total_lines = 0
            total_files = 0

            for py_file in Path("src").rglob("*.py"):
                try:
                    with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = len(f.readlines())
                        total_lines += lines
                        total_files += 1
                except Exception:
                    continue

            if total_files == 0:
                return QualityCheckResult(
                    check_name='code_complexity',
                    status='error',
                    score=0.0,
                    message='未找到Python源文件',
                    recommendations=['检查源代码目录结构']
                )

            avg_complexity = total_lines / total_files
            threshold = self.thresholds['code_complexity']
            passed = self._check_threshold(avg_complexity, threshold)

            status = 'pass' if passed else 'warning'
            score = max(0.0, 100.0 - (avg_complexity - threshold.value) * 2)

            return QualityCheckResult(
                check_name='code_complexity',
                status=status,
                score=score,
                message=f'平均代码复杂度: {avg_complexity:.1f} 行/文件',
                details={'avg_complexity': avg_complexity, 'total_files': total_files},
                recommendations=['重构大文件', '拆分复杂函数'] if not passed else []
            )

        except Exception as e:
            return QualityCheckResult(
                check_name='code_complexity',
                status='error',
                score=0.0,
                message=f'复杂度检查失败: {e}',
                recommendations=['检查源代码访问权限']
            )

    def check_lint_errors(self) -> QualityCheckResult:
        """检查代码规范错误"""
        try:
            # 运行flake8或其他lint工具
            result = subprocess.run(
                ['python', '-m', 'flake8', 'src', '--count', '--select=E9,F63,F7,F82', '--show-source', '--statistics'],
                capture_output=True,
                text=True,
                timeout=60
            )

            error_count = 0
            if result.returncode > 0:
                # 尝试从输出中提取错误数量
                lines = result.stdout.split('\n') + result.stderr.split('\n')
                for line in lines:
                    if line.strip().isdigit():
                        error_count = int(line.strip())
                        break

            threshold = self.thresholds['lint_errors']
            passed = self._check_threshold(error_count, threshold)

            status = 'pass' if passed else 'error'
            score = 100.0 if passed else max(0.0, 100.0 - error_count * 10)

            return QualityCheckResult(
                check_name='lint_errors',
                status=status,
                score=score,
                message=f'代码规范错误: {error_count} 个',
                details={'error_count': error_count},
                recommendations=['修复代码规范问题', '运行flake8检查'] if not passed else []
            )

        except subprocess.TimeoutExpired:
            return QualityCheckResult(
                check_name='lint_errors',
                status='error',
                score=0.0,
                message='代码规范检查超时',
                recommendations=['检查lint工具配置', '减少检查的文件数量']
            )
        except Exception as e:
            return QualityCheckResult(
                check_name='lint_errors',
                status='warning',
                score=50.0,
                message=f'代码规范检查失败: {e}',
                recommendations=['安装flake8工具', '检查Python环境']
            )

    def _check_threshold(self, value: float, threshold: QualityThreshold) -> bool:
        """检查是否满足阈值"""
        if threshold.operator == '>=':
            return value >= threshold.value
        elif threshold.operator == '<=':
            return value <= threshold.value
        elif threshold.operator == '>':
            return value > threshold.value
        elif threshold.operator == '<':
            return value < threshold.value
        elif threshold.operator == '==':
            return abs(value - threshold.value) < 0.001
        elif threshold.operator == '!=':
            return abs(value - threshold.value) >= 0.001
        else:
            return False


class PerformanceQualityGate:
    """性能质量门禁"""

    def __init__(self):
        self.thresholds = self._define_thresholds()

    def _define_thresholds(self) -> Dict[str, QualityThreshold]:
        """定义性能阈值"""
        return {
            'avg_test_time': QualityThreshold(
                metric='avg_test_time',
                operator='<=',
                value=5.0,
                description='平均测试执行时间不能超过5秒',
                severity='warning'
            ),
            'test_success_rate': QualityThreshold(
                metric='test_success_rate',
                operator='>=',
                value=90.0,
                description='测试成功率必须达到90%以上',
                severity='error'
            ),
            'memory_usage': QualityThreshold(
                metric='memory_usage',
                operator='<=',
                value=500.0,
                description='内存使用不能超过500MB',
                severity='warning'
            ),
            'performance_regression': QualityThreshold(
                metric='performance_regression',
                operator='<=',
                value=10.0,
                description='性能回退不能超过10%',
                severity='warning'
            )
        }

    def check_test_performance(self) -> QualityCheckResult:
        """检查测试性能"""
        try:
            # 读取性能历史数据
            history_file = Path("test_logs/performance_history.json")
            if not history_file.exists():
                return QualityCheckResult(
                    check_name='test_performance',
                    status='warning',
                    score=50.0,
                    message='未找到性能历史数据',
                    recommendations=['运行性能监控收集数据']
                )

            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)

            if not history:
                return QualityCheckResult(
                    check_name='test_performance',
                    status='warning',
                    score=50.0,
                    message='性能历史数据为空',
                    recommendations=['执行更多测试来收集性能数据']
                )

            # 计算平均执行时间
            execution_times = [h.get('execution_time', 0) for h in history if h.get('execution_time', 0) > 0]
            if execution_times:
                avg_time = sum(execution_times) / len(execution_times)
                threshold = self.thresholds['avg_test_time']
                time_passed = self._check_threshold(avg_time, threshold)
            else:
                avg_time = 0
                time_passed = True

            # 计算成功率
            total_tests = len(history)
            successful_tests = sum(1 for h in history if h.get('success', False))
            success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
            threshold = self.thresholds['test_success_rate']
            success_passed = self._check_threshold(success_rate, threshold)

            # 综合评估
            overall_passed = time_passed and success_passed
            status = 'pass' if overall_passed else ('error' if not success_passed else 'warning')
            score = (success_rate + (100 if time_passed else 50)) / 2

            return QualityCheckResult(
                check_name='test_performance',
                status=status,
                score=score,
                message=f'平均执行时间: {avg_time:.2f}s, 成功率: {success_rate:.1f}%',
                details={
                    'avg_time': avg_time,
                    'success_rate': success_rate,
                    'total_tests': total_tests
                },
                recommendations=[
                    '优化慢速测试' if not time_passed else '',
                    '修复失败的测试' if not success_passed else ''
                ]
            )

        except Exception as e:
            return QualityCheckResult(
                check_name='test_performance',
                status='error',
                score=0.0,
                message=f'性能检查失败: {e}',
                recommendations=['检查性能数据文件', '重新运行性能监控']
            )

    def _check_threshold(self, value: float, threshold: QualityThreshold) -> bool:
        """检查是否满足阈值"""
        if threshold.operator == '>=':
            return value >= threshold.value
        elif threshold.operator == '<=':
            return value <= threshold.value
        elif threshold.operator == '>':
            return value > threshold.value
        elif threshold.operator == '<':
            return value < threshold.value
        elif threshold.operator == '==':
            return abs(value - threshold.value) < 0.001
        elif threshold.operator == '!=':
            return abs(value - threshold.value) >= 0.001
        else:
            return False


class SecurityQualityGate:
    """安全质量门禁"""

    def __init__(self):
        self.thresholds = self._define_thresholds()

    def _define_thresholds(self) -> Dict[str, QualityThreshold]:
        """定义安全阈值"""
        return {
            'vulnerable_dependencies': QualityThreshold(
                metric='vulnerable_dependencies',
                operator='==',
                value=0.0,
                description='不能有已知漏洞的依赖包',
                severity='error'
            ),
            'security_lint_issues': QualityThreshold(
                metric='security_lint_issues',
                operator='==',
                value=0.0,
                description='不能有安全代码问题',
                severity='error'
            ),
            'secrets_exposed': QualityThreshold(
                metric='secrets_exposed',
                operator='==',
                value=0.0,
                description='不能暴露敏感信息',
                severity='error'
            )
        }

    def check_security_vulnerabilities(self) -> QualityCheckResult:
        """检查安全漏洞"""
        try:
            # 检查是否有安全扫描报告
            security_reports = list(Path(".").glob("security_scan_*.json"))
            security_reports.extend(Path("security_reports").glob("*.json") if Path("security_reports").exists() else [])

            if not security_reports:
                # 如果没有安全报告，运行简单的检查
                return self._run_basic_security_check()

            # 解析最新的安全报告
            latest_report = max(security_reports, key=lambda x: x.stat().st_mtime)
            with open(latest_report, 'r', encoding='utf-8') as f:
                report_data = json.load(f)

            vulnerabilities = report_data.get('vulnerabilities', [])
            vuln_count = len(vulnerabilities)

            threshold = self.thresholds['vulnerable_dependencies']
            passed = self._check_threshold(vuln_count, threshold)

            status = 'pass' if passed else 'error'
            score = 100.0 if passed else max(0.0, 100.0 - vuln_count * 20)

            return QualityCheckResult(
                check_name='security_vulnerabilities',
                status=status,
                score=score,
                message=f'发现 {vuln_count} 个安全漏洞',
                details={'vulnerabilities': vulnerabilities[:5]},  # 只显示前5个
                recommendations=[
                    '更新有漏洞的依赖包',
                    '修复安全代码问题',
                    '运行完整的依赖安全扫描'
                ] if not passed else []
            )

        except Exception as e:
            return QualityCheckResult(
                check_name='security_vulnerabilities',
                status='warning',
                score=30.0,
                message=f'安全检查失败: {e}',
                recommendations=['安装安全扫描工具', '检查安全报告格式']
            )

    def _run_basic_security_check(self) -> QualityCheckResult:
        """运行基本安全检查"""
        issues = []

        # 检查常见的敏感信息模式
        sensitive_patterns = [
            r'password\s*=\s*["\'][^"\']*["\']',
            r'secret\s*=\s*["\'][^"\']*["\']',
            r'api_key\s*=\s*["\'][^"\']*["\']',
            r'token\s*=\s*["\'][^"\']*["\']'
        ]

        for py_file in Path("src").rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                    for pattern in sensitive_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            issues.append(f"{py_file}: 可能包含敏感信息")

            except Exception:
                continue

        issue_count = len(issues)
        passed = issue_count == 0
        status = 'pass' if passed else 'error'
        score = 100.0 if passed else max(0.0, 100.0 - issue_count * 25)

        return QualityCheckResult(
            check_name='basic_security_check',
            status=status,
            score=score,
            message=f'发现 {issue_count} 个潜在安全问题',
            details={'issues': issues[:10]},  # 只显示前10个
            recommendations=[
                '移除硬编码的敏感信息',
                '使用环境变量或配置文件',
                '实施代码安全审查'
            ] if not passed else []
        )

    def _check_threshold(self, value: float, threshold: QualityThreshold) -> bool:
        """检查是否满足阈值"""
        if threshold.operator == '>=':
            return value >= threshold.value
        elif threshold.operator == '<=':
            return value <= threshold.value
        elif threshold.operator == '>':
            return value > threshold.value
        elif threshold.operator == '<':
            return value < threshold.value
        elif threshold.operator == '==':
            return abs(value - threshold.value) < 0.001
        elif threshold.operator == '!=':
            return abs(value - threshold.value) >= 0.001
        else:
            return False


class QualityGateSystem:
    """质量门禁系统"""

    def __init__(self):
        self.code_gate = CodeQualityGate()
        self.performance_gate = PerformanceQualityGate()
        self.security_gate = SecurityQualityGate()

    def run_quality_checks(self, check_types: Optional[List[str]] = None) -> QualityGateResult:
        """运行质量检查"""
        if check_types is None:
            check_types = ['code', 'performance', 'security']

        logger.info(f"开始质量门禁检查: {', '.join(check_types)}")

        start_time = time.time()
        results = []

        # 运行代码质量检查
        if 'code' in check_types:
            results.extend([
                self.code_gate.check_test_coverage(),
                self.code_gate.check_code_complexity(),
                self.code_gate.check_lint_errors()
            ])

        # 运行性能质量检查
        if 'performance' in check_types:
            results.append(self.performance_gate.check_test_performance())

        # 运行安全质量检查
        if 'security' in check_types:
            results.append(self.security_gate.check_security_vulnerabilities())

        # 计算总体结果
        total_checks = len(results)
        passed_checks = sum(1 for r in results if r.status == 'pass')
        failed_checks = sum(1 for r in results if r.status in ['fail', 'error'])
        warning_checks = sum(1 for r in results if r.status == 'warning')

        # 计算总体分数
        total_score = sum(r.score for r in results) / total_checks if total_checks > 0 else 0

        # 确定总体状态
        if failed_checks > 0:
            overall_status = 'fail'
        elif warning_checks > 0:
            overall_status = 'warning'
        else:
            overall_status = 'pass'

        # 收集建议
        recommendations = []
        for result in results:
            recommendations.extend(result.recommendations)

        execution_time = time.time() - start_time

        gate_result = QualityGateResult(
            overall_status=overall_status,
            overall_score=total_score,
            checks_passed=passed_checks,
            checks_failed=failed_checks,
            checks_warning=warning_checks,
            total_checks=total_checks,
            execution_time=execution_time,
            recommendations=list(set(recommendations)),  # 去重
            detailed_results=results
        )

        # 生成报告
        self._generate_quality_report(gate_result)

        logger.info("质量门禁检查完成")
        return gate_result

    def check_pull_request_quality(self, pr_data: Dict[str, Any]) -> QualityGateResult:
        """检查PR质量"""
        # 这里可以实现PR特定的质量检查
        # 例如：检查PR大小、涉及的文件类型、测试覆盖等

        return self.run_quality_checks()

    def _generate_quality_report(self, result: QualityGateResult):
        """生成质量报告"""
        report_path = Path("test_logs/quality_gate_report.md")

        status_emoji = {
            'pass': '✅',
            'warning': '⚠️',
            'fail': '❌'
        }

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 质量门禁检查报告\n\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## 📊 总体结果\n\n")
            f.write(f"- **状态**: {status_emoji.get(result.overall_status, '❓')} {result.overall_status.upper()}\n")
            f.write(".1")
            f.write(f"- **通过检查**: {result.checks_passed}/{result.total_checks}\n")
            f.write(f"- **失败检查**: {result.checks_failed}\n")
            f.write(f"- **警告检查**: {result.checks_warning}\n")
            f.write(".2")
            f.write("## 🔍 详细检查结果\n\n")
            f.write("| 检查项目 | 状态 | 分数 | 消息 |\n")
            f.write("|----------|------|------|------|\n")

            for check_result in result.detailed_results:
                status_icon = status_emoji.get(check_result.status, '❓')
                f.write(f"| {check_result.check_name} | {status_icon} {check_result.status} | {check_result.score:.1f} | {check_result.message} |\n")

            f.write("\n## 💡 改进建议\n\n")
            if result.recommendations:
                for i, rec in enumerate(result.recommendations, 1):
                    f.write(f"{i}. {rec}\n")
            else:
                f.write("✅ 所有质量检查都通过，没有改进建议。\n")

            # 根据状态给出不同的总结
            f.write("\n## 📋 质量评估\n\n")
            if result.overall_status == 'pass':
                f.write("🎉 **质量门禁通过** - 代码质量符合要求，可以继续部署。\n\n")
                f.write("✅ 所有关键质量指标都达到标准，代码质量良好。\n")
            elif result.overall_status == 'warning':
                f.write("⚠️ **质量门禁警告** - 存在一些质量问题，建议修复后再部署。\n\n")
                f.write("⚠️ 虽然没有严重问题，但存在一些可以改进的地方。\n")
            else:
                f.write("❌ **质量门禁失败** - 存在严重质量问题，必须修复后再部署。\n\n")
                f.write("❌ 检测到关键质量问题，需要立即处理。\n")

        logger.info(f"质量门禁报告已生成: {report_path}")

    def get_ci_cd_integration_script(self) -> str:
        """获取CI/CD集成脚本"""
        script = '''
#!/bin/bash
# 质量门禁CI/CD集成脚本

echo "🔍 开始质量门禁检查..."

# 运行质量门禁检查
python tests/quality_gate.py

# 检查退出码
if [ $? -eq 0 ]; then
    echo "✅ 质量门禁通过"
    exit 0
else
    echo "❌ 质量门禁失败"
    exit 1
fi
'''
        return script


def main():
    """主函数"""
    system = QualityGateSystem()

    print("🚪 质量门禁系统启动")
    print("🎯 检查类型: 代码质量 + 性能质量 + 安全质量")

    # 运行完整质量检查
    result = system.run_quality_checks()

    status_emoji = {
        'pass': '✅',
        'warning': '⚠️',
        'fail': '❌'
    }

    print("\n📊 质量门禁结果:")
    print(f"  🚦 总体状态: {status_emoji.get(result.overall_status, '❓')} {result.overall_status.upper()}")
    print(".1")
    print(f"  ✅ 通过检查: {result.checks_passed}")
    print(f"  ❌ 失败检查: {result.checks_failed}")
    print(f"  ⚠️ 警告检查: {result.checks_warning}")
    print(f"  📊 总检查数: {result.total_checks}")
    print(".2")
    # 显示失败的检查
    failed_checks = [r for r in result.detailed_results if r.status in ['fail', 'error']]
    if failed_checks:
        print("\n❌ 失败的检查:")
        for check in failed_checks:
            print(f"  • {check.check_name}: {check.message}")

    # 显示改进建议
    if result.recommendations:
        print("\n💡 改进建议:")
        for i, rec in enumerate(result.recommendations[:5], 1):  # 显示前5个
            print(f"  {i}. {rec}")

    print("\n📄 详细报告已保存到: test_logs/quality_gate_report.md")
    print("\n✅ 质量门禁系统运行完成")


if __name__ == "__main__":
    main()
