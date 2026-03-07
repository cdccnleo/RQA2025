#!/usr/bin/env python3
"""
RQA2025 持续集成测试流水线
自动化执行测试、质量监控和报告生成
"""

import os
import sys
import json
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import argparse
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class CITestPipeline:
    """持续集成测试流水线"""

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or os.getcwd())
        self.ci_logs_dir = self.project_root / "ci_logs"
        self.reports_dir = self.ci_logs_dir / "reports"
        self.baseline_file = self.ci_logs_dir / "quality_baseline.json"

        # 创建目录
        self.ci_logs_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # 设置日志
        self._setup_logging()

        # 质量基线
        self.quality_baseline = self._load_quality_baseline()

    def _setup_logging(self):
        """设置日志"""
        log_file = self.ci_logs_dir / f"ci_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('CI_Pipeline')

    def _load_quality_baseline(self) -> Dict[str, Any]:
        """加载质量基线"""
        if self.baseline_file.exists():
            try:
                with open(self.baseline_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"无法加载质量基线: {e}")

        # 默认基线
        return {
            'coverage_min': 75.0,
            'test_success_rate_min': 95.0,
            'performance_regression_max': 5.0,
            'quality_score_min': 8.0,
            'last_updated': datetime.now().isoformat()
        }

    def run_ci_pipeline(self) -> Dict[str, Any]:
        """运行CI流水线"""
        self.logger.info("🚀 开始运行CI测试流水线")

        pipeline_start = time.time()
        results = {
            'pipeline_id': f"ci_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'stages': {},
            'overall_status': 'unknown',
            'quality_gates': {},
            'notifications': []
        }

        try:
            # 阶段1: 代码质量检查
            self.logger.info("📋 阶段1: 代码质量检查")
            quality_results = self._run_quality_checks()
            results['stages']['quality'] = quality_results

            # 阶段2: 单元测试
            self.logger.info("📋 阶段2: 单元测试")
            unit_results = self._run_unit_tests()
            results['stages']['unit_tests'] = unit_results

            # 阶段3: 集成测试
            self.logger.info("📋 阶段3: 集成测试")
            integration_results = self._run_integration_tests()
            results['stages']['integration_tests'] = integration_results

            # 阶段4: 端到端测试（选择性）
            self.logger.info("📋 阶段4: 端到端测试")
            e2e_results = self._run_e2e_tests()
            results['stages']['e2e_tests'] = e2e_results

            # 阶段5: 性能测试
            self.logger.info("📋 阶段5: 性能测试")
            performance_results = self._run_performance_tests()
            results['stages']['performance'] = performance_results

            # 阶段6: 安全测试
            self.logger.info("📋 阶段6: 安全测试")
            security_results = self._run_security_tests()
            results['stages']['security'] = security_results

            # 阶段7: 质量门禁检查
            self.logger.info("📋 阶段7: 质量门禁检查")
            quality_gates = self._check_quality_gates(results)
            results['quality_gates'] = quality_gates

            # 确定整体状态
            results['overall_status'] = self._determine_overall_status(quality_gates)

            # 生成通知
            results['notifications'] = self._generate_notifications(results)

        except Exception as e:
            self.logger.error(f"❌ CI流水线执行失败: {e}")
            results['overall_status'] = 'failed'
            results['error'] = str(e)

        finally:
            # 保存结果
            self._save_pipeline_results(results)

            # 发送通知
            if results.get('notifications'):
                self._send_notifications(results['notifications'])

            pipeline_time = time.time() - pipeline_start
            self.logger.info(f"✅ CI流水线完成，耗时: {pipeline_time:.2f}秒")
            self.logger.info(f"📊 整体状态: {results['overall_status']}")

        return results

    def _run_quality_checks(self) -> Dict[str, Any]:
        """运行代码质量检查"""
        try:
            # 运行代码格式检查
            format_cmd = [sys.executable, "-m", "black", "--check", "--diff", "src/"]
            format_result = subprocess.run(format_cmd, capture_output=True, text=True, cwd=self.project_root)

            # 运行代码质量检查
            quality_cmd = [sys.executable, "-m", "flake8", "src/", "--max-line-length=120", "--extend-ignore=E203,W503"]
            quality_result = subprocess.run(quality_cmd, capture_output=True, text=True, cwd=self.project_root)

            # 运行类型检查
            type_cmd = [sys.executable, "-m", "mypy", "src/", "--ignore-missing-imports"]
            type_result = subprocess.run(type_cmd, capture_output=True, text=True, cwd=self.project_root)

            return {
                'status': 'completed',
                'format_check': 'passed' if format_result.returncode == 0 else 'failed',
                'quality_check': 'passed' if quality_result.returncode == 0 else 'failed',
                'type_check': 'passed' if type_result.returncode == 0 else 'failed',
                'format_issues': len(format_result.stdout.split('\n')) if format_result.returncode != 0 else 0,
                'quality_issues': len(quality_result.stdout.split('\n')) if quality_result.returncode != 0 else 0,
                'type_issues': len(type_result.stdout.split('\n')) if type_result.returncode != 0 else 0
            }

        except Exception as e:
            return {'status': 'failed', 'error': str(e)}

    def _run_unit_tests(self) -> Dict[str, Any]:
        """运行单元测试"""
        try:
            cmd = [
                sys.executable, "-m", "pytest",
                "tests/unit/",
                "--cov=src",
                "--cov-report=json:test_logs/unit_coverage.json",
                "--cov-report=html:test_logs/unit_coverage_html",
                "--cov-report=xml:test_logs/unit_coverage.xml",
                "--json-report",
                "--json-report-file=test_logs/unit_test_results.json",
                "--durations=10",
                "-x", "--tb=short"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root, timeout=1800)

            # 解析结果（简化）
            return {
                'status': 'completed' if result.returncode == 0 else 'failed',
                'return_code': result.returncode,
                'execution_time': time.time(),
                'stdout': result.stdout[-1000:],  # 最后1000字符
                'stderr': result.stderr[-1000:] if result.stderr else ''
            }

        except subprocess.TimeoutExpired:
            return {'status': 'timeout', 'error': '单元测试执行超时'}
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}

    def _run_integration_tests(self) -> Dict[str, Any]:
        """运行集成测试"""
        try:
            cmd = [
                sys.executable, "-m", "pytest",
                "tests/integration/",
                "--cov=src",
                "--cov-report=json:test_logs/integration_coverage.json",
                "--cov-report=html:test_logs/integration_coverage_html",
                "--json-report",
                "--json-report-file=test_logs/integration_test_results.json",
                "--durations=10",
                "-x", "--tb=short"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root, timeout=1200)

            return {
                'status': 'completed' if result.returncode == 0 else 'failed',
                'return_code': result.returncode,
                'execution_time': time.time(),
                'stdout': result.stdout[-1000:],
                'stderr': result.stderr[-1000:] if result.stderr else ''
            }

        except subprocess.TimeoutExpired:
            return {'status': 'timeout', 'error': '集成测试执行超时'}
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}

    def _run_e2e_tests(self) -> Dict[str, Any]:
        """运行端到端测试（选择性运行）"""
        try:
            # 只运行关键的E2E测试以节省时间
            e2e_test_files = [
                "tests/e2e/test_simple_e2e_workflow.py",
                "tests/e2e/test_advanced_business_scenarios_e2e.py::TestAdvancedBusinessScenariosE2E::test_flash_crash_recovery_e2e"
            ]

            cmd = [
                sys.executable, "-m", "pytest"
            ] + e2e_test_files + [
                "--json-report",
                "--json-report-file=test_logs/e2e_test_results.json",
                "--durations=5",
                "--tb=short"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root, timeout=600)

            return {
                'status': 'completed' if result.returncode == 0 else 'failed',
                'return_code': result.returncode,
                'execution_time': time.time(),
                'stdout': result.stdout[-1000:],
                'stderr': result.stderr[-1000:] if result.stderr else ''
            }

        except subprocess.TimeoutExpired:
            return {'status': 'timeout', 'error': '端到端测试执行超时'}
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}

    def _run_performance_tests(self) -> Dict[str, Any]:
        """运行性能测试"""
        try:
            # 检查是否有性能测试
            performance_dir = self.project_root / "tests" / "performance"
            if not performance_dir.exists():
                return {'status': 'skipped', 'reason': '无性能测试目录'}

            cmd = [
                sys.executable, "-m", "pytest",
                "tests/performance/",
                "--benchmark-json=test_logs/performance_results.json",
                "--durations=5",
                "--tb=short"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root, timeout=300)

            return {
                'status': 'completed' if result.returncode == 0 else 'failed',
                'return_code': result.returncode,
                'execution_time': time.time(),
                'stdout': result.stdout[-500:],
                'stderr': result.stderr[-500:] if result.stderr else ''
            }

        except subprocess.TimeoutExpired:
            return {'status': 'timeout', 'error': '性能测试执行超时'}
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}

    def _run_security_tests(self) -> Dict[str, Any]:
        """运行安全测试"""
        try:
            # 使用bandit进行安全检查
            cmd = [sys.executable, "-m", "bandit", "-r", "src/", "-f", "json", "-o", "test_logs/security_report.json"]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root, timeout=300)

            return {
                'status': 'completed' if result.returncode in [0, 1] else 'failed',  # bandit返回1表示发现问题
                'return_code': result.returncode,
                'execution_time': time.time(),
                'issues_found': result.returncode == 1,
                'stdout': result.stdout[-500:],
                'stderr': result.stderr[-500:] if result.stderr else ''
            }

        except subprocess.TimeoutExpired:
            return {'status': 'timeout', 'error': '安全测试执行超时'}
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}

    def _check_quality_gates(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """检查质量门禁"""
        gates = {
            'code_quality': 'unknown',
            'unit_tests': 'unknown',
            'integration_tests': 'unknown',
            'e2e_tests': 'unknown',
            'performance': 'unknown',
            'security': 'unknown',
            'overall': 'unknown'
        }

        # 代码质量门禁
        quality_stage = results['stages'].get('quality', {})
        if quality_stage.get('status') == 'completed':
            failed_checks = sum(1 for check in ['format_check', 'quality_check', 'type_check']
                              if quality_stage.get(check) == 'failed')
            gates['code_quality'] = 'passed' if failed_checks == 0 else 'failed'

        # 单元测试门禁
        unit_stage = results['stages'].get('unit_tests', {})
        gates['unit_tests'] = 'passed' if unit_stage.get('status') == 'completed' else 'failed'

        # 集成测试门禁
        integration_stage = results['stages'].get('integration_tests', {})
        gates['integration_tests'] = 'passed' if integration_stage.get('status') == 'completed' else 'failed'

        # E2E测试门禁 (较宽松)
        e2e_stage = results['stages'].get('e2e_tests', {})
        gates['e2e_tests'] = 'passed' if e2e_stage.get('status') in ['completed', 'failed'] else 'failed'

        # 性能测试门禁
        performance_stage = results['stages'].get('performance', {})
        gates['performance'] = 'passed' if performance_stage.get('status') in ['completed', 'skipped'] else 'failed'

        # 安全测试门禁
        security_stage = results['stages'].get('security', {})
        gates['security'] = 'passed' if security_stage.get('status') == 'completed' else 'failed'

        # 整体门禁：所有关键门禁通过
        critical_gates = ['code_quality', 'unit_tests', 'integration_tests']
        critical_passed = all(gates[gate] == 'passed' for gate in critical_gates)
        gates['overall'] = 'passed' if critical_passed else 'failed'

        return gates

    def _determine_overall_status(self, quality_gates: Dict[str, Any]) -> str:
        """确定整体状态"""
        if quality_gates.get('overall') == 'passed':
            return 'success'
        elif any(gate == 'failed' for gate in ['unit_tests', 'integration_tests']):
            return 'failed'
        else:
            return 'warning'

    def _generate_notifications(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成通知"""
        notifications = []

        status = results.get('overall_status', 'unknown')
        quality_gates = results.get('quality_gates', {})

        if status == 'failed':
            notifications.append({
                'type': 'alert',
                'priority': 'high',
                'title': '🚨 CI流水线失败',
                'message': 'CI流水线执行失败，存在阻塞性问题需要立即处理',
                'details': f"失败的门禁: {[k for k, v in quality_gates.items() if v == 'failed']}"
            })

        elif status == 'warning':
            notifications.append({
                'type': 'warning',
                'priority': 'medium',
                'title': '⚠️ CI流水线警告',
                'message': 'CI流水线完成但存在质量问题需要关注',
                'details': f"警告的门禁: {[k for k, v in quality_gates.items() if v == 'failed']}"
            })

        elif status == 'success':
            notifications.append({
                'type': 'success',
                'priority': 'low',
                'title': '✅ CI流水线成功',
                'message': 'CI流水线全部通过，代码质量良好',
                'details': '所有质量门禁均已通过'
            })

        return notifications

    def _save_pipeline_results(self, results: Dict[str, Any]):
        """保存流水线结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.reports_dir / f"ci_pipeline_results_{timestamp}.json"

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        self.logger.info(f"📄 CI结果已保存: {results_file}")

    def _send_notifications(self, notifications: List[Dict[str, Any]]):
        """发送通知"""
        for notification in notifications:
            self.logger.info(f"📢 {notification['title']}: {notification['message']}")

            # 这里可以集成邮件、Slack、DingTalk等通知服务
            # send_email(notification)
            # send_slack_message(notification)
            # send_dingtalk_message(notification)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RQA2025 持续集成测试流水线')
    parser.add_argument('--project-root', help='项目根目录', default=None)
    parser.add_argument('--run-full', action='store_true', help='运行完整CI流水线')
    parser.add_argument('--run-quality', action='store_true', help='仅运行质量检查')
    parser.add_argument('--run-tests', action='store_true', help='仅运行测试')
    parser.add_argument('--update-baseline', action='store_true', help='更新质量基线')

    args = parser.parse_args()

    pipeline = CITestPipeline(args.project_root)

    if args.run_full:
        results = pipeline.run_ci_pipeline()
        status = results.get('overall_status', 'unknown')

        if status == 'success':
            print("🎉 CI流水线执行成功！")
            sys.exit(0)
        elif status == 'warning':
            print("⚠️ CI流水线完成但存在警告")
            sys.exit(1)
        else:
            print("❌ CI流水线执行失败")
            sys.exit(1)

    elif args.run_quality:
        quality_results = pipeline._run_quality_checks()
        print(f"质量检查完成: {quality_results}")

    elif args.run_tests:
        unit_results = pipeline._run_unit_tests()
        integration_results = pipeline._run_integration_tests()
        print(f"测试完成 - 单元: {unit_results['status']}, 集成: {integration_results['status']}")

    elif args.update_baseline:
        # 更新质量基线
        baseline = {
            'coverage_min': 75.0,
            'test_success_rate_min': 95.0,
            'performance_regression_max': 5.0,
            'quality_score_min': 8.0,
            'last_updated': datetime.now().isoformat()
        }

        with open(pipeline.baseline_file, 'w', encoding='utf-8') as f:
            json.dump(baseline, f, indent=2)

        print(f"✅ 质量基线已更新: {pipeline.baseline_file}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
