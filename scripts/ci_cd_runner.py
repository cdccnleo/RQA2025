#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 CI/CD 运行器
本地CI/CD流水线执行工具

作者: AI Assistant
创建日期: 2025年9月13日
"""

import os
import sys
import subprocess
import time
from typing import Dict, List, Any
import json
import argparse
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CICDRunner:
    """CI/CD运行器"""

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or os.getcwd())
        self.results = {}
        self.start_time = None

    def run_pipeline(self, stages: List[str] = None) -> Dict[str, Any]:
        """
        运行CI/CD流水线

        Args:
            stages: 要运行的阶段列表

        Returns:
            流水线执行结果
        """
        if stages is None:
            stages = ['lint', 'test', 'build', 'security', 'deploy']

        self.start_time = time.time()
        logger.info("🚀 开始RQA2025 CI/CD流水线")

        results = {}

        for stage in stages:
            logger.info(f"📋 执行阶段: {stage}")
            try:
                if stage == 'lint':
                    results[stage] = self.run_lint()
                elif stage == 'test':
                    results[stage] = self.run_tests()
                elif stage == 'build':
                    results[stage] = self.run_build()
                elif stage == 'security':
                    results[stage] = self.run_security_scan()
                elif stage == 'deploy':
                    results[stage] = self.run_deploy()
                else:
                    logger.warning(f"未知阶段: {stage}")
                    continue

                if results[stage]['success']:
                    logger.info(f"✅ 阶段 {stage} 执行成功")
                else:
                    logger.error(f"❌ 阶段 {stage} 执行失败")
                    break

            except Exception as e:
                logger.error(f"阶段 {stage} 执行异常: {e}")
                results[stage] = {'success': False, 'error': str(e)}
                break

        # 计算总时间
        total_time = time.time() - self.start_time
        overall_success = all(r.get('success', False) for r in results.values())

        final_result = {
            'success': overall_success,
            'total_time': total_time,
            'stages': results,
            'timestamp': time.time()
        }

        logger.info(f"🏁 CI/CD流水线执行完成，总耗时: {total_time:.2f}秒")
        return final_result

    def run_lint(self) -> Dict[str, Any]:
        """运行代码质量检查"""
        logger.info("🔍 执行代码质量检查")

        try:
            # 检查flake8
            result = subprocess.run([
                sys.executable, '-m', 'flake8',
                'src', 'tests', 'scripts',
                '--count', '--select=E9,F63,F7,F82',
                '--show-source', '--statistics'
            ], capture_output=True, text=True, cwd=self.project_root)

            if result.returncode != 0:
                return {
                    'success': False,
                    'error': '代码质量检查失败',
                    'details': result.stdout + result.stderr
                }

            # 检查black格式
            result = subprocess.run([
                sys.executable, '-m', 'black',
                '--check', '--diff',
                'src', 'tests', 'scripts'
            ], capture_output=True, text=True, cwd=self.project_root)

            if result.returncode != 0:
                return {
                    'success': False,
                    'error': '代码格式检查失败',
                    'details': result.stdout + result.stderr
                }

            # 检查isort导入排序
            result = subprocess.run([
                sys.executable, '-m', 'isort',
                '--check-only', '--diff',
                'src', 'tests', 'scripts'
            ], capture_output=True, text=True, cwd=self.project_root)

            if result.returncode != 0:
                return {
                    'success': False,
                    'error': '导入排序检查失败',
                    'details': result.stdout + result.stderr
                }

            return {
                'success': True,
                'message': '代码质量检查通过'
            }

        except FileNotFoundError as e:
            return {
                'success': False,
                'error': f'缺少必要的工具: {e}',
                'suggestion': '请安装: pip install flake8 black isort'
            }

    def run_tests(self) -> Dict[str, Any]:
        """运行测试"""
        logger.info("🧪 执行测试")

        try:
            # 创建测试结果目录
            test_results_dir = self.project_root / 'test-results'
            test_results_dir.mkdir(exist_ok=True)

            # 运行单元测试
            result = subprocess.run([
                sys.executable, '-m', 'pytest',
                'tests/unit/', 'tests/integration/',
                '-v', '--tb=short',
                '--cov=src', '--cov-report=term-missing',
                '--cov-report=xml', '--cov-report=html',
                f'--junitxml={test_results_dir}/test-results.xml'
            ], capture_output=True, text=True, cwd=self.project_root, timeout=300)

            # 解析覆盖率报告
            coverage_file = self.project_root / 'coverage.xml'
            coverage_percent = 0.0
            if coverage_file.exists():
                # 简单解析覆盖率
                try:
                    with open(coverage_file, 'r') as f:
                        content = f.read()
                        # 查找覆盖率百分比（简化处理）
                        import re
                        match = re.search(r'line-rate="([^"]*)"', content)
                        if match:
                            coverage_percent = float(match.group(1)) * 100
                except Exception as e:
                    logger.warning(f"解析覆盖率文件失败: {e}")

            test_result = {
                'success': result.returncode == 0,
                'return_code': result.returncode,
                'coverage': coverage_percent,
                'stdout': result.stdout[-1000:],  # 只保留最后1000字符
                'stderr': result.stderr[-1000:]
            }

            if result.returncode != 0:
                test_result['error'] = '测试执行失败'

            return test_result

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': '测试执行超时'
            }
        except FileNotFoundError:
            return {
                'success': False,
                'error': 'pytest未安装',
                'suggestion': '请安装: pip install pytest pytest-cov'
            }

    def run_build(self) -> Dict[str, Any]:
        """运行构建"""
        logger.info("🔨 执行构建")

        try:
            # 构建Python包
            result = subprocess.run([
                sys.executable, '-m', 'build'
            ], capture_output=True, text=True, cwd=self.project_root)

            if result.returncode != 0:
                return {
                    'success': False,
                    'error': '构建失败',
                    'details': result.stderr
                }

            # 检查构建产物
            dist_dir = self.project_root / 'dist'
            if not dist_dir.exists():
                return {
                    'success': False,
                    'error': '构建产物未生成'
                }

            build_files = list(dist_dir.glob('*'))
            if not build_files:
                return {
                    'success': False,
                    'error': '构建产物为空'
                }

            return {
                'success': True,
                'message': '构建成功',
                'build_files': [str(f.name) for f in build_files]
            }

        except FileNotFoundError:
            return {
                'success': False,
                'error': 'build工具未安装',
                'suggestion': '请安装: pip install build'
            }

    def run_security_scan(self) -> Dict[str, Any]:
        """运行安全扫描"""
        logger.info("🔒 执行安全扫描")

        try:
            # 使用bandit进行安全扫描
            result = subprocess.run([
                sys.executable, '-m', 'bandit',
                '-r', 'src',
                '-f', 'json',
                '-o', 'security-report.json'
            ], capture_output=True, text=True, cwd=self.project_root)

            security_report = {}
            if (self.project_root / 'security-report.json').exists():
                try:
                    with open(self.project_root / 'security-report.json', 'r') as f:
                        security_report = json.load(f)
                except Exception as e:
                    logger.warning(f"解析安全报告失败: {e}")

            # 分析安全问题
            high_severity = 0
            medium_severity = 0

            if 'results' in security_report:
                for result_item in security_report['results']:
                    for issue in result_item.get('issues', []):
                        severity = issue.get('issue_severity', '').lower()
                        if severity == 'high':
                            high_severity += 1
                        elif severity == 'medium':
                            medium_severity += 1

            security_result = {
                'success': high_severity == 0,  # 高危问题为0才算成功
                'high_severity_issues': high_severity,
                'medium_severity_issues': medium_severity,
                'total_issues': high_severity + medium_severity
            }

            if high_severity > 0:
                security_result['warning'] = f'发现{high_severity}个高危安全问题'

            return security_result

        except FileNotFoundError:
            return {
                'success': False,
                'error': 'bandit未安装',
                'suggestion': '请安装: pip install bandit'
            }

    def run_deploy(self) -> Dict[str, Any]:
        """运行部署"""
        logger.info("🚀 执行部署")

        try:
            # 这里是模拟部署，实际部署需要根据环境配置
            deploy_script = self.project_root / 'scripts' / 'deploy.py'

            if deploy_script.exists():
                result = subprocess.run([
                    sys.executable, str(deploy_script)
                ], capture_output=True, text=True, cwd=self.project_root)

                return {
                    'success': result.returncode == 0,
                    'message': '部署脚本执行完成',
                    'details': result.stdout + result.stderr
                }
            else:
                # 模拟部署
                logger.info("模拟部署到测试环境...")
                time.sleep(2)  # 模拟部署时间

                return {
                    'success': True,
                    'message': '模拟部署成功',
                    'environment': 'test'
                }

        except Exception as e:
            return {
                'success': False,
                'error': f'部署失败: {e}'
            }

    def generate_report(self, results: Dict[str, Any]) -> str:
        """生成报告"""
        report_lines = [
            "# RQA2025 CI/CD 执行报告",
            f"执行时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"总耗时: {results['total_time']:.2f}秒",
            f"整体状态: {'✅ 成功' if results['success'] else '❌ 失败'}",
            "",
            "## 各阶段结果",
            ""
        ]

        for stage, result in results['stages'].items():
            status = "✅" if result.get('success', False) else "❌"
            report_lines.append(f"### {stage} {status}")

            if 'message' in result:
                report_lines.append(f"- 消息: {result['message']}")

            if 'error' in result:
                report_lines.append(f"- 错误: {result['error']}")

            if 'coverage' in result:
                report_lines.append(f"- 覆盖率: {result['coverage']:.2f}%")

            if 'build_files' in result:
                report_lines.append(f"- 构建文件: {', '.join(result['build_files'])}")

            if 'high_severity_issues' in result:
                report_lines.append(f"- 高危安全问题: {result['high_severity_issues']}")

            report_lines.append("")

        return "\n".join(report_lines)

    def save_report(self, results: Dict[str, Any], filename: str = 'ci_cd_report.md'):
        """保存报告"""
        report = self.generate_report(results)

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"📄 CI/CD报告已保存到: {filename}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RQA2025 CI/CD 运行器')
    parser.add_argument('--stages', nargs='+',
                        choices=['lint', 'test', 'build', 'security', 'deploy'],
                        default=['lint', 'test', 'build', 'security', 'deploy'],
                        help='要执行的阶段')
    parser.add_argument('--project-root', default=None,
                        help='项目根目录')
    parser.add_argument('--output', default='ci_cd_report.md',
                        help='输出报告文件')

    args = parser.parse_args()

    runner = CICDRunner(args.project_root)
    results = runner.run_pipeline(args.stages)
    runner.save_report(results, args.output)

    # 输出最终结果
    if results['success']:
        print("✅ CI/CD流水线执行成功！")
        sys.exit(0)
    else:
        print("❌ CI/CD流水线执行失败！")
        sys.exit(1)


if __name__ == '__main__':
    main()
