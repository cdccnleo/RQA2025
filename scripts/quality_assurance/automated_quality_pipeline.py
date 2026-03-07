#!/usr/bin/env python3
"""
RQA2025 自动化质量流水线

自动执行完整的质量检查流水线，包括：
- 代码文档一致性检查
- 文档自动同步
- 版本一致性检查
- 质量门禁评估
- 报告生成和通知
"""

import subprocess
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AutomatedQualityPipeline:
    """自动化质量流水线"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.reports_dir = self.project_root / "reports"
        self.scripts_dir = self.project_root / "scripts"

        # 确保报告目录存在
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def run_pipeline(self) -> Dict[str, Any]:
        """运行完整质量流水线"""

        logger.info("🚀 开始执行RQA2025自动化质量流水线")

        pipeline_results = {
            'timestamp': datetime.now().isoformat(),
            'pipeline_status': 'running',
            'stages': {},
            'summary': {},
            'recommendations': []
        }

        try:
            # 阶段1: 代码文档一致性检查
            logger.info("📋 阶段1: 执行代码文档一致性检查")
            consistency_result = self._run_consistency_check()
            pipeline_results['stages']['consistency_check'] = consistency_result

            # 阶段2: 文档自动同步
            logger.info("📝 阶段2: 执行文档自动同步")
            doc_sync_result = self._run_documentation_sync()
            pipeline_results['stages']['documentation_sync'] = doc_sync_result

            # 阶段3: 版本一致性检查
            logger.info("🔢 阶段3: 执行版本一致性检查")
            version_result = self._run_version_check()
            pipeline_results['stages']['version_check'] = version_result

            # 阶段4: 生成质量门禁报告
            logger.info("🚪 阶段4: 生成质量门禁报告")
            quality_gate_result = self._generate_quality_report()
            pipeline_results['stages']['quality_gate'] = quality_gate_result

            # 计算综合结果
            pipeline_results['summary'] = self._calculate_summary(pipeline_results['stages'])
            pipeline_results['recommendations'] = self._generate_recommendations(
                pipeline_results['stages'])

            pipeline_results['pipeline_status'] = 'completed'

            logger.info("✅ 质量流水线执行完成")

        except Exception as e:
            logger.error(f"❌ 质量流水线执行失败: {e}")
            pipeline_results['pipeline_status'] = 'failed'
            pipeline_results['error'] = str(e)

        # 保存流水线结果
        self._save_pipeline_report(pipeline_results)

        return pipeline_results

    def _run_consistency_check(self) -> Dict[str, Any]:
        """运行一致性检查"""
        try:
            cmd = [
                sys.executable,
                str(self.scripts_dir / "quality_assurance" / "consistency_checker.py"),
                "--project-root", str(self.project_root),
                "--output-format", "json",
                "--output-file", str(self.reports_dir / "consistency_report.json"),
                "--threshold", "95"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            return {
                'success': result.returncode == 0,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'report_file': str(self.reports_dir / "consistency_report.json")
            }

        except subprocess.TimeoutExpired:
            logger.error("一致性检查超时")
            return {'success': False, 'error': 'timeout'}
        except Exception as e:
            logger.error(f"一致性检查执行失败: {e}")
            return {'success': False, 'error': str(e)}

    def _run_documentation_sync(self) -> Dict[str, Any]:
        """运行文档同步"""
        try:
            cmd = [
                sys.executable,
                str(self.scripts_dir / "documentation_automation" / "doc_sync.py"),
                "--project-root", str(self.project_root),
                "--generate-report",
                "--output", str(self.reports_dir / "doc_sync_report.json")
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            return {
                'success': result.returncode == 0,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'report_file': str(self.reports_dir / "doc_sync_report.json")
            }

        except subprocess.TimeoutExpired:
            logger.error("文档同步超时")
            return {'success': False, 'error': 'timeout'}
        except Exception as e:
            logger.error(f"文档同步执行失败: {e}")
            return {'success': False, 'error': str(e)}

    def _run_version_check(self) -> Dict[str, Any]:
        """运行版本检查"""
        try:
            cmd = [
                sys.executable,
                str(self.scripts_dir / "version_management" / "version_manager.py"),
                "--project-root", str(self.project_root),
                "--check-consistency",
                "--output", str(self.reports_dir / "version_report.json")
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            return {
                'success': result.returncode == 0,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'report_file': str(self.reports_dir / "version_report.json")
            }

        except subprocess.TimeoutExpired:
            logger.error("版本检查超时")
            return {'success': False, 'error': 'timeout'}
        except Exception as e:
            logger.error(f"版本检查执行失败: {e}")
            return {'success': False, 'error': str(e)}

    def _generate_quality_report(self) -> Dict[str, Any]:
        """生成质量门禁报告"""
        try:
            cmd = [
                sys.executable,
                str(self.scripts_dir / "quality_assurance" / "generate_quality_report.py"),
                "--consistency-report", str(self.reports_dir / "consistency_report.json"),
                "--doc-sync-report", str(self.reports_dir / "doc_sync_report.json"),
                "--version-report", str(self.reports_dir / "version_report.json"),
                "--output", str(self.reports_dir / "quality_gate_report.json")
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            return {
                'success': result.returncode == 0,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'report_file': str(self.reports_dir / "quality_gate_report.json")
            }

        except subprocess.TimeoutExpired:
            logger.error("质量报告生成超时")
            return {'success': False, 'error': 'timeout'}
        except Exception as e:
            logger.error(f"质量报告生成失败: {e}")
            return {'success': False, 'error': str(e)}

    def _calculate_summary(self, stages: Dict[str, Any]) -> Dict[str, Any]:
        """计算流水线摘要"""

        summary = {
            'total_stages': len(stages),
            'successful_stages': sum(1 for s in stages.values() if s.get('success', False)),
            'failed_stages': sum(1 for s in stages.values() if not s.get('success', False)),
            'overall_success': all(s.get('success', False) for s in stages.values())
        }

        # 尝试从质量门禁报告中获取更详细的指标
        quality_gate_file = self.reports_dir / "quality_gate_report.json"
        if quality_gate_file.exists():
            try:
                with open(quality_gate_file, 'r', encoding='utf-8') as f:
                    quality_report = json.load(f)
                summary.update(quality_report.get('summary', {}))
            except Exception as e:
                logger.warning(f"无法读取质量报告: {e}")

        return summary

    def _generate_recommendations(self, stages: Dict[str, Any]) -> List[str]:
        """生成改进建议"""

        recommendations = []

        # 检查各个阶段的失败情况
        failed_stages = [name for name,
                         result in stages.items() if not result.get('success', False)]

        if failed_stages:
            recommendations.append(f"❌ 发现失败阶段: {', '.join(failed_stages)}")

        for stage_name, stage_result in stages.items():
            if not stage_result.get('success', False):
                if stage_name == 'consistency_check':
                    recommendations.append("🔧 修复一致性检查：检查consistency_checker.py脚本和配置")
                elif stage_name == 'documentation_sync':
                    recommendations.append("🔧 修复文档同步：检查doc_sync.py脚本和模板配置")
                elif stage_name == 'version_check':
                    recommendations.append("🔧 修复版本检查：检查version_manager.py脚本和版本配置")
                elif stage_name == 'quality_gate':
                    recommendations.append("🔧 修复质量门禁：检查generate_quality_report.py脚本")

        # 检查质量门禁结果
        quality_gate_file = self.reports_dir / "quality_gate_report.json"
        if quality_gate_file.exists():
            try:
                with open(quality_gate_file, 'r', encoding='utf-8') as f:
                    quality_report = json.load(f)

                failed_gates = quality_report.get('failed_gates', [])
                for gate in failed_gates:
                    recommendations.append(f"🚨 {gate.get('message', '质量门禁失败')}")

                # 添加质量报告中的建议
                report_recommendations = quality_report.get('recommendations', [])
                recommendations.extend(report_recommendations)

            except Exception as e:
                logger.warning(f"无法读取质量门禁报告: {e}")

        return recommendations

    def _save_pipeline_report(self, pipeline_results: Dict[str, Any]):
        """保存流水线报告"""

        report_file = self.reports_dir / "automated_quality_pipeline_report.json"
        summary_file = self.reports_dir / "pipeline_summary.txt"

        # 保存详细JSON报告
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(pipeline_results, f, indent=2, ensure_ascii=False)

        # 生成文本摘要
        summary_lines = [
            "=" * 50,
            "RQA2025 自动化质量流水线执行报告",
            "=" * 50,
            f"执行时间: {pipeline_results['timestamp']}",
            f"流水线状态: {pipeline_results['pipeline_status']}",
            ""
        ]

        summary = pipeline_results.get('summary', {})
        if summary:
            summary_lines.extend([
                "=== 执行摘要 ===",
                f"总阶段数: {summary.get('total_stages', 0)}",
                f"成功阶段: {summary.get('successful_stages', 0)}",
                f"失败阶段: {summary.get('failed_stages', 0)}",
                f"整体成功: {'✅' if summary.get('overall_success', False) else '❌'}",
                ""
            ])

            # 添加关键指标
            if 'consistency_score' in summary:
                summary_lines.append(f"一致性评分: {summary['consistency_score']}%")
            if 'sync_success_rate' in summary:
                summary_lines.append(f"同步成功率: {summary['sync_success_rate']}%")
            if 'version_consistency' in summary:
                summary_lines.append(f"版本一致性: {'✅' if summary['version_consistency'] else '❌'}")
            if 'passed_gates' in summary and 'total_gates' in summary:
                summary_lines.append(f"质量门禁: {summary['passed_gates']}/{summary['total_gates']} 通过")

        recommendations = pipeline_results.get('recommendations', [])
        if recommendations:
            summary_lines.extend([
                "",
                "=== 改进建议 ==="
            ])
            summary_lines.extend(f"• {rec}" for rec in recommendations)

        summary_lines.extend([
            "",
            f"详细报告: {report_file}",
            "=" * 50
        ])

        # 保存文本摘要
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))

        logger.info(f"流水线报告已保存: {report_file}")
        logger.info(f"摘要报告已保存: {summary_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RQA2025 自动化质量流水线')
    parser.add_argument('--project-root', default='.',
                        help='项目根目录路径')
    parser.add_argument('--output-dir', default='reports',
                        help='输出目录')

    args = parser.parse_args()

    pipeline = AutomatedQualityPipeline(args.project_root)
    results = pipeline.run_pipeline()

    # 输出执行结果
    summary = results.get('summary', {})
    print("=== 流水线执行结果 ===")
    print(f"状态: {results['pipeline_status']}")
    print(f"成功阶段: {summary.get('successful_stages', 0)}/{summary.get('total_stages', 0)}")

    if summary.get('consistency_score'):
        print(f"一致性评分: {summary['consistency_score']}%")
    if summary.get('passed_gates') is not None:
        print(f"质量门禁: {summary['passed_gates']}/{summary['total_gates']} 通过")

    if results['pipeline_status'] == 'completed' and summary.get('overall_success'):
        print("🎉 所有质量检查通过！")
        sys.exit(0)
    else:
        print("⚠️  发现质量问题，请查看详细报告")
        sys.exit(1)


if __name__ == "__main__":
    main()
