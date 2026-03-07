#!/usr/bin/env python3
"""
RQA2025持续覆盖率监控系统

提供持续的测试覆盖率监控、趋势分析和自动告警功能
支持CI/CD集成和自动化质量门禁
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ContinuousCoverageMonitor:
    """持续覆盖率监控器"""

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or Path(__file__).parent.parent)
        self.reports_dir = self.project_root / "test_logs"
        self.history_file = self.reports_dir / "coverage_history.json"
        self.alerts_file = self.reports_dir / "coverage_alerts.json"

        # 覆盖率阈值配置
        self.thresholds = {
            "production": 70.0,    # 生产环境最低要求
            "warning": 60.0,       # 警告阈值
            "critical": 50.0,      # 严重阈值
            "target": 80.0         # 目标覆盖率
        }

        # 初始化历史数据
        self._load_history()

    def _load_history(self):
        """加载历史数据"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
            except Exception as e:
                logger.warning(f"加载历史数据失败: {e}")
                self.history = []
        else:
            self.history = []

    def _save_history(self):
        """保存历史数据"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存历史数据失败: {e}")

    def run_coverage_analysis(self) -> Dict[str, Any]:
        """
        运行覆盖率分析

        Returns:
            覆盖率分析结果
        """
        logger.info("开始覆盖率分析...")

        try:
            # 运行pytest覆盖率测试
            cmd = [
                sys.executable, "-m", "pytest",
                "tests/unit/",  # 只测试单元测试
                "--cov=src",
                "--cov-report=json:coverage.json",
                "--cov-report=term-missing",
                "--tb=no",
                "-q"
            ]

            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300
            )

            # 解析覆盖率结果
            coverage_data = self._parse_coverage_result(result)

            # 添加时间戳
            coverage_data['timestamp'] = datetime.now().isoformat()
            coverage_data['test_exit_code'] = result.returncode

            # 保存到历史记录
            self.history.append(coverage_data)
            if len(self.history) > 100:  # 保留最近100次记录
                self.history = self.history[-100:]

            self._save_history()

            logger.info(f"覆盖率分析完成: {coverage_data.get('total_coverage', 0):.1f}%")
            return coverage_data

        except Exception as e:
            logger.error(f"覆盖率分析失败: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'total_coverage': 0.0
            }

    def _parse_coverage_result(self, result: subprocess.CompletedProcess) -> Dict[str, Any]:
        """
        解析覆盖率测试结果

        Args:
            result: pytest运行结果

        Returns:
            解析后的覆盖率数据
        """
        coverage_data = {
            'total_coverage': 0.0,
            'layer_coverage': {},
            'test_results': {
                'passed': 0,
                'failed': 0,
                'skipped': 0,
                'errors': 0
            }
        }

        # 尝试读取coverage.json文件
        coverage_json = self.project_root / "coverage.json"
        if coverage_json.exists():
            try:
                with open(coverage_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 解析总覆盖率
                if 'totals' in data and 'percent_covered' in data['totals']:
                    coverage_data['total_coverage'] = data['totals']['percent_covered']

                # 解析各文件覆盖率
                if 'files' in data:
                    coverage_data['file_coverage'] = {}
                    for file_path, file_data in data['files'].items():
                        if 'summary' in file_data and 'percent_covered' in file_data['summary']:
                            coverage_data['file_coverage'][file_path] = file_data['summary']['percent_covered']

            except Exception as e:
                logger.warning(f"解析coverage.json失败: {e}")

        # 解析测试结果
        output_lines = result.stdout.split('\n')
        for line in output_lines:
            if 'passed' in line and 'failed' in line:
                # 尝试解析类似 "10 passed, 2 failed, 1 skipped" 的行
                parts = line.replace(',', '').split()
                for i, part in enumerate(parts):
                    if part.isdigit():
                        if i + 1 < len(parts):
                            next_word = parts[i + 1].lower()
                            if 'passed' in next_word:
                                coverage_data['test_results']['passed'] = int(part)
                            elif 'failed' in next_word:
                                coverage_data['test_results']['failed'] = int(part)
                            elif 'skipped' in next_word:
                                coverage_data['test_results']['skipped'] = int(part)
                            elif 'error' in next_word:
                                coverage_data['test_results']['errors'] = int(part)

        return coverage_data

    def analyze_trends(self) -> Dict[str, Any]:
        """
        分析覆盖率趋势

        Returns:
            趋势分析结果
        """
        if len(self.history) < 2:
            return {'trend': 'insufficient_data', 'message': '需要至少2次测量才能分析趋势'}

        # 计算趋势
        recent_measurements = self.history[-10:]  # 最近10次测量
        coverage_values = [h.get('total_coverage', 0) for h in recent_measurements if 'error' not in h]

        if len(coverage_values) < 2:
            return {'trend': 'no_valid_data', 'message': '没有足够的有效覆盖率数据'}

        # 计算趋势斜率
        n = len(coverage_values)
        x = list(range(n))
        y = coverage_values

        # 简单线性回归
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_xx = sum(xi * xi for xi in x)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x) if (n * sum_xx - sum_x * sum_x) != 0 else 0

        # 判断趋势
        if slope > 0.1:
            trend = 'improving'
            message = f'覆盖率呈上升趋势 (斜率: {slope:.3f})'
        elif slope < -0.1:
            trend = 'declining'
            message = f'覆盖率呈下降趋势 (斜率: {slope:.3f})'
        else:
            trend = 'stable'
            message = f'覆盖率保持稳定 (斜率: {slope:.3f})'

        return {
            'trend': trend,
            'slope': slope,
            'message': message,
            'current_coverage': coverage_values[-1],
            'average_coverage': sum(coverage_values) / len(coverage_values),
            'min_coverage': min(coverage_values),
            'max_coverage': max(coverage_values)
        }

    def check_quality_gates(self, coverage_data: Dict[str, Any]) -> List[str]:
        """
        检查质量门禁

        Args:
            coverage_data: 覆盖率数据

        Returns:
            质量门禁检查结果
        """
        issues = []
        total_coverage = coverage_data.get('total_coverage', 0)

        if total_coverage < self.thresholds['critical']:
            issues.append(f"🚨 严重问题: 覆盖率仅为{total_coverage:.1f}%，远低于最低要求{self.thresholds['critical']}%")
        elif total_coverage < self.thresholds['warning']:
            issues.append(f"⚠️ 警告: 覆盖率{total_coverage:.1f}%低于生产要求{self.thresholds['production']}%")
        elif total_coverage < self.thresholds['production']:
            issues.append(f"📢 注意: 覆盖率{total_coverage:.1f}%未达到生产标准{self.thresholds['production']}%")
        elif total_coverage >= self.thresholds['target']:
            issues.append(f"✅ 优秀: 覆盖率{total_coverage:.1f}%达到目标标准{self.thresholds['target']}%")
        else:
            issues.append(f"👍 良好: 覆盖率{total_coverage:.1f}%达到生产要求{self.thresholds['production']}%")

        # 检查测试失败
        test_results = coverage_data.get('test_results', {})
        failed_tests = test_results.get('failed', 0) + test_results.get('errors', 0)

        if failed_tests > 0:
            issues.append(f"❌ 测试失败: {failed_tests}个测试未通过")

        return issues

    def generate_report(self) -> str:
        """
        生成监控报告

        Returns:
            格式化的报告字符串
        """
        # 获取最新覆盖率数据
        if not self.history:
            return "暂无覆盖率数据，请先运行分析"

        latest_data = self.history[-1]
        trend_analysis = self.analyze_trends()
        quality_issues = self.check_quality_gates(latest_data)

        # 生成报告
        report = []
        report.append("🔍 RQA2025持续覆盖率监控报告")
        report.append("=" * 50)
        report.append("")

        report.append(f"📊 报告时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"🎯 当前覆盖率: {latest_data.get('total_coverage', 0):.1f}%")
        report.append("")

        # 趋势分析
        report.append("📈 趋势分析:")
        report.append(f"   状态: {trend_analysis['trend']}")
        report.append(f"   信息: {trend_analysis['message']}")
        if 'average_coverage' in trend_analysis:
            report.append(f"   平均覆盖率: {trend_analysis['average_coverage']:.1f}%")
            report.append(f"   覆盖率范围: {trend_analysis['min_coverage']:.1f}% - {trend_analysis['max_coverage']:.1f}%")
        report.append("")

        # 质量门禁
        report.append("🚪 质量门禁检查:")
        for issue in quality_issues:
            report.append(f"   {issue}")
        report.append("")

        # 测试结果
        test_results = latest_data.get('test_results', {})
        report.append("🧪 测试执行结果:")
        report.append(f"   通过: {test_results.get('passed', 0)}")
        report.append(f"   失败: {test_results.get('failed', 0)}")
        report.append(f"   跳过: {test_results.get('skipped', 0)}")
        report.append(f"   错误: {test_results.get('errors', 0)}")
        report.append("")

        # 历史数据统计
        report.append("📚 历史统计:")
        report.append(f"   总测量次数: {len(self.history)}")
        if self.history:
            valid_measurements = [h for h in self.history if 'error' not in h]
            if valid_measurements:
                coverages = [h.get('total_coverage', 0) for h in valid_measurements]
                report.append(f"   历史最高: {max(coverages):.1f}%")
                report.append(f"   历史最低: {min(coverages):.1f}%")
                report.append(f"   历史平均: {sum(coverages)/len(coverages):.1f}%")

        return "\n".join(report)

    def run_monitoring_cycle(self) -> bool:
        """
        执行完整的监控周期

        Returns:
            是否成功完成监控
        """
        try:
            logger.info("开始监控周期...")

            # 运行覆盖率分析
            coverage_data = self.run_coverage_analysis()

            # 生成报告
            report = self.generate_report()

            # 保存报告
            report_file = self.reports_dir / f"coverage_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)

            # 打印报告
            print(report)

            # 检查是否需要告警
            quality_issues = self.check_quality_gates(coverage_data)
            critical_issues = [issue for issue in quality_issues if '🚨' in issue or '❌' in issue]

            if critical_issues:
                logger.warning("发现关键质量问题:")
                for issue in critical_issues:
                    logger.warning(f"  {issue}")
                return False

            logger.info("监控周期完成")
            return True

        except Exception as e:
            logger.error(f"监控周期失败: {e}")
            return False

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='RQA2025持续覆盖率监控')
    parser.add_argument('--project-root', help='项目根目录', default=None)
    parser.add_argument('--continuous', action='store_true', help='持续监控模式')
    parser.add_argument('--interval', type=int, default=3600, help='监控间隔(秒)，默认1小时')

    args = parser.parse_args()

    monitor = ContinuousCoverageMonitor(args.project_root)

    if args.continuous:
        logger.info(f"启动持续监控模式，间隔: {args.interval}秒")
        while True:
            success = monitor.run_monitoring_cycle()
            if not success:
                logger.warning("监控周期出现问题，继续运行...")

            time.sleep(args.interval)
    else:
        # 单次运行
        success = monitor.run_monitoring_cycle()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()




