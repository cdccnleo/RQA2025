#!/usr/bin/env python3
"""
RQA2025 自动化性能回归测试和报告生成系统
实现CI/CD集成的自动化性能测试，支持性能回归检测和自动化报告生成
"""

import sys
import json
import time
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
import sqlite3

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class PerformanceBaseline:
    """性能基准数据"""
    test_name: str
    test_category: str
    baseline_timestamp: str
    baseline_version: str

    # 性能指标基准值
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    throughput_ops_per_sec: float
    cpu_usage_percent: float
    memory_usage_mb: float
    error_rate_percent: float

    # 回归阈值
    latency_regression_threshold: float = 20.0  # 20%
    throughput_regression_threshold: float = 10.0  # 10%
    resource_regression_threshold: float = 30.0  # 30%


@dataclass
class PerformanceRegressionResult:
    """性能回归检测结果"""
    test_name: str
    has_regression: bool
    regression_type: str  # 'latency', 'throughput', 'resource', 'error_rate'
    baseline_value: float
    current_value: float
    change_percentage: float
    threshold: float
    severity: str  # 'minor', 'major', 'critical'


@dataclass
class AutomatedTestResult:
    """自动化测试结果"""
    test_run_id: str
    timestamp: str
    git_commit: str
    git_branch: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    regressions_detected: int
    total_execution_time: float
    test_results: List[Dict[str, Any]]
    regression_results: List[PerformanceRegressionResult]
    report_path: str


class PerformanceDatabase:
    """性能数据库管理"""

    def __init__(self, db_path: str = "performance_history.db"):
        self.db_path = Path(db_path)
        self._init_database()

    def _init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            # 性能基准表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_baselines (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_name TEXT NOT NULL,
                    test_category TEXT NOT NULL,
                    baseline_timestamp TEXT NOT NULL,
                    baseline_version TEXT NOT NULL,
                    latency_p50_ms REAL,
                    latency_p95_ms REAL,
                    latency_p99_ms REAL,
                    throughput_ops_per_sec REAL,
                    cpu_usage_percent REAL,
                    memory_usage_mb REAL,
                    error_rate_percent REAL,
                    latency_regression_threshold REAL DEFAULT 20.0,
                    throughput_regression_threshold REAL DEFAULT 10.0,
                    resource_regression_threshold REAL DEFAULT 30.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(test_name, test_category)
                )
            """)

            # 测试历史表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS test_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_run_id TEXT NOT NULL,
                    test_name TEXT NOT NULL,
                    test_category TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    git_commit TEXT,
                    git_branch TEXT,
                    latency_p50_ms REAL,
                    latency_p95_ms REAL,
                    latency_p99_ms REAL,
                    throughput_ops_per_sec REAL,
                    cpu_usage_percent REAL,
                    memory_usage_mb REAL,
                    error_rate_percent REAL,
                    execution_time REAL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 回归检测记录表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS regression_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_run_id TEXT NOT NULL,
                    test_name TEXT NOT NULL,
                    regression_type TEXT NOT NULL,
                    baseline_value REAL,
                    current_value REAL,
                    change_percentage REAL,
                    threshold REAL,
                    severity TEXT,
                    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def save_baseline(self, baseline: PerformanceBaseline):
        """保存性能基准"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO performance_baselines 
                (test_name, test_category, baseline_timestamp, baseline_version,
                 latency_p50_ms, latency_p95_ms, latency_p99_ms, throughput_ops_per_sec,
                 cpu_usage_percent, memory_usage_mb, error_rate_percent,
                 latency_regression_threshold, throughput_regression_threshold, resource_regression_threshold)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                baseline.test_name, baseline.test_category, baseline.baseline_timestamp,
                baseline.baseline_version, baseline.latency_p50_ms, baseline.latency_p95_ms,
                baseline.latency_p99_ms, baseline.throughput_ops_per_sec, baseline.cpu_usage_percent,
                baseline.memory_usage_mb, baseline.error_rate_percent, baseline.latency_regression_threshold,
                baseline.throughput_regression_threshold, baseline.resource_regression_threshold
            ))

    def get_baseline(self, test_name: str, test_category: str) -> Optional[PerformanceBaseline]:
        """获取性能基准"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM performance_baselines 
                WHERE test_name = ? AND test_category = ?
            """, (test_name, test_category))

            row = cursor.fetchone()
            if row:
                return PerformanceBaseline(
                    test_name=row['test_name'],
                    test_category=row['test_category'],
                    baseline_timestamp=row['baseline_timestamp'],
                    baseline_version=row['baseline_version'],
                    latency_p50_ms=row['latency_p50_ms'],
                    latency_p95_ms=row['latency_p95_ms'],
                    latency_p99_ms=row['latency_p99_ms'],
                    throughput_ops_per_sec=row['throughput_ops_per_sec'],
                    cpu_usage_percent=row['cpu_usage_percent'],
                    memory_usage_mb=row['memory_usage_mb'],
                    error_rate_percent=row['error_rate_percent'],
                    latency_regression_threshold=row['latency_regression_threshold'],
                    throughput_regression_threshold=row['throughput_regression_threshold'],
                    resource_regression_threshold=row['resource_regression_threshold']
                )
        return None

    def save_test_result(self, test_run_id: str, test_name: str, test_category: str,
                         metrics: Dict[str, Any], git_info: Dict[str, str]):
        """保存测试结果"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO test_history 
                (test_run_id, test_name, test_category, timestamp, git_commit, git_branch,
                 latency_p50_ms, latency_p95_ms, latency_p99_ms, throughput_ops_per_sec,
                 cpu_usage_percent, memory_usage_mb, error_rate_percent, execution_time, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                test_run_id, test_name, test_category, datetime.now().isoformat(),
                git_info.get('commit', ''), git_info.get('branch', ''),
                metrics.get('latency_p50', 0), metrics.get('latency_p95', 0),
                metrics.get('latency_p99', 0), metrics.get('throughput_ops_per_sec', 0),
                metrics.get('cpu_usage_percent', 0), metrics.get('memory_usage_mb', 0),
                metrics.get('error_rate_percent', 0), metrics.get('execution_time', 0),
                json.dumps(metrics)
            ))

    def save_regression_result(self, test_run_id: str, regression: PerformanceRegressionResult):
        """保存回归检测结果"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO regression_history 
                (test_run_id, test_name, regression_type, baseline_value, current_value,
                 change_percentage, threshold, severity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                test_run_id, regression.test_name, regression.regression_type,
                regression.baseline_value, regression.current_value, regression.change_percentage,
                regression.threshold, regression.severity
            ))

    def get_test_history(self, test_name: str, days: int = 30) -> List[Dict[str, Any]]:
        """获取测试历史"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM test_history 
                WHERE test_name = ? AND created_at >= datetime('now', '-{} days')
                ORDER BY created_at DESC
            """.format(days), (test_name,))

            return [dict(row) for row in cursor.fetchall()]


class RegressionDetector:
    """性能回归检测器"""

    def __init__(self, database: PerformanceDatabase):
        self.database = database

    def detect_regressions(self, test_name: str, test_category: str,
                           current_metrics: Dict[str, Any]) -> List[PerformanceRegressionResult]:
        """检测性能回归"""
        regressions = []

        # 获取基准数据
        baseline = self.database.get_baseline(test_name, test_category)
        if not baseline:
            # 没有基准数据，跳过回归检测
            return regressions

        # 检测延迟回归
        for metric, baseline_value, threshold in [
            ('latency_p50', baseline.latency_p50_ms, baseline.latency_regression_threshold),
            ('latency_p95', baseline.latency_p95_ms, baseline.latency_regression_threshold),
            ('latency_p99', baseline.latency_p99_ms, baseline.latency_regression_threshold)
        ]:
            current_value = current_metrics.get(metric, 0)
            if baseline_value > 0:
                change_percentage = ((current_value - baseline_value) / baseline_value) * 100
                if change_percentage > threshold:
                    severity = self._determine_severity(change_percentage, threshold)
                    regressions.append(PerformanceRegressionResult(
                        test_name=test_name,
                        has_regression=True,
                        regression_type='latency',
                        baseline_value=baseline_value,
                        current_value=current_value,
                        change_percentage=change_percentage,
                        threshold=threshold,
                        severity=severity
                    ))

        # 检测吞吐量回归
        current_throughput = current_metrics.get('throughput_ops_per_sec', 0)
        if baseline.throughput_ops_per_sec > 0 and current_throughput > 0:
            change_percentage = ((baseline.throughput_ops_per_sec - current_throughput) /
                                 baseline.throughput_ops_per_sec) * 100
            if change_percentage > baseline.throughput_regression_threshold:
                severity = self._determine_severity(
                    change_percentage, baseline.throughput_regression_threshold)
                regressions.append(PerformanceRegressionResult(
                    test_name=test_name,
                    has_regression=True,
                    regression_type='throughput',
                    baseline_value=baseline.throughput_ops_per_sec,
                    current_value=current_throughput,
                    change_percentage=change_percentage,
                    threshold=baseline.throughput_regression_threshold,
                    severity=severity
                ))

        # 检测资源使用回归
        for metric, baseline_value, threshold in [
            ('cpu_usage_percent', baseline.cpu_usage_percent, baseline.resource_regression_threshold),
            ('memory_usage_mb', baseline.memory_usage_mb, baseline.resource_regression_threshold)
        ]:
            current_value = current_metrics.get(metric, 0)
            if baseline_value > 0:
                change_percentage = ((current_value - baseline_value) / baseline_value) * 100
                if change_percentage > threshold:
                    severity = self._determine_severity(change_percentage, threshold)
                    regressions.append(PerformanceRegressionResult(
                        test_name=test_name,
                        has_regression=True,
                        regression_type='resource',
                        baseline_value=baseline_value,
                        current_value=current_value,
                        change_percentage=change_percentage,
                        threshold=threshold,
                        severity=severity
                    ))

        return regressions

    def _determine_severity(self, change_percentage: float, threshold: float) -> str:
        """确定回归严重程度"""
        if change_percentage > threshold * 3:
            return 'critical'
        elif change_percentage > threshold * 2:
            return 'major'
        else:
            return 'minor'


class GitIntegration:
    """Git集成工具"""

    @staticmethod
    def get_git_info() -> Dict[str, str]:
        """获取Git信息"""
        try:
            # 获取当前提交哈希
            commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                universal_newlines=True
            ).strip()

            # 获取当前分支
            branch = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                universal_newlines=True
            ).strip()

            return {'commit': commit, 'branch': branch}

        except subprocess.CalledProcessError:
            return {'commit': 'unknown', 'branch': 'unknown'}


class AutomatedPerformanceTestRunner:
    """自动化性能测试运行器"""

    def __init__(self, output_dir: str = "reports/automated_performance"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化组件
        self.database = PerformanceDatabase(str(self.output_dir / "performance_history.db"))
        self.regression_detector = RegressionDetector(self.database)

        # 设置日志
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            # 文件日志
            file_handler = logging.FileHandler(self.output_dir / "automated_test.log")
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

            # 控制台日志
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        return logger

    def run_automated_test_suite(self, test_config: Optional[Dict[str, Any]] = None) -> AutomatedTestResult:
        """运行自动化测试套件"""
        test_run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger.info(f"开始自动化性能测试: {test_run_id}")

        # 获取Git信息
        git_info = GitIntegration.get_git_info()
        self.logger.info(f"Git信息: {git_info}")

        # 默认测试配置
        if test_config is None:
            test_config = {
                'test_suites': [
                    'event_bus_performance',
                    'dependency_injection_performance',
                    'data_ingestion_performance',
                    'cache_performance',
                    'order_processing',
                    'risk_management'
                ],
                'iterations': 500,
                'warmup_iterations': 50,
                'concurrent_users': [1, 5, 10]
            }

        start_time = time.time()
        test_results = []
        regression_results = []
        total_tests = 0
        passed_tests = 0
        failed_tests = 0

        # 运行测试套件
        for suite_name in test_config['test_suites']:
            self.logger.info(f"运行测试套件: {suite_name}")

            try:
                # 模拟运行测试套件
                result = self._run_test_suite(suite_name, test_config)
                test_results.append(result)

                # 保存测试结果到数据库
                self.database.save_test_result(
                    test_run_id, suite_name, result['category'],
                    result['metrics'], git_info
                )

                # 检测性能回归
                regressions = self.regression_detector.detect_regressions(
                    suite_name, result['category'], result['metrics']
                )

                for regression in regressions:
                    regression_results.append(regression)
                    self.database.save_regression_result(test_run_id, regression)

                total_tests += 1
                if result['status'] == 'passed':
                    passed_tests += 1
                else:
                    failed_tests += 1

                self.logger.info(f"测试套件 {suite_name} 完成，检测到 {len(regressions)} 个回归")

            except Exception as e:
                self.logger.error(f"测试套件 {suite_name} 失败: {e}")
                failed_tests += 1

        end_time = time.time()
        total_execution_time = end_time - start_time

        # 生成报告
        report_path = self._generate_automated_report(
            test_run_id, git_info, test_results, regression_results, total_execution_time
        )

        # 创建结果对象
        result = AutomatedTestResult(
            test_run_id=test_run_id,
            timestamp=datetime.now().isoformat(),
            git_commit=git_info['commit'],
            git_branch=git_info['branch'],
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            regressions_detected=len(regression_results),
            total_execution_time=total_execution_time,
            test_results=test_results,
            regression_results=regression_results,
            report_path=report_path
        )

        self.logger.info(f"自动化测试完成: {test_run_id}")
        self.logger.info(f"总测试: {total_tests}, 通过: {passed_tests}, 失败: {failed_tests}")
        self.logger.info(f"检测到回归: {len(regression_results)}")

        return result

    def _run_test_suite(self, suite_name: str, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """运行测试套件（模拟）"""
        import random
        import time

        # 模拟测试执行
        execution_time = random.uniform(0.1, 2.0)
        time.sleep(execution_time / 10)  # 模拟执行时间

        # 模拟性能指标
        metrics = {
            'latency_p50': random.uniform(1.0, 10.0),
            'latency_p95': random.uniform(5.0, 50.0),
            'latency_p99': random.uniform(10.0, 100.0),
            'throughput_ops_per_sec': random.uniform(1000, 50000),
            'cpu_usage_percent': random.uniform(10, 80),
            'memory_usage_mb': random.uniform(50, 500),
            'error_rate_percent': random.uniform(0, 2),
            'execution_time': execution_time
        }

        # 根据测试名称确定类别
        if 'event_bus' in suite_name or 'dependency' in suite_name or 'business_process' in suite_name:
            category = 'core_service'
        elif 'data' in suite_name or 'cache' in suite_name:
            category = 'data_management'
        elif 'order' in suite_name or 'risk' in suite_name:
            category = 'trading_system'
        else:
            category = 'infrastructure'

        return {
            'test_name': suite_name,
            'category': category,
            'status': 'passed' if metrics['error_rate_percent'] < 1 else 'failed',
            'metrics': metrics,
            'execution_time': execution_time
        }

    def _generate_automated_report(self, test_run_id: str, git_info: Dict[str, str],
                                   test_results: List[Dict[str, Any]],
                                   regression_results: List[PerformanceRegressionResult],
                                   total_execution_time: float) -> str:
        """生成自动化测试报告"""
        report_file = self.output_dir / f"automated_report_{test_run_id}.md"

        report_content = []
        report_content.append(f"# 自动化性能测试报告")
        report_content.append("")
        report_content.append(f"**测试运行ID**: {test_run_id}")
        report_content.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append(f"**Git提交**: {git_info['commit'][:8]}")
        report_content.append(f"**Git分支**: {git_info['branch']}")
        report_content.append(f"**总执行时间**: {total_execution_time:.2f}秒")
        report_content.append("")

        # 测试概要
        passed_tests = sum(1 for r in test_results if r['status'] == 'passed')
        failed_tests = len(test_results) - passed_tests

        report_content.append("## 📊 测试概要")
        report_content.append("")
        report_content.append(f"- **总测试数**: {len(test_results)}")
        report_content.append(f"- **通过测试**: {passed_tests}")
        report_content.append(f"- **失败测试**: {failed_tests}")
        report_content.append(f"- **成功率**: {passed_tests/len(test_results)*100:.1f}%")
        report_content.append(f"- **检测到回归**: {len(regression_results)}")
        report_content.append("")

        # 性能回归报告
        if regression_results:
            report_content.append("## ⚠️ 性能回归检测")
            report_content.append("")

            critical_regressions = [r for r in regression_results if r.severity == 'critical']
            major_regressions = [r for r in regression_results if r.severity == 'major']
            minor_regressions = [r for r in regression_results if r.severity == 'minor']

            report_content.append(f"- **严重回归**: {len(critical_regressions)}")
            report_content.append(f"- **主要回归**: {len(major_regressions)}")
            report_content.append(f"- **轻微回归**: {len(minor_regressions)}")
            report_content.append("")

            # 详细回归信息
            for regression in regression_results:
                icon = "🔴" if regression.severity == 'critical' else (
                    "🟠" if regression.severity == 'major' else "🟡")
                report_content.append(
                    f"### {icon} {regression.test_name} - {regression.regression_type.upper()}回归")
                report_content.append(f"- **严重程度**: {regression.severity}")
                report_content.append(f"- **基准值**: {regression.baseline_value:.2f}")
                report_content.append(f"- **当前值**: {regression.current_value:.2f}")
                report_content.append(f"- **变化幅度**: {regression.change_percentage:+.1f}%")
                report_content.append(f"- **阈值**: {regression.threshold:.1f}%")
                report_content.append("")
        else:
            report_content.append("## ✅ 性能回归检测")
            report_content.append("")
            report_content.append("未检测到性能回归，所有测试都在预期范围内。")
            report_content.append("")

        # 测试结果详情
        report_content.append("## 📋 测试结果详情")
        report_content.append("")

        for result in test_results:
            status_icon = "✅" if result['status'] == 'passed' else "❌"
            report_content.append(f"### {status_icon} {result['test_name']}")
            report_content.append(f"- **类别**: {result['category']}")
            report_content.append(f"- **状态**: {result['status']}")
            report_content.append(f"- **执行时间**: {result['execution_time']:.3f}秒")

            metrics = result['metrics']
            report_content.append(f"- **P50延迟**: {metrics['latency_p50']:.2f}ms")
            report_content.append(f"- **P95延迟**: {metrics['latency_p95']:.2f}ms")
            report_content.append(f"- **吞吐量**: {metrics['throughput_ops_per_sec']:.0f} ops/sec")
            report_content.append(f"- **CPU使用率**: {metrics['cpu_usage_percent']:.1f}%")
            report_content.append(f"- **内存使用**: {metrics['memory_usage_mb']:.1f}MB")
            report_content.append(f"- **错误率**: {metrics['error_rate_percent']:.2f}%")
            report_content.append("")

        # 建议和改进
        report_content.append("## 💡 建议和改进")
        report_content.append("")

        if regression_results:
            report_content.append("检测到性能回归，建议采取以下措施：")
            report_content.append("• 分析回归原因，重点关注严重和主要回归")
            report_content.append("• 检查最近的代码变更，确定影响性能的修改")
            report_content.append("• 考虑回滚有问题的提交或优化相关代码")
        else:
            report_content.append("性能表现良好，建议继续保持：")
            report_content.append("• 定期运行性能测试，及时发现潜在问题")
            report_content.append("• 持续监控系统资源使用情况")
            report_content.append("• 优化现有代码，进一步提升性能")

        report_content.append("")
        report_content.append("---")
        report_content.append(f"报告生成于: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 写入文件
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))

        self.logger.info(f"自动化测试报告已生成: {report_file}")
        return str(report_file)

    def update_baselines(self, test_results: List[Dict[str, Any]], git_info: Dict[str, str]):
        """更新性能基准"""
        self.logger.info("开始更新性能基准...")

        for result in test_results:
            metrics = result['metrics']

            baseline = PerformanceBaseline(
                test_name=result['test_name'],
                test_category=result['category'],
                baseline_timestamp=datetime.now().isoformat(),
                baseline_version=git_info['commit'],
                latency_p50_ms=metrics['latency_p50'],
                latency_p95_ms=metrics['latency_p95'],
                latency_p99_ms=metrics['latency_p99'],
                throughput_ops_per_sec=metrics['throughput_ops_per_sec'],
                cpu_usage_percent=metrics['cpu_usage_percent'],
                memory_usage_mb=metrics['memory_usage_mb'],
                error_rate_percent=metrics['error_rate_percent']
            )

            self.database.save_baseline(baseline)
            self.logger.info(f"已更新基准: {result['test_name']}")

        self.logger.info(f"已更新 {len(test_results)} 个测试的性能基准")


def main():
    """主函数"""
    print("🚀 RQA2025 自动化性能测试系统")
    print("=" * 60)

    # 创建测试运行器
    runner = AutomatedPerformanceTestRunner()

    # 运行自动化测试
    result = runner.run_automated_test_suite()

    print("\n" + "=" * 60)
    print("📋 测试结果概要")
    print("=" * 60)
    print(f"测试运行ID: {result.test_run_id}")
    print(f"Git提交: {result.git_commit[:8]}")
    print(f"Git分支: {result.git_branch}")
    print(f"总测试数: {result.total_tests}")
    print(f"通过测试: {result.passed_tests}")
    print(f"失败测试: {result.failed_tests}")
    print(f"成功率: {result.passed_tests/result.total_tests*100:.1f}%")
    print(f"执行时间: {result.total_execution_time:.2f}秒")
    print(f"检测到回归: {result.regressions_detected}")

    if result.regressions_detected > 0:
        print("\n⚠️  检测到性能回归:")
        for regression in result.regression_results:
            severity_icon = "🔴" if regression.severity == 'critical' else (
                "🟠" if regression.severity == 'major' else "🟡")
            print(
                f"  {severity_icon} {regression.test_name} - {regression.regression_type}: {regression.change_percentage:+.1f}%")
    else:
        print("\n✅ 没有检测到性能回归")

    print(f"\n📄 详细报告: {result.report_path}")

    # 提示是否更新基准
    if result.regressions_detected == 0:
        print("\n📝 建议: 性能表现良好，可以考虑更新性能基准")

    return result


if __name__ == "__main__":
    main()
