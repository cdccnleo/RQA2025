#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试性能监控体系

建立全面的测试执行性能监控：
- 实时性能指标收集
- 历史趋势分析
- 性能瓶颈识别
- 优化建议生成
- 可视化报告
"""

import os
import time
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
import json
import logging
from datetime import datetime, timedelta
import statistics

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """性能指标"""
    timestamp: datetime
    test_file: str
    execution_time: float
    memory_usage: float  # MB
    cpu_usage: float     # %
    success: bool
    test_count: int
    passed: int
    failed: int
    errors: int
    skipped: int


@dataclass
class PerformanceSnapshot:
    """性能快照"""
    timestamp: datetime
    total_tests: int
    total_duration: float
    avg_test_time: float
    success_rate: float
    memory_peak: float
    cpu_avg: float
    throughput: float  # tests/second


@dataclass
class PerformanceAnalysis:
    """性能分析结果"""
    trends: Dict[str, Any]
    bottlenecks: List[str]
    recommendations: List[str]
    health_score: float
    risk_areas: List[str]


class TestPerformanceMonitor:
    """测试性能监控器"""

    def __init__(self, history_file: str = "test_logs/performance_history.json"):
        self.history_file = Path(history_file)
        self.current_session: List[PerformanceMetrics] = []
        self.session_start_time = datetime.now()
        self.monitoring_active = False
        self.system_monitor_thread: Optional[threading.Thread] = None

        # 系统资源监控
        self.system_stats = {
            'memory_baseline': psutil.virtual_memory().available / (1024**2),  # MB
            'cpu_baseline': psutil.cpu_percent(interval=None),
        }

        # 历史数据缓存
        self._history_cache: Optional[List[PerformanceMetrics]] = None

        logger.info("测试性能监控器初始化完成")

    def start_monitoring(self):
        """开始性能监控"""
        if self.monitoring_active:
            logger.warning("性能监控已在运行")
            return

        self.monitoring_active = True
        self.session_start_time = datetime.now()
        self.current_session = []

        # 启动系统资源监控线程
        self.system_monitor_thread = threading.Thread(target=self._monitor_system_resources, daemon=True)
        self.system_monitor_thread.start()

        logger.info("性能监控已启动")

    def stop_monitoring(self) -> PerformanceSnapshot:
        """停止性能监控并生成快照"""
        if not self.monitoring_active:
            logger.warning("性能监控未启动")
            return self._create_empty_snapshot()

        self.monitoring_active = False

        # 等待监控线程结束
        if self.system_monitor_thread and self.system_monitor_thread.is_alive():
            self.system_monitor_thread.join(timeout=5)

        # 保存会话数据
        self._save_session_data()

        # 生成性能快照
        snapshot = self._create_snapshot()
        logger.info("性能监控已停止")

        return snapshot

    def record_test_execution(self, test_file: str, execution_time: float, success: bool,
                            test_results: Dict[str, int]):
        """记录测试执行指标"""
        if not self.monitoring_active:
            return

        # 获取当前系统资源使用情况
        memory_usage = self._get_memory_usage()
        cpu_usage = self._get_cpu_usage()

        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            test_file=test_file,
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            success=success,
            test_count=test_results.get('passed', 0) + test_results.get('failed', 0) +
                    test_results.get('errors', 0) + test_results.get('skipped', 0),
            passed=test_results.get('passed', 0),
            failed=test_results.get('failed', 0),
            errors=test_results.get('errors', 0),
            skipped=test_results.get('skipped', 0)
        )

        self.current_session.append(metrics)

    def analyze_performance(self, days: int = 7) -> PerformanceAnalysis:
        """分析性能趋势"""
        logger.info(f"开始性能分析（最近{days}天）...")

        # 获取历史数据
        history_data = self._load_history_data(days)

        # 趋势分析
        trends = self._analyze_trends(history_data)

        # 瓶颈识别
        bottlenecks = self._identify_bottlenecks(history_data)

        # 生成建议
        recommendations = self._generate_recommendations(trends, bottlenecks)

        # 计算健康评分
        health_score = self._calculate_health_score(trends, bottlenecks)

        # 识别风险区域
        risk_areas = self._identify_risk_areas(history_data)

        analysis = PerformanceAnalysis(
            trends=trends,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
            health_score=health_score,
            risk_areas=risk_areas
        )

        logger.info("性能分析完成")
        return analysis

    def generate_performance_report(self, snapshot: PerformanceSnapshot,
                                analysis: PerformanceAnalysis):
        """生成性能报告"""
        report_path = Path("test_logs/performance_report.md")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 测试性能监控报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## 📊 当前会话概览\n\n")
            f.write(f"- **测试总数**: {snapshot.total_tests}\n")
            f.write(".2")
            f.write(".2")
            f.write(".1")
            f.write(".1")
            f.write(".2")
            f.write(".1")
            f.write("## 📈 性能趋势分析\n\n")

            trends = analysis.trends
            if 'execution_time_trend' in trends:
                trend = trends['execution_time_trend']
                direction = "📈 上升" if trend['slope'] > 0 else "📉 下降"
                f.write(f"- **执行时间趋势**: {direction} ({trend['change_percent']:.1f}%)\n")

            if 'success_rate_trend' in trends:
                trend = trends['success_rate_trend']
                direction = "📈 改善" if trend['slope'] > 0 else "📉 恶化"
                f.write(f"- **成功率趋势**: {direction} ({trend['change_percent']:.1f}%)\n")

            if 'memory_trend' in trends:
                trend = trends['memory_trend']
                direction = "📈 增加" if trend['slope'] > 0 else "📉 减少"
                f.write(f"- **内存使用趋势**: {direction} ({trend['change_percent']:.1f}%)\n")

            f.write("\n## 🚧 性能瓶颈\n\n")
            if analysis.bottlenecks:
                for bottleneck in analysis.bottlenecks:
                    f.write(f"- ⚠️ {bottleneck}\n")
            else:
                f.write("✅ 未发现明显性能瓶颈\n")

            f.write("\n## 💡 优化建议\n\n")
            if analysis.recommendations:
                for i, rec in enumerate(analysis.recommendations, 1):
                    f.write(f"{i}. {rec}\n")
            else:
                f.write("✅ 当前性能表现良好，无需特别优化\n")

            f.write("\n## 🏥 健康评分\n\n")
            health_score = analysis.health_score
            if health_score >= 80:
                status = "🟢 健康"
            elif health_score >= 60:
                status = "🟡 一般"
            else:
                status = "🔴 需要关注"

            f.write(f"- **整体健康评分**: {health_score:.1f}/100 ({status})\n")

            f.write("\n## ⚠️ 风险区域\n\n")
            if analysis.risk_areas:
                for risk in analysis.risk_areas:
                    f.write(f"- 🚨 {risk}\n")
            else:
                f.write("✅ 未发现高风险区域\n")

            f.write("\n## 📋 详细指标\n\n")
            if self.current_session:
                f.write("### 慢速测试Top 5\n\n")
                slow_tests = sorted(self.current_session, key=lambda x: x.execution_time, reverse=True)[:5]
                f.write("| 测试文件 | 执行时间 | 状态 |\n")
                f.write("|----------|----------|------|\n")
                for test in slow_tests:
                    status = "✅" if test.success else "❌"
                    f.write(f"| `{Path(test.test_file).name}` | {test.execution_time:.2f}s | {status} |\n")

                f.write("\n### 资源消耗Top 5\n\n")
                high_memory_tests = sorted(self.current_session, key=lambda x: x.memory_usage, reverse=True)[:5]
                f.write("| 测试文件 | 内存使用 | CPU使用 |\n")
                f.write("|----------|----------|--------|\n")
                for test in high_memory_tests:
                    f.write(f"| `{Path(test.test_file).name}` | {test.memory_usage:.1f}MB | {test.cpu_usage:.1f}% |\n")

        logger.info(f"性能报告已生成: {report_path}")

    def _monitor_system_resources(self):
        """监控系统资源使用情况"""
        while self.monitoring_active:
            try:
                # 每秒更新一次系统资源统计
                time.sleep(1)
            except Exception as e:
                logger.debug(f"系统资源监控异常: {e}")

    def _get_memory_usage(self) -> float:
        """获取内存使用量（MB）"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024**2)  # Bytes to MB
        except Exception:
            return 0.0

    def _get_cpu_usage(self) -> float:
        """获取CPU使用率"""
        try:
            process = psutil.Process()
            return process.cpu_percent(interval=0.1)
        except Exception:
            return 0.0

    def _create_snapshot(self) -> PerformanceSnapshot:
        """创建性能快照"""
        if not self.current_session:
            return self._create_empty_snapshot()

        total_tests = sum(m.test_count for m in self.current_session)
        total_duration = sum(m.execution_time for m in self.current_session)
        successful_tests = sum(m.passed for m in self.current_session)
        total_test_count = len(self.current_session)

        return PerformanceSnapshot(
            timestamp=datetime.now(),
            total_tests=total_tests,
            total_duration=total_duration,
            avg_test_time=total_duration / total_test_count if total_test_count > 0 else 0,
            success_rate=successful_tests / total_tests * 100 if total_tests > 0 else 0,
            memory_peak=max((m.memory_usage for m in self.current_session), default=0),
            cpu_avg=sum(m.cpu_usage for m in self.current_session) / total_test_count if total_test_count > 0 else 0,
            throughput=total_tests / total_duration if total_duration > 0 else 0
        )

    def _create_empty_snapshot(self) -> PerformanceSnapshot:
        """创建空的性能快照"""
        return PerformanceSnapshot(
            timestamp=datetime.now(),
            total_tests=0,
            total_duration=0.0,
            avg_test_time=0.0,
            success_rate=0.0,
            memory_peak=0.0,
            cpu_avg=0.0,
            throughput=0.0
        )

    def _save_session_data(self):
        """保存会话数据到历史文件"""
        try:
            # 加载现有历史数据
            history = self._load_history_data(30)  # 最近30天

            # 添加当前会话数据
            history.extend(self.current_session)

            # 只保留最近30天的历史数据
            cutoff_date = datetime.now() - timedelta(days=30)
            history = [m for m in history if m.timestamp > cutoff_date]

            # 保存到文件
            history_dicts = [
                {
                    'timestamp': m.timestamp.isoformat(),
                    'test_file': m.test_file,
                    'execution_time': m.execution_time,
                    'memory_usage': m.memory_usage,
                    'cpu_usage': m.cpu_usage,
                    'success': m.success,
                    'test_count': m.test_count,
                    'passed': m.passed,
                    'failed': m.failed,
                    'errors': m.errors,
                    'skipped': m.skipped
                }
                for m in history
            ]

            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history_dicts, f, indent=2, ensure_ascii=False)

            logger.info(f"历史数据已保存，共 {len(history)} 条记录")

        except Exception as e:
            logger.warning(f"保存历史数据失败: {e}")

    def _load_history_data(self, days: int = 7) -> List[PerformanceMetrics]:
        """加载历史数据"""
        if self._history_cache is not None:
            return self._history_cache

        try:
            if not self.history_file.exists():
                return []

            with open(self.history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            cutoff_date = datetime.now() - timedelta(days=days)
            history = []

            for item in data:
                timestamp = datetime.fromisoformat(item['timestamp'])
                if timestamp > cutoff_date:
                    history.append(PerformanceMetrics(
                        timestamp=timestamp,
                        test_file=item['test_file'],
                        execution_time=item['execution_time'],
                        memory_usage=item.get('memory_usage', 0.0),
                        cpu_usage=item.get('cpu_usage', 0.0),
                        success=item['success'],
                        test_count=item['test_count'],
                        passed=item['passed'],
                        failed=item['failed'],
                        errors=item['errors'],
                        skipped=item['skipped']
                    ))

            self._history_cache = history
            return history

        except Exception as e:
            logger.warning(f"加载历史数据失败: {e}")
            return []

    def _analyze_trends(self, history: List[PerformanceMetrics]) -> Dict[str, Any]:
        """分析性能趋势"""
        trends = {}

        if len(history) < 5:  # 需要足够的数据点
            return trends

        try:
            # 执行时间趋势
            times = [m.execution_time for m in history[-20:]]  # 最近20个数据点
            if len(times) >= 5:
                slope, intercept = self._linear_regression(range(len(times)), times)
                avg_recent = statistics.mean(times[-5:])
                avg_older = statistics.mean(times[:5]) if len(times) >= 10 else statistics.mean(times)
                change_percent = (avg_recent - avg_older) / avg_older * 100 if avg_older > 0 else 0

                trends['execution_time_trend'] = {
                    'slope': slope,
                    'change_percent': change_percent,
                    'avg_recent': avg_recent,
                    'avg_older': avg_older
                }

            # 成功率趋势
            success_rates = [(m.passed / m.test_count * 100) if m.test_count > 0 else 0 for m in history[-20:]]
            if len(success_rates) >= 5:
                slope, intercept = self._linear_regression(range(len(success_rates)), success_rates)
                avg_recent = statistics.mean(success_rates[-5:])
                avg_older = statistics.mean(success_rates[:5]) if len(success_rates) >= 10 else statistics.mean(success_rates)
                change_percent = (avg_recent - avg_older) / avg_older * 100 if avg_older > 0 else 0

                trends['success_rate_trend'] = {
                    'slope': slope,
                    'change_percent': change_percent,
                    'avg_recent': avg_recent,
                    'avg_older': avg_older
                }

            # 内存使用趋势
            memory_usage = [m.memory_usage for m in history[-20:]]
            if len(memory_usage) >= 5 and any(m > 0 for m in memory_usage):
                slope, intercept = self._linear_regression(range(len(memory_usage)), memory_usage)
                avg_recent = statistics.mean([m for m in memory_usage[-5:] if m > 0] or [0])
                avg_older = statistics.mean([m for m in memory_usage[:5] if m > 0] or [0])
                change_percent = (avg_recent - avg_older) / avg_older * 100 if avg_older > 0 else 0

                trends['memory_trend'] = {
                    'slope': slope,
                    'change_percent': change_percent,
                    'avg_recent': avg_recent,
                    'avg_older': avg_older
                }

        except Exception as e:
            logger.warning(f"趋势分析失败: {e}")

        return trends

    def _identify_bottlenecks(self, history: List[PerformanceMetrics]) -> List[str]:
        """识别性能瓶颈"""
        bottlenecks = []

        if not history:
            return bottlenecks

        try:
            # 慢速测试识别
            execution_times = [m.execution_time for m in history]
            if execution_times:
                avg_time = statistics.mean(execution_times)
                std_dev = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
                threshold = avg_time + 2 * std_dev

                slow_tests = [m for m in history if m.execution_time > threshold]
                if slow_tests:
                    bottlenecks.append(f"发现 {len(slow_tests)} 个异常慢速测试 (>{threshold:.2f}s)")

            # 高内存消耗测试
            memory_usage = [m.memory_usage for m in history if m.memory_usage > 0]
            if memory_usage:
                avg_memory = statistics.mean(memory_usage)
                threshold = avg_memory * 2

                high_memory_tests = [m for m in history if m.memory_usage > threshold]
                if high_memory_tests:
                    bottlenecks.append(f"发现 {len(high_memory_tests)} 个高内存消耗测试 (>{threshold:.1f}MB)")

            # 失败率高的测试
            failed_tests = [m for m in history if not m.success]
            if failed_tests and len(history) > 10:
                failure_rate = len(failed_tests) / len(history)
                if failure_rate > 0.3:  # 30%失败率
                    bottlenecks.append(".1")            # 资源使用不稳定
            cpu_usage = [m.cpu_usage for m in history if m.cpu_usage > 0]
            if cpu_usage and len(cpu_usage) > 5:
                std_dev = statistics.stdev(cpu_usage)
                if std_dev > 20:  # CPU使用标准差大于20%
                    bottlenecks.append("CPU使用不稳定，存在资源竞争")

        except Exception as e:
            logger.warning(f"瓶颈识别失败: {e}")

        return bottlenecks

    def _generate_recommendations(self, trends: Dict[str, Any], bottlenecks: List[str]) -> List[str]:
        """生成优化建议"""
        recommendations = []

        # 基于趋势的建议
        if 'execution_time_trend' in trends:
            trend = trends['execution_time_trend']
            if trend['slope'] > 0.1:  # 执行时间在增加
                recommendations.append("执行时间呈上升趋势，建议检查是否有性能回归")
            elif trend['slope'] < -0.1:  # 执行时间在减少
                recommendations.append("执行时间优化效果良好，继续保持")

        if 'success_rate_trend' in trends:
            trend = trends['success_rate_trend']
            if trend['slope'] < -0.5:  # 成功率在下降
                recommendations.append("测试成功率呈下降趋势，需要关注测试稳定性")
            elif trend['slope'] > 0.5:  # 成功率在改善
                recommendations.append("测试稳定性有所改善，建议总结经验")

        # 基于瓶颈的建议
        for bottleneck in bottlenecks:
            if "慢速测试" in bottleneck:
                recommendations.append("优化慢速测试：考虑使用更高效的测试策略或并行执行")
            if "高内存消耗" in bottleneck:
                recommendations.append("优化内存使用：检查是否存在内存泄漏或不必要的资源占用")
            if "失败率" in bottleneck:
                recommendations.append("提高测试稳定性：分析失败原因并修复不稳定的测试")
            if "CPU使用不稳定" in bottleneck:
                recommendations.append("优化资源使用：减少并发冲突或调整资源分配策略")

        # 通用建议
        if not recommendations:
            recommendations.append("当前性能表现良好，建议继续监控并维护")

        recommendations.append("考虑实施增量测试策略，减少全量测试执行时间")
        recommendations.append("定期审查和清理过时的测试用例")

        return recommendations

    def _calculate_health_score(self, trends: Dict[str, Any], bottlenecks: List[str]) -> float:
        """计算健康评分"""
        score = 100.0

        # 基于趋势扣分
        if 'execution_time_trend' in trends:
            trend = trends['execution_time_trend']
            if trend['slope'] > 0:  # 时间增加
                score -= min(abs(trend['change_percent']), 20)

        if 'success_rate_trend' in trends:
            trend = trends['success_rate_trend']
            if trend['slope'] < 0:  # 成功率下降
                score -= min(abs(trend['change_percent']), 30)

        # 基于瓶颈扣分
        score -= len(bottlenecks) * 10

        return max(0.0, min(100.0, score))

    def _identify_risk_areas(self, history: List[PerformanceMetrics]) -> List[str]:
        """识别风险区域"""
        risk_areas = []

        if not history:
            return risk_areas

        try:
            # 按测试文件分组统计
            file_stats = {}
            for m in history:
                file = m.test_file
                if file not in file_stats:
                    file_stats[file] = {'count': 0, 'failures': 0, 'total_time': 0}

                file_stats[file]['count'] += 1
                file_stats[file]['total_time'] += m.execution_time
                if not m.success:
                    file_stats[file]['failures'] += 1

            # 识别高风险文件
            for file, stats in file_stats.items():
                failure_rate = stats['failures'] / stats['count']
                avg_time = stats['total_time'] / stats['count']

                if failure_rate > 0.5:  # 失败率超过50%
                    risk_areas.append(f"高失败率测试: {Path(file).name} ({failure_rate:.1f})")
                elif avg_time > 10:  # 平均执行时间超过10秒
                    risk_areas.append(f"慢速测试: {Path(file).name} ({avg_time:.1f}s)")

        except Exception as e:
            logger.warning(f"风险区域识别失败: {e}")

        return risk_areas

    def _linear_regression(self, x: List[float], y: List[float]) -> Tuple[float, float]:
        """简单线性回归"""
        try:
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(xi * yi for xi, yi in zip(x, y))
            sum_x2 = sum(xi * xi for xi in x)

            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            intercept = (sum_y - slope * sum_x) / n

            return slope, intercept
        except Exception:
            return 0.0, 0.0


class PerformanceMonitorManager:
    """性能监控管理器"""

    def __init__(self):
        self.monitor = TestPerformanceMonitor()

    def run_performance_monitoring_cycle(self) -> Dict[str, Any]:
        """运行完整的性能监控周期"""
        logger.info("开始性能监控周期...")

        # 启动监控
        self.monitor.start_monitoring()

        try:
            # 这里可以集成实际的测试执行
            # 例如运行增量测试或指定的测试套件

            # 模拟一些测试执行（实际使用时需要替换）
            time.sleep(2)  # 模拟执行时间

            # 停止监控
            snapshot = self.monitor.stop_monitoring()

            # 分析性能
            analysis = self.monitor.analyze_performance()

            # 生成报告
            self.monitor.generate_performance_report(snapshot, analysis)

            result = {
                'status': 'success',
                'snapshot': {
                    'total_tests': snapshot.total_tests,
                    'total_duration': snapshot.total_duration,
                    'avg_test_time': snapshot.avg_test_time,
                    'success_rate': snapshot.success_rate,
                    'memory_peak': snapshot.memory_peak,
                    'cpu_avg': snapshot.cpu_avg,
                    'throughput': snapshot.throughput
                },
                'analysis': {
                    'health_score': analysis.health_score,
                    'bottlenecks_count': len(analysis.bottlenecks),
                    'recommendations_count': len(analysis.recommendations),
                    'risk_areas_count': len(analysis.risk_areas)
                }
            }

            logger.info("性能监控周期完成")
            return result

        except Exception as e:
            logger.error(f"性能监控周期失败: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def integrate_with_incremental_tester(self, incremental_tester):
        """与增量测试器集成"""
        # 这里可以添加与增量测试器的集成逻辑
        # 例如：在增量测试执行时自动启动性能监控

        logger.info("已准备好与增量测试器集成")


def main():
    """主函数"""
    manager = PerformanceMonitorManager()

    print("📊 性能监控体系启动")
    print("🎯 功能: 实时监控 + 趋势分析 + 瓶颈识别 + 优化建议")

    # 运行性能监控周期
    result = manager.run_performance_monitoring_cycle()

    print("\n📈 监控结果:")
    if result['status'] == 'success':
        snapshot = result['snapshot']
        analysis = result['analysis']

        print(f"  📊 测试总数: {snapshot['total_tests']}")
        print(".2")
        print(".2")
        print(".1")
        print(".1")
        print(".2")
        print(f"  🏥 健康评分: {analysis['health_score']:.1f}/100")
        print(f"  🚧 瓶颈数量: {analysis['bottlenecks_count']}")
        print(f"  💡 建议数量: {analysis['recommendations_count']}")
        print(f"  ⚠️ 风险区域: {analysis['risk_areas_count']}")
    else:
        print(f"  ❌ 监控失败: {result.get('message', '未知错误')}")

    print("\n📄 详细报告已保存到: test_logs/performance_report.md")
    print("\n✅ 性能监控体系运行完成")


if __name__ == "__main__":
    main()
