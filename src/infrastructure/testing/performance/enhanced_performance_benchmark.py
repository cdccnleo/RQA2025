#!/usr/bin/env python3
"""
RQA2025 增强版性能基准测试框架 - 主框架
"""

from .enhanced_performance_benchmark_core import *
from typing import Dict, List, Any, Callable


class PerformanceBenchmarkFramework:
    """增强版性能基准测试框架"""

    def __init__(self, output_dir: str = "reports/performance_benchmark_enhanced"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 设置日志
        self.logger = self._setup_logger()

        # 初始化测量器
        self.collector = PerformanceCollector()
        self.latency_measurer = LatencyMeasurer()
        self.throughput_measurer = ThroughputMeasurer()
        self.concurrency_tester = ConcurrencyTester()

        # 性能阈值定义
        self.thresholds = self._define_performance_thresholds()

        # 测试套件注册表
        self.test_suites = {}

    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            # 文件日志
            file_handler = logging.FileHandler(self.output_dir / "performance_test.log")
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

    def _define_performance_thresholds(self) -> Dict[TestCategory, PerformanceThreshold]:
        """定义性能阈值"""
        thresholds = {}

        # 核心服务层阈值
        thresholds[TestCategory.CORE_SERVICE] = PerformanceThreshold(
            test_category=TestCategory.CORE_SERVICE,
            latency_p50_ms={
                PerformanceLevel.EXCELLENT: 0.5,
                PerformanceLevel.GOOD: 1.0,
                PerformanceLevel.ACCEPTABLE: 5.0,
                PerformanceLevel.POOR: 10.0,
                PerformanceLevel.CRITICAL: 50.0
            },
            latency_p95_ms={
                PerformanceLevel.EXCELLENT: 2.0,
                PerformanceLevel.GOOD: 5.0,
                PerformanceLevel.ACCEPTABLE: 20.0,
                PerformanceLevel.POOR: 50.0,
                PerformanceLevel.CRITICAL: 200.0
            },
            latency_p99_ms={
                PerformanceLevel.EXCELLENT: 5.0,
                PerformanceLevel.GOOD: 10.0,
                PerformanceLevel.ACCEPTABLE: 50.0,
                PerformanceLevel.POOR: 100.0,
                PerformanceLevel.CRITICAL: 500.0
            },
            min_throughput_ops_per_sec={
                PerformanceLevel.EXCELLENT: 50000.0,
                PerformanceLevel.GOOD: 10000.0,
                PerformanceLevel.ACCEPTABLE: 1000.0,
                PerformanceLevel.POOR: 100.0,
                PerformanceLevel.CRITICAL: 10.0
            },
            max_cpu_usage_percent={
                PerformanceLevel.EXCELLENT: 30.0,
                PerformanceLevel.GOOD: 50.0,
                PerformanceLevel.ACCEPTABLE: 70.0,
                PerformanceLevel.POOR: 85.0,
                PerformanceLevel.CRITICAL: 95.0
            },
            max_memory_usage_mb={
                PerformanceLevel.EXCELLENT: 100.0,
                PerformanceLevel.GOOD: 200.0,
                PerformanceLevel.ACCEPTABLE: 500.0,
                PerformanceLevel.POOR: 1000.0,
                PerformanceLevel.CRITICAL: 2000.0
            },
            max_error_rate_percent={
                PerformanceLevel.EXCELLENT: 0.01,
                PerformanceLevel.GOOD: 0.1,
                PerformanceLevel.ACCEPTABLE: 1.0,
                PerformanceLevel.POOR: 5.0,
                PerformanceLevel.CRITICAL: 10.0
            }
        )

        # 交易系统阈值
        thresholds[TestCategory.TRADING_SYSTEM] = PerformanceThreshold(
            test_category=TestCategory.TRADING_SYSTEM,
            latency_p50_ms={
                PerformanceLevel.EXCELLENT: 0.1,
                PerformanceLevel.GOOD: 0.5,
                PerformanceLevel.ACCEPTABLE: 1.0,
                PerformanceLevel.POOR: 2.0,
                PerformanceLevel.CRITICAL: 10.0
            },
            latency_p95_ms={
                PerformanceLevel.EXCELLENT: 0.5,
                PerformanceLevel.GOOD: 2.0,
                PerformanceLevel.ACCEPTABLE: 5.0,
                PerformanceLevel.POOR: 10.0,
                PerformanceLevel.CRITICAL: 50.0
            },
            latency_p99_ms={
                PerformanceLevel.EXCELLENT: 1.0,
                PerformanceLevel.GOOD: 5.0,
                PerformanceLevel.ACCEPTABLE: 10.0,
                PerformanceLevel.POOR: 20.0,
                PerformanceLevel.CRITICAL: 100.0
            },
            min_throughput_ops_per_sec={
                PerformanceLevel.EXCELLENT: 100000.0,
                PerformanceLevel.GOOD: 50000.0,
                PerformanceLevel.ACCEPTABLE: 10000.0,
                PerformanceLevel.POOR: 1000.0,
                PerformanceLevel.CRITICAL: 100.0
            },
            max_cpu_usage_percent={
                PerformanceLevel.EXCELLENT: 20.0,
                PerformanceLevel.GOOD: 40.0,
                PerformanceLevel.ACCEPTABLE: 60.0,
                PerformanceLevel.POOR: 80.0,
                PerformanceLevel.CRITICAL: 95.0
            },
            max_memory_usage_mb={
                PerformanceLevel.EXCELLENT: 50.0,
                PerformanceLevel.GOOD: 100.0,
                PerformanceLevel.ACCEPTABLE: 200.0,
                PerformanceLevel.POOR: 500.0,
                PerformanceLevel.CRITICAL: 1000.0
            },
            max_error_rate_percent={
                PerformanceLevel.EXCELLENT: 0.001,
                PerformanceLevel.GOOD: 0.01,
                PerformanceLevel.ACCEPTABLE: 0.1,
                PerformanceLevel.POOR: 1.0,
                PerformanceLevel.CRITICAL: 5.0
            }
        )

        return thresholds

    def register_test_suite(self, name: str, test_func: Callable,
                            category: TestCategory, config: Optional[Dict[str, Any]] = None):
        """注册测试套件"""
        self.test_suites[name] = {
            'func': test_func,
            'category': category,
            'config': config or {}
        }
        self.logger.info(f"注册测试套件: {name} ({category.value})")

    def run_benchmark_suite(self, suite_name: str, iterations: int = 1000,
                            warmup_iterations: int = 100,
                            concurrent_users: Optional[List[int]] = None) -> BenchmarkResult:
        """运行基准测试套件"""
        if suite_name not in self.test_suites:
            raise ValueError(f"未找到测试套件: {suite_name}")

        suite_info = self.test_suites[suite_name]
        test_func = suite_info['func']
        category = suite_info['category']
        config = suite_info['config']

        self.logger.info(f"开始运行基准测试套件: {suite_name}")

        start_time = time.time()
        metrics_list = []
        passed_tests = 0
        failed_tests = 0

        # 预热
        self.logger.info(f"预热阶段: {warmup_iterations} 次迭代")
        for _ in range(warmup_iterations):
            try:
                test_func()
            except Exception as e:
                self.logger.warning(f"预热迭代失败: {e}")

        # 重置测量器
        self.latency_measurer.reset()
        self.throughput_measurer.start()

        # 开始性能监控
        self.collector.start_monitoring()

        try:
            # 基础性能测试
            metrics = self._run_basic_performance_test(
                test_func, suite_name, category, iterations
            )
            metrics_list.append(metrics)
            passed_tests += 1

            # 并发性能测试
            if concurrent_users:
                for users in concurrent_users:
                    try:
                        metrics = self._run_concurrent_performance_test(
                            test_func, suite_name, category, users, iterations // 10
                        )
                        metrics_list.append(metrics)
                        passed_tests += 1
                    except Exception as e:
                        self.logger.error(f"并发测试失败 (users={users}): {e}")
                        failed_tests += 1

        except Exception as e:
            self.logger.error(f"基础性能测试失败: {e}")
            failed_tests += 1

        finally:
            # 停止性能监控
            monitor_data = self.collector.stop_monitoring()

        end_time = time.time()
        execution_time = end_time - start_time

        # 评估性能水平
        performance_level = self._evaluate_performance_level(metrics_list, category)

        # 生成建议
        recommendations = self._generate_recommendations(metrics_list, category, monitor_data)

        # 创建结果
        result = BenchmarkResult(
            test_suite_name=suite_name,
            execution_time=execution_time,
            total_tests=passed_tests + failed_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            performance_level=performance_level,
            metrics=metrics_list,
            summary={
                'category': category.value,
                'iterations': iterations,
                'warmup_iterations': warmup_iterations,
                'monitor_data': monitor_data,
                'config': config
            },
            recommendations=recommendations
        )

        self.logger.info(f"基准测试套件完成: {suite_name}, 性能水平: {performance_level.value}")

        return result

    def _run_basic_performance_test(self, test_func: Callable, test_name: str,
                                    category: TestCategory, iterations: int) -> PerformanceMetrics:
        """运行基础性能测试"""
        self.logger.info(f"运行基础性能测试: {iterations} 次迭代")

        error_count = 0

        for i in range(iterations):
            try:
                _, latency = self.latency_measurer.measure(test_func)
                self.throughput_measurer.record_operation()
                self.throughput_measurer.record_request()
            except Exception as e:
                error_count += 1
                if error_count <= 10:  # 只记录前10个错误
                    self.logger.warning(f"测试迭代失败 ({i+1}/{iterations}): {e}")

        # 获取延迟统计
        latency_stats = self.latency_measurer.get_percentiles()

        # 获取吞吐量统计
        throughput_stats = self.throughput_measurer.get_throughput()

        # 获取资源使用情况
        process = psutil.Process()
        cpu_percent = process.cpu_percent()
        memory_mb = process.memory_info().rss / (1024 * 1024)

        # 计算错误率
        error_rate = (error_count / iterations) * 100
        availability = 100 - error_rate

        return PerformanceMetrics(
            test_name=f"{test_name}_basic",
            test_category=category,
            timestamp=time.time(),
            latency_p50=latency_stats.get('p50', 0),
            latency_p95=latency_stats.get('p95', 0),
            latency_p99=latency_stats.get('p99', 0),
            latency_p999=latency_stats.get('p999', 0),
            throughput_ops_per_sec=throughput_stats.get('ops_per_sec', 0),
            requests_per_sec=throughput_stats.get('requests_per_sec', 0),
            cpu_usage_percent=cpu_percent,
            memory_usage_mb=memory_mb,
            disk_io_mb_per_sec=0,
            network_io_mb_per_sec=0,
            concurrent_users=1,
            max_concurrent_supported=1,
            error_rate_percent=error_rate,
            availability_percent=availability,
            metadata={
                'iterations': iterations,
                'error_count': error_count,
                'latency_details': latency_stats,
                'throughput_details': throughput_stats
            }
        )

    def _run_concurrent_performance_test(self, test_func: Callable, test_name: str,
                                         category: TestCategory, concurrent_users: int,
                                         iterations_per_user: int) -> PerformanceMetrics:
        """运行并发性能测试"""
        self.logger.info(f"运行并发性能测试: {concurrent_users} 用户, {iterations_per_user} 次/用户")

        # 准备任务参数
        args_list = [()] * (concurrent_users * iterations_per_user)

        # 运行并发测试
        concurrent_result = self.concurrency_tester.test_concurrent_execution(
            test_func, args_list, concurrent_users
        )

        # 计算指标
        total_operations = concurrent_result['successful_tasks']
        duration = concurrent_result['duration']
        error_rate = concurrent_result['error_rate']

        # 估算延迟（基于吞吐量）
        if total_operations > 0 and duration > 0:
            avg_latency_ms = (duration * 1000) / total_operations
        else:
            avg_latency_ms = 0

        return PerformanceMetrics(
            test_name=f"{test_name}_concurrent_{concurrent_users}",
            test_category=category,
            timestamp=time.time(),
            latency_p50=avg_latency_ms,
            latency_p95=avg_latency_ms * 2,
            latency_p99=avg_latency_ms * 3,
            latency_p999=avg_latency_ms * 5,
            throughput_ops_per_sec=concurrent_result.get('throughput', 0),
            requests_per_sec=concurrent_result.get('throughput', 0),
            cpu_usage_percent=0,
            memory_usage_mb=0,
            disk_io_mb_per_sec=0,
            network_io_mb_per_sec=0,
            concurrent_users=concurrent_users,
            max_concurrent_supported=concurrent_users,
            error_rate_percent=error_rate,
            availability_percent=100 - error_rate,
            metadata=concurrent_result
        )

    def _evaluate_performance_level(self, metrics_list: List[PerformanceMetrics],
                                    category: TestCategory) -> PerformanceLevel:
        """评估性能水平"""
        if category not in self.thresholds:
            return PerformanceLevel.ACCEPTABLE

        threshold = self.thresholds[category]

        # 基于主要指标评估
        if metrics_list:
            primary_metrics = metrics_list[0]  # 使用基础测试结果

            # 检查延迟指标
            if primary_metrics.latency_p50 <= threshold.latency_p50_ms[PerformanceLevel.EXCELLENT]:
                return PerformanceLevel.EXCELLENT
            elif primary_metrics.latency_p50 <= threshold.latency_p50_ms[PerformanceLevel.GOOD]:
                return PerformanceLevel.GOOD
            elif primary_metrics.latency_p50 <= threshold.latency_p50_ms[PerformanceLevel.ACCEPTABLE]:
                return PerformanceLevel.ACCEPTABLE
            elif primary_metrics.latency_p50 <= threshold.latency_p50_ms[PerformanceLevel.POOR]:
                return PerformanceLevel.POOR
            else:
                return PerformanceLevel.CRITICAL

        return PerformanceLevel.ACCEPTABLE

    def _generate_recommendations(self, metrics_list: List[PerformanceMetrics],
                                  category: TestCategory, monitor_data: Dict[str, Any]) -> List[str]:
        """生成性能优化建议"""
        recommendations = []

        if not metrics_list:
            return recommendations

        primary_metrics = metrics_list[0]

        # 延迟相关建议
        if primary_metrics.latency_p99 > 100:  # 超过100ms
            recommendations.append("考虑优化算法复杂度以减少P99延迟")

        # 内存相关建议
        if primary_metrics.memory_usage_mb > 500:
            recommendations.append("内存使用较高，建议检查内存泄漏和优化数据结构")

        # CPU相关建议
        if primary_metrics.cpu_usage_percent > 80:
            recommendations.append("CPU使用率较高，建议优化计算密集型操作")

        # 错误率相关建议
        if primary_metrics.error_rate_percent > 1:
            recommendations.append("错误率较高，建议增强错误处理和重试机制")

        # 吞吐量相关建议
        if primary_metrics.throughput_ops_per_sec < 1000:
            recommendations.append("吞吐量较低，考虑并行处理和批量操作")

        return recommendations
