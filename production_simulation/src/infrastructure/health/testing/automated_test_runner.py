"""
automated_test_runner 模块

提供 automated_test_runner 相关功能和接口。
"""

import logging

import threading
import time

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Callable
"""
自动化测试运行器

提供自动化测试执行功能
"""

logger = logging.getLogger(__name__)


class TestExecutionStatus(Enum):
    """测试执行状态"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


@dataclass
class TestResult:
    """测试结果"""
    test_name: str
    status: TestExecutionStatus
    execution_time: float
    start_time: float
    end_time: float
    error_message: Optional[str] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class AutomatedTestRunner:
    """自动化测试运行器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.test_results: List[TestResult] = []
        self._execution_lock = threading.Lock()
        self._stop_event = threading.Event()

    def add_test(self, test_name: str, test_func: Callable, **kwargs):
        """添加测试到队列"""
        metadata = kwargs.copy()
        metadata['test_func'] = test_func

        with self._execution_lock:
            self.test_results.append(TestResult(
                test_name=test_name,
                status=TestExecutionStatus.PENDING,
                execution_time=0.0,
                start_time=0.0,
                end_time=0.0,
                metadata=metadata
            ))

        logger.info(f"测试已添加到队列: {test_name}")

    def run_tests(self) -> List[TestResult]:
        """运行所有测试"""
        if not self.test_results:
            logger.warning("没有测试需要运行")
            return []

        logger.info(f"开始运行测试套件，测试数量: {len(self.test_results)}")

        try:
            # 顺序执行测试
            for test_result in self.test_results:
                if self._stop_event.is_set():
                    logger.info("测试执行被中断")
                    break

                self._execute_single_test(test_result)

        finally:
            pass

        return self.test_results

    def _execute_single_test(self, test_result: TestResult) -> None:
        """执行单个测试"""
        if test_result.status != TestExecutionStatus.PENDING:
            return

        test_result.status = TestExecutionStatus.RUNNING
        test_result.start_time = time.time()

        try:
            logger.info(f"开始执行测试: {test_result.test_name}")

            # 获取测试函数
            test_func = test_result.metadata.get('test_func')
            if not test_func:
                raise ValueError(f"测试函数未找到: {test_result.test_name}")

            # 执行测试
            start_time = time.time()
            test_func(**{k: v for k, v in test_result.metadata.items() if k != 'test_func'})
            execution_time = time.time() - start_time

            # 更新测试结果
            test_result.end_time = time.time()
            test_result.execution_time = execution_time
            test_result.status = TestExecutionStatus.PASSED

            logger.info(f"测试执行成功: {test_result.test_name}, 耗时: {execution_time:.3f}s")

        except Exception as e:
            test_result.end_time = time.time()
            test_result.execution_time = test_result.end_time - test_result.start_time
            test_result.status = TestExecutionStatus.FAILED
            test_result.error_message = str(e)

            logger.error(f"测试执行失败: {test_result.test_name}, 错误: {e}")

    def stop_execution(self) -> None:
        """停止测试执行"""
        logger.info("收到停止信号，正在停止测试执行...")
        self._stop_event.set()

    def get_execution_status(self) -> Dict[str, Any]:
        """获取执行状态"""
        total = len(self.test_results)
        running = len([r for r in self.test_results if r.status == TestExecutionStatus.RUNNING])
        completed = len([
            r for r in self.test_results
            if r.status in [TestExecutionStatus.PASSED, TestExecutionStatus.FAILED]
        ])

        return {
            'total_tests': total,
            'running_tests': running,
            'completed_tests': completed,
            'pending_tests': total - running - completed,
            'stop_requested': self._stop_event.is_set()
        }

    def check_health(self) -> Dict[str, Any]:
        """执行整体健康检查

        Returns:
            Dict[str, Any]: 健康检查结果
        """
        try:
            logger.info("开始自动化测试运行器健康检查")

            health_checks = {
                "execution_status": self.check_execution_health(),
                "test_configuration": self.check_test_configuration(),
                "performance_metrics": self.check_performance_health(),
                "resource_usage": self.check_resource_usage_health()
            }

            # 综合健康状态
            overall_healthy = all(check.get("healthy", False) for check in health_checks.values())

            result = {
                "healthy": overall_healthy,
                "timestamp": datetime.now().isoformat(),
                "service": "automated_test_runner",
                "checks": health_checks
            }

            if not overall_healthy:
                logger.warning("自动化测试运行器健康检查发现问题")
                result["issues"] = [
                    name for name, check in health_checks.items()
                    if not check.get("healthy", False)
                ]

            logger.info(f"自动化测试运行器健康检查完成，状态: {'健康' if overall_healthy else '异常'}")
            return result

        except Exception as e:
            logger.error(f"自动化测试运行器健康检查失败: {str(e)}", exc_info=True)
            return {
                "healthy": False,
                "timestamp": datetime.now().isoformat(),
                "service": "automated_test_runner",
                "error": str(e)
            }

    def check_execution_health(self) -> Dict[str, Any]:
        """检查测试执行健康状态

        Returns:
            Dict[str, Any]: 执行健康状态检查结果
        """
        try:
            status = self.get_execution_status()
            total_tests = status['total_tests']

            # 检查是否有测试
            has_tests = total_tests > 0

            # 检查执行状态合理性
            execution_sane = (
                status['running_tests'] >= 0 and
                status['completed_tests'] >= 0 and
                status['pending_tests'] >= 0 and
                status['running_tests'] + status['completed_tests'] +
                status['pending_tests'] == total_tests
            )

            # 检查是否异常终止
            abnormal_termination = self._stop_event.is_set() and status['running_tests'] > 0

            return {
                "healthy": has_tests and execution_sane and not abnormal_termination,
                "has_tests": has_tests,
                "execution_sane": execution_sane,
                "abnormal_termination": abnormal_termination,
                "execution_status": status
            }
        except Exception as e:
            logger.error(f"执行健康检查失败: {str(e)}")
            return {"healthy": False, "error": str(e)}

    def check_test_configuration(self) -> Dict[str, Any]:
        """检查测试配置有效性

        Returns:
            Dict[str, Any]: 测试配置检查结果
        """
        try:
            # 检查配置参数
            timeout = self.config.get('timeout', 30)
            max_workers = self.config.get('max_workers', 4)
            retry_count = self.config.get('retry_count', 0)

            # 验证配置合理性
            valid_timeout = 1 <= timeout <= 3600  # 1秒到1小时
            valid_workers = 1 <= max_workers <= 20  # 1到20个工作线程
            valid_retry = 0 <= retry_count <= 10  # 0到10次重试

            # 检查测试函数是否可用
            test_functions_available = len(self.test_results) > 0 or hasattr(self, '_tests')

            return {
                "healthy": valid_timeout and valid_workers and valid_retry and test_functions_available,
                "config_validation": {
                    "timeout_valid": valid_timeout,
                    "workers_valid": valid_workers,
                    "retry_valid": valid_retry
                },
                "test_functions_available": test_functions_available,
                "config": {
                    "timeout": timeout,
                    "max_workers": max_workers,
                    "retry_count": retry_count
                }
            }
        except Exception as e:
            logger.error(f"测试配置检查失败: {str(e)}")
            return {"healthy": False, "error": str(e)}

    def check_performance_health(self) -> Dict[str, Any]:
        """检查性能健康状态

        Returns:
            Dict[str, Any]: 性能健康检查结果
        """
        try:
            if not self.test_results:
                return {"healthy": True, "reason": "no_test_results"}

            # 计算性能指标
            execution_times = [r.execution_time for r in self.test_results if r.execution_time > 0]
            if not execution_times:
                return {"healthy": False, "reason": "no_execution_times"}

            avg_time = sum(execution_times) / len(execution_times)
            max_time = max(execution_times)
            min_time = min(execution_times)

            # 计算成功率
            total_tests = len(self.test_results)
            passed_tests = len([r for r in self.test_results if r.status ==
                               TestExecutionStatus.PASSED])
            success_rate = passed_tests / total_tests if total_tests > 0 else 0

            # 性能阈值检查
            acceptable_avg_time = avg_time < 300  # 平均执行时间 < 5分钟
            acceptable_success_rate = success_rate > 0.5  # 成功率 > 50%

            return {
                "healthy": acceptable_avg_time and acceptable_success_rate,
                "performance_metrics": {
                    "average_time": avg_time,
                    "max_time": max_time,
                    "min_time": min_time,
                    "success_rate": success_rate,
                    "total_tests": total_tests,
                    "passed_tests": passed_tests
                },
                "thresholds": {
                    "acceptable_avg_time": acceptable_avg_time,
                    "acceptable_success_rate": acceptable_success_rate
                }
            }
        except Exception as e:
            logger.error(f"性能健康检查失败: {str(e)}")
            return {"healthy": False, "error": str(e)}

    def check_resource_usage_health(self) -> Dict[str, Any]:
        """检查资源使用健康状态

        Returns:
            Dict[str, Any]: 资源使用健康检查结果
        """
        try:
            # 估算内存使用（简化检查）
            memory_usage = len(self.test_results) * 1024  # 每个测试结果约1KB
            acceptable_memory = memory_usage < 100 * 1024 * 1024  # 100MB

            # 检查线程资源
            thread_count = threading.active_count()
            reasonable_threads = thread_count < 50  # 线程数不应过多

            # 检查停止事件状态
            stop_requested = self._stop_event.is_set()

            return {
                "healthy": acceptable_memory and reasonable_threads,
                "memory_usage_kb": memory_usage / 1024,
                "thread_count": thread_count,
                "stop_requested": stop_requested,
                "resource_limits": {
                    "acceptable_memory": acceptable_memory,
                    "reasonable_threads": reasonable_threads
                }
            }
        except Exception as e:
            logger.error(f"资源使用健康检查失败: {str(e)}")
            return {"healthy": False, "error": str(e)}

    def health_status(self) -> Dict[str, Any]:
        """获取健康状态摘要

        Returns:
            Dict[str, Any]: 健康状态摘要
        """
        try:
            execution_status = self.get_execution_status()
            health_check = self.check_health()

            return {
                "status": "healthy" if health_check["healthy"] else "unhealthy",
                "execution_status": execution_status,
                "health_check": health_check,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"获取健康状态摘要失败: {str(e)}")
            return {"status": "error", "error": str(e)}

    def health_summary(self) -> Dict[str, Any]:
        """获取健康摘要报告

        Returns:
            Dict[str, Any]: 健康摘要报告
        """
        try:
            health_check = self.check_health()
            execution_status = self.get_execution_status()

            # 计算测试统计
            total_tests = len(self.test_results)
            passed = len([r for r in self.test_results if r.status == TestExecutionStatus.PASSED])
            failed = len([r for r in self.test_results if r.status == TestExecutionStatus.FAILED])
            running = len([r for r in self.test_results if r.status == TestExecutionStatus.RUNNING])

            return {
                "overall_health": "healthy" if health_check["healthy"] else "unhealthy",
                "test_statistics": {
                    "total": total_tests,
                    "passed": passed,
                    "failed": failed,
                    "running": running,
                    "success_rate": passed / total_tests if total_tests > 0 else 0
                },
                "execution_status": execution_status,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"获取健康摘要报告失败: {str(e)}")
            return {"overall_health": "error", "error": str(e)}

    def monitor_test_execution(self) -> Dict[str, Any]:
        """监控测试执行状态

        Returns:
            Dict[str, Any]: 测试执行监控结果
        """
        try:
            execution_status = self.get_execution_status()

            # 计算执行进度
            total = execution_status['total_tests']
            completed = execution_status['completed_tests']
            progress = completed / total if total > 0 else 0

            # 检查执行效率
            avg_execution_time = 0
            if self.test_results:
                times = [r.execution_time for r in self.test_results if r.execution_time > 0]
                if times:
                    avg_execution_time = sum(times) / len(times)

            # 检查是否有超时测试
            timeout_tests = len([r for r in self.test_results if r.status ==
                                TestExecutionStatus.TIMEOUT])

            return {
                "healthy": progress > 0 or total == 0,  # 没有测试或有进度则健康
                "execution_progress": {
                    "total": total,
                    "completed": completed,
                    "progress_percentage": progress * 100,
                    "avg_execution_time": avg_execution_time
                },
                "issues": {
                    "timeout_tests": timeout_tests,
                    "stop_requested": execution_status['stop_requested']
                }
            }
        except Exception as e:
            logger.error(f"测试执行监控失败: {str(e)}")
            return {"healthy": False, "error": str(e)}

    def monitor_test_performance(self) -> Dict[str, Any]:
        """监控测试性能指标

        Returns:
            Dict[str, Any]: 性能监控结果
        """
        try:
            if not self.test_results:
                return {"healthy": True, "reason": "no_tests_executed"}

            # 分析性能趋势
            execution_times = [r.execution_time for r in self.test_results if r.execution_time > 0]
            if len(execution_times) < 2:
                return {"healthy": True, "reason": "insufficient_data"}

            # 计算统计信息
            avg_time = sum(execution_times) / len(execution_times)
            median_time = sorted(execution_times)[len(execution_times) // 2]

            # 检查性能异常
            outliers = [t for t in execution_times if t > avg_time * 2]  # 超过平均时间2倍的异常值

            return {
                "healthy": len(outliers) < len(execution_times) * 0.1,  # 异常值少于10%
                "performance_stats": {
                    "average_time": avg_time,
                    "median_time": median_time,
                    "min_time": min(execution_times),
                    "max_time": max(execution_times),
                    "total_tests": len(execution_times)
                },
                "anomalies": {
                    "outlier_count": len(outliers),
                    "outlier_percentage": len(outliers) / len(execution_times) * 100
                }
            }
        except Exception as e:
            logger.error(f"测试性能监控失败: {str(e)}")
            return {"healthy": False, "error": str(e)}

    def validate_test_runner_config(self) -> Dict[str, Any]:
        """验证测试运行器配置

        Returns:
            Dict[str, Any]: 配置验证结果
        """
        try:
            validation_results = {
                "timeout_config": self._validate_timeout_config(),
                "worker_config": self._validate_worker_config(),
                "retry_config": self._validate_retry_config(),
                "test_functions": self._validate_test_functions()
            }

            overall_valid = all(result.get("valid", False)
                                for result in validation_results.values())

            return {
                "valid": overall_valid,
                "validation_results": validation_results,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"测试运行器配置验证失败: {str(e)}")
            return {"valid": False, "error": str(e)}

    def _validate_timeout_config(self) -> Dict[str, Any]:
        """验证超时配置"""
        timeout = self.config.get('timeout', 30)
        valid = 1 <= timeout <= 3600  # 1秒到1小时

        return {
            "valid": valid,
            "current_value": timeout,
            "valid_range": "1-3600 seconds"
        }

    def _validate_worker_config(self) -> Dict[str, Any]:
        """验证工作线程配置"""
        max_workers = self.config.get('max_workers', 4)
        valid = 1 <= max_workers <= 20  # 1到20个线程

        return {
            "valid": valid,
            "current_value": max_workers,
            "valid_range": "1-20 workers"
        }

    def _validate_retry_config(self) -> Dict[str, Any]:
        """验证重试配置"""
        retry_count = self.config.get('retry_count', 0)
        valid = 0 <= retry_count <= 10  # 0到10次重试

        return {
            "valid": valid,
            "current_value": retry_count,
            "valid_range": "0-10 retries"
        }

    def _validate_test_functions(self) -> Dict[str, Any]:
        """验证测试函数"""
        # 检查是否有测试函数注册（简化检查）
        has_tests = len(self.test_results) > 0 or hasattr(self, '_tests')

        return {
            "valid": has_tests,
            "has_registered_tests": has_tests,
            "test_count": len(self.test_results)
        }


def run_tests():
    """运行测试的便捷函数"""
    runner = AutomatedTestRunner()
    return runner.run_tests()
