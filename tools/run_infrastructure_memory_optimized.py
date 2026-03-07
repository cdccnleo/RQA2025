#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
内存优化的基础设施测试运行脚本
专门解决内存暴涨问题
"""

import os
import sys
import subprocess
import argparse
import time
import psutil
from pathlib import Path
from typing import List, Dict, Any
import logging
import gc
import tempfile

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryOptimizedTestRunner:
    """内存优化的测试运行器"""

    def __init__(self, max_memory_mb: int = 1024):
        self.max_memory_mb = max_memory_mb
        self.process = psutil.Process()
        self.temp_dir = None

    def setup_environment(self):
        """设置优化环境"""
        # 设置环境变量
        os.environ.update({
            'LIGHTWEIGHT_TEST': 'true',
            'DISABLE_HEAVY_IMPORTS': 'true',
            'PYTEST_DISABLE_PLUGIN_AUTOLOAD': 'true',
            'MPLBACKEND': 'Agg',
            'DISABLE_LOGGING': 'true',
            'DISABLE_MONITORING': 'true',
            'PYTHONPATH': str(Path(__file__).parent.parent.parent)
        })

        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        os.environ['TEMP_DIR'] = self.temp_dir

    def cleanup(self):
        """清理资源"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)

        # 强制垃圾回收
        gc.collect()

    def get_memory_usage(self) -> float:
        """获取内存使用量(MB)"""
        return self.process.memory_info().rss / 1024 / 1024

    def check_memory_limit(self) -> bool:
        """检查内存限制"""
        memory_mb = self.get_memory_usage()
        if memory_mb > self.max_memory_mb:
            logger.warning(f"内存使用过高: {memory_mb:.1f}MB > {self.max_memory_mb}MB")
            return False
        return True

    def run_single_test_file(self, test_file: str, timeout: int = 120) -> Dict[str, Any]:
        """运行单个测试文件"""
        logger.info(f"运行测试文件: {test_file}")

        # 构建pytest命令
        cmd = [
            sys.executable, "-m", "pytest",
            f"tests/unit/infrastructure/{test_file}",
            "-v",
            "--tb=short",
            "--disable-warnings",
            "--no-header",
            "--no-summary",
            "--maxfail=3",
            "-x",
            "--durations=0",
            "--disable-pytest-warnings",
            "--strict-markers"
        ]

        try:
            # 启动进程
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=os.environ.copy()
            )

            # 监控进程
            start_time = time.time()
            while process.poll() is None:
                if time.time() - start_time > timeout:
                    logger.error(f"测试超时 ({timeout}秒)")
                    process.terminate()
                    return {"success": False, "error": "timeout"}

                # 检查内存使用
                if not self.check_memory_limit():
                    logger.error("内存使用过高，终止测试")
                    process.terminate()
                    return {"success": False, "error": "memory_limit_exceeded"}

                time.sleep(1)

            # 获取输出
            stdout, stderr = process.communicate()

            return {
                "success": process.returncode == 0,
                "returncode": process.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "memory_usage": self.get_memory_usage()
            }

        except Exception as e:
            logger.error(f"运行测试时出错: {e}")
            return {"success": False, "error": str(e)}

    def get_test_files_by_priority(self) -> Dict[str, List[str]]:
        """按优先级分组测试文件"""
        # 轻量级测试文件（优先运行）
        lightweight_files = [
            "test_infrastructure.py",
            "test_infrastructure_core.py",
            "test_version.py",
            "test_metrics.py",
            "test_event.py",
            "test_error_handler.py",
            "test_config_exceptions.py",
            "test_network_exceptions.py",
            "test_storage_exceptions.py",
            "test_alert_manager.py",
            "test_backtest_monitor.py",
            "test_factory.py",
            "test_yaml_loader.py",
            "test_json_loader.py",
            "test_env_loader.py",
            "test_config_loader_service.py",
            "test_behavior_monitor.py",
            "test_storage_monitor.py",
            "test_resource_api.py",
            "test_deployment_manager.py",
            "test_config_version.py",
            "test_connection_pool.py"
        ]

        # 中等复杂度测试文件
        medium_files = [
            "test_monitoring.py",
            "test_logging_coverage.py",
            "test_config_coverage.py",
            "test_deployment_validator.py",
            "test_init_infrastructure.py",
            "test_app_factory.py",
            "test_unified_config_manager.py",
            "test_standard_interfaces.py",
            "test_validators.py",
            "test_performance_monitor.py",
            "test_schema_validator.py",
            "test_resource_dashboard.py",
            "test_model_monitor.py",
            "test_network_monitor.py",
            "test_lock.py",
            "test_gpu_manager.py",
            "test_circuit_breaker.py",
            "test_prometheus_monitor.py",
            "test_quota_manager.py",
            "test_event_service.py",
            "test_data_sync.py",
            "test_load_balancer.py",
            "test_final_deployment_check.py",
            "test_application_monitor.py",
            "test_circuit_breaker_manager.py",
            "test_service_launcher.py",
            "test_visual_monitor.py",
            "test_metrics_collector.py",
            "test_notification.py",
            "test_system_monitor.py",
            "test_resource_manager.py",
            "test_disaster_recovery.py",
            "test_degradation_manager.py",
            "test_thread_management.py",
            "test_circuit_breaker_tester.py",
            "test_market_aware_retry_test.py",
            "test_persistent_error_handler_test.py"
        ]

        # 重量级测试文件（最后运行）
        heavy_files = [
            "test_coverage_improvement.py",
            "test_unified_interface_manager.py",
            "test_unified_hot_reload.py",
            "test_document_management.py",
            "test_optimization_modules.py",
            "test_async_inference_engine_top20.py",
            "test_async_inference_engine.py"
        ]

        return {
            "lightweight": lightweight_files,
            "medium": medium_files,
            "heavy": heavy_files
        }

    def run_tests_by_priority(self) -> Dict[str, Any]:
        """按优先级运行测试"""
        test_groups = self.get_test_files_by_priority()
        results = {}

        for priority, files in test_groups.items():
            logger.info(f"\n=== 运行 {priority.upper()} 测试 ===")
            group_results = []

            for test_file in files:
                logger.info(f"运行: {test_file}")

                # 根据优先级设置不同的内存限制
                if priority == "lightweight":
                    max_memory = 512
                    timeout = 60
                elif priority == "medium":
                    max_memory = 768
                    timeout = 120
                else:  # heavy
                    max_memory = 1024
                    timeout = 180

                self.max_memory_mb = max_memory
                result = self.run_single_test_file(test_file, timeout)
                group_results.append({"file": test_file, "result": result})

                # 强制垃圾回收
                gc.collect()
                time.sleep(1)

                # 检查是否需要停止
                if not result["success"] and "memory_limit_exceeded" in result.get("error", ""):
                    logger.warning(f"内存限制超出，跳过剩余 {priority} 测试")
                    break

            results[priority] = group_results

        return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="内存优化的基础设施测试运行器")
    parser.add_argument("--max-memory", type=int, default=1024, help="最大内存使用(MB)")
    parser.add_argument("--priority", choices=["lightweight",
                        "medium", "heavy"], help="只运行指定优先级的测试")
    parser.add_argument("--file", help="运行指定测试文件")
    parser.add_argument("--timeout", type=int, default=180, help="超时时间(秒)")

    args = parser.parse_args()

    runner = MemoryOptimizedTestRunner(max_memory_mb=args.max_memory)

    try:
        runner.setup_environment()

        if args.file:
            # 运行单个文件
            result = runner.run_single_test_file(args.file, args.timeout)
            if result["success"]:
                logger.info("✓ 测试通过")
            else:
                logger.error("✗ 测试失败")
                if "error" in result:
                    logger.error(f"错误: {result['error']}")

            logger.info(f"内存使用: {result.get('memory_usage', 0):.1f}MB")

        elif args.priority:
            # 运行指定优先级的测试
            test_groups = runner.get_test_files_by_priority()
            if args.priority in test_groups:
                logger.info(f"运行 {args.priority} 优先级测试")
                for test_file in test_groups[args.priority]:
                    result = runner.run_single_test_file(test_file, args.timeout)
                    status = "✓" if result["success"] else "✗"
                    memory = result.get("memory_usage", 0)
                    logger.info(f"  {status} {test_file} (内存: {memory:.1f}MB)")

                    gc.collect()
                    time.sleep(1)
            else:
                logger.error(f"未知优先级: {args.priority}")

        else:
            # 按优先级运行所有测试
            results = runner.run_tests_by_priority()

            # 输出结果
            logger.info("\n=== 测试结果汇总 ===")
            for priority, tests in results.items():
                logger.info(f"\n{priority.upper()} 测试:")
                passed = 0
                failed = 0
                total_memory = 0

                for test in tests:
                    if test["result"]["success"]:
                        passed += 1
                        status = "✓"
                    else:
                        failed += 1
                        status = "✗"

                    memory = test["result"].get("memory_usage", 0)
                    total_memory += memory
                    logger.info(f"  {status} {test['file']} (内存: {memory:.1f}MB)")

                logger.info(f"  通过: {passed}, 失败: {failed}, 总内存: {total_memory:.1f}MB")

    finally:
        runner.cleanup()


if __name__ == "__main__":
    main()
