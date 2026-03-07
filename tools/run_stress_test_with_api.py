#!/usr/bin/env python3
"""
RQA2025 改进的压力测试脚本
包含API服务器启动、性能监控和优化建议
"""

import sys
import time
import json
import asyncio
import aiohttp
import threading
import statistics
import logging
import subprocess
import requests
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class StressTestConfig:
    """压力测试配置"""
    base_url: str = "http://localhost:5000"
    concurrent_users: int = 10  # 降低并发数进行测试
    requests_per_user: int = 20  # 降低请求数
    test_duration_seconds: int = 60
    ramp_up_time_seconds: int = 10
    target_rps: int = 50  # 降低目标RPS
    api_server_timeout: int = 30  # API服务器启动超时


@dataclass
class TestResult:
    """测试结果"""
    url: str
    method: str
    status_code: int
    response_time: float
    timestamp: float
    error: Optional[str] = None


class APIServerManager:
    """API服务器管理器"""

    def __init__(self, port: int = 5000):
        self.port = port
        self.process = None
        self.server_url = f"http://localhost:{port}"

    def start_server(self) -> bool:
        """启动API服务器"""
        try:
            logger.info(f"启动API服务器在端口 {self.port}...")

            # 启动API服务器
            self.process = subprocess.Popen([
                sys.executable, "-m", "flask", "run",
                "--host", "0.0.0.0", "--port", str(self.port)
            ], cwd=project_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # 等待服务器启动
            for _ in range(self.config.api_server_timeout):
                try:
                    response = requests.get(f"{self.server_url}/health", timeout=1)
                    if response.status_code == 200:
                        logger.info("API服务器启动成功")
                        return True
                except requests.exceptions.RequestException:
                    time.sleep(1)
                    continue

            logger.error("API服务器启动超时")
            return False

        except Exception as e:
            logger.error(f"启动API服务器失败: {e}")
            return False

    def stop_server(self):
        """停止API服务器"""
        if self.process:
            logger.info("停止API服务器...")
            self.process.terminate()
            self.process.wait()
            logger.info("API服务器已停止")


class OptimizedStressTestRunner:
    """优化的压力测试运行器"""

    def __init__(self, config: StressTestConfig):
        self.config = config
        self.results: List[TestResult] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.lock = threading.Lock()
        self.api_manager = APIServerManager()

    async def run_stress_test(self) -> Dict[str, Any]:
        """运行压力测试"""
        logger.info(f"开始优化压力测试: {self.config.concurrent_users} 并发用户, "
                    f"{self.config.requests_per_user} 请求/用户")

        # 启动API服务器
        if not self.api_manager.start_server():
            return {"error": "API服务器启动失败"}

        try:
            self.start_time = time.time()

            # 创建测试任务
            tasks = []
            for user_id in range(self.config.concurrent_users):
                task = asyncio.create_task(
                    self._user_workload(user_id)
                )
                tasks.append(task)

            # 等待所有任务完成
            await asyncio.gather(*tasks)

            self.end_time = time.time()

            # 生成测试报告
            return self._generate_report()

        finally:
            self.api_manager.stop_server()

    async def _user_workload(self, user_id: int):
        """单个用户的工作负载"""
        # 使用连接池优化性能
        connector = aiohttp.TCPConnector(
            limit=100,  # 连接池大小
            limit_per_host=30,  # 每个主机的连接数
            ttl_dns_cache=300,  # DNS缓存时间
            use_dns_cache=True
        )

        timeout = aiohttp.ClientTimeout(total=10)  # 10秒超时

        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': f'StressTest-{user_id}'}
        ) as session:
            for request_id in range(self.config.requests_per_user):
                # 随机选择测试端点
                endpoint = self._get_random_endpoint()

                try:
                    start_time = time.time()

                    if endpoint["method"] == "GET":
                        async with session.get(endpoint["url"]) as response:
                            response_time = time.time() - start_time
                            await self._record_result(
                                endpoint["url"], "GET", response.status,
                                response_time, None
                            )
                    elif endpoint["method"] == "POST":
                        data = self._generate_test_data()
                        async with session.post(endpoint["url"], json=data) as response:
                            response_time = time.time() - start_time
                            await self._record_result(
                                endpoint["url"], "POST", response.status,
                                response_time, None
                            )

                    # 添加随机延迟模拟真实用户行为
                    await asyncio.sleep(0.01 + (user_id % 10) * 0.001)

                except Exception as e:
                    response_time = time.time() - start_time
                    await self._record_result(
                        endpoint["url"], endpoint["method"], 0,
                        response_time, str(e)
                    )

    def _get_random_endpoint(self) -> Dict[str, str]:
        """获取随机测试端点"""
        endpoints = [
            {"url": f"{self.config.base_url}/health", "method": "GET"},
            {"url": f"{self.config.base_url}/api/v1/analyze", "method": "POST"},
            {"url": f"{self.config.base_url}/api/v1/finetune", "method": "POST"},
            {"url": f"{self.config.base_url}/api/v1/models", "method": "GET"},
            {"url": f"{self.config.base_url}/monitoring/metrics", "method": "GET"},
        ]
        return endpoints[hash(str(time.time())) % len(endpoints)]

    def _generate_test_data(self) -> Dict[str, Any]:
        """生成测试数据"""
        return {
            "text": f"压力测试文本 {time.time()}",
            "task_type": "classification",
            "model_name": "test_model"
        }

    async def _record_result(self, url: str, method: str, status_code: int,
                             response_time: float, error: Optional[str]):
        """记录测试结果"""
        result = TestResult(
            url=url,
            method=method,
            status_code=status_code,
            response_time=response_time,
            timestamp=time.time(),
            error=error
        )

        with self.lock:
            self.results.append(result)

    def _generate_report(self) -> Dict[str, Any]:
        """生成压力测试报告"""
        if not self.results:
            return {"error": "没有测试结果"}

        # 计算基本统计信息
        response_times = [r.response_time for r in self.results]
        status_codes = [r.status_code for r in self.results]
        errors = [r for r in self.results if r.error]

        # 按端点分组统计
        endpoint_stats = {}
        for result in self.results:
            key = f"{result.method} {result.url}"
            if key not in endpoint_stats:
                endpoint_stats[key] = []
            endpoint_stats[key].append(result.response_time)

        # 计算性能指标
        total_requests = len(self.results)
        successful_requests = len([r for r in self.results if r.status_code == 200])
        failed_requests = total_requests - successful_requests
        error_rate = (failed_requests / total_requests) * 100 if total_requests > 0 else 0

        # 响应时间统计
        avg_response_time = statistics.mean(response_times)
        median_response_time = statistics.median(response_times)
        p95_response_time = sorted(response_times)[int(len(response_times) * 0.95)]
        p99_response_time = sorted(response_times)[int(len(response_times) * 0.99)]

        # 吞吐量计算
        test_duration = self.end_time - self.start_time
        requests_per_second = total_requests / test_duration if test_duration > 0 else 0

        # 生成优化建议
        optimization_suggestions = self._generate_optimization_suggestions(
            avg_response_time, error_rate, requests_per_second
        )

        report = {
            "test_config": {
                "concurrent_users": self.config.concurrent_users,
                "requests_per_user": self.config.requests_per_user,
                "test_duration_seconds": test_duration,
                "target_rps": self.config.target_rps
            },
            "summary": {
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "error_rate_percent": error_rate,
                "requests_per_second": requests_per_second
            },
            "performance": {
                "avg_response_time_ms": avg_response_time * 1000,
                "median_response_time_ms": median_response_time * 1000,
                "p95_response_time_ms": p95_response_time * 1000,
                "p99_response_time_ms": p99_response_time * 1000,
                "min_response_time_ms": min(response_times) * 1000,
                "max_response_time_ms": max(response_times) * 1000
            },
            "endpoint_performance": {
                endpoint: {
                    "avg_response_time_ms": statistics.mean(times) * 1000,
                    "median_response_time_ms": statistics.median(times) * 1000,
                    "request_count": len(times)
                }
                for endpoint, times in endpoint_stats.items()
            },
            "errors": [
                {
                    "url": r.url,
                    "method": r.method,
                    "error": r.error,
                    "timestamp": r.timestamp
                }
                for r in errors[:20]  # 只显示前20个错误
            ],
            "optimization_suggestions": optimization_suggestions
        }

        return report

    def _generate_optimization_suggestions(self, avg_response_time: float,
                                           error_rate: float, rps: float) -> List[str]:
        """生成优化建议"""
        suggestions = []

        if avg_response_time > 1.0:  # 响应时间超过1秒
            suggestions.append("响应时间过长，建议优化API处理逻辑")
            suggestions.append("考虑实现缓存机制减少重复计算")
            suggestions.append("优化数据库查询和模型推理性能")

        if error_rate > 5.0:  # 错误率超过5%
            suggestions.append("错误率过高，建议检查API服务器稳定性")
            suggestions.append("增加错误处理和重试机制")
            suggestions.append("优化资源分配和内存管理")

        if rps < 10:  # 吞吐量低于10 RPS
            suggestions.append("吞吐量较低，建议增加并发处理能力")
            suggestions.append("考虑使用异步处理和连接池")
            suggestions.append("优化网络I/O和数据库连接")

        if not suggestions:
            suggestions.append("系统性能良好，可以逐步增加负载测试")

        return suggestions

    def save_report(self, report: Dict[str, Any], filename: str = "optimized_stress_test_report.json"):
        """保存测试报告"""
        # 确保日志目录存在
        log_dir = project_root / "test_logs"
        log_dir.mkdir(exist_ok=True)

        report_path = log_dir / filename
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"压力测试报告已保存到: {report_path}")


def run_stress_test_sync(config: StressTestConfig) -> Dict[str, Any]:
    """同步运行压力测试"""
    runner = OptimizedStressTestRunner(config)
    return asyncio.run(runner.run_stress_test())


def main():
    """主函数"""
    logger.info("启动RQA2025优化压力测试...")

    # 配置压力测试
    config = StressTestConfig()

    # 运行压力测试
    report = run_stress_test_sync(config)

    # 保存报告
    runner = OptimizedStressTestRunner(config)
    runner.save_report(report)

    # 输出结果摘要
    logger.info("压力测试完成!")
    logger.info(f"总请求数: {report.get('summary', {}).get('total_requests', 0)}")
    logger.info(f"成功率: {100 - report.get('summary', {}).get('error_rate_percent', 100):.2f}%")
    logger.info(f"平均响应时间: {report.get('performance', {}).get('avg_response_time_ms', 0):.2f}ms")
    logger.info(f"P95响应时间: {report.get('performance', {}).get('p95_response_time_ms', 0):.2f}ms")
    logger.info(f"吞吐量: {report.get('summary', {}).get('requests_per_second', 0):.2f} RPS")

    # 显示优化建议
    suggestions = report.get('optimization_suggestions', [])
    if suggestions:
        logger.info("优化建议:")
        for suggestion in suggestions:
            logger.info(f"  - {suggestion}")

    success_rate = 100 - report.get('summary', {}).get('error_rate_percent', 100)
    if success_rate < 90:
        logger.warning("⚠️ 压力测试未完全通过，请检查系统性能")
    else:
        logger.info("✅ 压力测试通过!")


if __name__ == "__main__":
    main()
