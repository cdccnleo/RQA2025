#!/usr/bin/env python3
"""
优化的压力测试脚本，测试优化后的API服务器性能
"""

import os
import sys
import time
import json
import asyncio
import aiohttp
import threading
import statistics
import logging
import requests
from typing import Dict, List, Any, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class OptimizedStressTestConfig:
    """优化的压力测试配置"""
    base_url: str = "http://localhost:5000"
    concurrent_users: int = 20  # 增加并发数
    requests_per_user: int = 50  # 增加请求数
    test_duration_seconds: int = 60
    target_rps: int = 100  # 提高目标RPS
    warmup_requests: int = 10  # 预热请求数

@dataclass
class TestResult:
    """测试结果"""
    url: str
    method: str
    status_code: int
    response_time: float
    timestamp: float
    error: Optional[str] = None

class OptimizedStressTestRunner:
    """优化的压力测试运行器"""
    
    def __init__(self, config: OptimizedStressTestConfig):
        self.config = config
        self.results: List[TestResult] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.lock = threading.Lock()
        
    async def run_stress_test(self) -> Dict[str, Any]:
        """运行优化的压力测试"""
        logger.info(f"开始优化压力测试: {self.config.concurrent_users} 并发用户, "
                   f"{self.config.requests_per_user} 请求/用户")
        
        # 检查API服务器
        if not self._check_api_server():
            return {"error": "API服务器未运行"}
        
        # 预热阶段
        logger.info("开始预热阶段...")
        await self._warmup_phase()
        
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
    
    def _check_api_server(self) -> bool:
        """检查API服务器是否运行"""
        try:
            response = requests.get(f"{self.config.base_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    async def _warmup_phase(self):
        """预热阶段，填充缓存"""
        async with aiohttp.ClientSession() as session:
            warmup_tasks = []
            for i in range(self.config.warmup_requests):
                # 预热健康检查
                task = session.get(f"{self.config.base_url}/health")
                warmup_tasks.append(task)
                
                # 预热模型列表
                task = session.get(f"{self.config.base_url}/api/v1/models")
                warmup_tasks.append(task)
                
                # 预热分析接口
                data = {"text": f"预热文本 {i}", "model_name": "finbert"}
                task = session.post(f"{self.config.base_url}/api/v1/analyze", json=data)
                warmup_tasks.append(task)
            
            await asyncio.gather(*warmup_tasks, return_exceptions=True)
        
        logger.info("预热阶段完成")
    
    async def _user_workload(self, user_id: int):
        """单个用户的工作负载"""
        # 使用连接池优化性能
        connector = aiohttp.TCPConnector(
            limit=200,  # 增加连接池大小
            limit_per_host=50,  # 增加每个主机的连接数
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=30
        )
        
        timeout = aiohttp.ClientTimeout(total=5, connect=2)
        
        async with aiohttp.ClientSession(
            connector=connector, 
            timeout=timeout,
            headers={'User-Agent': f'OptimizedStressTest-{user_id}'}
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
                        data = self._generate_test_data(user_id, request_id)
                        async with session.post(endpoint["url"], json=data) as response:
                            response_time = time.time() - start_time
                            await self._record_result(
                                endpoint["url"], "POST", response.status,
                                response_time, None
                            )
                    
                    # 减少延迟，提高并发
                    await asyncio.sleep(0.001 + (user_id % 5) * 0.0001)
                    
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
            {"url": f"{self.config.base_url}/api/v1/cache/stats", "method": "GET"},
        ]
        return endpoints[hash(str(time.time())) % len(endpoints)]
    
    def _generate_test_data(self, user_id: int, request_id: int) -> Dict[str, Any]:
        """生成测试数据"""
        return {
            "text": f"优化压力测试文本 {user_id}_{request_id}_{time.time()}",
            "task_type": "classification",
            "model_name": "finbert",
            "user_id": user_id,
            "request_id": request_id
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
        """生成优化的压力测试报告"""
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
        
        # 性能改进对比
        performance_improvement = self._calculate_performance_improvement(
            avg_response_time, error_rate, requests_per_second
        )
        
        report = {
            "test_config": {
                "concurrent_users": self.config.concurrent_users,
                "requests_per_user": self.config.requests_per_user,
                "test_duration_seconds": test_duration,
                "target_rps": self.config.target_rps,
                "warmup_requests": self.config.warmup_requests
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
                for r in errors[:10]  # 只显示前10个错误
            ],
            "optimization_suggestions": optimization_suggestions,
            "performance_improvement": performance_improvement
        }
        
        return report
    
    def _generate_optimization_suggestions(self, avg_response_time: float, 
                                         error_rate: float, rps: float) -> List[str]:
        """生成优化建议"""
        suggestions = []
        
        if avg_response_time > 0.1:  # 响应时间超过100ms
            suggestions.append("响应时间仍然较高，建议进一步优化缓存策略")
            suggestions.append("考虑使用Redis等外部缓存")
            suggestions.append("优化数据库查询和模型推理性能")
        
        if error_rate > 2.0:  # 错误率超过2%
            suggestions.append("错误率需要进一步降低，建议增加重试机制")
            suggestions.append("优化错误处理和资源分配")
            suggestions.append("增加连接池大小和超时设置")
        
        if rps < 50:  # 吞吐量低于50 RPS
            suggestions.append("吞吐量需要提升，建议增加服务器资源")
            suggestions.append("考虑使用负载均衡和集群部署")
            suggestions.append("优化网络I/O和并发处理")
        
        if not suggestions:
            suggestions.append("系统性能良好，可以进一步增加负载测试")
            suggestions.append("考虑实现自动扩缩容机制")
        
        return suggestions
    
    def _calculate_performance_improvement(self, avg_response_time: float, 
                                         error_rate: float, rps: float) -> Dict[str, Any]:
        """计算性能改进"""
        # 基准值（优化前）
        baseline_response_time = 0.305  # 305ms
        baseline_error_rate = 18.0  # 18%
        baseline_rps = 11.32  # 11.32 RPS
        
        # 计算改进百分比
        response_time_improvement = ((baseline_response_time - avg_response_time) / baseline_response_time) * 100
        error_rate_improvement = ((baseline_error_rate - error_rate) / baseline_error_rate) * 100
        rps_improvement = ((rps - baseline_rps) / baseline_rps) * 100
        
        return {
            "response_time_improvement_percent": response_time_improvement,
            "error_rate_improvement_percent": error_rate_improvement,
            "rps_improvement_percent": rps_improvement,
            "overall_improvement": (response_time_improvement + error_rate_improvement + rps_improvement) / 3
        }
    
    def save_report(self, report: Dict[str, Any], filename: str = "optimized_stress_test_report.json"):
        """保存测试报告"""
        # 确保日志目录存在
        log_dir = project_root / "test_logs"
        log_dir.mkdir(exist_ok=True)
        
        report_path = log_dir / filename
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"优化压力测试报告已保存到: {report_path}")

def run_stress_test_sync(config: OptimizedStressTestConfig) -> Dict[str, Any]:
    """同步运行优化的压力测试"""
    runner = OptimizedStressTestRunner(config)
    return asyncio.run(runner.run_stress_test())

def main():
    """主函数"""
    logger.info("启动RQA2025优化压力测试...")
    
    # 配置优化的压力测试
    config = OptimizedStressTestConfig()
    
    # 运行压力测试
    report = run_stress_test_sync(config)
    
    # 保存报告
    runner = OptimizedStressTestRunner(config)
    runner.save_report(report)
    
    # 输出结果摘要
    logger.info("优化压力测试完成!")
    logger.info(f"总请求数: {report.get('summary', {}).get('total_requests', 0)}")
    logger.info(f"成功率: {100 - report.get('summary', {}).get('error_rate_percent', 100):.2f}%")
    logger.info(f"平均响应时间: {report.get('performance', {}).get('avg_response_time_ms', 0):.2f}ms")
    logger.info(f"P95响应时间: {report.get('performance', {}).get('p95_response_time_ms', 0):.2f}ms")
    logger.info(f"吞吐量: {report.get('summary', {}).get('requests_per_second', 0):.2f} RPS")
    
    # 显示性能改进
    improvement = report.get('performance_improvement', {})
    if improvement:
        logger.info("性能改进:")
        logger.info(f"  响应时间改进: {improvement.get('response_time_improvement_percent', 0):.1f}%")
        logger.info(f"  错误率改进: {improvement.get('error_rate_improvement_percent', 0):.1f}%")
        logger.info(f"  吞吐量改进: {improvement.get('rps_improvement_percent', 0):.1f}%")
        logger.info(f"  总体改进: {improvement.get('overall_improvement', 0):.1f}%")
    
    # 显示优化建议
    suggestions = report.get('optimization_suggestions', [])
    if suggestions:
        logger.info("优化建议:")
        for suggestion in suggestions:
            logger.info(f"  - {suggestion}")
    
    success_rate = 100 - report.get('summary', {}).get('error_rate_percent', 100)
    if success_rate < 95:
        logger.warning("⚠️ 压力测试未完全通过，请检查系统性能")
    else:
        logger.info("✅ 优化压力测试通过!")

if __name__ == "__main__":
    main() 