#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025系统性能基准测试脚本

执行单用户场景下的性能基准测试，建立系统性能基线。
"""

import time
import requests
import statistics
import json
import logging
from typing import Dict, List, Any
from datetime import datetime
import psutil
import os
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PerformanceBaselineTester:
    """性能基准测试器"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'test_environment': {
                'base_url': base_url,
                'python_version': os.sys.version,
                'system_info': self._get_system_info()
            },
            'api_tests': {},
            'database_tests': {},
            'cache_tests': {},
            'resource_usage': {}
        }

    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'disk_usage': psutil.disk_usage('/').percent
        }

    def _measure_response_time(self, func, *args, **kwargs) -> Dict[str, Any]:
        """测量函数执行时间和资源使用"""
        # 记录开始时的资源使用
        start_cpu = psutil.cpu_percent()
        start_memory = psutil.virtual_memory().percent

        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)

        end_time = time.time()
        response_time = (end_time - start_time) * 1000  # 转换为毫秒

        # 记录结束时的资源使用
        end_cpu = psutil.cpu_percent()
        end_memory = psutil.virtual_memory().percent

        return {
            'response_time_ms': response_time,
            'success': success,
            'error': error,
            'cpu_usage_start': start_cpu,
            'cpu_usage_end': end_cpu,
            'memory_usage_start': start_memory,
            'memory_usage_end': end_memory,
            'result': result
        }

    def test_health_check(self) -> Dict[str, Any]:
        """测试健康检查接口"""
        def health_request():
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()

        return self._measure_response_time(health_request)

    def test_market_data_api(self) -> Dict[str, Any]:
        """测试市场数据API"""
        def market_data_request():
            response = self.session.get(f"{self.base_url}/api/market/data")
            response.raise_for_status()
            return response.json()

        return self._measure_response_time(market_data_request)

    def test_user_registration(self) -> Dict[str, Any]:
        """测试用户注册"""
        def registration_request():
            user_data = {
                "username": f"test_user_{int(time.time())}",
                "email": f"test_{int(time.time())}@example.com",
                "password": "TestPassword123!",
                "initial_balance": 10000.0
            }
            response = self.session.post(
                f"{self.base_url}/api/user/register",
                json=user_data
            )
            response.raise_for_status()
            return response.json()

        return self._measure_response_time(registration_request)

    def test_portfolio_balance(self) -> Dict[str, Any]:
        """测试投资组合余额查询"""
        # 首先需要登录获取token
        login_data = {
            "username": "test_user",
            "password": "TestPassword123!"
        }

        try:
            login_response = self.session.post(
                f"{self.base_url}/api/auth/login",
                json=login_data
            )
            if login_response.status_code == 200:
                token = login_response.json().get('access_token')
                self.session.headers.update({'Authorization': f'Bearer {token}'})
        except:
            pass  # 登录失败，继续测试

        def portfolio_request():
            response = self.session.get(f"{self.base_url}/api/portfolio/balance")
            response.raise_for_status()
            return response.json()

        return self._measure_response_time(portfolio_request)

    def test_trading_order(self) -> Dict[str, Any]:
        """测试交易订单"""
        # 确保已登录
        if 'Authorization' not in self.session.headers:
            self.test_portfolio_balance()  # 尝试登录

        def order_request():
            order_data = {
                "symbol": "AAPL",
                "quantity": 10,
                "order_type": "market",
                "side": "buy",
                "price": 150.0
            }
            response = self.session.post(
                f"{self.base_url}/api/trading/order",
                json=order_data
            )
            response.raise_for_status()
            return response.json()

        return self._measure_response_time(order_request)

    def run_multiple_tests(self, test_func, iterations: int = 10) -> Dict[str, Any]:
        """运行多次测试并统计结果"""
        results = []
        for i in range(iterations):
            result = test_func()
            results.append(result)
            time.sleep(0.1)  # 短暂延迟避免过于频繁的请求

        response_times = [r['response_time_ms'] for r in results if r['success']]
        success_rate = sum(1 for r in results if r['success']) / len(results)

        return {
            'iterations': iterations,
            'success_rate': success_rate,
            'response_times': {
                'min': min(response_times) if response_times else None,
                'max': max(response_times) if response_times else None,
                'mean': statistics.mean(response_times) if response_times else None,
                'median': statistics.median(response_times) if response_times else None,
                'p95': sorted(response_times)[int(len(response_times) * 0.95)] if len(response_times) >= 20 else None
            },
            'individual_results': results
        }

    def run_all_baseline_tests(self) -> Dict[str, Any]:
        """运行所有基准测试"""
        logger.info("开始执行性能基准测试...")

        # API测试
        logger.info("执行健康检查测试...")
        self.results['api_tests']['health_check'] = self.run_multiple_tests(
            self.test_health_check, 5
        )

        logger.info("执行市场数据API测试...")
        self.results['api_tests']['market_data'] = self.run_multiple_tests(
            self.test_market_data_api, 10
        )

        logger.info("执行用户注册测试...")
        self.results['api_tests']['user_registration'] = self.run_multiple_tests(
            self.test_user_registration, 5
        )

        logger.info("执行投资组合查询测试...")
        self.results['api_tests']['portfolio_balance'] = self.run_multiple_tests(
            self.test_portfolio_balance, 10
        )

        logger.info("执行交易订单测试...")
        self.results['api_tests']['trading_order'] = self.run_multiple_tests(
            self.test_trading_order, 5
        )

        # 资源使用总结
        self.results['resource_usage'] = self._get_system_info()

        logger.info("性能基准测试完成")
        return self.results

    def save_results(self, output_file: str):
        """保存测试结果"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        logger.info(f"测试结果已保存到: {output_file}")

    def print_summary(self):
        """打印测试总结"""
        print("\n" + "="*80)
        print("🏁 RQA2025系统性能基准测试结果总结")
        print("="*80)

        print(f"测试时间: {self.results['timestamp']}")
        print(f"测试环境: {self.results['test_environment']['base_url']}")

        print("\n📊 API性能测试结果:")
        for test_name, test_result in self.results['api_tests'].items():
            success_rate = test_result['success_rate'] * 100
            mean_time = test_result['response_times']['mean']
            print(f"  • {test_name}: 成功率 {success_rate:.1f}%, 平均响应时间 {mean_time:.2f}ms")

        print("\n💻 系统资源使用:")
        resource = self.results['resource_usage']
        print(f"  • CPU核心数: {resource['cpu_count']}")
        print(f"  • CPU使用率: {resource['cpu_percent']:.1f}%")
        print(f"  • 内存使用率: {resource['memory_percent']:.1f}%")
        print(f"  • 磁盘使用率: {resource['disk_usage_percent']:.1f}%")
        print("\n🎯 性能评估:")
        all_response_times = []
        for test_result in self.results['api_tests'].values():
            if test_result['response_times']['mean']:
                all_response_times.append(test_result['response_times']['mean'])

        if all_response_times:
            avg_response_time = statistics.mean(all_response_times)
            print(f"  • 平均响应时间: {avg_response_time:.2f}ms")
            if avg_response_time < 200:
                print("  ✅ 性能表现优秀")
            elif avg_response_time < 500:
                print("  ⚠️ 性能表现良好，需要优化")
            else:
                print("  ❌ 性能表现不佳，需要重点优化")

        print("="*80)

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='RQA2025系统性能基准测试')
    parser.add_argument('--url', default='http://localhost:8000',
                       help='系统基础URL')
    parser.add_argument('--output', default='performance_baseline_results.json',
                       help='输出文件路径')
    parser.add_argument('--iterations', type=int, default=10,
                       help='每个测试的迭代次数')

    args = parser.parse_args()

    # 检查系统是否运行
    try:
        response = requests.get(f"{args.url}/health", timeout=5)
        if response.status_code != 200:
            logger.error(f"系统健康检查失败: HTTP {response.status_code}")
            return
    except requests.exceptions.RequestException as e:
        logger.error(f"无法连接到系统: {e}")
        logger.error("请确保RQA2025系统正在运行在指定的URL")
        return

    # 执行测试
    tester = PerformanceBaselineTester(args.url)
    results = tester.run_all_baseline_tests()

    # 保存结果
    tester.save_results(args.output)

    # 打印总结
    tester.print_summary()

    logger.info("性能基准测试执行完成")

if __name__ == "__main__":
    main()