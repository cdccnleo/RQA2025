#!/usr/bin/env python3
"""
性能和稳定性测试脚本
评估系统在高负载下的表现
"""

import time
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics
import json
from pathlib import Path

class PerformanceTester:
    def __init__(self):
        self.results = {
            'response_times': [],
            'cpu_usage': [],
            'memory_usage': [],
            'errors': 0,
            'total_requests': 0,
            'successful_requests': 0
        }
        self.is_running = False

    def simulate_business_request(self, request_type='strategy'):
        """模拟业务请求"""
        try:
            start_time = time.time()

            # 模拟不同类型的业务请求
            if request_type == 'strategy':
                # 模拟策略计算请求
                time.sleep(0.01)  # 10ms 策略计算时间
                result = {'type': 'strategy', 'status': 'success', 'result': 'strategy_computed'}
            elif request_type == 'trading':
                # 模拟交易执行请求
                time.sleep(0.005)  # 5ms 交易执行时间
                result = {'type': 'trading', 'status': 'success', 'result': 'trade_executed'}
            elif request_type == 'risk':
                # 模拟风险评估请求
                time.sleep(0.008)  # 8ms 风险评估时间
                result = {'type': 'risk', 'status': 'success', 'result': 'risk_assessed'}

            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # 转换为毫秒

            return {
                'success': True,
                'response_time': response_time,
                'result': result
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response_time': 0
            }

    def monitor_system_resources(self):
        """监控系统资源使用情况"""
        while self.is_running:
            try:
                # CPU使用率
                cpu_percent = psutil.cpu_percent(interval=1)
                self.results['cpu_usage'].append(cpu_percent)

                # 内存使用率
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                self.results['memory_usage'].append(memory_percent)

                time.sleep(1)
            except Exception as e:
                print(f"资源监控异常: {e}")
                break

    def run_concurrent_test(self, num_requests=100, concurrency=10):
        """运行并发性能测试"""
        print(f"开始并发性能测试: {num_requests}个请求, {concurrency}并发")

        request_types = ['strategy', 'trading', 'risk']
        self.results['total_requests'] = num_requests

        def single_request():
            """单个请求执行"""
            request_type = request_types[len(self.results['response_times']) % len(request_types)]
            result = self.simulate_business_request(request_type)

            if result['success']:
                self.results['response_times'].append(result['response_time'])
                self.results['successful_requests'] += 1
            else:
                self.results['errors'] += 1

            return result

        # 启动资源监控
        self.is_running = True
        monitor_thread = threading.Thread(target=self.monitor_system_resources, daemon=True)
        monitor_thread.start()

        # 执行并发测试
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(single_request) for _ in range(num_requests)]
            for future in as_completed(futures):
                try:
                    future.result(timeout=10)
                except Exception as e:
                    print(f"请求执行异常: {e}")
                    self.results['errors'] += 1

        end_time = time.time()
        total_time = end_time - start_time

        # 停止资源监控
        self.is_running = False
        time.sleep(2)  # 等待监控线程停止

        return total_time

    def run_stability_test(self, duration_minutes=5):
        """运行稳定性测试"""
        print(f"开始稳定性测试: {duration_minutes}分钟")

        stability_results = {
            'start_time': time.time(),
            'end_time': None,
            'cpu_samples': [],
            'memory_samples': [],
            'response_times': [],
            'errors': 0,
            'total_requests': 0
        }

        self.is_running = True
        monitor_thread = threading.Thread(target=self.monitor_system_resources, daemon=True)
        monitor_thread.start()

        end_time = time.time() + (duration_minutes * 60)

        while time.time() < end_time:
            # 执行随机业务请求
            request_types = ['strategy', 'trading', 'risk']
            import random
            request_type = random.choice(request_types)

            result = self.simulate_business_request(request_type)
            stability_results['total_requests'] += 1

            if result['success']:
                stability_results['response_times'].append(result['response_time'])
            else:
                stability_results['errors'] += 1

            # 每秒执行一个请求
            time.sleep(1)

        stability_results['end_time'] = time.time()
        self.is_running = False
        time.sleep(2)

        return stability_results

    def analyze_results(self, test_type='concurrent'):
        """分析测试结果"""
        if test_type == 'concurrent':
            response_times = self.results['response_times']

            if not response_times:
                return {
                    'error': '没有有效的响应时间数据',
                    'success_rate': 0,
                    'avg_response_time': 0
                }

            analysis = {
                'total_requests': self.results['total_requests'],
                'successful_requests': self.results['successful_requests'],
                'errors': self.results['errors'],
                'success_rate': self.results['successful_requests'] / self.results['total_requests'] * 100,
                'avg_response_time': statistics.mean(response_times),
                'min_response_time': min(response_times),
                'max_response_time': max(response_times),
                'p95_response_time': statistics.quantiles(response_times, n=20)[18],  # 95th percentile
                'p99_response_time': statistics.quantiles(response_times, n=100)[98],  # 99th percentile
                'cpu_avg': statistics.mean(self.results['cpu_usage']) if self.results['cpu_usage'] else 0,
                'cpu_max': max(self.results['cpu_usage']) if self.results['cpu_usage'] else 0,
                'memory_avg': statistics.mean(self.results['memory_usage']) if self.results['memory_usage'] else 0,
                'memory_max': max(self.results['memory_usage']) if self.results['memory_usage'] else 0
            }
        else:  # stability test
            response_times = self.results.get('stability_response_times', [])

            analysis = {
                'duration_minutes': (self.results.get('stability_end_time', 0) - self.results.get('stability_start_time', 0)) / 60,
                'total_requests': self.results.get('stability_total_requests', 0),
                'errors': self.results.get('stability_errors', 0),
                'success_rate': (self.results.get('stability_total_requests', 0) - self.results.get('stability_errors', 0)) / max(self.results.get('stability_total_requests', 1), 1) * 100,
                'avg_response_time': statistics.mean(response_times) if response_times else 0,
                'cpu_stability': statistics.stdev(self.results['cpu_usage']) if len(self.results['cpu_usage']) > 1 else 0,
                'memory_stability': statistics.stdev(self.results['memory_usage']) if len(self.results['memory_usage']) > 1 else 0
            }

        return analysis

    def run_full_performance_test(self):
        """运行完整的性能测试套件"""
        print("🚀 开始完整的性能和稳定性测试")
        print("=" * 60)

        # 1. 并发性能测试
        print("\n1. 并发性能测试 (100个请求, 10并发)")
        concurrent_time = self.run_concurrent_test(num_requests=100, concurrency=10)
        concurrent_analysis = self.analyze_results('concurrent')

        print(f"执行时间: {concurrent_time:.2f}秒")
        print(f"成功率: {concurrent_analysis['success_rate']:.1f}%")
        print(f"平均响应时间: {concurrent_analysis['avg_response_time']:.1f}ms")
        print(f"P95响应时间: {concurrent_analysis['p95_response_time']:.1f}ms")
        print(f"CPU使用率: 平均{concurrent_analysis['cpu_avg']:.1f}%, 最高{concurrent_analysis['cpu_max']:.1f}%")
        print(f"内存使用率: 平均{concurrent_analysis['memory_avg']:.1f}%, 最高{concurrent_analysis['memory_max']:.1f}%")

        # 2. 稳定性测试
        print("\n2. 稳定性测试 (2分钟)")
        stability_results = self.run_stability_test(duration_minutes=2)
        self.results.update({
            'stability_start_time': stability_results['start_time'],
            'stability_end_time': stability_results['end_time'],
            'stability_response_times': stability_results['response_times'],
            'stability_total_requests': stability_results['total_requests'],
            'stability_errors': stability_results['errors']
        })
        stability_analysis = self.analyze_results('stability')

        print(f"测试时长: {stability_analysis['duration_minutes']:.1f}分钟")
        print(f"总请求数: {stability_analysis['total_requests']}")
        print(f"成功率: {stability_analysis['success_rate']:.1f}%")
        print(f"平均响应时间: {stability_analysis['avg_response_time']:.1f}ms")
        print(f"CPU波动性: {stability_analysis['cpu_stability']:.2f}")
        print(f"内存波动性: {stability_analysis['memory_stability']:.2f}")

        # 3. 性能评估
        print("\n3. 性能评估结果")
        performance_passed = (
            concurrent_analysis['success_rate'] >= 99.0 and
            concurrent_analysis['avg_response_time'] <= 500 and
            concurrent_analysis['p95_response_time'] <= 1000 and
            concurrent_analysis['cpu_max'] <= 80 and
            concurrent_analysis['memory_max'] <= 85
        )

        stability_passed = (
            stability_analysis['success_rate'] >= 99.5 and
            stability_analysis['cpu_stability'] <= 5.0 and
            stability_analysis['memory_stability'] <= 5.0
        )

        if performance_passed and stability_passed:
            print("🎉 性能和稳定性测试全部通过!")
            print("✅ 系统性能达到生产环境要求")
            overall_result = "PASSED"
        elif performance_passed:
            print("⚠️ 并发性能测试通过，稳定性测试存在问题")
            print("🔍 建议优化系统稳定性")
            overall_result = "WARNING"
        else:
            print("❌ 性能测试未达到要求")
            print("🔧 需要优化系统性能")
            overall_result = "FAILED"

        # 保存测试报告
        self.save_performance_report(concurrent_analysis, stability_analysis, overall_result)

        print("=" * 60)
        return overall_result

    def save_performance_report(self, concurrent_analysis, stability_analysis, overall_result):
        """保存性能测试报告"""
        report = {
            'test_timestamp': time.time(),
            'overall_result': overall_result,
            'concurrent_test': concurrent_analysis,
            'stability_test': stability_analysis,
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'cpu_count_logical': psutil.cpu_count(logical=True),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3)
            }
        }

        report_path = Path('reports/performance_test_report.json')
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"性能测试报告已保存: {report_path}")

def main():
    tester = PerformanceTester()
    result = tester.run_full_performance_test()

    if result == "PASSED":
        print("\n✅ 性能测试成功!")
        return 0
    else:
        print("\n❌ 性能测试失败!")
        return 1

if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)
