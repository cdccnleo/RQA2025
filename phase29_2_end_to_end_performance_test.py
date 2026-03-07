#!/usr/bin/env python3
"""
RQA2025 Phase 29.2 端到端性能测试脚本

测试系统整体性能表现，包括：
1. 响应时间测试
2. 吞吐量测试
3. 并发处理能力测试
4. 内存使用监控
5. CPU使用监控
"""

import time
import psutil
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import tracemalloc
import gc

class PerformanceTestSuite:
    """端到端性能测试套件"""

    def __init__(self):
        self.results = {
            'response_time': [],
            'throughput': [],
            'memory_usage': [],
            'cpu_usage': [],
            'concurrent_users': [],
            'error_rate': []
        }
        self.test_start_time = None
        self.test_end_time = None

    def run_full_performance_test(self) -> Dict[str, Any]:
        """运行完整的性能测试"""
        print('🚀 RQA2025 Phase 29.2 端到端性能测试启动')
        print('=' * 70)

        self.test_start_time = time.time()

        try:
            # 1. 系统预热
            print('🔥 1. 系统预热...')
            self._warm_up_system()

            # 2. 响应时间测试
            print('⚡ 2. 响应时间测试...')
            response_times = self._test_response_time()

            # 3. 吞吐量测试
            print('📈 3. 吞吐量测试...')
            throughput_results = self._test_throughput()

            # 4. 并发处理能力测试
            print('👥 4. 并发处理能力测试...')
            concurrent_results = self._test_concurrent_processing()

            # 5. 资源使用监控
            print('📊 5. 资源使用监控...')
            resource_usage = self._monitor_resource_usage()

            # 6. 稳定性测试
            print('🔄 6. 稳定性测试...')
            stability_results = self._test_stability()

            # 生成测试报告
            self.test_end_time = time.time()
            report = self._generate_performance_report(
                response_times, throughput_results, concurrent_results,
                resource_usage, stability_results
            )

            print('✅ 端到端性能测试完成！')
            return report

        except Exception as e:
            print(f'❌ 测试过程中发生错误: {str(e)}')
            import traceback
            traceback.print_exc()
            return {'status': 'failed', 'error': str(e)}

    def _warm_up_system(self):
        """系统预热"""
        print('   • 执行系统预热操作...')
        # 这里可以添加一些预热操作，如初始化缓存、建立连接等
        time.sleep(2)
        print('   ✅ 系统预热完成')

    def _test_response_time(self) -> Dict[str, Any]:
        """测试响应时间"""
        response_times = []

        # 模拟100次请求
        for i in range(100):
            start_time = time.time()
            # 模拟一个简单的操作
            self._simulate_operation()
            end_time = time.time()

            response_time = (end_time - start_time) * 1000  # 转换为毫秒
            response_times.append(response_time)

        # 计算统计指标
        avg_response_time = statistics.mean(response_times)
        median_response_time = statistics.median(response_times)
        p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
        p99_response_time = statistics.quantiles(response_times, n=100)[98]  # 99th percentile
        min_response_time = min(response_times)
        max_response_time = max(response_times)

        results = {
            'average': round(avg_response_time, 2),
            'median': round(median_response_time, 2),
            'p95': round(p95_response_time, 2),
            'p99': round(p99_response_time, 2),
            'min': round(min_response_time, 2),
            'max': round(max_response_time, 2),
            'samples': len(response_times)
        }

        print(f'   ✅ 响应时间测试完成: 平均 {results["average"]}ms, P95 {results["p95"]}ms')
        return results

    def _test_throughput(self) -> Dict[str, Any]:
        """测试吞吐量"""
        test_duration = 10  # 10秒测试
        operations_completed = 0

        start_time = time.time()
        end_time = start_time + test_duration

        while time.time() < end_time:
            self._simulate_operation()
            operations_completed += 1

        actual_duration = time.time() - start_time
        throughput = operations_completed / actual_duration  # ops/sec

        results = {
            'operations_per_second': round(throughput, 2),
            'total_operations': operations_completed,
            'test_duration': round(actual_duration, 2)
        }

        print(f'   ✅ 吞吐量测试完成: {results["operations_per_second"]} ops/sec')
        return results

    def _test_concurrent_processing(self) -> Dict[str, Any]:
        """测试并发处理能力"""
        concurrent_users = [10, 50, 100, 200]  # 不同并发用户数
        results = {}

        for users in concurrent_users:
            print(f'   • 测试 {users} 并发用户...')

            response_times = []
            errors = 0

            def single_user_test(user_id: int):
                nonlocal errors
                try:
                    start_time = time.time()
                    self._simulate_operation()
                    end_time = time.time()
                    response_times.append((end_time - start_time) * 1000)
                except Exception as e:
                    errors += 1

            # 使用线程池模拟并发用户
            with ThreadPoolExecutor(max_workers=users) as executor:
                futures = [executor.submit(single_user_test, i) for i in range(users)]
                for future in as_completed(futures):
                    pass  # 等待所有任务完成

            if response_times:
                avg_response = statistics.mean(response_times)
                success_rate = ((users - errors) / users) * 100

                results[str(users)] = {
                    'avg_response_time': round(avg_response, 2),
                    'success_rate': round(success_rate, 2),
                    'errors': errors
                }

                print(f'     - 平均响应时间: {round(avg_response, 2)}ms')
                print(f'     - 成功率: {round(success_rate, 2)}%')
            else:
                results[str(users)] = {'error': 'No successful operations'}

        return results

    def _monitor_resource_usage(self) -> Dict[str, Any]:
        """监控资源使用情况"""
        print('   • 监控系统资源使用...')

        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)

        # 内存使用
        memory = psutil.virtual_memory()
        memory_usage = {
            'total': memory.total,
            'available': memory.available,
            'used': memory.used,
            'percent': memory.percent
        }

        # 磁盘使用
        disk = psutil.disk_usage('/')
        disk_usage = {
            'total': disk.total,
            'used': disk.used,
            'free': disk.free,
            'percent': disk.percent
        }

        results = {
            'cpu_percent': cpu_percent,
            'memory': memory_usage,
            'disk': disk_usage
        }

        print(f'   ✅ 资源监控完成: CPU {cpu_percent}%, 内存 {memory_usage["percent"]}%')
        return results

    def _test_stability(self) -> Dict[str, Any]:
        """测试系统稳定性"""
        print('   • 执行稳定性测试...')

        stability_duration = 30  # 30秒稳定性测试
        check_interval = 5  # 每5秒检查一次

        response_times = []
        error_count = 0
        checks = 0

        start_time = time.time()
        end_time = start_time + stability_duration

        while time.time() < end_time:
            checks += 1
            try:
                op_start = time.time()
                self._simulate_operation()
                op_end = time.time()
                response_times.append((op_end - op_start) * 1000)
            except Exception as e:
                error_count += 1

            time.sleep(check_interval)

        avg_response_time = statistics.mean(response_times) if response_times else 0
        error_rate = (error_count / checks) * 100 if checks > 0 else 0

        results = {
            'test_duration': stability_duration,
            'total_checks': checks,
            'successful_operations': len(response_times),
            'errors': error_count,
            'avg_response_time': round(avg_response_time, 2),
            'error_rate': round(error_rate, 2)
        }

        print(f'   ✅ 稳定性测试完成: 错误率 {results["error_rate"]}%, 平均响应 {results["avg_response_time"]}ms')
        return results

    def _simulate_operation(self):
        """模拟一个系统操作"""
        # 这里可以根据实际系统添加真实的测试操作
        # 目前使用简单的计算操作作为模拟
        import random
        import math

        # 模拟一些计算密集型操作
        data = [random.random() for _ in range(1000)]
        result = sum(math.sin(x) * math.cos(x) for x in data)

        # 模拟一些I/O操作
        time.sleep(0.001)  # 1ms delay

        return result

    def _generate_performance_report(self, response_times: Dict, throughput: Dict,
                                   concurrent: Dict, resources: Dict, stability: Dict) -> Dict[str, Any]:
        """生成性能测试报告"""
        report = {
            'status': 'completed',
            'timestamp': datetime.now().isoformat(),
            'test_duration': round(self.test_end_time - self.test_start_time, 2),
            'performance_metrics': {
                'response_time': response_times,
                'throughput': throughput,
                'concurrent_processing': concurrent,
                'resource_usage': resources,
                'stability': stability
            }
        }

        # 性能评估
        evaluation = self._evaluate_performance(response_times, throughput, concurrent, resources, stability)
        report['evaluation'] = evaluation

        return report

    def _evaluate_performance(self, response_times: Dict, throughput: Dict,
                            concurrent: Dict, resources: Dict, stability: Dict) -> Dict[str, Any]:
        """评估性能测试结果"""
        evaluation = {
            'overall_score': 0,
            'grade': 'Unknown',
            'recommendations': [],
            'critical_issues': [],
            'strengths': []
        }

        score = 0

        # 响应时间评估 (30%)
        if response_times['average'] < 50:  # < 50ms 优秀
            score += 30
            evaluation['strengths'].append('响应时间优秀')
        elif response_times['average'] < 100:  # < 100ms 良好
            score += 20
            evaluation['strengths'].append('响应时间良好')
        elif response_times['average'] < 200:  # < 200ms 可接受
            score += 10
            evaluation['recommendations'].append('响应时间需要优化')
        else:
            evaluation['critical_issues'].append('响应时间过长')

        # 吞吐量评估 (25%)
        if throughput['operations_per_second'] > 1000:  # > 1000 ops/sec 优秀
            score += 25
            evaluation['strengths'].append('吞吐量表现优秀')
        elif throughput['operations_per_second'] > 500:  # > 500 ops/sec 良好
            score += 15
            evaluation['strengths'].append('吞吐量表现良好')
        elif throughput['operations_per_second'] > 100:  # > 100 ops/sec 可接受
            score += 5
            evaluation['recommendations'].append('吞吐量需要提升')
        else:
            evaluation['critical_issues'].append('吞吐量不足')

        # 并发处理评估 (20%)
        concurrent_100 = concurrent.get('100', {})
        if concurrent_100.get('success_rate', 0) > 95:  # > 95% 成功率优秀
            score += 20
            evaluation['strengths'].append('并发处理能力优秀')
        elif concurrent_100.get('success_rate', 0) > 85:  # > 85% 成功率良好
            score += 10
            evaluation['strengths'].append('并发处理能力良好')
        else:
            evaluation['recommendations'].append('并发处理能力需要改进')

        # 资源使用评估 (15%)
        if resources['cpu_percent'] < 70 and resources['memory']['percent'] < 80:
            score += 15
            evaluation['strengths'].append('资源使用效率良好')
        elif resources['cpu_percent'] < 85 and resources['memory']['percent'] < 90:
            score += 7
            evaluation['recommendations'].append('资源使用需要优化')
        else:
            evaluation['critical_issues'].append('资源使用效率低下')

        # 稳定性评估 (10%)
        if stability['error_rate'] < 1:  # < 1% 错误率优秀
            score += 10
            evaluation['strengths'].append('系统稳定性优秀')
        elif stability['error_rate'] < 5:  # < 5% 错误率良好
            score += 5
            evaluation['recommendations'].append('稳定性需要进一步提升')
        else:
            evaluation['critical_issues'].append('系统稳定性不足')

        evaluation['overall_score'] = score

        # 等级评定
        if score >= 90:
            evaluation['grade'] = '优秀 (A)'
        elif score >= 75:
            evaluation['grade'] = '良好 (B)'
        elif score >= 60:
            evaluation['grade'] = '可接受 (C)'
        else:
            evaluation['grade'] = '需要改进 (D)'

        return evaluation

def main():
    """主函数"""
    tester = PerformanceTestSuite()
    report = tester.run_full_performance_test()

    if report.get('status') == 'completed':
        # 保存测试报告
        with open('RQA2025_PHASE29_2_PERFORMANCE_TEST_REPORT.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print('\n📊 性能测试报告已保存到: RQA2025_PHASE29_2_PERFORMANCE_TEST_REPORT.json')

        # 输出关键指标
        metrics = report['performance_metrics']
        eval_result = report['evaluation']

        print('\n🎯 关键性能指标:')
        print(f"响应时间: {metrics['response_time']['average']}ms (平均)")
        print(f"吞吐量: {metrics['throughput']['operations_per_second']} ops/sec")
        print(f"并发处理: 100用户成功率 {metrics['concurrent_processing'].get('100', {}).get('success_rate', 'N/A')}%")
        print(f"资源使用: CPU {metrics['resource_usage']['cpu_percent']}%, 内存 {metrics['resource_usage']['memory']['percent']}%")
        print(f"稳定性: 错误率 {metrics['stability']['error_rate']}%")

        print(f'\n🏆 综合评分: {eval_result["overall_score"]}/100 - {eval_result["grade"]}')

        if eval_result['critical_issues']:
            print('\n⚠️ 关键问题:')
            for issue in eval_result['critical_issues']:
                print(f'  • {issue}')

        if eval_result['recommendations']:
            print('\n💡 优化建议:')
            for rec in eval_result['recommendations']:
                print(f'  • {rec}')

        if eval_result['strengths']:
            print('\n✅ 性能优势:')
            for strength in eval_result['strengths']:
                print(f'  • {strength}')

    return report

if __name__ == '__main__':
    main()
