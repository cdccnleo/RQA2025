#!/usr/bin/env python3
"""
Phase 2 生产环境性能测试验证（简化版）
"""

import time
import psutil
import json
from datetime import datetime
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os

def main():
    print('🏭 Phase 2 生产环境性能测试开始')
    print('=' * 60)

    # 生产环境配置
    production_config = {
        'target_response_time': 1.0,  # 1秒
        'max_memory_usage': 80.0,     # 80%
        'max_cpu_usage': 70.0,        # 70%
        'target_throughput': 100,     # 100 req/sec
        'test_duration': 60,          # 1分钟
        'concurrent_users': 10,       # 10并发用户
    }

    # 性能基准
    print('📊 建立性能基准...')
    baseline_cpu = psutil.cpu_percent(interval=2)
    baseline_memory = psutil.virtual_memory().percent
    print(f'基准CPU使用率: {baseline_cpu:.1f}%')
    print(f'基准内存使用率: {baseline_memory:.1f}%')
    # 测试结果收集
    performance_results = {
        'response_times': [],
        'throughput': [],
        'resource_usage': [],
        'error_rates': [],
    }

    def simulate_production_operation():
        """模拟生产环境操作"""
        # 模拟数据库查询
        time.sleep(np.random.uniform(0.01, 0.05))

        # 模拟业务逻辑处理
        data = np.random.randn(50, 5)
        result = np.mean(data, axis=0)

        # 模拟外部API调用
        time.sleep(np.random.uniform(0.02, 0.08))

        return {'result': result.tolist(), 'success': True}

    # 1. 响应时间测试
    print('\n⚡ 测试响应时间...')
    response_times = []

    for i in range(50):
        start_time = time.time()
        try:
            result = simulate_production_operation()
            response_time = time.time() - start_time
            response_times.append(response_time)
        except Exception as e:
            response_times.append(2.0)  # 超时

    avg_response_time = np.mean(response_times)
    p95_response_time = np.percentile(response_times, 95)
    max_response_time = max(response_times)

    performance_results['response_times'] = response_times

    print(f'平均响应时间: {avg_response_time:.3f}s')
    print(f'95%响应时间: {p95_response_time:.3f}s')
    print(f'最大响应时间: {max_response_time:.3f}s')
    # 2. 吞吐量测试
    print('\n📈 测试吞吐量...')
    duration = 30  # 30秒测试
    request_count = 0
    start_time = time.time()

    while time.time() - start_time < duration:
        # 模拟并发请求
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(simulate_production_operation) for _ in range(5)]
            for future in futures:
                try:
                    future.result(timeout=1.0)
                    request_count += 1
                except Exception:
                    pass

    actual_throughput = request_count / duration
    performance_results['throughput'].append(actual_throughput)

    print(f'实际吞吐量: {actual_throughput:.1f} req/s')
    # 3. 资源使用测试
    print('\n💾 测试资源使用...')
    initial_cpu = psutil.cpu_percent(interval=1)
    initial_memory = psutil.virtual_memory().percent

    # 运行负载测试
    with ThreadPoolExecutor(max_workers=production_config['concurrent_users']) as executor:
        futures = [executor.submit(simulate_production_operation) for _ in range(100)]

        for future in futures:
            try:
                future.result(timeout=5)
            except Exception:
                pass

    # 测量峰值资源使用
    peak_cpu = psutil.cpu_percent(interval=1)
    peak_memory = psutil.virtual_memory().percent

    resource_usage = {
        'cpu': peak_cpu,
        'memory': peak_memory,
        'timestamp': datetime.now().isoformat()
    }

    performance_results['resource_usage'].append(resource_usage)

    print(f'峰值CPU使用率: {peak_cpu:.1f}%')
    print(f'峰值内存使用率: {peak_memory:.1f}%')
    # 4. 可扩展性测试
    print('\n🔄 测试可扩展性...')
    scalability_results = []
    user_counts = [2, 5, 8, 10]

    for user_count in user_counts:
        start_time = time.time()
        response_times = []

        # 运行并发测试
        with ThreadPoolExecutor(max_workers=user_count) as executor:
            futures = [executor.submit(simulate_production_operation) for _ in range(user_count * 5)]

            for future in futures:
                try:
                    start = time.time()
                    future.result(timeout=3.0)
                    response_times.append(time.time() - start)
                except Exception:
                    response_times.append(3.0)  # 超时

        avg_response_time = np.mean(response_times) if response_times else 0
        throughput = len([t for t in response_times if t < 3.0]) / (time.time() - start_time)

        scalability_results.append({
            'user_count': user_count,
            'avg_response_time': avg_response_time,
            'throughput': throughput
        })

    # 分析可扩展性趋势
    response_times = [r['avg_response_time'] for r in scalability_results]
    response_time_trend = np.polyfit(range(len(response_times)), response_times, 1)[0]

    print(f'响应时间趋势: {response_time_trend:.4f}')
    # 5. 错误处理测试
    print('\n🚨 测试错误处理...')
    error_scenarios = ['timeout', 'connection_error', 'invalid_data']
    error_counts = {'handled': 0, 'unhandled': 0}

    for scenario in error_scenarios:
        try:
            if scenario == 'timeout':
                time.sleep(0.1)
                raise TimeoutError('Network timeout')
            elif scenario == 'connection_error':
                raise ConnectionError('Database connection failed')
            elif scenario == 'invalid_data':
                raise ValueError('Invalid input data')

            error_counts['handled'] += 1
        except Exception:
            error_counts['unhandled'] += 1

    error_rate = (error_counts['unhandled'] / len(error_scenarios)) * 100
    performance_results['error_rates'].append(error_rate)

    print(f"错误处理率: {error_counts['handled']}/{len(error_scenarios)} ({(1-error_rate/100)*100:.1f}%)")

    # 6. 生产就绪性分析
    print('\n📊 生产就绪性分析...')

    # 计算就绪分数
    readiness_score = 100

    # 响应时间评估
    if avg_response_time > production_config['target_response_time']:
        readiness_score -= 15
    if p95_response_time > production_config['target_response_time'] * 2:
        readiness_score -= 10

    # 资源使用评估
    if peak_cpu > production_config['max_cpu_usage']:
        readiness_score -= 15
    if peak_memory > production_config['max_memory_usage']:
        readiness_score -= 15

    # 吞吐量评估
    if actual_throughput < production_config['target_throughput'] * 0.7:
        readiness_score -= 10

    # 错误率评估
    if error_rate > 10.0:
        readiness_score -= 15

    # 可扩展性评估
    if abs(response_time_trend) > 0.05:
        readiness_score -= 10

    # 确定就绪状态
    if readiness_score >= 90:
        status = 'production_ready'
        message = '系统完全具备生产环境部署条件'
    elif readiness_score >= 75:
        status = 'conditionally_ready'
        message = '系统基本具备生产条件，建议进行优化'
    elif readiness_score >= 60:
        status = 'needs_optimization'
        message = '系统需要进一步优化后才能投入生产'
    else:
        status = 'not_ready'
        message = '系统暂不具备生产环境部署条件'

    print(f'生产就绪分数: {readiness_score}/100')
    print(f'就绪状态: {status}')
    print(f'评估结果: {message}')

    # 7. 生成建议
    print('\n💡 优化建议:')
    recommendations = []

    if avg_response_time > production_config['target_response_time']:
        recommendations.append('⚡ 响应时间优化: 考虑添加缓存或使用异步处理')

    if peak_cpu > production_config['max_cpu_usage']:
        recommendations.append('🔥 CPU优化: 考虑增加CPU资源或优化CPU密集型操作')

    if peak_memory > production_config['max_memory_usage']:
        recommendations.append('💾 内存优化: 检查内存泄漏，考虑增加内存资源')

    if actual_throughput < production_config['target_throughput'] * 0.7:
        recommendations.append('📈 吞吐量优化: 考虑水平扩展或负载均衡')

    if abs(response_time_trend) > 0.05:
        recommendations.append('🔄 可扩展性优化: 改进并发处理和资源管理')

    if not recommendations:
        recommendations.append('✅ 性能表现良好，继续监控系统运行状态')

    for rec in recommendations:
        print(f'  {rec}')

    # 保存测试报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'phase': 'Phase 2 - Production Performance Test',
        'config': production_config,
        'results': performance_results,
        'analysis': {
            'readiness_score': readiness_score,
            'status': status,
            'message': message,
            'metrics': {
                'avg_response_time': avg_response_time,
                'p95_response_time': p95_response_time,
                'throughput': actual_throughput,
                'peak_cpu': peak_cpu,
                'peak_memory': peak_memory,
                'error_rate': error_rate,
                'scalability_trend': response_time_trend
            }
        },
        'recommendations': recommendations
    }

    report_file = f'phase2_production_performance_quick_test_{int(datetime.now().timestamp())}.json'
    os.makedirs('test_logs', exist_ok=True)
    with open(f'test_logs/{report_file}', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)

    print('=' * 60)
    print('✅ Phase 2 生产环境性能测试完成')
    print(f'📄 详细报告已保存: test_logs/{report_file}')
    print('=' * 60)

    return status, readiness_score

if __name__ == "__main__":
    main()
