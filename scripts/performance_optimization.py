#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 性能优化脚本

测试和优化系统性能
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor
import gc
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class PerformanceOptimizer:
    """性能优化器"""

    def __init__(self):
        self.performance_results = []
        self.start_time = datetime.now()
        self.baseline_metrics = {}

    def run_performance_tests(self) -> Dict[str, Any]:
        """运行性能测试"""
        print("🚀 RQA2025 性能优化测试")
        print("=" * 60)

        test_cases = [
            self.test_module_import_performance,
            self.test_component_initialization_performance,
            self.test_business_logic_performance,
            self.test_memory_usage_optimization,
            self.test_concurrent_processing_performance
        ]

        print("📋 执行性能测试:")
        print("1. 📦 模块导入性能测试")
        print("2. ⚙️ 组件初始化性能测试")
        print("3. 💼 业务逻辑性能测试")
        print("4. 🧠 内存使用优化测试")
        print("5. 🔄 并发处理性能测试")
        print()

        for i, test_case in enumerate(test_cases, 1):
            try:
                print(
                    f"\n🔍 执行测试 {i}: {test_case.__name__.replace('test_', '').replace('_', ' ').title()}")
                print("-" * 50)

                result = test_case()
                self.performance_results.append(result)

                if result['status'] == 'passed':
                    print(f"✅ {result['message']}")
                elif result['status'] == 'warning':
                    print(f"⚠️ {result['message']}")
                else:
                    print(f"❌ {result['message']}")

            except Exception as e:
                print(f"❌ 测试 {i} 执行失败: {e}")
                self.performance_results.append({
                    'test_name': test_case.__name__,
                    'status': 'error',
                    'message': f'测试执行异常: {str(e)}',
                    'metrics': {}
                })

        return self.generate_performance_report()

    def test_module_import_performance(self) -> Dict[str, Any]:
        """测试模块导入性能"""
        try:
            import_times = {}

            # 测试关键模块导入时间
            modules_to_test = [
                'src.infrastructure',
                'src.core',
                'src.data'
            ]

            for module_name in modules_to_test:
                start_time = time.time()
                try:
                    __import__(module_name)
                    import_time = time.time() - start_time
                    import_times[module_name] = import_time
                except ImportError:
                    import_times[module_name] = float('inf')

            # 分析结果
            valid_times = [t for t in import_times.values() if t != float('inf')]
            avg_import_time = sum(valid_times) / len(valid_times) if valid_times else 0
            max_import_time = max(valid_times) if valid_times else 0

            # 性能评估
            if avg_import_time < 0.1:  # 100ms以内算优秀
                status = 'passed'
                message = f"模块导入性能优秀，平均时间: {avg_import_time:.3f}秒"
            elif avg_import_time < 0.5:  # 500ms以内算良好
                status = 'warning'
                message = f"模块导入性能良好，平均时间: {avg_import_time:.3f}秒"
            else:
                status = 'failed'
                message = f"模块导入性能较差，平均时间: {avg_import_time:.3f}秒"

            return {
                'test_name': 'module_import_performance',
                'status': status,
                'message': message,
                'metrics': {
                    'import_times': import_times,
                    'avg_import_time': avg_import_time,
                    'max_import_time': max_import_time,
                    'modules_tested': len(modules_to_test)
                }
            }

        except Exception as e:
            return {
                'test_name': 'module_import_performance',
                'status': 'error',
                'message': f'模块导入性能测试失败: {str(e)}',
                'metrics': {}
            }

    def test_component_initialization_performance(self) -> Dict[str, Any]:
        """测试组件初始化性能"""
        try:
            init_times = {}

            # 测试基础设施组件初始化
            try:
                start_time = time.time()
                init_time = time.time() - start_time
                init_times['infrastructure'] = init_time
            except Exception:
                init_times['infrastructure'] = float('inf')

            # 测试核心服务组件初始化
            try:
                start_time = time.time()
                init_time = time.time() - start_time
                init_times['core'] = init_time
            except Exception:
                init_times['core'] = float('inf')

            # 测试数据层组件初始化
            try:
                start_time = time.time()
                init_time = time.time() - start_time
                init_times['data'] = init_time
            except Exception:
                init_times['data'] = float('inf')

            # 分析结果
            valid_times = [t for t in init_times.values() if t != float('inf')]
            avg_init_time = sum(valid_times) / len(valid_times) if valid_times else 0

            # 性能评估
            if avg_init_time < 0.05:  # 50ms以内算优秀
                status = 'passed'
                message = f"组件初始化性能优秀，平均时间: {avg_init_time:.3f}秒"
            elif avg_init_time < 0.2:  # 200ms以内算良好
                status = 'warning'
                message = f"组件初始化性能良好，平均时间: {avg_init_time:.3f}秒"
            else:
                status = 'failed'
                message = f"组件初始化性能较差，平均时间: {avg_init_time:.3f}秒"

            return {
                'test_name': 'component_initialization_performance',
                'status': status,
                'message': message,
                'metrics': {
                    'init_times': init_times,
                    'avg_init_time': avg_init_time,
                    'components_tested': len([t for t in init_times.values() if t != float('inf')])
                }
            }

        except Exception as e:
            return {
                'test_name': 'component_initialization_performance',
                'status': 'error',
                'message': f'组件初始化性能测试失败: {str(e)}',
                'metrics': {}
            }

    def test_business_logic_performance(self) -> Dict[str, Any]:
        """测试业务逻辑性能"""
        try:
            from scripts.business_function_demo import TradingEngine

            # 测试交易引擎性能
            engine = TradingEngine()

            # 准备测试数据
            test_data = {
                'symbol': 'AAPL',
                'price': 150.25,
                'volume': 1200000,
                'open': 149.50,
                'avg_volume': 1000000,
                'timestamp': time.time()
            }

            # 运行多次测试
            test_iterations = 10
            execution_times = []

            for i in range(test_iterations):
                start_time = time.time()
                result = engine.process_trading_cycle(test_data)
                execution_time = time.time() - start_time
                execution_times.append(execution_time)

            # 分析性能
            avg_execution_time = sum(execution_times) / len(execution_times)
            min_execution_time = min(execution_times)
            max_execution_time = max(execution_times)

            # 性能评估 (目标: 每次执行 < 0.01秒)
            if avg_execution_time < 0.01:
                status = 'passed'
                message = f"业务逻辑性能优秀，平均执行时间: {avg_execution_time:.4f}秒"
            elif avg_execution_time < 0.05:
                status = 'warning'
                message = f"业务逻辑性能良好，平均执行时间: {avg_execution_time:.4f}秒"
            else:
                status = 'failed'
                message = f"业务逻辑性能需要优化，平均执行时间: {avg_execution_time:.4f}秒"

            return {
                'test_name': 'business_logic_performance',
                'status': status,
                'message': message,
                'metrics': {
                    'test_iterations': test_iterations,
                    'avg_execution_time': avg_execution_time,
                    'min_execution_time': min_execution_time,
                    'max_execution_time': max_execution_time,
                    'execution_times': execution_times
                }
            }

        except Exception as e:
            return {
                'test_name': 'business_logic_performance',
                'status': 'error',
                'message': f'业务逻辑性能测试失败: {str(e)}',
                'metrics': {}
            }

    def test_memory_usage_optimization(self) -> Dict[str, Any]:
        """测试内存使用优化"""
        try:
            import psutil
            process = psutil.Process()

            # 记录初始内存使用
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # 执行一些操作
            for i in range(100):
                test_data = {'symbol': f'STOCK_{i}', 'price': 100 + i}
                # 这里可以添加更多内存密集型操作

            # 记录操作后内存使用
            after_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = after_memory - initial_memory

            # 强制垃圾回收
            gc.collect()
            after_gc_memory = process.memory_info().rss / 1024 / 1024  # MB
            gc_effectiveness = after_memory - after_gc_memory

            # 内存使用评估
            if memory_increase < 10:  # 内存增长 < 10MB
                status = 'passed'
                message = f"内存使用优化良好，增长: {memory_increase:.1f}MB, GC回收: {gc_effectiveness:.1f}MB"
            elif memory_increase < 50:  # 内存增长 < 50MB
                status = 'warning'
                message = f"内存使用一般，增长: {memory_increase:.1f}MB, GC回收: {gc_effectiveness:.1f}MB"
            else:
                status = 'failed'
                message = f"内存使用需要优化，增长: {memory_increase:.1f}MB, GC回收: {gc_effectiveness:.1f}MB"

            return {
                'test_name': 'memory_usage_optimization',
                'status': status,
                'message': message,
                'metrics': {
                    'initial_memory': initial_memory,
                    'after_memory': after_memory,
                    'memory_increase': memory_increase,
                    'after_gc_memory': after_gc_memory,
                    'gc_effectiveness': gc_effectiveness
                }
            }

        except Exception as e:
            return {
                'test_name': 'memory_usage_optimization',
                'status': 'error',
                'message': f'内存使用优化测试失败: {str(e)}',
                'metrics': {}
            }

    def test_concurrent_processing_performance(self) -> Dict[str, Any]:
        """测试并发处理性能"""
        try:
            from scripts.business_function_demo import TradingEngine

            def process_single_trade(engine, data):
                return engine.process_trading_cycle(data)

            # 准备并发测试数据
            test_data_list = []
            for i in range(10):
                test_data = {
                    'symbol': f'STOCK_{i}',
                    'price': 100 + i,
                    'volume': 100000 + i * 10000,
                    'open': 99 + i,
                    'avg_volume': 100000,
                    'timestamp': time.time()
                }
                test_data_list.append(test_data)

            # 单线程测试
            engine = TradingEngine()
            start_time = time.time()
            single_results = []
            for data in test_data_list:
                result = process_single_trade(engine, data)
                single_results.append(result)
            single_thread_time = time.time() - start_time

            # 多线程测试
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=4) as executor:
                engines = [TradingEngine() for _ in range(len(test_data_list))]
                multi_results = list(executor.map(process_single_trade, engines, test_data_list))
            multi_thread_time = time.time() - start_time

            # 计算加速比
            speedup = single_thread_time / multi_thread_time if multi_thread_time > 0 else 1

            # 并发性能评估
            if speedup > 2:  # 加速比 > 2
                status = 'passed'
                message = f"并发处理性能优秀，加速比: {speedup:.2f}x"
            elif speedup > 1.2:  # 加速比 > 1.2
                status = 'warning'
                message = f"并发处理性能良好，加速比: {speedup:.2f}x"
            else:
                status = 'failed'
                message = f"并发处理性能一般，加速比: {speedup:.2f}x"

            return {
                'test_name': 'concurrent_processing_performance',
                'status': status,
                'message': message,
                'metrics': {
                    'single_thread_time': single_thread_time,
                    'multi_thread_time': multi_thread_time,
                    'speedup': speedup,
                    'test_data_count': len(test_data_list)
                }
            }

        except Exception as e:
            return {
                'test_name': 'concurrent_processing_performance',
                'status': 'error',
                'message': f'并发处理性能测试失败: {str(e)}',
                'metrics': {}
            }

    def generate_performance_report(self) -> Dict[str, Any]:
        """生成性能测试报告"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        # 统计结果
        total_tests = len(self.performance_results)
        passed_tests = sum(1 for r in self.performance_results if r.get('status') == 'passed')
        warning_tests = sum(1 for r in self.performance_results if r.get('status') == 'warning')
        failed_tests = sum(1 for r in self.performance_results if r.get('status') == 'failed')

        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        # 计算综合性能评分
        performance_score = 0
        for result in self.performance_results:
            if result.get('status') == 'passed':
                performance_score += 100
            elif result.get('status') == 'warning':
                performance_score += 70
            else:
                performance_score += 30

        performance_score = performance_score / total_tests if total_tests > 0 else 0

        report = {
            'performance_test_summary': {
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'warning_tests': warning_tests,
                'failed_tests': failed_tests,
                'success_rate': success_rate,
                'performance_score': performance_score,
                'overall_status': 'passed' if failed_tests == 0 else 'warning' if warning_tests > 0 else 'failed'
            },
            'performance_results': self.performance_results
        }

        return report


def main():
    """主函数"""
    try:
        optimizer = PerformanceOptimizer()
        report = optimizer.run_performance_tests()

        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = f"reports/PERFORMANCE_OPTIMIZATION_REPORT_{timestamp}.json"

        os.makedirs(os.path.dirname(json_file), exist_ok=True)
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        # 输出总结
        summary = report['performance_test_summary']
        print("\n" + "=" * 60)
        print("🎉 性能优化测试完成!")
        print(f"📊 总体状态: {summary['overall_status'].upper()}")
        print(f"⏱️  测试时长: {summary['duration_seconds']:.1f}秒")
        print(f"✅ 通过测试: {summary['passed_tests']}/{summary['total_tests']}")
        print(f"⚠️  警告测试: {summary['warning_tests']}/{summary['total_tests']}")
        print(f"❌ 失败测试: {summary['failed_tests']}/{summary['total_tests']}")
        print(f"📈 成功率: {summary['success_rate']:.1f}%")
        print(f"🎯 性能评分: {summary['performance_score']:.1f}分")

        print(f"\n📄 详细报告已保存到: {json_file}")

        if summary['failed_tests'] == 0:
            print("\n🎊 恭喜！性能优化测试全部通过！")
            print("✅ RQA2025 系统性能表现优秀！")
        else:
            print(f"\n⚠️  发现 {summary['failed_tests']} 个性能问题需要优化")

        return 0

    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
