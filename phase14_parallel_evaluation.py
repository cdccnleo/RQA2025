#!/usr/bin/env python3
"""
Phase 14.1: pytest并行执行配置评估脚本
评估当前并行执行配置和性能瓶颈
"""

import time
import subprocess
import sys
import os
from pathlib import Path
import multiprocessing
import psutil
import json
from typing import Dict, List, Any

class ParallelExecutionEvaluator:
    """pytest并行执行评估器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.results = {}

    def check_environment(self) -> Dict[str, Any]:
        """检查测试环境配置"""
        print("🔍 检查测试环境配置...")

        env_info = {
            'python_version': sys.version,
            'cpu_cores': multiprocessing.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'available_memory_gb': psutil.virtual_memory().available / (1024**3)
        }

        # 检查pytest-xdist
        try:
            import xdist
            env_info['pytest_xdist'] = 'installed'
            env_info['xdist_version'] = getattr(xdist, '__version__', 'unknown')
        except ImportError:
            env_info['pytest_xdist'] = 'not_installed'

        # 检查pytest版本
        try:
            import pytest
            env_info['pytest_version'] = pytest.__version__
        except ImportError:
            env_info['pytest_version'] = 'not_found'

        print(f"  📊 CPU核心数: {env_info['cpu_cores']}")
        print(".1f"        print(f"  📦 pytest版本: {env_info['pytest_version']}")
        print(f"  📦 pytest-xdist: {env_info['pytest_xdist']}")

        return env_info

    def analyze_test_structure(self) -> Dict[str, Any]:
        """分析测试文件结构"""
        print("🔍 分析测试文件结构...")

        test_structure = {
            'total_test_files': 0,
            'test_files_by_type': {},
            'test_files_by_layer': {},
            'estimated_test_count': 0
        }

        # 统计测试文件
        for root, dirs, files in os.walk(self.project_root / 'tests'):
            for file in files:
                if file.endswith('_test.py') or file.startswith('test_'):
                    test_structure['total_test_files'] += 1

                    # 按类型分类
                    if 'unit' in root:
                        test_structure['test_files_by_type']['unit'] = test_structure['test_files_by_type'].get('unit', 0) + 1
                    elif 'integration' in root:
                        test_structure['test_files_by_type']['integration'] = test_structure['test_files_by_type'].get('integration', 0) + 1
                    elif 'e2e' in root:
                        test_structure['test_files_by_type']['e2e'] = test_structure['test_files_by_type'].get('e2e', 0) + 1

                    # 按层级分类
                    path_parts = Path(root).relative_to(self.project_root / 'tests').parts
                    if len(path_parts) > 1:
                        layer = path_parts[1] if len(path_parts) > 1 else 'other'
                        test_structure['test_files_by_layer'][layer] = test_structure['test_files_by_layer'].get(layer, 0) + 1

        # 估算测试数量（每个文件平均20个测试）
        test_structure['estimated_test_count'] = test_structure['total_test_files'] * 20

        print(f"  📊 总测试文件数: {test_structure['total_test_files']}")
        print(f"  📊 预估测试总数: {test_structure['estimated_test_count']}")
        print(f"  📊 测试类型分布: {test_structure['test_files_by_type']}")

        return test_structure

    def run_performance_test(self, test_files: List[str], workers: int = 1) -> Dict[str, Any]:
        """运行性能测试"""
        print(f"🔍 运行性能测试 (workers={workers})...")

        cmd = [
            sys.executable, '-m', 'pytest'
        ] + test_files + [
            '--tb=no', '-q',
            '--cov-report=',  # 禁用覆盖率报告以加速测试
            '--durations=0'   # 显示最慢的测试
        ]

        if workers > 1:
            cmd.extend(['-n', str(workers), '--dist=loadscope'])

        start_time = time.time()
        start_memory = psutil.virtual_memory().used / (1024**2)  # MB

        result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)

        end_time = time.time()
        end_memory = psutil.virtual_memory().used / (1024**2)  # MB

        performance_data = {
            'execution_time_seconds': end_time - start_time,
            'memory_used_mb': end_memory - start_memory,
            'return_code': result.returncode,
            'workers': workers,
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0
        }

        # 解析输出
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if 'passed' in line and 'failed' in line:
                    # 解析类似 "5 passed, 0 failed" 的行
                    parts = line.split(',')
                    for part in parts:
                        part = part.strip()
                        if 'passed' in part:
                            performance_data['tests_passed'] = int(part.split()[0])
                        elif 'failed' in part:
                            performance_data['tests_failed'] = int(part.split()[0])

        performance_data['tests_run'] = performance_data['tests_passed'] + performance_data['tests_failed']

        print(f"  ⏱️ 执行时间: {performance_data['execution_time_seconds']:.2f}秒")
        print(f"  🧠 内存使用: {performance_data['memory_used_mb']:.1f}MB")
        print(f"  ✅ 通过测试: {performance_data['tests_passed']}")
        print(f"  ❌ 失败测试: {performance_data['tests_failed']}")

        return performance_data

    def identify_bottlenecks(self, performance_results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """识别性能瓶颈"""
        print("🔍 识别性能瓶颈...")

        bottlenecks = {
            'scalability_limit': 0,
            'memory_pressure': False,
            'io_contention': False,
            'module_import_conflicts': False,
            'recommendations': []
        }

        # 检查扩展性
        if len(performance_results) >= 2:
            single_worker = performance_results.get(1, {})
            multi_worker = performance_results.get(max(performance_results.keys()), {})

            if single_worker.get('execution_time_seconds', 0) > 0 and multi_worker.get('execution_time_seconds', 0) > 0:
                speedup = single_worker['execution_time_seconds'] / multi_worker['execution_time_seconds']
                efficiency = speedup / max(performance_results.keys())

                print(f"  📊 并行效率: {efficiency:.2f} (理想值为1.0)")

                if efficiency < 0.5:
                    bottlenecks['scalability_limit'] = max(performance_results.keys())
                    bottlenecks['recommendations'].append('并行效率低下，建议减少worker数量或检查资源竞争')
                elif efficiency > 1.2:
                    bottlenecks['recommendations'].append('并行效率优秀，可以考虑增加worker数量')

        # 检查内存压力
        for worker_count, data in performance_results.items():
            if data.get('memory_used_mb', 0) > 1000:  # 1GB
                bottlenecks['memory_pressure'] = True
                bottlenecks['recommendations'].append('检测到内存压力，建议增加系统内存或减少并发数')
                break

        # 检查模块导入冲突（通过失败测试数判断）
        failed_tests = sum(data.get('tests_failed', 0) for data in performance_results.values())
        total_tests = sum(data.get('tests_run', 0) for data in performance_results.values())

        if failed_tests > total_tests * 0.1:  # 失败率超过10%
            bottlenecks['module_import_conflicts'] = True
            bottlenecks['recommendations'].append('测试失败率较高，可能存在模块导入冲突或依赖问题')

        return bottlenecks

    def generate_optimization_plan(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成优化计划"""
        print("🎯 生成优化计划...")

        plan = {
            'recommended_workers': 2,
            'configuration_changes': [],
            'monitoring_setup': [],
            'follow_up_actions': []
        }

        env_info = evaluation_results.get('environment', {})
        structure_info = evaluation_results.get('structure', {})
        performance_results = evaluation_results.get('performance', {})
        bottlenecks = evaluation_results.get('bottlenecks', {})

        # 推荐worker数量
        cpu_cores = env_info.get('cpu_cores', 4)
        memory_gb = env_info.get('memory_gb', 8)

        if cpu_cores >= 8 and memory_gb >= 16:
            plan['recommended_workers'] = 4
        elif cpu_cores >= 4 and memory_gb >= 8:
            plan['recommended_workers'] = 2
        else:
            plan['recommended_workers'] = 1

        # 避免超过瓶颈
        if bottlenecks.get('scalability_limit', 0) > 0:
            plan['recommended_workers'] = min(plan['recommended_workers'], bottlenecks['scalability_limit'])

        print(f"  📊 推荐worker数量: {plan['recommended_workers']}")

        # 配置优化建议
        plan['configuration_changes'].extend([
            f"设置 -n={plan['recommended_workers']} --dist=loadscope",
            "启用 --durations=10 显示慢测试",
            "配置 --maxfail=5 失败后停止",
            "添加 --strict-markers 严格标记检查"
        ])

        # 监控设置
        plan['monitoring_setup'].extend([
            "设置pytest执行时间监控",
            "配置内存使用跟踪",
            "启用测试失败原因分析",
            "建立性能基准线"
        ])

        # 后续行动
        plan['follow_up_actions'].extend([
            "实施配置优化",
            "运行完整测试套件验证",
            "监控性能改进效果",
            "根据结果调整配置"
        ])

        return plan

    def run_evaluation(self) -> Dict[str, Any]:
        """运行完整评估"""
        print("🚀 Phase 14.1: pytest并行执行配置评估")
        print("=" * 60)

        # 1. 环境检查
        self.results['environment'] = self.check_environment()

        # 2. 结构分析
        self.results['structure'] = self.analyze_test_structure()

        # 3. 性能测试
        test_files = [
            'tests/unit/infrastructure/test_config_low_coverage.py',
            'tests/unit/infrastructure/test_cache_low_coverage.py',
            'tests/unit/data/test_data_loader_core_coverage.py'
        ]

        print("\n🔬 运行性能测试...")
        performance_results = {}

        # 串行测试
        performance_results[1] = self.run_performance_test(test_files, workers=1)

        # 并行测试 (2 workers)
        if self.results['environment'].get('cpu_cores', 1) >= 2:
            performance_results[2] = self.run_performance_test(test_files, workers=2)

        # 并行测试 (4 workers)
        if self.results['environment'].get('cpu_cores', 1) >= 4:
            performance_results[4] = self.run_performance_test(test_files, workers=4)

        self.results['performance'] = performance_results

        # 4. 瓶颈识别
        self.results['bottlenecks'] = self.identify_bottlenecks(performance_results)

        # 5. 优化计划
        self.results['optimization_plan'] = self.generate_optimization_plan(self.results)

        # 保存结果
        self.save_results()

        print("\n" + "=" * 60)
        print("✅ Phase 14.1 评估完成")
        print("=" * 60)

        return self.results

    def save_results(self):
        """保存评估结果"""
        results_file = self.project_root / 'test_logs' / 'phase14_parallel_evaluation.json'

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"📄 评估结果已保存到: {results_file}")

    def print_summary(self):
        """打印评估摘要"""
        if not self.results:
            return

        print("\n📊 评估摘要:")
        print(f"  环境状态: {'✅ 良好' if self.results.get('environment', {}).get('pytest_xdist') == 'installed' else '❌ 需要安装pytest-xdist'}")
        print(f"  测试文件: {self.results.get('structure', {}).get('total_test_files', 0)} 个")
        print(f"  预估测试: {self.results.get('structure', {}).get('estimated_test_count', 0)} 个")

        perf = self.results.get('performance', {})
        if perf:
            fastest = min(perf.values(), key=lambda x: x.get('execution_time_seconds', float('inf')))
            print(".1f"
        plan = self.results.get('optimization_plan', {})
        print(f"  推荐配置: -n={plan.get('recommended_workers', 2)}")


def main():
    """主函数"""
    project_root = Path(__file__).parent
    evaluator = ParallelExecutionEvaluator(project_root)
    results = evaluator.run_evaluation()
    evaluator.print_summary()


if __name__ == '__main__':
    main()
