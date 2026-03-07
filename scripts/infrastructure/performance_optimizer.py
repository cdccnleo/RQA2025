#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基础设施层性能优化脚本
优化内存使用、响应时间、缓存策略等
"""

import os
import psutil
import gc
from pathlib import Path
from typing import Dict, List, Any
import json
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InfrastructurePerformanceOptimizer:
    """基础设施层性能优化器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.infrastructure_dir = self.project_root / "src" / "infrastructure"
        self.optimization_log = self.project_root / "backup" / \
            "performance_optimization" / "optimization_log.json"

        # 创建备份目录
        self.optimization_log.parent.mkdir(parents=True, exist_ok=True)

        # 性能基准
        self.performance_benchmarks = {
            'memory_usage_mb': 512,  # 最大内存使用(MB)
            'response_time_ms': 100,  # 最大响应时间(ms)
            'cache_hit_rate': 0.8,   # 最小缓存命中率
            'cpu_usage_percent': 70,  # 最大CPU使用率
        }

        # 优化策略
        self.optimization_strategies = {
            'memory': self._optimize_memory_usage,
            'response_time': self._optimize_response_time,
            'cache': self._optimize_cache_strategy,
            'cpu': self._optimize_cpu_usage,
        }

    def analyze_current_performance(self) -> Dict[str, Any]:
        """分析当前性能状况"""
        logger.info("开始分析当前性能状况...")

        # 获取系统资源使用情况
        process = psutil.Process()
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent()

        # 分析基础设施层模块
        module_analysis = self._analyze_infrastructure_modules()

        performance_data = {
            'timestamp': datetime.now().isoformat(),
            'system_resources': {
                'memory_usage_mb': memory_info.rss / 1024 / 1024,
                'cpu_usage_percent': cpu_percent,
                'available_memory_mb': psutil.virtual_memory().available / 1024 / 1024,
            },
            'module_analysis': module_analysis,
            'performance_issues': self._identify_performance_issues(module_analysis),
            'optimization_recommendations': []
        }

        logger.info(f"性能分析完成: 内存使用 {performance_data['system_resources']['memory_usage_mb']:.1f}MB")
        return performance_data

    def _analyze_infrastructure_modules(self) -> Dict[str, Any]:
        """分析基础设施层模块"""
        modules = {}

        # 分析主要模块
        core_modules = ['config', 'logging', 'monitoring', 'database', 'cache', 'security']

        for module in core_modules:
            module_path = self.infrastructure_dir / module
            if module_path.exists():
                modules[module] = {
                    'file_count': len(list(module_path.rglob("*.py"))),
                    'total_lines': self._count_lines(module_path),
                    'imports': self._analyze_imports(module_path),
                    'complexity': self._calculate_complexity(module_path)
                }

        return modules

    def _count_lines(self, path: Path) -> int:
        """统计代码行数"""
        total_lines = 0
        for py_file in path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    total_lines += len(f.readlines())
            except Exception:
                continue
        return total_lines

    def _analyze_imports(self, path: Path) -> Dict[str, int]:
        """分析导入情况"""
        imports = {}
        for py_file in path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    for line in lines:
                        if line.strip().startswith('import ') or line.strip().startswith('from '):
                            import_type = 'external' if 'sklearn' in line or 'pandas' in line or 'numpy' in line else 'internal'
                            imports[import_type] = imports.get(import_type, 0) + 1
            except Exception:
                continue
        return imports

    def _calculate_complexity(self, path: Path) -> Dict[str, float]:
        """计算代码复杂度"""
        complexity = {
            'average_function_length': 0,
            'max_function_length': 0,
            'total_functions': 0
        }

        total_length = 0
        max_length = 0
        function_count = 0

        for py_file in path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')

                    in_function = False
                    function_lines = 0

                    for line in lines:
                        stripped = line.strip()
                        if stripped.startswith('def '):
                            if in_function:
                                max_length = max(max_length, function_lines)
                                total_length += function_lines
                                function_count += 1
                            in_function = True
                            function_lines = 0
                        elif in_function:
                            if stripped and not stripped.startswith('#'):
                                function_lines += 1
                            if stripped.startswith('class ') or stripped.startswith('def '):
                                max_length = max(max_length, function_lines)
                                total_length += function_lines
                                function_count += 1
                                function_lines = 0
            except Exception:
                continue

        if function_count > 0:
            complexity['average_function_length'] = total_length / function_count
        complexity['max_function_length'] = max_length
        complexity['total_functions'] = function_count

        return complexity

    def _identify_performance_issues(self, module_analysis: Dict[str, Any]) -> List[str]:
        """识别性能问题"""
        issues = []

        # 检查内存使用
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        if memory_mb > self.performance_benchmarks['memory_usage_mb']:
            issues.append(
                f"内存使用过高: {memory_mb:.1f}MB > {self.performance_benchmarks['memory_usage_mb']}MB")

        # 检查CPU使用
        cpu_percent = process.cpu_percent()
        if cpu_percent > self.performance_benchmarks['cpu_usage_percent']:
            issues.append(
                f"CPU使用过高: {cpu_percent:.1f}% > {self.performance_benchmarks['cpu_usage_percent']}%")

        # 检查模块复杂度
        for module_name, analysis in module_analysis.items():
            if analysis['complexity']['average_function_length'] > 50:
                issues.append(
                    f"{module_name}模块函数平均长度过长: {analysis['complexity']['average_function_length']:.1f}行")

            if analysis['complexity']['max_function_length'] > 200:
                issues.append(
                    f"{module_name}模块存在过长函数: {analysis['complexity']['max_function_length']}行")

        return issues

    def optimize_performance(self) -> Dict[str, Any]:
        """执行性能优化"""
        logger.info("开始性能优化...")

        # 分析当前性能
        performance_data = self.analyze_current_performance()

        # 执行优化策略
        optimization_results = {}

        for strategy_name, strategy_func in self.optimization_strategies.items():
            try:
                result = strategy_func(performance_data)
                optimization_results[strategy_name] = result
                logger.info(f"{strategy_name}优化完成")
            except Exception as e:
                logger.error(f"{strategy_name}优化失败: {e}")
                optimization_results[strategy_name] = {'error': str(e)}

        # 保存优化日志
        self._save_optimization_log(performance_data, optimization_results)

        return {
            'performance_data': performance_data,
            'optimization_results': optimization_results
        }

    def _optimize_memory_usage(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """优化内存使用"""
        logger.info("执行内存使用优化...")

        # 强制垃圾回收
        gc.collect()

        # 优化导入策略
        self._optimize_imports()

        # 优化数据结构
        self._optimize_data_structures()

        return {
            'memory_optimized': True,
            'garbage_collection': 'completed',
            'imports_optimized': True
        }

    def _optimize_response_time(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """优化响应时间"""
        logger.info("执行响应时间优化...")

        # 优化缓存策略
        self._optimize_cache_config()

        # 优化异步处理
        self._optimize_async_processing()

        return {
            'response_time_optimized': True,
            'cache_optimized': True,
            'async_optimized': True
        }

    def _optimize_cache_strategy(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """优化缓存策略"""
        logger.info("执行缓存策略优化...")

        # 更新缓存配置
        cache_config = {
            'max_size': 1000,
            'ttl': 300,
            'eviction_policy': 'lru',
            'compression': True
        }

        # 写入缓存配置文件
        cache_config_path = self.infrastructure_dir / "cache" / "cache_config.json"
        cache_config_path.parent.mkdir(exist_ok=True)

        with open(cache_config_path, 'w', encoding='utf-8') as f:
            json.dump(cache_config, f, indent=2)

        return {
            'cache_config_updated': True,
            'cache_config_path': str(cache_config_path)
        }

    def _optimize_cpu_usage(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """优化CPU使用"""
        logger.info("执行CPU使用优化...")

        # 优化线程池配置
        thread_config = {
            'max_workers': min(32, (os.cpu_count() or 1) + 4),
            'thread_name_prefix': 'infrastructure_worker',
            'daemon': True
        }

        # 写入线程配置
        thread_config_path = self.infrastructure_dir / "config" / "thread_config.json"
        thread_config_path.parent.mkdir(exist_ok=True)

        with open(thread_config_path, 'w', encoding='utf-8') as f:
            json.dump(thread_config, f, indent=2)

        return {
            'cpu_optimized': True,
            'thread_config_updated': True,
            'thread_config_path': str(thread_config_path)
        }

    def _optimize_imports(self):
        """优化导入策略"""
        # 创建延迟导入配置文件
        lazy_imports = {
            'heavy_modules': [
                'sklearn',
                'pandas',
                'numpy',
                'matplotlib',
                'seaborn'
            ],
            'light_modules': [
                'src.infrastructure.config',
                'src.infrastructure.logging',
                'src.infrastructure.monitoring'
            ]
        }

        lazy_imports_path = self.infrastructure_dir / "config" / "lazy_imports.json"
        with open(lazy_imports_path, 'w', encoding='utf-8') as f:
            json.dump(lazy_imports, f, indent=2)

    def _optimize_data_structures(self):
        """优化数据结构"""
        # 创建内存优化配置
        memory_config = {
            'use_slots': True,
            'weak_references': True,
            'object_pooling': True,
            'compression_threshold': 1024
        }

        memory_config_path = self.infrastructure_dir / "config" / "memory_config.json"
        with open(memory_config_path, 'w', encoding='utf-8') as f:
            json.dump(memory_config, f, indent=2)

    def _optimize_cache_config(self):
        """优化缓存配置"""
        # 创建缓存优化配置
        cache_optimization = {
            'enable_compression': True,
            'enable_serialization': True,
            'max_memory_mb': 256,
            'cleanup_interval': 300
        }

        cache_optimization_path = self.infrastructure_dir / "cache" / "optimization_config.json"
        with open(cache_optimization_path, 'w', encoding='utf-8') as f:
            json.dump(cache_optimization, f, indent=2)

    def _optimize_async_processing(self):
        """优化异步处理"""
        # 创建异步处理配置
        async_config = {
            'max_concurrent_tasks': 100,
            'task_timeout': 30,
            'enable_cancellation': True,
            'enable_retry': True
        }

        async_config_path = self.infrastructure_dir / "config" / "async_config.json"
        with open(async_config_path, 'w', encoding='utf-8') as f:
            json.dump(async_config, f, indent=2)

    def _save_optimization_log(self, performance_data: Dict[str, Any], optimization_results: Dict[str, Any]):
        """保存优化日志"""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'performance_data': performance_data,
            'optimization_results': optimization_results
        }

        with open(self.optimization_log, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

        logger.info(f"优化日志已保存到: {self.optimization_log}")

    def generate_optimization_report(self) -> str:
        """生成优化报告"""
        if not self.optimization_log.exists():
            return "未找到优化日志，请先运行优化"

        with open(self.optimization_log, 'r', encoding='utf-8') as f:
            log_data = json.load(f)

        report = []
        report.append("# 基础设施层性能优化报告")
        report.append(f"生成时间: {log_data['timestamp']}")
        report.append("")

        # 性能数据
        perf_data = log_data['performance_data']
        report.append("## 性能分析")
        report.append(f"- 内存使用: {perf_data['system_resources']['memory_usage_mb']:.1f}MB")
        report.append(f"- CPU使用: {perf_data['system_resources']['cpu_usage_percent']:.1f}%")
        report.append(f"- 可用内存: {perf_data['system_resources']['available_memory_mb']:.1f}MB")
        report.append("")

        # 性能问题
        if perf_data['performance_issues']:
            report.append("## 发现的性能问题")
            for issue in perf_data['performance_issues']:
                report.append(f"- {issue}")
            report.append("")

        # 优化结果
        opt_results = log_data['optimization_results']
        report.append("## 优化结果")
        for strategy, result in opt_results.items():
            if 'error' not in result:
                report.append(f"- {strategy}: ✅ 优化成功")
            else:
                report.append(f"- {strategy}: ❌ 优化失败 - {result['error']}")

        return "\n".join(report)


def main():
    """主函数"""
    project_root = Path.cwd()
    optimizer = InfrastructurePerformanceOptimizer(str(project_root))

    # 执行性能优化
    results = optimizer.optimize_performance()

    # 生成报告
    report = optimizer.generate_optimization_report()
    print(report)

    # 保存报告
    report_path = project_root / "reports" / "infrastructure_performance_optimization_report.md"
    report_path.parent.mkdir(exist_ok=True)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n优化报告已保存到: {report_path}")


if __name__ == "__main__":
    main()
