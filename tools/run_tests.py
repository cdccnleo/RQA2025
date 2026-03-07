#!/usr/bin/env python3
"""
智能测试运行器 - 增强版
支持优先级分组、并行执行、智能失败处理、覆盖率检查、结果持久化
"""

import os
import sys
import time
import subprocess
import concurrent.futures
import json
import sqlite3
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import argparse
from datetime import datetime, timedelta


class TestPriority(Enum):
    """测试优先级枚举"""
    STABLE = "stable"      # 稳定测试 - 核心功能
    MODERATE = "moderate"  # 中等测试 - 重要功能
    EXPERIMENTAL = "experimental"  # 实验性测试 - 新功能
    FEATURES = "features"  # 特征处理层测试 - 专门分组


@dataclass
class TestResult:
    """测试结果数据类"""
    file_path: str
    priority: TestPriority
    passed: int = 0
    failed: int = 0
    errors: int = 0
    duration: float = 0.0
    output: str = ""
    success: bool = False
    coverage: Optional[float] = None
    performance_metrics: Optional[Dict] = None
    layer: Optional[str] = None  # 添加层标识


class TestDatabase:
    """测试结果数据库管理"""

    def __init__(self, db_path: str = "test_results.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 检查现有表结构
        cursor.execute("PRAGMA table_info(test_results)")
        existing_columns = [row[1] for row in cursor.fetchall()]

        # 创建测试结果表（如果不存在）
        if 'test_results' not in [row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]:
            cursor.execute('''
                CREATE TABLE test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    layer TEXT,
                    passed INTEGER NOT NULL,
                    failed INTEGER NOT NULL,
                    errors INTEGER NOT NULL,
                    duration REAL NOT NULL,
                    success BOOLEAN NOT NULL,
                    coverage REAL,
                    performance_metrics TEXT
                )
            ''')
        else:
            # 如果表存在但缺少layer列，添加它
            if 'layer' not in existing_columns:
                try:
                    cursor.execute('ALTER TABLE test_results ADD COLUMN layer TEXT')
                    print("✅ 已添加 layer 列到现有数据库")
                except Exception as e:
                    print(f"⚠️  添加 layer 列失败: {e}")

        # 创建测试覆盖率表（如果不存在）
        if 'coverage_history' not in [row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]:
            cursor.execute('''
                CREATE TABLE coverage_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    overall_coverage REAL NOT NULL,
                    priority_coverage TEXT NOT NULL,
                    layer_coverage TEXT,
                    total_tests INTEGER NOT NULL
                )
            ''')
        else:
            # 如果表存在但缺少layer_coverage列，添加它
            cursor.execute("PRAGMA table_info(coverage_history)")
            coverage_columns = [row[1] for row in cursor.fetchall()]
            if 'layer_coverage' not in coverage_columns:
                try:
                    cursor.execute('ALTER TABLE coverage_history ADD COLUMN layer_coverage TEXT')
                    print("✅ 已添加 layer_coverage 列到现有数据库")
                except Exception as e:
                    print(f"⚠️  添加 layer_coverage 列失败: {e}")

        conn.commit()
        conn.close()

    def save_test_result(self, result: TestResult):
        """保存测试结果"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO test_results 
            (timestamp, file_path, priority, layer, passed, failed, errors, duration, success, coverage, performance_metrics)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            result.file_path,
            result.priority.value,
            result.layer,
            result.passed,
            result.failed,
            result.errors,
            result.duration,
            result.success,
            result.coverage,
            json.dumps(result.performance_metrics) if result.performance_metrics else None
        ))

        conn.commit()
        conn.close()

    def get_layer_coverage_stats(self, layer: str) -> Dict:
        """获取特定层的覆盖率统计"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT 
                COUNT(*) as total_files,
                AVG(coverage) as avg_coverage,
                SUM(passed) as total_passed,
                SUM(failed) as total_failed,
                SUM(errors) as total_errors,
                AVG(duration) as avg_duration
            FROM test_results 
            WHERE layer = ? AND coverage IS NOT NULL
        ''', (layer,))

        result = cursor.fetchone()
        conn.close()

        if result and result[0] > 0:
            return {
                'total_files': result[0],
                'avg_coverage': round(result[1], 2),
                'total_passed': result[2],
                'total_failed': result[3],
                'total_errors': result[4],
                'avg_duration': round(result[5], 2)
            }
        return {}

    def get_coverage_trend(self, days: int = 7) -> Dict:
        """获取覆盖率趋势"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        since_date = (datetime.now() - timedelta(days=days)).isoformat()

        cursor.execute('''
            SELECT timestamp, overall_coverage, total_tests
            FROM coverage_history
            WHERE timestamp >= ?
            ORDER BY timestamp
        ''', (since_date,))

        results = cursor.fetchall()
        conn.close()

        return {
            'dates': [row[0] for row in results],
            'coverage': [row[1] for row in results],
            'total_tests': [row[2] for row in results]
        }


class SmartTestRunner:
    """智能测试运行器"""

    def __init__(self, max_workers: int = 4, verbose: bool = False,
                 enable_coverage: bool = True, save_results: bool = True):
        self.max_workers = max_workers
        self.verbose = verbose
        self.enable_coverage = enable_coverage
        self.save_results = save_results
        self.results: List[TestResult] = []
        self.db = TestDatabase() if save_results else None

    def get_test_groups(self) -> Dict[TestPriority, List[str]]:
        """获取测试分组"""
        return {
            TestPriority.STABLE: [
                "tests/unit/infrastructure/test_infrastructure.py",
                "tests/unit/infrastructure/test_error_handler.py",
                "tests/unit/infrastructure/test_event.py",
                "tests/unit/data/test_data_manager.py",
                "tests/unit/data/test_data_loader.py"
            ],
            TestPriority.MODERATE: [
                "tests/unit/infrastructure/test_version.py",
                "tests/unit/infrastructure/test_factory_patterns.py",
                "tests/unit/infrastructure/test_unified_infrastructure.py",
                "tests/unit/data/test_validator.py",
                "tests/unit/data/test_multiprocess_loader.py"
            ],
            TestPriority.EXPERIMENTAL: [
                "tests/unit/infrastructure/test_optimization.py",
                "tests/unit/infrastructure/test_architecture_optimization.py",
                "tests/unit/data/test_data_critical.py",
                "tests/unit/data/quality/test_quality_metrics.py",
                "tests/unit/data/adapters/test_adapter_registry.py"
            ],
            TestPriority.FEATURES: [
                "tests/unit/features/test_feature_manager.py",
                "tests/unit/features/test_feature_engineer.py",
                "tests/unit/features/test_feature_selector.py",
                "tests/unit/features/test_performance_optimizer.py",
                "tests/unit/features/test_feature_quality_assessor.py",
                "tests/unit/features/test_distributed_processor.py",
                "tests/unit/features/test_plugin_manager.py",
                "tests/unit/features/test_plugin_loader.py",
                "tests/unit/features/test_plugin_validator.py",
                "tests/unit/features/test_gpu_technical_processor.py",
                "tests/unit/features/test_core_manager.py",
                "tests/unit/features/test_feature_store.py",
                "tests/unit/features/test_advanced_feature_selector.py",
                "tests/unit/features/test_feature_engine.py",
                "tests/unit/features/test_feature_pipeline.py"
            ]
        }

    def validate_test_path(self, test_path: str) -> bool:
        """验证测试路径是否有效"""
        if not test_path:
            return False

        # 检查文件是否存在
        if os.path.isfile(test_path):
            return test_path.endswith('.py') and 'test_' in os.path.basename(test_path)

        # 检查目录是否存在
        if os.path.isdir(test_path):
            # 检查目录中是否包含测试文件
            for root, dirs, files in os.walk(test_path):
                if any(f.endswith('.py') and 'test_' in f for f in files):
                    return True
            return False

        return False

    def expand_test_path(self, test_path: str) -> List[str]:
        """展开测试路径，支持通配符和目录"""
        if not test_path:
            return []

        expanded_paths = []

        # 如果是文件路径
        if os.path.isfile(test_path):
            if test_path.endswith('.py') and 'test_' in os.path.basename(test_path):
                expanded_paths.append(test_path)
            return expanded_paths

        # 如果是目录路径
        if os.path.isdir(test_path):
            for root, dirs, files in os.walk(test_path):
                for file in files:
                    if file.endswith('.py') and 'test_' in file:
                        full_path = os.path.join(root, file)
                        expanded_paths.append(full_path)
            return expanded_paths

        # 支持通配符模式
        import glob
        try:
            # 支持常见的通配符模式
            if '*' in test_path or '?' in test_path:
                matched_paths = glob.glob(test_path, recursive=True)
                for path in matched_paths:
                    if os.path.isfile(path) and path.endswith('.py') and 'test_' in os.path.basename(path):
                        expanded_paths.append(path)
                    elif os.path.isdir(path):
                        # 递归展开目录
                        expanded_paths.extend(self.expand_test_path(path))
            else:
                # 尝试作为相对路径处理
                if not os.path.isabs(test_path):
                    # 相对于当前工作目录
                    abs_path = os.path.abspath(test_path)
                    if os.path.isfile(abs_path) and abs_path.endswith('.py') and 'test_' in os.path.basename(abs_path):
                        expanded_paths.append(abs_path)
                    elif os.path.isdir(abs_path):
                        expanded_paths.extend(self.expand_test_path(abs_path))
        except Exception as e:
            print(f"⚠️  路径展开失败: {test_path}, 错误: {e}")

        return expanded_paths

    def run_specific_tests(self, test_paths: List[str]) -> List[TestResult]:
        """运行指定的测试文件"""
        print(f"\n{'='*60}")
        print(f"运行指定测试文件")
        print(f"{'='*60}")

        results = []
        valid_paths = []

        # 验证和展开所有路径
        for path in test_paths:
            # 对于目录路径，总是尝试展开
            if os.path.isdir(path):
                expanded = self.expand_test_path(path)
                if expanded:
                    valid_paths.extend(expanded)
                else:
                    print(f"⚠️  目录中没有找到测试文件: {path}")
            elif self.validate_test_path(path):
                valid_paths.append(path)
            else:
                expanded = self.expand_test_path(path)
                if expanded:
                    valid_paths.extend(expanded)
                else:
                    print(f"⚠️  无效的测试路径: {path}")

        if not valid_paths:
            print("❌ 没有找到有效的测试文件")
            return results

        # 去重并排序
        valid_paths = sorted(list(set(valid_paths)))

        print(f"找到 {len(valid_paths)} 个测试文件:")
        for path in valid_paths:
            print(f"  📁 {path}")

        # 并行执行测试
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {
                executor.submit(self.run_single_test, path, TestPriority.EXPERIMENTAL): path
                for path in valid_paths
            }

            for future in concurrent.futures.as_completed(future_to_path):
                test_path = future_to_path[future]
                print(f"\n执行测试: {test_path}")
                try:
                    result = future.result()
                    results.append(result)
                    self.print_test_result(result)
                except Exception as e:
                    error_result = TestResult(
                        file_path=test_path,
                        priority=TestPriority.EXPERIMENTAL,
                        errors=1,
                        output=f"执行异常: {str(e)}",
                        success=False
                    )
                    results.append(error_result)
                    self.print_test_result(error_result)

        return results

    def get_module_test_files(self, module_name: str) -> List[str]:
        """根据模块名获取对应的测试文件列表"""
        module_mapping = {
            'feature_manager': [
                "tests/unit/features/test_feature_manager.py",
                "tests/unit/features/test_feature_manager_enhanced.py",
                "tests/unit/features/test_optimized_feature_manager.py"
            ],
            'feature_engineer': [
                "tests/unit/features/test_feature_engineer.py",
                "tests/unit/features/test_feature_engineer_enhanced.py",
                "tests/unit/features/test_feature_engineer_coverage_enhanced.py"
            ],
            'feature_selector': [
                "tests/unit/features/test_feature_selector.py",
                "tests/unit/features/test_advanced_feature_selector.py",
                "tests/unit/features/test_feature_selector_coverage_enhanced.py"
            ],
            'performance_optimizer': [
                "tests/unit/features/test_performance_optimizer.py",
                "tests/unit/features/test_high_freq_optimizer.py",
                "tests/unit/features/test_rfecv_performance.py"
            ],
            'quality_assessor': [
                "tests/unit/features/test_feature_quality_assessor.py",
                "tests/unit/features/test_feature_stability.py",
                "tests/unit/features/test_feature_correlation.py"
            ],
            'distributed_processor': [
                "tests/unit/features/test_distributed_processor.py",
                "tests/unit/features/test_distributed_processor_enhanced.py",
                "tests/unit/features/test_distributed_feature_processor.py"
            ],
            'plugin_system': [
                "tests/unit/features/test_plugin_manager.py",
                "tests/unit/features/test_plugin_loader.py",
                "tests/unit/features/test_plugin_validator.py",
                "tests/unit/features/test_base_plugin.py"
            ],
            'gpu_processor': [
                "tests/unit/features/test_gpu_technical_processor.py",
                "tests/unit/features/test_multi_gpu_processor.py"
            ],
            'core_manager': [
                "tests/unit/features/test_core_manager.py",
                "tests/unit/features/test_worker_manager_enhanced.py"
            ],
            'feature_store': [
                "tests/unit/features/test_feature_store.py",
                "tests/unit/features/test_feature_store_coverage_enhanced.py",
                "tests/unit/features/test_feature_cache.py"
            ],
            'feature_engine': [
                "tests/unit/features/test_feature_engine.py",
                "tests/unit/features/test_feature_pipeline.py",
                "tests/unit/features/test_parallel_feature_processor.py"
            ],
            'benchmark': [
                "tests/unit/features/test_benchmark_runner.py",
                "tests/unit/features/test_performance.py"
            ],
            'monitoring': [
                "tests/unit/features/test_features_monitor.py",
                "tests/unit/features/test_monitoring_integration.py"
            ]
        }

        # 支持模糊匹配
        if module_name in module_mapping:
            return module_mapping[module_name]

        # 模糊匹配：包含关键词的模块
        matched_modules = []
        for key, files in module_mapping.items():
            if module_name.lower() in key.lower():
                matched_modules.extend(files)

        if matched_modules:
            return matched_modules

        # 如果没有匹配，返回所有特征处理层测试文件
        print(f"⚠️  未找到模块 '{module_name}' 的测试文件，将运行所有特征处理层测试")
        return self.get_test_groups()[TestPriority.FEATURES]

    def get_module_coverage_analysis(self, module_name: str) -> Dict:
        """获取特定模块的覆盖率分析"""
        if not self.db:
            return {}

        # 获取模块对应的测试文件
        module_test_files = self.get_module_test_files(module_name)

        # 从数据库获取这些文件的测试结果
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()

        # 构建查询条件
        file_conditions = " OR ".join(
            [f"file_path LIKE '%{os.path.basename(f)}%'" for f in module_test_files])

        cursor.execute(f'''
            SELECT 
                file_path,
                passed,
                failed,
                errors,
                duration,
                coverage,
                timestamp
            FROM test_results 
            WHERE {file_conditions}
            ORDER BY timestamp DESC
        ''')

        results = cursor.fetchall()
        conn.close()

        if not results:
            return {
                'module_name': module_name,
                'status': 'no_data',
                'message': f'模块 {module_name} 暂无测试数据'
            }

        # 分析数据
        total_files = len(set(r[0] for r in results))  # 去重文件数
        total_tests = sum(r[1] + r[2] + r[3] for r in results)
        passed_tests = sum(r[1] for r in results)
        failed_tests = sum(r[2] for r in results)
        error_tests = sum(r[3] for r in results)

        # 覆盖率统计
        coverage_results = [r[5] for r in results if r[5] is not None]
        avg_coverage = sum(coverage_results) / len(coverage_results) if coverage_results else 0

        # 按文件分组统计
        file_stats = {}
        for result in results:
            file_path = result[0]
            filename = os.path.basename(file_path)
            if filename not in file_stats:
                file_stats[filename] = {
                    'passed': 0,
                    'failed': 0,
                    'errors': 0,
                    'coverage': 0,
                    'duration': 0,
                    'last_run': result[6]
                }

            file_stats[filename]['passed'] += result[1]
            file_stats[filename]['failed'] += result[2]
            file_stats[filename]['errors'] += result[3]
            if result[5] is not None:
                file_stats[filename]['coverage'] = result[5]  # 最新覆盖率
            file_stats[filename]['duration'] += result[4]

        # 计算成功率
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        # 覆盖率分布
        coverage_distribution = self.get_coverage_distribution(coverage_results)

        # 性能分析
        durations = [r[4] for r in results]
        avg_duration = sum(durations) / len(durations) if durations else 0
        max_duration = max(durations) if durations else 0
        min_duration = min(durations) if durations else 0

        return {
            'module_name': module_name,
            'status': 'success',
            'overview': {
                'total_files': total_files,
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'error_tests': error_tests,
                'success_rate': round(success_rate, 2),
                'avg_coverage': round(avg_coverage, 2)
            },
            'file_stats': file_stats,
            'coverage_distribution': coverage_distribution,
            'performance': {
                'avg_duration': round(avg_duration, 2),
                'max_duration': round(max_duration, 2),
                'min_duration': round(min_duration, 2),
                'total_duration': round(sum(durations), 2)
            },
            'test_files': module_test_files
        }

    def print_module_analysis(self, analysis: Dict):
        """打印模块分析结果"""
        if not analysis:
            print("❌ 无法获取模块分析数据")
            return

        if analysis.get('status') == 'no_data':
            print(f"⚠️  {analysis.get('message', '')}")
            return

        module_name = analysis.get('module_name', 'Unknown')
        overview = analysis.get('overview', {})
        file_stats = analysis.get('file_stats', {})
        coverage_dist = analysis.get('coverage_distribution', {})
        performance = analysis.get('performance', {})
        test_files = analysis.get('test_files', [])

        print(f"\n{'='*80}")
        print(f"🎯 模块分析报告: {module_name.upper()}")
        print(f"{'='*80}")

        # 概览统计
        print(f"\n📊 概览统计:")
        print(f"  测试文件数: {overview.get('total_files', 0)}")
        print(f"  总测试数: {overview.get('total_tests', 0)}")
        print(f"  测试通过率: {overview.get('success_rate', 0):.1f}%")
        print(f"  平均覆盖率: {overview.get('avg_coverage', 0):.1f}%")

        # 文件详细统计
        if file_stats:
            print(f"\n📁 文件详细统计:")
            for filename, stats in file_stats.items():
                total_tests = stats['passed'] + stats['failed'] + stats['errors']
                success_rate = (stats['passed'] / total_tests * 100) if total_tests > 0 else 0
                print(f"  {filename}:")
                print(f"    测试: {stats['passed']}/{total_tests} ({success_rate:.1f}%)")
                print(f"    覆盖率: {stats['coverage']:.1f}%")
                print(f"    耗时: {stats['duration']:.2f}秒")
                print(f"    最后运行: {stats['last_run']}")

        # 覆盖率分布
        if coverage_dist:
            print(f"\n📈 覆盖率分布:")
            print(f"    优秀(90%+): {coverage_dist.get('excellent', 0)} 个文件")
            print(f"    良好(70-89%): {coverage_dist.get('good', 0)} 个文件")
            print(f"    一般(50-69%): {coverage_dist.get('fair', 0)} 个文件")
            print(f"    较差(30-49%): {coverage_dist.get('poor', 0)} 个文件")
            print(f"    很差(<30%): {coverage_dist.get('very_poor', 0)} 个文件")

        # 性能统计
        if performance:
            print(f"\n⚡ 性能统计:")
            print(f"  平均耗时: {performance.get('avg_duration', 0):.2f}秒")
            print(f"  最长耗时: {performance.get('max_duration', 0):.2f}秒")
            print(f"  最短耗时: {performance.get('min_duration', 0):.2f}秒")
            print(f"  总耗时: {performance.get('total_duration', 0):.2f}秒")

        # 测试文件列表
        if test_files:
            print(f"\n🔍 相关测试文件:")
            for test_file in test_files:
                if os.path.exists(test_file):
                    print(f"  ✅ {test_file}")
                else:
                    print(f"  ❌ {test_file} (文件不存在)")

        # 建议
        print(f"\n💡 改进建议:")
        avg_coverage = overview.get('avg_coverage', 0)
        success_rate = overview.get('success_rate', 0)

        if avg_coverage < 70:
            print(f"  📈 覆盖率偏低 ({avg_coverage:.1f}%)，建议补充测试用例")
        if success_rate < 95:
            print(f"  🚨 测试通过率偏低 ({success_rate:.1f}%)，建议修复失败的测试")
        if avg_coverage >= 90 and success_rate >= 95:
            print(f"  🎉 模块测试质量优秀！")

    def run_single_test(self, test_file: str, priority: TestPriority) -> TestResult:
        """运行单个测试文件"""
        start_time = time.time()

        # 检查文件是否存在
        if not os.path.exists(test_file):
            return TestResult(
                file_path=test_file,
                priority=priority,
                errors=1,
                output=f"文件不存在: {test_file}",
                success=False
            )

        # 构建pytest命令
        cmd = [
            sys.executable, "-m", "pytest",
            test_file,
            "--maxfail=5",  # 允许部分失败
            "--tb=short",   # 简短回溯
            "-v",           # 详细输出
            "--disable-warnings"  # 禁用警告
        ]

        # 添加覆盖率检查
        if self.enable_coverage:
            # 使用项目根目录作为覆盖率收集路径
            cmd.extend([
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov"
            ])

        try:
            # 执行测试，设置环境变量避免编码问题
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=300,  # 5分钟超时
                cwd=os.getcwd(),
                env=env
            )

            duration = time.time() - start_time

            # 解析输出
            passed, failed, errors = self.parse_pytest_output(result.stdout)
            coverage = self.parse_coverage_output(result.stdout) if self.enable_coverage else None
            performance_metrics = self.extract_performance_metrics(result.stdout)

            # 判断是否成功
            success = failed == 0 and errors == 0

            # 确定测试文件所属的层
            layer = self.determine_test_layer(test_file)

            test_result = TestResult(
                file_path=test_file,
                priority=priority,
                passed=passed,
                failed=failed,
                errors=errors,
                duration=duration,
                output=result.stdout,
                success=success,
                coverage=coverage,
                performance_metrics=performance_metrics,
                layer=layer
            )

            # 保存结果到数据库
            if self.save_results and self.db:
                self.db.save_test_result(test_result)

            return test_result

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return TestResult(
                file_path=test_file,
                priority=priority,
                errors=1,
                duration=duration,
                output="测试超时",
                success=False
            )
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                file_path=test_file,
                priority=priority,
                errors=1,
                duration=duration,
                output=f"执行异常: {str(e)}",
                success=False
            )

    def parse_pytest_output(self, output: str) -> Tuple[int, int, int]:
        """解析pytest输出，提取测试统计"""
        passed = failed = errors = 0

        if not output:
            return passed, failed, errors

        lines = output.split('\n')

        # 查找包含测试结果的最后几行
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue

            # 匹配 pytest 的最终统计行 - 支持多种格式
            if 'passed' in line or 'failed' in line or 'error' in line:
                # 解析类似 "19 passed, 12 failed" 或 "19 passed, 12 failed in 1.27s" 的行
                try:
                    # 移除时间信息
                    if ' in ' in line:
                        line = line.split(' in ')[0]

                    # 移除等号装饰
                    line = line.strip('=')

                    parts = line.split(',')
                    for part in parts:
                        part = part.strip()
                        if 'passed' in part and not any(x in part for x in ['failed', 'error']):
                            passed = int(part.split()[0])
                        elif 'failed' in part and not any(x in part for x in ['passed', 'error']):
                            failed = int(part.split()[0])
                        elif 'error' in part and not any(x in part for x in ['passed', 'failed']):
                            errors = int(part.split()[0])

                    # 如果找到了任何结果，就跳出循环
                    if passed > 0 or failed > 0 or errors > 0:
                        break

                except (ValueError, IndexError):
                    continue

        return passed, failed, errors

    def parse_coverage_output(self, output: str) -> Optional[float]:
        """解析覆盖率输出"""
        if not output:
            return None

        lines = output.split('\n')
        for line in lines:
            # 匹配 "TOTAL 125126 121279 3.07%" 格式
            if 'TOTAL' in line and len(line.split()) >= 4:
                try:
                    parts = line.split()
                    # 查找包含百分号的最后一部分
                    for part in parts:
                        if '%' in part:
                            return float(part.replace('%', ''))
                except (ValueError, IndexError):
                    continue
            # 匹配 "lines-valid=14435 lines-covered=7966 line-rate=0.5519" 格式
            elif 'line-rate=' in line:
                try:
                    rate_part = line.split('line-rate=')[1].split()[0]
                    return float(rate_part) * 100
                except (ValueError, IndexError):
                    continue
            # 匹配 "7966 of 14435 lines" 格式
            elif ' of ' in line and ' lines' in line:
                try:
                    parts = line.split(' of ')
                    covered = int(parts[0])
                    total = int(parts[1].split()[0])
                    if total > 0:
                        return (covered / total) * 100
                except (ValueError, IndexError):
                    continue
            # 匹配 "Total coverage: 3.07%" 格式
            elif 'Total coverage:' in line and '%' in line:
                try:
                    coverage_part = line.split('Total coverage:')[1].strip()
                    return float(coverage_part.replace('%', ''))
                except (ValueError, IndexError):
                    continue

        return None

    def extract_performance_metrics(self, output: str) -> Optional[Dict]:
        """提取性能指标"""
        if not output:
            return None

        metrics = {}
        lines = output.split('\n')

        for line in lines:
            if 'test session starts' in line:
                # 提取测试会话开始时间
                pass
            elif 'passed in' in line and 's' in line:
                # 提取总耗时
                try:
                    time_part = line.split('passed in ')[1].split('s')[0]
                    metrics['total_duration'] = float(time_part)
                except (ValueError, IndexError):
                    pass

        return metrics if metrics else None

    def determine_test_layer(self, test_file: str) -> str:
        """确定测试文件所属的层"""
        if 'features' in test_file:
            return 'features'
        elif 'infrastructure' in test_file:
            return 'infrastructure'
        elif 'data' in test_file:
            return 'data'
        elif 'models' in test_file:
            return 'models'
        elif 'trading' in test_file:
            return 'trading'
        elif 'risk' in test_file:
            return 'risk'
        elif 'monitoring' in test_file:
            return 'monitoring'
        else:
            return 'unknown'

    def get_features_layer_coverage(self) -> Dict:
        """获取特征处理层覆盖率统计"""
        if not self.db:
            return {}

        # 获取特征处理层统计
        features_stats = self.db.get_layer_coverage_stats('features')

        # 获取特征处理层测试文件列表
        features_tests = [r for r in self.results if r.layer == 'features']

        if not features_tests:
            return {}

        # 计算详细统计
        total_files = len(features_tests)
        total_tests = sum(r.passed + r.failed + r.errors for r in features_tests)
        passed_tests = sum(r.passed for r in features_tests)
        failed_tests = sum(r.failed for r in features_tests)
        error_tests = sum(r.errors for r in features_tests)

        # 覆盖率统计
        coverage_results = [r.coverage for r in features_tests if r.coverage is not None]
        avg_coverage = sum(coverage_results) / len(coverage_results) if coverage_results else 0

        # 按模块分组统计
        module_stats = {}
        for result in features_tests:
            module_name = self.extract_module_name(result.file_path)
            if module_name not in module_stats:
                module_stats[module_name] = {
                    'files': 0,
                    'tests': 0,
                    'passed': 0,
                    'failed': 0,
                    'errors': 0,
                    'coverage': 0,
                    'total_coverage': 0
                }

            module_stats[module_name]['files'] += 1
            module_stats[module_name]['tests'] += result.passed + result.failed + result.errors
            module_stats[module_name]['passed'] += result.passed
            module_stats[module_name]['failed'] += result.failed
            module_stats[module_name]['errors'] += result.errors

            if result.coverage is not None:
                module_stats[module_name]['total_coverage'] += result.coverage

        # 计算每个模块的平均覆盖率
        for module in module_stats.values():
            if module['files'] > 0:
                module['coverage'] = round(module['total_coverage'] / module['files'], 2)

        return {
            'overview': {
                'total_files': total_files,
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'error_tests': error_tests,
                'success_rate': round((passed_tests / total_tests * 100), 2) if total_tests > 0 else 0,
                'avg_coverage': round(avg_coverage, 2)
            },
            'module_stats': module_stats,
            'coverage_distribution': self.get_coverage_distribution(coverage_results)
        }

    def extract_module_name(self, file_path: str) -> str:
        """从文件路径提取模块名称"""
        # 从 tests/unit/features/test_feature_manager.py 提取 feature_manager
        filename = os.path.basename(file_path)
        if filename.startswith('test_'):
            return filename[5:-3]  # 移除 'test_' 前缀和 '.py' 后缀
        return filename[:-3] if filename.endswith('.py') else filename

    def get_coverage_distribution(self, coverage_values: List[float]) -> Dict:
        """获取覆盖率分布统计"""
        if not coverage_values:
            return {}

        distribution = {
            'excellent': 0,  # 90%+
            'good': 0,       # 70-89%
            'fair': 0,       # 50-69%
            'poor': 0,       # 30-49%
            'very_poor': 0   # <30%
        }

        for coverage in coverage_values:
            if coverage >= 90:
                distribution['excellent'] += 1
            elif coverage >= 70:
                distribution['good'] += 1
            elif coverage >= 50:
                distribution['fair'] += 1
            elif coverage >= 30:
                distribution['poor'] += 1
            else:
                distribution['very_poor'] += 1

        return distribution

    def run_test_group(self, priority: TestPriority, test_files: List[str]) -> List[TestResult]:
        """运行测试组"""
        print(f"\n{'='*60}")
        print(f"运行 {priority.value.upper()} 优先级测试组")
        print(f"{'='*60}")

        results = []

        if priority == TestPriority.STABLE:
            # 稳定测试串行执行
            for test_file in test_files:
                print(f"\n执行测试: {test_file}")
                result = self.run_single_test(test_file, priority)
                results.append(result)
                self.print_test_result(result)
        else:
            # 其他优先级并行执行
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_file = {
                    executor.submit(self.run_single_test, test_file, priority): test_file
                    for test_file in test_files
                }

                for future in concurrent.futures.as_completed(future_to_file):
                    test_file = future_to_file[future]
                    print(f"\n执行测试: {test_file}")
                    try:
                        result = future.result()
                        results.append(result)
                        self.print_test_result(result)
                    except Exception as e:
                        error_result = TestResult(
                            file_path=test_file,
                            priority=priority,
                            errors=1,
                            output=f"执行异常: {str(e)}",
                            success=False
                        )
                        results.append(error_result)
                        self.print_test_result(error_result)

        return results

    def print_test_result(self, result: TestResult):
        """打印测试结果"""
        status = "✅ 通过" if result.success else "❌ 失败"
        print(f"  {status} - {result.file_path}")
        print(f"    通过: {result.passed}, 失败: {result.failed}, 错误: {result.errors}")
        print(f"    耗时: {result.duration:.2f}秒")

        if not result.success and self.verbose:
            print(f"    输出: {result.output[:200]}...")

    def run_tests(self) -> Dict[str, any]:
        """运行所有测试"""
        print("🚀 启动智能测试运行器")
        print(f"最大并行数: {self.max_workers}")

        test_groups = self.get_test_groups()

        # 按优先级顺序执行
        for priority in [TestPriority.STABLE, TestPriority.MODERATE, TestPriority.EXPERIMENTAL, TestPriority.FEATURES]:
            test_files = test_groups[priority]
            group_results = self.run_test_group(priority, test_files)
            self.results.extend(group_results)

        # 生成总结报告
        summary = self.generate_summary()
        self.print_summary(summary)

        # 显示覆盖率趋势
        if self.save_results and self.db:
            self.show_coverage_trend()

        return summary

    def show_coverage_trend(self):
        """显示覆盖率趋势"""
        try:
            trend = self.db.get_coverage_trend(days=7)
            if trend['dates']:
                print(f"\n📊 覆盖率趋势 (最近7天):")
                print(f"  日期: {', '.join(trend['dates'][-3:])}")  # 显示最近3天
                print(f"  覆盖率: {', '.join([f'{c:.1f}%' for c in trend['coverage'][-3:]])}")

                # 计算趋势
                if len(trend['coverage']) >= 2:
                    recent_trend = trend['coverage'][-1] - trend['coverage'][-2]
                    trend_arrow = "📈" if recent_trend > 0 else "📉" if recent_trend < 0 else "➡️"
                    trend_text = "上升" if recent_trend > 0 else "下降" if recent_trend < 0 else "稳定"
                    print(f"  趋势: {trend_arrow} {trend_text} ({recent_trend:+.1f}%)")
        except Exception as e:
            print(f"  无法获取覆盖率趋势: {e}")

    def generate_summary(self) -> Dict[str, any]:
        """生成测试总结"""
        summary = {
            'total_files': len(self.results),
            'total_passed': 0,
            'total_failed': 0,
            'total_errors': 0,
            'total_duration': 0.0,
            'by_priority': {},
            'success_rate': 0.0,
            'coverage_summary': {},
            'health_score': 0.0,
            'performance_summary': {}
        }

        # 按优先级统计
        for priority in TestPriority:
            priority_results = [r for r in self.results if r.priority == priority]
            if priority_results:
                summary['by_priority'][priority.value] = {
                    'count': len(priority_results),
                    'passed': sum(r.passed for r in priority_results),
                    'failed': sum(r.failed for r in priority_results),
                    'errors': sum(r.errors for r in priority_results),
                    'success_files': sum(1 for r in priority_results if r.success),
                    'duration': sum(r.duration for r in priority_results),
                    'avg_coverage': self.calculate_average_coverage(priority_results)
                }

        # 总体统计
        summary['total_passed'] = sum(r.passed for r in self.results)
        summary['total_failed'] = sum(r.failed for r in self.results)
        summary['total_errors'] = sum(r.errors for r in self.results)
        summary['total_duration'] = sum(r.duration for r in self.results)

        total_tests = summary['total_passed'] + summary['total_failed'] + summary['total_errors']
        if total_tests > 0:
            summary['success_rate'] = (summary['total_passed'] / total_tests) * 100

        # 覆盖率统计
        summary['coverage_summary'] = self.generate_coverage_summary()

        # 特征处理层覆盖率统计
        summary['features_coverage'] = self.get_features_layer_coverage()

        # 健康度评分
        summary['health_score'] = self.calculate_health_score()

        # 性能统计
        summary['performance_summary'] = self.generate_performance_summary()

        return summary

    def calculate_average_coverage(self, results: List[TestResult]) -> float:
        """计算平均覆盖率"""
        coverage_values = [r.coverage for r in results if r.coverage is not None]
        if not coverage_values:
            return 0.0
        return sum(coverage_values) / len(coverage_values)

    def generate_coverage_summary(self) -> Dict:
        """生成覆盖率总结"""
        coverage_results = [r for r in self.results if r.coverage is not None]
        if not coverage_results:
            return {'overall': 0.0, 'by_priority': {}}

        overall_coverage = sum(r.coverage for r in coverage_results) / len(coverage_results)

        by_priority = {}
        for priority in TestPriority:
            priority_results = [r for r in coverage_results if r.priority == priority]
            if priority_results:
                by_priority[priority.value] = sum(
                    r.coverage for r in priority_results) / len(priority_results)

        return {
            'overall': overall_coverage,
            'by_priority': by_priority
        }

    def calculate_health_score(self) -> float:
        """计算系统健康度评分"""
        if not self.results:
            return 0.0

        # 基础分数：测试通过率
        total_tests = sum(r.passed + r.failed + r.errors for r in self.results)
        if total_tests == 0:
            return 0.0

        pass_rate = sum(r.passed for r in self.results) / total_tests

        # 稳定性分数：稳定测试的成功率
        stable_results = [r for r in self.results if r.priority == TestPriority.STABLE]
        stability_score = 0.0
        if stable_results:
            stable_success = sum(1 for r in stable_results if r.success)
            stability_score = stable_success / len(stable_results)

        # 覆盖率分数
        coverage_results = [r for r in self.results if r.coverage is not None]
        coverage_score = 0.0
        if coverage_results:
            avg_coverage = sum(r.coverage for r in coverage_results) / len(coverage_results)
            coverage_score = avg_coverage / 100.0

        # 综合评分：通过率40% + 稳定性30% + 覆盖率30%
        health_score = (pass_rate * 0.4 + stability_score * 0.3 + coverage_score * 0.3) * 100

        return round(health_score, 2)

    def generate_performance_summary(self) -> Dict:
        """生成性能总结"""
        if not self.results:
            return {}

        durations = [r.duration for r in self.results]
        avg_duration = sum(durations) / len(durations)
        max_duration = max(durations)
        min_duration = min(durations)

        return {
            'avg_duration': round(avg_duration, 2),
            'max_duration': round(max_duration, 2),
            'min_duration': round(min_duration, 2),
            'total_duration': round(sum(durations), 2)
        }

    def print_summary(self, summary: Dict[str, any]):
        """打印测试总结"""
        print(f"\n{'='*60}")
        print("📊 测试总结报告")
        print(f"{'='*60}")

        # 按优先级显示
        for priority in TestPriority:
            if priority.value in summary['by_priority']:
                stats = summary['by_priority'][priority.value]
                print(f"\n{priority.value.upper()} 优先级:")
                print(f"  文件数: {stats['count']}")
                print(f"  成功文件: {stats['success_files']}/{stats['count']}")
                print(f"  测试通过: {stats['passed']}, 失败: {stats['failed']}, 错误: {stats['errors']}")
                print(f"  总耗时: {stats['duration']:.2f}秒")
                if 'avg_coverage' in stats:
                    print(f"  平均覆盖率: {stats['avg_coverage']:.1f}%")

        # 总体统计
        print(f"\n总体统计:")
        print(f"  总文件数: {summary['total_files']}")
        print(
            f"  总测试数: {summary['total_passed'] + summary['total_failed'] + summary['total_errors']}")
        print(f"  通过率: {summary['success_rate']:.1f}%")
        print(f"  总耗时: {summary['total_duration']:.2f}秒")

        # 覆盖率统计
        if 'coverage_summary' in summary and summary['coverage_summary']:
            coverage = summary['coverage_summary']
            print(f"\n📈 覆盖率统计:")
            print(f"  整体覆盖率: {coverage.get('overall', 0):.1f}%")
            for priority, cov in coverage.get('by_priority', {}).items():
                print(f"  {priority} 优先级: {cov:.1f}%")

        # 特征处理层覆盖率统计
        if 'features_coverage' in summary and summary['features_coverage']:
            features = summary['features_coverage']
            if features:
                print(f"\n🎯 特征处理层覆盖率分析:")
                overview = features.get('overview', {})
                if overview:
                    print(f"  总文件数: {overview.get('total_files', 0)}")
                    print(f"  总测试数: {overview.get('total_tests', 0)}")
                    print(f"  测试通过率: {overview.get('success_rate', 0):.1f}%")
                    print(f"  平均覆盖率: {overview.get('avg_coverage', 0):.1f}%")

                # 模块统计
                module_stats = features.get('module_stats', {})
                if module_stats:
                    print(f"  📊 模块覆盖率详情:")
                    for module, stats in module_stats.items():
                        print(f"    {module}: {stats['coverage']:.1f}% ({stats['tests']} 测试)")

                # 覆盖率分布
                coverage_dist = features.get('coverage_distribution', {})
                if coverage_dist:
                    print(f"  📈 覆盖率分布:")
                    print(f"    优秀(90%+): {coverage_dist.get('excellent', 0)} 个模块")
                    print(f"    良好(70-89%): {coverage_dist.get('good', 0)} 个模块")
                    print(f"    一般(50-69%): {coverage_dist.get('fair', 0)} 个模块")
                    print(f"    较差(30-49%): {coverage_dist.get('poor', 0)} 个模块")
                    print(f"    很差(<30%): {coverage_dist.get('very_poor', 0)} 个模块")

        # 系统健康度
        if 'health_score' in summary:
            health_score = summary['health_score']
            health_status = "🟢 优秀" if health_score >= 90 else "🟡 良好" if health_score >= 70 else "🔴 需要改进"
            print(f"\n🏥 系统健康度: {health_score:.1f}/100 {health_status}")

        # 性能统计
        if 'performance_summary' in summary and summary['performance_summary']:
            perf = summary['performance_summary']
            print(f"\n⚡ 性能统计:")
            print(f"  平均耗时: {perf.get('avg_duration', 0):.2f}秒")
            print(f"  最长耗时: {perf.get('max_duration', 0):.2f}秒")
            print(f"  最短耗时: {perf.get('min_duration', 0):.2f}秒")

        # 成功/失败统计
        stable_stats = summary['by_priority'].get('stable', {})
        stable_success = stable_stats.get('success_files', 0)
        stable_total = stable_stats.get('count', 0)

        if stable_total > 0:
            print(
                f"\n稳定测试成功率: {stable_success}/{stable_total} ({stable_success/stable_total*100:.1f}%)")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="智能测试运行器 - 支持特征处理层模块分析",
        epilog="""
可用模块选项:
  feature_manager      - 特征管理器模块
  feature_engineer     - 特征工程模块  
  feature_selector     - 特征选择模块
  performance_optimizer - 性能优化模块
  quality_assessor     - 质量评估模块
  distributed_processor - 分布式处理模块
  plugin_system        - 插件系统模块
  gpu_processor        - GPU处理模块
  core_manager         - 核心管理模块
  feature_store        - 特征存储模块
  feature_engine       - 特征引擎模块
  benchmark            - 基准测试模块
  monitoring           - 监控模块

示例用法:
  # 模块分析
  python run_tests.py --module feature_manager
  python run_tests.py --module gpu_processor --enable-coverage
  
  # 指定路径测试
  python run_tests.py --path tests/unit/features/test_feature_manager.py
  python run_tests.py --path tests/unit/features/ --enable-coverage
  python run_tests.py --path "tests/unit/features/test_*.py" --save-results
  
  # 优先级测试
  python run_tests.py --priority features --enable-coverage --save-results
        """
    )
    parser.add_argument("--max-workers", type=int, default=4, help="最大并行数")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    parser.add_argument("--priority", choices=["stable", "moderate", "experimental", "features"],
                        help="只运行指定优先级的测试")
    parser.add_argument("--module", type=str, help="指定要分析的模块名称")
    parser.add_argument("--path", type=str, nargs='+', help="指定要运行的测试文件或目录路径 (支持通配符)")
    parser.add_argument("--enable-coverage", action="store_true", help="启用覆盖率检查")
    parser.add_argument("--save-results", action="store_true", help="保存测试结果到数据库")

    args = parser.parse_args()

    # 创建测试运行器
    runner = SmartTestRunner(
        max_workers=args.max_workers,
        verbose=args.verbose,
        enable_coverage=args.enable_coverage,
        save_results=args.save_results
    )

    try:
        # 如果指定了路径参数，运行指定的测试文件
        if args.path:
            print(f"🔍 运行指定路径的测试: {', '.join(args.path)}")
            results = runner.run_specific_tests(args.path)
            if results:
                # 生成总结报告
                runner.results = results
                summary = runner.generate_summary()
                runner.print_summary(summary)

                # 显示覆盖率趋势
                if args.save_results and runner.db:
                    runner.show_coverage_trend()

                # 设置退出码
                success_count = sum(1 for r in results if r.success)
                total_count = len(results)
                if total_count > 0 and success_count == 0:
                    print("\n❌ 所有指定的测试都失败了！")
                    sys.exit(1)
                else:
                    print(f"\n✅ 测试执行完成: {success_count}/{total_count} 成功")
                    sys.exit(0)
            else:
                print("\n❌ 没有找到有效的测试文件")
                sys.exit(1)

        # 如果指定了模块参数，进行模块分析
        if args.module:
            print(f"🔍 分析模块: {args.module}")
            analysis = runner.get_module_coverage_analysis(args.module)
            runner.print_module_analysis(analysis)
            sys.exit(0)

        # 运行测试
        summary = runner.run_tests()

        # 设置退出码
        stable_stats = summary['by_priority'].get('stable', {})
        stable_success = stable_stats.get('success_files', 0)
        stable_total = stable_stats.get('count', 0)

        # 只有当所有稳定测试都失败时才返回错误码
        if stable_total > 0 and stable_success == 0:
            print("\n❌ 所有稳定测试都失败了！")
            sys.exit(1)
        else:
            print("\n✅ 测试执行完成")
            sys.exit(0)

    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n💥 测试运行器异常: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
