#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pytest深度分析脚本
自动检测conftest.py、fixture依赖链、pytest插件冲突等问题
"""
import os
import sys
import time
import subprocess
import ast
import threading
import importlib
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).parent.parent
TEST_DIR = PROJECT_ROOT / 'tests' / 'unit' / 'infrastructure'
TIMEOUT = 30


class PytestDeepAnalyzer:
    def __init__(self):
        self.conftest_files = []
        self.fixture_dependencies = {}
        self.plugin_conflicts = []
        self.import_issues = []
        self.global_setup_issues = []

    def find_conftest_files(self) -> List[Path]:
        """查找所有conftest.py文件"""
        conftest_files = []
        for root, dirs, files in os.walk(TEST_DIR):
            if 'conftest.py' in files:
                conftest_files.append(Path(root) / 'conftest.py')
        return conftest_files

    def analyze_conftest_file(self, conftest_path: Path) -> Dict:
        """分析单个conftest.py文件"""
        result = {
            'path': str(conftest_path),
            'fixtures': [],
            'hooks': [],
            'imports': [],
            'global_code': [],
            'issues': []
        }

        try:
            with open(conftest_path, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content, filename=str(conftest_path))

            # 分析imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for n in node.names:
                        result['imports'].append(n.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        result['imports'].append(node.module)

            # 分析fixtures和hooks
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if any(decorator.id == 'pytest.fixture' for decorator in node.decorator_list
                           if isinstance(decorator, ast.Name)):
                        result['fixtures'].append(node.name)
                    elif any('pytest' in getattr(decorator, 'id', '') for decorator in node.decorator_list
                             if isinstance(decorator, ast.Name)):
                        result['hooks'].append(node.name)

            # 检查全局代码
            for node in ast.walk(tree):
                if isinstance(node, (ast.Expr, ast.Assign)) and node.lineno < 50:  # 顶层代码
                    result['global_code'].append(
                        f"Line {node.lineno}: {ast.unparse(node)[:100]}...")

            # 检查潜在问题
            if len(result['global_code']) > 5:
                result['issues'].append("存在大量全局代码，可能导致import时阻塞")

            heavy_imports = ['requests', 'sqlalchemy', 'psycopg2',
                             'redis', 'pymongo', 'sklearn', 'tensorflow', 'torch']
            for imp in result['imports']:
                if any(heavy in imp for heavy in heavy_imports):
                    result['issues'].append(f"存在重量级import: {imp}")

        except Exception as e:
            result['issues'].append(f"分析失败: {e}")

        return result

    def analyze_fixture_dependencies(self, conftest_analysis: Dict) -> Dict:
        """分析fixture依赖链"""
        dependencies = {}
        for fixture_name in conftest_analysis['fixtures']:
            try:
                # 这里可以进一步分析fixture的具体依赖
                dependencies[fixture_name] = {
                    'type': 'unknown',
                    'dependencies': [],
                    'estimated_cost': 'low'
                }
            except Exception:
                dependencies[fixture_name] = {
                    'type': 'error',
                    'dependencies': [],
                    'estimated_cost': 'unknown'
                }
        return dependencies

    def check_pytest_plugins(self) -> List[Dict]:
        """检查pytest插件冲突"""
        plugin_conflicts = []

        # 检查常见冲突插件组合
        try:
            result = subprocess.run([sys.executable, '-m', 'pytest', '--version'],
                                    capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version_output = result.stdout
                if 'pytest-xdist' in version_output and 'pytest-cov' in version_output:
                    plugin_conflicts.append({
                        'type': 'plugin_conflict',
                        'description': 'pytest-xdist与pytest-cov可能存在冲突',
                        'severity': 'medium'
                    })
        except Exception as e:
            plugin_conflicts.append({
                'type': 'plugin_error',
                'description': f'无法检查pytest插件: {e}',
                'severity': 'high'
            })

        return plugin_conflicts

    def test_pytest_collection_with_options(self) -> Dict:
        """测试不同pytest选项的收集性能"""
        test_results = {}

        # 测试选项组合
        test_options = [
            ['--collect-only'],
            ['--collect-only', '--tb=short'],
            ['--collect-only', '--disable-warnings'],
            ['--collect-only', '--import-mode=importlib'],
            ['--collect-only', '--import-mode=prepend'],
        ]

        for i, options in enumerate(test_options):
            cmd = [sys.executable, '-m', 'pytest'] + options + [str(TEST_DIR)]
            try:
                start_time = time.time()
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True)
                timer = threading.Timer(TIMEOUT, proc.kill)
                timer.start()
                stdout, stderr = proc.communicate()
                timer.cancel()
                duration = time.time() - start_time

                test_results[f'test_{i}'] = {
                    'options': options,
                    'duration': duration,
                    'success': proc.returncode == 0,
                    'stdout': stdout[:500],
                    'stderr': stderr[:500]
                }
            except Exception as e:
                test_results[f'test_{i}'] = {
                    'options': options,
                    'duration': TIMEOUT,
                    'success': False,
                    'error': str(e)
                }

        return test_results

    def analyze_import_chain(self, module_name: str) -> Dict:
        """分析模块import链"""
        try:
            module = importlib.import_module(module_name)
            return {
                'module': module_name,
                'file': getattr(module, '__file__', 'unknown'),
                'imports': list(module.__dict__.keys()) if hasattr(module, '__dict__') else []
            }
        except Exception as e:
            return {
                'module': module_name,
                'error': str(e)
            }

    def run_full_analysis(self) -> Dict:
        """运行完整分析"""
        print("🔍 开始深度分析pytest相关问题...\n")

        # 1. 查找conftest.py文件
        print("1. 查找conftest.py文件...")
        conftest_files = self.find_conftest_files()
        print(f"   找到 {len(conftest_files)} 个conftest.py文件")

        # 2. 分析conftest.py文件
        print("\n2. 分析conftest.py文件...")
        conftest_analyses = []
        for cf in conftest_files:
            print(f"   分析: {cf}")
            analysis = self.analyze_conftest_file(cf)
            conftest_analyses.append(analysis)
            if analysis['issues']:
                print(f"   ⚠️  发现问题: {analysis['issues']}")

        # 3. 检查pytest插件
        print("\n3. 检查pytest插件冲突...")
        plugin_conflicts = self.check_pytest_plugins()

        # 4. 测试不同收集选项
        print("\n4. 测试pytest收集选项...")
        collection_tests = self.test_pytest_collection_with_options()

        # 5. 分析fixture依赖
        print("\n5. 分析fixture依赖...")
        fixture_deps = {}
        for analysis in conftest_analyses:
            fixture_deps.update(self.analyze_fixture_dependencies(analysis))

        return {
            'conftest_files': conftest_analyses,
            'plugin_conflicts': plugin_conflicts,
            'collection_tests': collection_tests,
            'fixture_dependencies': fixture_deps
        }

    def generate_report(self, analysis_results: Dict):
        """生成分析报告"""
        print("\n" + "="*60)
        print("📊 PYTEST深度分析报告")
        print("="*60)

        # Conftest分析
        print("\n📁 CONFTEST.PY分析:")
        for cf in analysis_results['conftest_files']:
            print(f"\n文件: {cf['path']}")
            print(f"  Fixtures: {len(cf['fixtures'])} 个")
            print(f"  Hooks: {len(cf['hooks'])} 个")
            print(f"  Imports: {len(cf['imports'])} 个")
            if cf['issues']:
                print(f"  ⚠️  问题: {cf['issues']}")

        # 插件冲突
        print("\n🔌 PYTEST插件分析:")
        if analysis_results['plugin_conflicts']:
            for conflict in analysis_results['plugin_conflicts']:
                print(f"  ⚠️  {conflict['description']} (严重性: {conflict['severity']})")
        else:
            print("  ✅ 未发现明显的插件冲突")

        # 收集测试结果
        print("\n⚡ 收集性能测试:")
        for test_name, result in analysis_results['collection_tests'].items():
            status = "✅" if result['success'] else "❌"
            print(f"  {status} 选项: {' '.join(result['options'])} - {result['duration']:.1f}s")
            if not result['success'] and 'error' in result:
                print(f"    错误: {result['error']}")

        # 修复建议
        print("\n🔧 修复建议:")
        print("1. 如果conftest.py中有大量全局代码，考虑移到fixture内部")
        print("2. 如果存在重量级import，考虑延迟加载或mock")
        print("3. 如果插件冲突，尝试禁用部分插件或更新版本")
        print("4. 如果收集慢，尝试使用 --import-mode=importlib")
        print("5. 考虑使用 --disable-warnings 减少输出")


def main():
    analyzer = PytestDeepAnalyzer()
    results = analyzer.run_full_analysis()
    analyzer.generate_report(results)


if __name__ == "__main__":
    main()
