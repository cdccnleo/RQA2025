#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增量测试选择器

基于代码变更智能选择需要执行的测试子集：
- 检测Git变更或文件修改时间
- 分析代码依赖关系
- 识别受影响的测试用例
- 提供快速反馈的测试执行策略
"""

import os
import git
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
import json
import logging
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ChangeImpact:
    """变更影响分析"""
    changed_files: List[str]
    affected_modules: Set[str]
    affected_tests: Set[str]
    risk_level: str
    estimated_tests: int
    execution_time_savings: float


@dataclass
class TestSelection:
    """测试选择结果"""
    critical_tests: List[str]  # 必须执行的测试
    related_tests: List[str]   # 相关测试
    smoke_tests: List[str]     # 冒烟测试
    priority_order: List[str]  # 执行优先级
    estimated_duration: float  # 预估执行时间


class IncrementalTester:
    """增量测试选择器"""

    def __init__(self, source_dir: str = "src", test_dir: str = "tests"):
        self.source_dir = Path(source_dir)
        self.test_dir = Path(test_dir)
        self.repo = self._init_git_repo()
        self.dependency_map = self._build_dependency_map()
        self.test_mapping = self._build_test_mapping()

    def _init_git_repo(self) -> Optional[git.Repo]:
        """初始化Git仓库"""
        try:
            return git.Repo(".")
        except git.InvalidGitRepositoryError:
            logger.warning("当前目录不是Git仓库，将使用文件修改时间检测变更")
            return None

    def _build_dependency_map(self) -> Dict[str, Set[str]]:
        """构建代码依赖关系图"""
        dependency_map = {}

        # 遍历源代码文件
        for py_file in self.source_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            try:
                dependencies = self._analyze_file_dependencies(py_file)
                module_name = self._file_to_module(py_file)
                dependency_map[module_name] = dependencies
            except Exception as e:
                logger.warning(f"分析依赖失败 {py_file}: {e}")

        return dependency_map

    def _analyze_file_dependencies(self, file_path: Path) -> Set[str]:
        """分析文件依赖关系"""
        dependencies = set()

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 分析导入语句
            import_lines = [line.strip() for line in content.split('\n')
                        if line.strip().startswith(('import ', 'from '))]

            for line in import_lines:
                if line.startswith('from src.'):
                    # 提取模块名
                    parts = line.split()
                    if len(parts) >= 2:
                        module_path = parts[1].replace('src.', '').split('.')[0]
                        dependencies.add(module_path)
                elif line.startswith('import src.'):
                    parts = line.split()
                    if len(parts) >= 2:
                        module_path = parts[1].replace('src.', '').split('.')[0]
                        dependencies.add(module_path)

        except Exception as e:
            logger.warning(f"分析文件依赖失败 {file_path}: {e}")

        return dependencies

    def _file_to_module(self, file_path: Path) -> str:
        """将文件路径转换为模块名"""
        relative_path = file_path.relative_to(self.source_dir)
        module_name = str(relative_path).replace('.py', '').replace('/', '.')
        return module_name

    def _build_test_mapping(self) -> Dict[str, Set[str]]:
        """构建测试映射关系"""
        test_mapping = {}

        # 遍历测试文件
        for test_file in self.test_dir.rglob("test_*.py"):
            try:
                related_modules = self._analyze_test_dependencies(test_file)
                test_path = str(test_file.relative_to(self.test_dir))

                for module in related_modules:
                    if module not in test_mapping:
                        test_mapping[module] = set()
                    test_mapping[module].add(test_path)

            except Exception as e:
                logger.warning(f"分析测试依赖失败 {test_file}: {e}")

        return test_mapping

    def _analyze_test_dependencies(self, test_file: Path) -> Set[str]:
        """分析测试文件的依赖关系"""
        dependencies = set()

        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 查找导入的模块
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('from src.') or line.startswith('import src.'):
                    # 提取模块名
                    if 'from src.' in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            module_full = parts[1].replace('src.', '')
                            module_base = module_full.split('.')[0]
                            dependencies.add(module_base)
                    elif 'import src.' in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            module_full = parts[1].replace('src.', '')
                            module_base = module_full.split('.')[0]
                            dependencies.add(module_base)

        except Exception as e:
            logger.warning(f"分析测试依赖失败 {test_file}: {e}")

        return dependencies

    def detect_changes(self) -> List[str]:
        """检测代码变更"""
        changed_files = []

        if self.repo:
            # 使用Git检测变更
            changed_files = self._detect_git_changes()
        else:
            # 使用文件修改时间检测
            changed_files = self._detect_file_changes()

        logger.info(f"检测到 {len(changed_files)} 个变更文件")
        return changed_files

    def _detect_git_changes(self) -> List[str]:
        """使用Git检测变更"""
        try:
            # 获取未提交的变更
            unstaged = [item.a_path for item in self.repo.index.diff(None)]
            staged = [item.a_path for item in self.repo.index.diff("HEAD")]

            # 获取未跟踪的文件
            untracked = self.repo.untracked_files

            all_changes = set(unstaged + staged + untracked)
            # 只关注源代码和测试文件的变更
            relevant_changes = [
                f for f in all_changes
                if f.startswith(('src/', 'tests/')) and f.endswith('.py')
            ]

            return relevant_changes

        except Exception as e:
            logger.warning(f"Git变更检测失败: {e}")
            return []

    def _detect_file_changes(self) -> List[str]:
        """使用文件修改时间检测变更"""
        # 获取最近5分钟内修改的文件
        cutoff_time = time.time() - 300  # 5分钟前
        changed_files = []

        for py_file in self.source_dir.rglob("*.py"):
            if py_file.stat().st_mtime > cutoff_time:
                changed_files.append(str(py_file.relative_to(Path("."))))

        for py_file in self.test_dir.rglob("*.py"):
            if py_file.stat().st_mtime > cutoff_time:
                changed_files.append(str(py_file.relative_to(Path("."))))

        return changed_files

    def analyze_impact(self, changed_files: List[str]) -> ChangeImpact:
        """分析变更影响"""
        affected_modules = set()
        affected_tests = set()

        # 分析直接变更的模块
        for file_path in changed_files:
            if file_path.startswith('src/'):
                module_name = file_path.replace('src/', '').replace('.py', '').replace('/', '.')
                module_base = module_name.split('.')[0]
                affected_modules.add(module_base)

        # 分析依赖关系传播
        all_affected_modules = set(affected_modules)
        for module in affected_modules:
            # 添加依赖此模块的其他模块
            for dep_module, deps in self.dependency_map.items():
                if module in deps:
                    all_affected_modules.add(dep_module)

        # 查找受影响的测试
        for module in all_affected_modules:
            if module in self.test_mapping:
                affected_tests.update(self.test_mapping[module])

        # 计算风险等级
        risk_level = self._calculate_risk_level(len(changed_files), len(all_affected_modules), len(affected_tests))

        # 预估执行时间节省
        total_test_files = len([f for f in self.test_dir.rglob("test_*.py")])
        time_savings = (total_test_files - len(affected_tests)) / total_test_files * 100 if total_test_files > 0 else 0

        return ChangeImpact(
            changed_files=changed_files,
            affected_modules=all_affected_modules,
            affected_tests=affected_tests,
            risk_level=risk_level,
            estimated_tests=len(affected_tests),
            execution_time_savings=time_savings
        )

    def _calculate_risk_level(self, changed_files: int, affected_modules: int, affected_tests: int) -> str:
        """计算风险等级"""
        score = changed_files * 2 + affected_modules * 1.5 + affected_tests * 0.5

        if score > 50:
            return "high"
        elif score > 20:
            return "medium"
        else:
            return "low"

    def select_tests(self, impact: ChangeImpact) -> TestSelection:
        """选择要执行的测试"""
        critical_tests = []
        related_tests = []
        smoke_tests = []

        # 识别关键测试（直接测试变更模块）
        for test_file in impact.affected_tests:
            # 简单的关键测试识别逻辑
            if any(keyword in test_file.lower() for keyword in ['core', 'main', 'critical', 'smoke']):
                critical_tests.append(test_file)
            else:
                related_tests.append(test_file)

        # 添加冒烟测试（如果有的话）
        smoke_tests = self._find_smoke_tests()

        # 确定执行优先级
        priority_order = critical_tests + smoke_tests + related_tests

        # 预估执行时间（基于测试文件数量估算）
        estimated_duration = len(priority_order) * 2.0  # 假设每个测试平均2秒

        return TestSelection(
            critical_tests=critical_tests,
            related_tests=related_tests,
            smoke_tests=smoke_tests,
            priority_order=priority_order,
            estimated_duration=estimated_duration
        )

    def _find_smoke_tests(self) -> List[str]:
        """查找冒烟测试"""
        smoke_tests = []

        # 查找标记为smoke的测试
        for test_file in self.test_dir.rglob("test_*.py"):
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'pytest.mark.smoke' in content or 'smoke' in content.lower():
                        smoke_tests.append(str(test_file.relative_to(self.test_dir)))
            except Exception:
                continue

        return smoke_tests

    def run_incremental_tests(self, selection: TestSelection) -> Dict[str, Any]:
        """运行增量测试"""
        logger.info(f"开始增量测试执行，共 {len(selection.priority_order)} 个测试文件")

        results = {
            'executed_tests': [],
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'skipped': 0,
            'total_duration': 0.0
        }

        # 按优先级执行测试
        for test_file in selection.priority_order:
            test_result = self._run_single_test(test_file)
            results['executed_tests'].append(test_result)
            results['passed'] += test_result.get('passed', 0)
            results['failed'] += test_result.get('failed', 0)
            results['errors'] += test_result.get('errors', 0)
            results['skipped'] += test_result.get('skipped', 0)
            results['total_duration'] += test_result.get('duration', 0)

        logger.info("增量测试执行完成")
        return results

    def _run_single_test(self, test_file: str) -> Dict[str, Any]:
        """运行单个测试文件"""
        import subprocess

        start_time = time.time()

        try:
            cmd = [
                "python", "-m", "pytest",
                f"{self.test_dir}/{test_file}",
                "--tb=short",
                "--quiet",
                "--disable-warnings"
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,  # 1分钟超时
                cwd=Path.cwd()
            )

            duration = time.time() - start_time

            # 解析结果
            passed, failed, errors, skipped = self._parse_pytest_output(result.stdout)

            return {
                'test_file': test_file,
                'duration': duration,
                'passed': passed,
                'failed': failed,
                'errors': errors,
                'skipped': skipped,
                'success': result.returncode == 0
            }

        except subprocess.TimeoutExpired:
            return {
                'test_file': test_file,
                'duration': time.time() - start_time,
                'passed': 0,
                'failed': 0,
                'errors': 1,
                'skipped': 0,
                'success': False
            }
        except Exception:
            return {
                'test_file': test_file,
                'duration': time.time() - start_time,
                'passed': 0,
                'failed': 0,
                'errors': 1,
                'skipped': 0,
                'success': False
            }

    def _parse_pytest_output(self, output: str) -> Tuple[int, int, int, int]:
        """解析pytest输出"""
        import re

        passed = failed = errors = skipped = 0

        # 查找总结行
        lines = output.split('\n')
        for line in reversed(lines):
            line = line.strip()
            if 'passed' in line.lower() and ('failed' in line.lower() or 'error' in line.lower() or 'skipped' in line.lower()):
                # 解析类似 "5 passed, 2 failed, 1 error, 3 skipped"
                match = re.search(r'(\d+)\s*passed.*?(\d+)\s*failed.*?(\d+)\s*error.*?(\d+)\s*skipped', line, re.IGNORECASE)
                if match:
                    passed = int(match.group(1))
                    failed = int(match.group(2))
                    errors = int(match.group(3))
                    skipped = int(match.group(4))
                    break

        return passed, failed, errors, skipped

    def generate_incremental_report(self, impact: ChangeImpact, selection: TestSelection, results: Dict[str, Any]):
        """生成增量测试报告"""
        report_path = Path("test_logs/incremental_test_report.md")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 增量测试报告\n\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## 变更分析\n\n")
            f.write(f"- **变更文件数**: {len(impact.changed_files)}\n")
            f.write(f"- **受影响模块数**: {len(impact.affected_modules)}\n")
            f.write(f"- **受影响测试数**: {len(impact.affected_tests)}\n")
            f.write(f"- **风险等级**: {impact.risk_level.upper()}\n")
            f.write(".1")
            f.write("### 变更文件\n\n")
            for file in impact.changed_files[:10]:  # 只显示前10个
                f.write(f"- `{file}`\n")
            if len(impact.changed_files) > 10:
                f.write(f"- ... 还有 {len(impact.changed_files) - 10} 个文件\n")

            f.write("\n## 测试选择\n\n")
            f.write(f"- **关键测试**: {len(selection.critical_tests)}\n")
            f.write(f"- **相关测试**: {len(selection.related_tests)}\n")
            f.write(f"- **冒烟测试**: {len(selection.smoke_tests)}\n")
            f.write(f"- **总计**: {len(selection.priority_order)}\n")
            f.write(".1")
            f.write("\n## 执行结果\n\n")
            f.write(f"- **执行测试数**: {len(results['executed_tests'])}\n")
            f.write(f"- **通过**: {results['passed']}\n")
            f.write(f"- **失败**: {results['failed']}\n")
            f.write(f"- **错误**: {results['errors']}\n")
            f.write(f"- **跳过**: {results['skipped']}\n")
            f.write(".1")
            if results['executed_tests']:
                f.write("### 测试详情\n\n")
                f.write("| 测试文件 | 状态 | 时间 |\n")
                f.write("|----------|------|------|\n")

                for test_result in results['executed_tests'][:20]:  # 只显示前20个
                    status = "✅" if test_result['success'] else "❌"
                    f.write(f"| `{test_result['test_file']}` | {status} | {test_result['duration']:.2f}s |\n")

            f.write("\n## 效率分析\n\n")
            f.write(f"- **测试减少比例**: {impact.execution_time_savings:.1f}%\n")
            f.write(f"- **预计时间节省**: {impact.execution_time_savings * selection.estimated_duration / 100:.1f}秒\n")
            f.write(f"- **反馈速度提升**: {100 / (100 - impact.execution_time_savings):.1f}倍\n")

        logger.info(f"增量测试报告已生成: {report_path}")

    def run_full_incremental_cycle(self) -> Dict[str, Any]:
        """运行完整的增量测试周期"""
        logger.info("开始增量测试周期...")

        # 1. 检测变更
        changed_files = self.detect_changes()

        if not changed_files:
            logger.info("未检测到变更，跳过测试")
            return {
                'status': 'no_changes',
                'message': '未检测到代码变更'
            }

        # 2. 分析影响
        impact = self.analyze_impact(changed_files)

        # 3. 选择测试
        selection = self.select_tests(impact)

        # 4. 执行测试
        results = self.run_incremental_tests(selection)

        # 5. 生成报告
        self.generate_incremental_report(impact, selection, results)

        summary = {
            'status': 'completed',
            'changed_files': len(changed_files),
            'affected_tests': len(impact.affected_tests),
            'executed_tests': len(results['executed_tests']),
            'passed': results['passed'],
            'failed': results['failed'],
            'time_savings': impact.execution_time_savings,
            'risk_level': impact.risk_level
        }

        logger.info("增量测试周期完成")
        return summary


def main():
    """主函数"""
    tester = IncrementalTester()

    print("🎯 增量测试选择器启动")
    print("🔍 检测代码变更和依赖关系...")

    # 运行完整增量测试周期
    summary = tester.run_full_incremental_cycle()

    print("\n📊 增量测试执行结果:")
    if summary['status'] == 'no_changes':
        print("  ℹ️ 未检测到代码变更，跳过测试")
    else:
        print(f"  📁 变更文件: {summary['changed_files']}")
        print(f"  🧪 受影响测试: {summary['affected_tests']}")
        print(f"  ▶️ 执行测试: {summary['executed_tests']}")
        print(f"  ✅ 通过: {summary['passed']}")
        print(f"  ❌ 失败: {summary['failed']}")
        print(".1")
        print(f"  ⚠️ 风险等级: {summary['risk_level'].upper()}")

    print("\n📄 详细报告已保存到: test_logs/incremental_test_report.md")
    print("\n✅ 增量测试选择器运行完成")


if __name__ == "__main__":
    main()
