#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化版增量测试选择器

解决性能和编码问题的优化版本：
- 智能并行控制，避免资源耗尽
- 正确的编码处理，解决Windows环境问题
- 高效的依赖分析，减少文件遍历
- 完善的超时和错误处理
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
import json
import logging
import hashlib
import subprocess
import platform

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


class OptimizedIncrementalTester:
    """优化版增量测试选择器"""

    def __init__(self, source_dir: str = "src", test_dir: str = "tests", max_workers: int = 2):
        self.source_dir = Path(source_dir)
        self.test_dir = Path(test_dir)
        self.max_workers = max_workers  # 限制最大并行数，避免资源耗尽

        # 编码配置 - 解决Windows环境问题
        self.encoding = 'utf-8' if platform.system() != 'Windows' else 'gbk'
        self.errors = 'replace'  # 遇到解码错误时替换

        # 缓存优化
        self._dependency_cache: Dict[str, Set[str]] = {}
        self._test_mapping_cache: Dict[str, Set[str]] = {}

        logger.info(f"优化版增量测试器初始化: max_workers={max_workers}, encoding={self.encoding}")

    def detect_changes_fast(self) -> List[str]:
        """快速检测变更文件"""
        logger.info("快速检测变更文件...")

        # 1. 检查Git状态（如果可用）
        git_changes = self._detect_git_changes()
        if git_changes:
            logger.info(f"Git检测到 {len(git_changes)} 个变更文件")
            return git_changes

        # 2. 检查最近修改的文件（过去10分钟）
        recent_changes = self._detect_recent_changes()
        logger.info(f"检测到 {len(recent_changes)} 个最近修改的文件")
        return recent_changes

    def _detect_git_changes(self) -> List[str]:
        """检测Git变更"""
        try:
            import git
            repo = git.Repo(".")
            changed_files = []

            # 获取工作区变更
            for item in repo.index.diff(None):
                if item.a_path.endswith('.py'):
                    changed_files.append(item.a_path)

            # 获取暂存区变更
            for item in repo.index.diff("HEAD"):
                if item.a_path.endswith('.py'):
                    changed_files.append(item.a_path)

            # 获取未跟踪的文件
            for file_path in repo.untracked_files:
                if file_path.endswith('.py'):
                    changed_files.append(file_path)

            return list(set(changed_files))  # 去重

        except Exception as e:
            logger.debug(f"Git检测失败: {e}")
            return []

    def _detect_recent_changes(self, minutes: int = 10) -> List[str]:
        """检测最近修改的文件"""
        cutoff_time = time.time() - (minutes * 60)
        changed_files = []

        # 检查源代码文件
        for py_file in self.source_dir.rglob("*.py"):
            try:
                if py_file.stat().st_mtime > cutoff_time:
                    changed_files.append(str(py_file.relative_to(Path("."))))
            except OSError:
                continue

        # 检查测试文件
        for py_file in self.test_dir.rglob("*.py"):
            try:
                if py_file.stat().st_mtime > cutoff_time:
                    changed_files.append(str(py_file.relative_to(Path("."))))
            except OSError:
                continue

        return changed_files

    def analyze_impact_efficient(self, changed_files: List[str]) -> ChangeImpact:
        """高效分析变更影响"""
        logger.info("高效分析变更影响...")

        affected_modules = set()
        affected_tests = set()

        # 快速模块映射
        for file_path in changed_files:
            if file_path.startswith('src/'):
                module = self._file_to_module_simple(file_path)
                affected_modules.add(module)

        # 使用缓存的测试映射
        if not self._test_mapping_cache:
            self._test_mapping_cache = self._build_test_mapping_cached()

        # 查找受影响的测试
        for module in affected_modules:
            if module in self._test_mapping_cache:
                affected_tests.update(self._test_mapping_cache[module])

        # 计算风险等级
        risk_level = self._calculate_risk_level(len(changed_files), len(affected_modules), len(affected_tests))

        # 估算时间节省
        total_test_files = self._count_total_tests()
        time_savings = (total_test_files - len(affected_tests)) / total_test_files * 100 if total_test_files > 0 else 0

        return ChangeImpact(
            changed_files=changed_files,
            affected_modules=affected_modules,
            affected_tests=affected_tests,
            risk_level=risk_level,
            estimated_tests=len(affected_tests),
            execution_time_savings=time_savings
        )

    def _file_to_module_simple(self, file_path: str) -> str:
        """简化版文件到模块转换"""
        if file_path.startswith('src/'):
            return file_path.replace('src/', '').replace('.py', '').split('/')[0]
        return ''

    def _build_test_mapping_cached(self) -> Dict[str, Set[str]]:
        """构建缓存的测试映射"""
        cache_file = Path(".test_cache.json")

        # 尝试从缓存加载
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    # 检查缓存是否过期（1小时）
                    if time.time() - cache_data.get('timestamp', 0) < 3600:
                        logger.info("使用缓存的测试映射")
                        return {k: set(v) for k, v in cache_data['mapping'].items()}
            except Exception as e:
                logger.debug(f"缓存加载失败: {e}")

        # 重新构建映射
        logger.info("重新构建测试映射...")
        mapping = {}
        test_count = 0

        for test_file in self.test_dir.rglob("test_*.py"):
            try:
                modules = self._analyze_test_dependencies_fast(test_file)
                test_path = str(test_file.relative_to(self.test_dir))

                for module in modules:
                    if module not in mapping:
                        mapping[module] = set()
                    mapping[module].add(test_path)

                test_count += 1
                if test_count % 100 == 0:
                    logger.debug(f"已处理 {test_count} 个测试文件")

            except Exception as e:
                logger.debug(f"分析测试文件失败 {test_file}: {e}")
                continue

        # 保存缓存
        try:
            cache_data = {
                'timestamp': time.time(),
                'mapping': {k: list(v) for k, v in mapping.items()}
            }
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
            logger.info(f"测试映射缓存已保存，共 {len(mapping)} 个模块")
        except Exception as e:
            logger.debug(f"缓存保存失败: {e}")

        return mapping

    def _analyze_test_dependencies_fast(self, test_file: Path) -> Set[str]:
        """快速分析测试依赖"""
        modules = set()

        try:
            # 只读取文件前100行，避免大文件处理过慢
            with open(test_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= 100:  # 只读取前100行
                        break
                    lines.append(line)

            content = ''.join(lines)

            # 快速查找导入语句
            import_lines = [line.strip() for line in content.split('\n')
                        if line.strip().startswith(('from src.', 'import src.'))]

            for line in import_lines:
                if 'from src.' in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        module_full = parts[1].replace('src.', '')
                        module_base = module_full.split('.')[0]
                        modules.add(module_base)
                elif 'import src.' in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        module_full = parts[1].replace('src.', '')
                        module_base = module_full.split('.')[0]
                        modules.add(module_base)

        except Exception as e:
            logger.debug(f"快速依赖分析失败 {test_file}: {e}")

        return modules

    def _count_total_tests(self) -> int:
        """统计总测试文件数"""
        try:
            return len(list(self.test_dir.rglob("test_*.py")))
        except Exception:
            return 100  # 默认值

    def _calculate_risk_level(self, changed_files: int, affected_modules: int, affected_tests: int) -> str:
        """计算风险等级"""
        score = changed_files * 2 + affected_modules * 1.5 + affected_tests * 0.5

        if score > 50:
            return "high"
        elif score > 20:
            return "medium"
        else:
            return "low"

    def select_tests_smart(self, impact: ChangeImpact) -> TestSelection:
        """智能选择测试"""
        logger.info("智能选择测试用例...")

        critical_tests = []
        related_tests = []
        smoke_tests = []

        # 转换测试文件路径
        affected_test_files = []
        for test_path in impact.affected_tests:
            full_path = self.test_dir / test_path
            if full_path.exists():
                affected_test_files.append(str(full_path.relative_to(Path("."))))

        # 分类测试（简化版本）
        for test_file in affected_test_files[:20]:  # 限制数量，避免过多
            if any(keyword in test_file.lower() for keyword in ['core', 'main', 'critical']):
                critical_tests.append(test_file)
            else:
                related_tests.append(test_file)

        # 查找冒烟测试
        smoke_tests = self._find_smoke_tests_limited()

        # 执行优先级
        priority_order = (critical_tests + smoke_tests + related_tests)[:15]  # 进一步限制

        # 预估执行时间
        estimated_duration = len(priority_order) * 1.5  # 更保守的估算

        return TestSelection(
            critical_tests=critical_tests,
            related_tests=related_tests,
            smoke_tests=smoke_tests,
            priority_order=priority_order,
            estimated_duration=estimated_duration
        )

    def _find_smoke_tests_limited(self) -> List[str]:
        """有限查找冒烟测试"""
        smoke_tests = []

        try:
            # 只检查前50个测试文件
            for i, test_file in enumerate(self.test_dir.rglob("test_*.py")):
                if i >= 50:
                    break

                try:
                    with open(test_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read(1000)  # 只读取前1000字符
                        if 'pytest.mark.smoke' in content or 'smoke' in content.lower():
                            smoke_tests.append(str(test_file.relative_to(Path("."))))
                except Exception:
                    continue

        except Exception as e:
            logger.debug(f"查找冒烟测试失败: {e}")

        return smoke_tests[:5]  # 最多5个冒烟测试

    def run_tests_sequentially(self, selection: TestSelection) -> Dict[str, Any]:
        """顺序执行测试，避免并行问题"""
        logger.info(f"顺序执行 {len(selection.priority_order)} 个测试...")

        results = {
            'executed_tests': [],
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'skipped': 0,
            'total_duration': 0.0
        }

        # 顺序执行，避免编码和资源问题
        for i, test_file in enumerate(selection.priority_order):
            logger.debug(f"执行测试 {i+1}/{len(selection.priority_order)}: {test_file}")

            test_result = self._run_single_test_safe(test_file)
            results['executed_tests'].append(test_result)
            results['passed'] += test_result.get('passed', 0)
            results['failed'] += test_result.get('failed', 0)
            results['errors'] += test_result.get('errors', 0)
            results['skipped'] += test_result.get('skipped', 0)
            results['total_duration'] += test_result.get('duration', 0)

            # 进度报告
            if (i + 1) % 5 == 0:
                logger.info(f"已执行 {i+1}/{len(selection.priority_order)} 个测试")

        logger.info("增量测试执行完成")
        return results

    def _run_single_test_safe(self, test_file: str) -> Dict[str, Any]:
        """安全执行单个测试"""
        start_time = time.time()

        try:
            # 构建命令
            cmd = [
                "python", "-m", "pytest",
                test_file,
                "--tb=no",  # 不显示traceback，减少输出
                "--quiet",
                "--disable-warnings",
                "-x",  # 遇到第一个失败就停止
                "--maxfail=3"  # 最多失败3次
            ]

            # 执行测试，设置更短的超时
            result = subprocess.run(
                cmd,
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                timeout=30,  # 30秒超时
                encoding=self.encoding,
                errors=self.errors
            )

            duration = time.time() - start_time

            # 解析结果
            passed, failed, errors, skipped = self._parse_pytest_output_safe(result.stdout)

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
            duration = time.time() - start_time
            logger.warning(f"测试超时: {test_file}")
            return {
                'test_file': test_file,
                'duration': duration,
                'passed': 0,
                'failed': 0,
                'errors': 1,
                'skipped': 0,
                'success': False
            }
        except Exception as e:
            duration = time.time() - start_time
            logger.warning(f"测试执行异常 {test_file}: {e}")
            return {
                'test_file': test_file,
                'duration': duration,
                'passed': 0,
                'failed': 0,
                'errors': 1,
                'skipped': 0,
                'success': False
            }

    def _parse_pytest_output_safe(self, output: str) -> Tuple[int, int, int, int]:
        """安全解析pytest输出"""
        passed = failed = errors = skipped = 0

        try:
            # 查找总结行
            lines = output.split('\n')
            for line in reversed(lines):
                line = line.strip()
                if 'passed' in line.lower() and ('failed' in line.lower() or 'error' in line.lower() or 'skipped' in line.lower()):
                    # 简单的数字提取
                    import re
                    numbers = re.findall(r'(\d+)', line)
                    if len(numbers) >= 4:
                        passed = int(numbers[0])
                        failed = int(numbers[1])
                        errors = int(numbers[2])
                        skipped = int(numbers[3])
                    break
        except Exception as e:
            logger.debug(f"解析pytest输出失败: {e}")

        return passed, failed, errors, skipped

    def generate_report_optimized(self, impact: ChangeImpact, selection: TestSelection, results: Dict[str, Any]):
        """生成优化版报告"""
        report_path = Path("test_logs/incremental_test_report_optimized.md")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 增量测试执行报告 (优化版)\n\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## 执行概览\n\n")
            f.write("- **检测方式**: Git检测 + 文件时间戳\n")
            f.write("- **并行策略**: 顺序执行 (避免编码问题)\n")
            f.write(f"- **编码处理**: {self.encoding} with error replacement\n")
            f.write(".1")
            f.write("## 变更分析\n\n")
            f.write(f"- **变更文件数**: {len(impact.changed_files)}\n")
            f.write(f"- **受影响模块数**: {len(impact.affected_modules)}\n")
            f.write(f"- **受影响测试数**: {len(impact.affected_tests)}\n")
            f.write(f"- **风险等级**: {impact.risk_level.upper()}\n")
            f.write(".1")
            f.write("## 测试选择\n\n")
            f.write(f"- **关键测试**: {len(selection.critical_tests)}\n")
            f.write(f"- **相关测试**: {len(selection.related_tests)}\n")
            f.write(f"- **冒烟测试**: {len(selection.smoke_tests)}\n")
            f.write(f"- **实际执行**: {len(selection.priority_order)}\n")
            f.write(".1")
            f.write("## 执行结果\n\n")
            f.write(f"- **成功执行**: {sum(1 for r in results['executed_tests'] if r['success'])}\n")
            f.write(f"- **失败执行**: {sum(1 for r in results['executed_tests'] if not r['success'])}\n")
            f.write(f"- **通过测试**: {results['passed']}\n")
            f.write(f"- **失败测试**: {results['failed']}\n")
            f.write(f"- **错误数**: {results['errors']}\n")
            f.write(f"- **跳过数**: {results['skipped']}\n")

            if results['executed_tests']:
                f.write("\n### 测试详情\n\n")
                f.write("| 测试文件 | 结果 | 时间 |\n")
                f.write("|----------|------|------|\n")

                for test_result in results['executed_tests'][:10]:  # 只显示前10个
                    status = "✅" if test_result['success'] else "❌"
                    f.write(f"| `{Path(test_result['test_file']).name}` | {status} | {test_result['duration']:.2f}s |\n")

            f.write("\n## 性能分析\n\n")
            f.write(f"- **时间节省**: {impact.execution_time_savings:.1f}%\n")
            f.write("- **执行效率**: 顺序执行，避免资源竞争\n")
            f.write("- **稳定性**: 编码错误处理 + 超时控制\n")

            f.write("\n## 优化亮点\n\n")
            f.write("1. **编码安全**: 正确处理Windows环境编码问题\n")
            f.write("2. **顺序执行**: 避免并行导致的资源耗尽\n")
            f.write("3. **缓存优化**: 测试映射结果缓存，提高后续运行速度\n")
            f.write("4. **超时控制**: 单个测试30秒超时，避免无限等待\n")
            f.write("5. **数量限制**: 智能限制执行的测试数量\n")

        logger.info(f"优化版报告已生成: {report_path}")

    def run_optimized_cycle(self) -> Dict[str, Any]:
        """运行优化版增量测试周期"""
        logger.info("开始优化版增量测试周期...")

        start_time = time.time()

        # 1. 快速检测变更
        changed_files = self.detect_changes_fast()

        if not changed_files:
            logger.info("未检测到变更，跳过测试")
            return {
                'status': 'no_changes',
                'message': '未检测到代码变更',
                'total_time': time.time() - start_time
            }

        # 2. 高效分析影响
        impact = self.analyze_impact_efficient(changed_files)

        # 3. 智能选择测试
        selection = self.select_tests_smart(impact)

        if not selection.priority_order:
            logger.info("没有需要执行的测试")
            return {
                'status': 'no_tests',
                'message': '没有匹配的测试用例',
                'total_time': time.time() - start_time
            }

        # 4. 顺序执行测试
        results = self.run_tests_sequentially(selection)

        # 5. 生成优化版报告
        self.generate_report_optimized(impact, selection, results)

        total_time = time.time() - start_time

        summary = {
            'status': 'completed',
            'changed_files': len(changed_files),
            'affected_tests': len(impact.affected_tests),
            'executed_tests': len(results['executed_tests']),
            'passed': results['passed'],
            'failed': results['failed'],
            'time_savings': impact.execution_time_savings,
            'risk_level': impact.risk_level,
            'total_time': total_time,
            'avg_test_time': results['total_duration'] / len(results['executed_tests']) if results['executed_tests'] else 0
        }

        logger.info(".2")
        return summary


def main():
    """主函数"""
    # 创建优化版测试器，限制并行度和资源使用
    tester = OptimizedIncrementalTester(max_workers=1)  # 完全顺序执行

    print("🚀 优化版增量测试选择器启动")
    print("🎯 特性: 编码安全 + 顺序执行 + 缓存优化 + 超时控制")

    # 运行优化版增量测试周期
    summary = tester.run_optimized_cycle()

    print("\n📊 执行结果:")
    if summary['status'] == 'no_changes':
        print("  ℹ️ 未检测到代码变更")
    elif summary['status'] == 'no_tests':
        print("  ℹ️ 没有需要执行的测试")
    else:
        print(f"  📁 变更文件: {summary['changed_files']}")
        print(f"  🧪 受影响测试: {summary['affected_tests']}")
        print(f"  ▶️ 执行测试: {summary['executed_tests']}")
        print(f"  ✅ 通过: {summary['passed']}")
        print(f"  ❌ 失败: {summary['failed']}")
        print(".1")
        print(".2")
        print(f"  ⚠️ 风险等级: {summary['risk_level'].upper()}")

    print("\n📄 详细报告已保存到: test_logs/incremental_test_report_optimized.md")
    print("\n✅ 优化版增量测试选择器运行完成")


if __name__ == "__main__":
    main()
