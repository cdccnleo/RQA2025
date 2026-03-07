#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多语言测试适配器

扩展测试框架支持多种编程语言：
- JavaScript/TypeScript (Node.js, Jest)
- Java (JUnit, TestNG)
- Go (testing包, Ginkgo)
- C# (.NET测试框架)
- 统一报告生成和CI/CD集成
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import platform

logger = logging.getLogger(__name__)


@dataclass
class LanguageConfig:
    """语言配置"""
    name: str
    extensions: List[str]
    test_commands: List[str]
    coverage_commands: List[str]
    report_formats: List[str]
    package_managers: List[str]
    runtime_requirements: List[str]


@dataclass
class TestResult:
    """测试结果"""
    language: str
    passed: int
    failed: int
    errors: int
    skipped: int
    total_time: float
    coverage: Optional[float]
    success: bool
    output: str
    report_path: Optional[str]


class LanguageAdapter(ABC):
    """语言适配器基类"""

    def __init__(self, config: LanguageConfig):
        self.config = config
        self.project_root = Path.cwd()

    @abstractmethod
    def detect_projects(self) -> List[Path]:
        """检测项目"""
        pass

    @abstractmethod
    def setup_environment(self, project_path: Path) -> bool:
        """设置环境"""
        pass

    @abstractmethod
    def run_tests(self, project_path: Path, coverage: bool = False) -> TestResult:
        """运行测试"""
        pass

    @abstractmethod
    def parse_test_output(self, output: str, returncode: int) -> Dict[str, Any]:
        """解析测试输出"""
        pass

    def is_available(self) -> bool:
        """检查语言运行时是否可用"""
        return all(self._check_runtime(req) for req in self.config.runtime_requirements)

    def _check_runtime(self, requirement: str) -> bool:
        """检查运行时要求"""
        try:
            if requirement.startswith("command:"):
                cmd = requirement.split(":", 1)[1]
                subprocess.run(cmd.split(), capture_output=True, timeout=5)
                return True
            elif requirement.startswith("file:"):
                file_path = requirement.split(":", 1)[1]
                return Path(file_path).exists()
            else:
                # 检查命令是否存在
                subprocess.run([requirement, "--version"], capture_output=True, timeout=5)
                return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False


class JavaScriptAdapter(LanguageAdapter):
    """JavaScript/TypeScript适配器"""

    def __init__(self):
        config = LanguageConfig(
            name="JavaScript/TypeScript",
            extensions=[".js", ".ts", ".jsx", ".tsx"],
            test_commands=[
                "npm test",
                "yarn test",
                "jest",
                "vitest"
            ],
            coverage_commands=[
                "npm run test:coverage",
                "yarn test:coverage",
                "jest --coverage",
                "vitest --coverage"
            ],
            report_formats=["lcov", "json", "html"],
            package_managers=["npm", "yarn", "pnpm"],
            runtime_requirements=["command:node --version", "command:npm --version"]
        )
        super().__init__(config)

    def detect_projects(self) -> List[Path]:
        """检测JS/TS项目"""
        projects = []

        # 查找package.json文件
        for package_json in self.project_root.rglob("package.json"):
            project_dir = package_json.parent

            # 检查是否有测试相关配置
            try:
                with open(package_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if any(key in data for key in ["scripts", "jest", "vitest"]):
                    projects.append(project_dir)
            except Exception:
                continue

        return projects

    def setup_environment(self, project_path: Path) -> bool:
        """设置JS环境"""
        try:
            # 检查node_modules是否存在
            if not (project_path / "node_modules").exists():
                logger.info(f"安装依赖: {project_path}")
                for pm in self.config.package_managers:
                    try:
                        if pm == "npm":
                            subprocess.run([pm, "install"], cwd=project_path, timeout=300)
                        elif pm == "yarn":
                            subprocess.run([pm, "install"], cwd=project_path, timeout=300)
                        break
                    except Exception:
                        continue

            return True
        except Exception as e:
            logger.error(f"JS环境设置失败: {e}")
            return False

    def run_tests(self, project_path: Path, coverage: bool = False) -> TestResult:
        """运行JS测试"""
        start_time = time.time()

        try:
            # 选择测试命令
            commands = self.config.coverage_commands if coverage else self.config.test_commands

            for cmd in commands:
                try:
                    logger.info(f"运行JS测试: {cmd}")
                    result = subprocess.run(
                        cmd.split(),
                        cwd=project_path,
                        capture_output=True,
                        text=True,
                        timeout=600  # 10分钟超时
                    )

                    duration = time.time() - start_time

                    # 解析结果
                    parsed = self.parse_test_output(result.stdout + result.stderr, result.returncode)

                    return TestResult(
                        language=self.config.name,
                        passed=parsed.get('passed', 0),
                        failed=parsed.get('failed', 0),
                        errors=parsed.get('errors', 0),
                        skipped=parsed.get('skipped', 0),
                        total_time=duration,
                        coverage=parsed.get('coverage'),
                        success=result.returncode == 0,
                        output=result.stdout + result.stderr,
                        report_path=parsed.get('report_path')
                    )

                except subprocess.TimeoutExpired:
                    logger.warning(f"JS测试超时: {cmd}")
                    continue
                except Exception as e:
                    logger.warning(f"JS测试命令失败 {cmd}: {e}")
                    continue

            # 如果所有命令都失败了
            return TestResult(
                language=self.config.name,
                passed=0, failed=0, errors=1, skipped=0,
                total_time=time.time() - start_time,
                coverage=None,
                success=False,
                output="所有测试命令都失败",
                report_path=None
            )

        except Exception as e:
            return TestResult(
                language=self.config.name,
                passed=0, failed=0, errors=1, skipped=0,
                total_time=time.time() - start_time,
                coverage=None,
                success=False,
                output=f"JS测试执行异常: {e}",
                report_path=None
            )

    def parse_test_output(self, output: str, returncode: int) -> Dict[str, Any]:
        """解析JS测试输出"""
        result = {
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'skipped': 0,
            'coverage': None,
            'report_path': None
        }

        try:
            lines = output.split('\n')

            # 查找Jest/Vitest风格的输出
            for line in lines:
                line = line.strip().lower()

                # Jest/Vitest格式: "Tests: 5 passed, 2 failed, 1 skipped"
                if 'tests:' in line and ('passed' in line or 'failed' in line):
                    import re
                    # 匹配数字
                    passed_match = re.search(r'(\d+)\s*passed', line)
                    failed_match = re.search(r'(\d+)\s*failed', line)
                    skipped_match = re.search(r'(\d+)\s*skipped', line)

                    if passed_match:
                        result['passed'] = int(passed_match.group(1))
                    if failed_match:
                        result['failed'] = int(failed_match.group(1))
                    if skipped_match:
                        result['skipped'] = int(skipped_match.group(1))
                    break

                # 覆盖率信息
                if 'coverage' in line and '%' in line:
                    cov_match = re.search(r'(\d+(?:\.\d+)?)%', line)
                    if cov_match:
                        result['coverage'] = float(cov_match.group(1))

        except Exception as e:
            logger.debug(f"JS输出解析失败: {e}")

        return result


class JavaAdapter(LanguageAdapter):
    """Java适配器"""

    def __init__(self):
        config = LanguageConfig(
            name="Java",
            extensions=[".java"],
            test_commands=[
                "mvn test",
                "gradle test",
                "./gradlew test",
                "mvn surefire:test"
            ],
            coverage_commands=[
                "mvn test jacoco:report",
                "gradle test jacocoTestReport",
                "./gradlew test jacocoTestReport"
            ],
            report_formats=["html", "xml", "json"],
            package_managers=["mvn", "gradle"],
            runtime_requirements=["command:java -version", "command:mvn --version"]
        )
        super().__init__(config)

    def detect_projects(self) -> List[Path]:
        """检测Java项目"""
        projects = []

        # 查找pom.xml或build.gradle
        for pom in self.project_root.rglob("pom.xml"):
            projects.append(pom.parent)

        for gradle in self.project_root.rglob("build.gradle"):
            projects.append(gradle.parent)

        for gradle_kts in self.project_root.rglob("build.gradle.kts"):
            projects.append(gradle_kts.parent)

        return list(set(projects))  # 去重

    def setup_environment(self, project_path: Path) -> bool:
        """设置Java环境"""
        try:
            # 对于Maven项目，依赖通常已经下载
            # 对于Gradle项目也是
            return True
        except Exception as e:
            logger.error(f"Java环境设置失败: {e}")
            return False

    def run_tests(self, project_path: Path, coverage: bool = False) -> TestResult:
        """运行Java测试"""
        import time
        start_time = time.time()

        try:
            commands = self.config.coverage_commands if coverage else self.config.test_commands

            for cmd in commands:
                try:
                    logger.info(f"运行Java测试: {cmd}")
                    result = subprocess.run(
                        cmd.split(),
                        cwd=project_path,
                        capture_output=True,
                        text=True,
                        timeout=900  # 15分钟超时
                    )

                    duration = time.time() - start_time
                    parsed = self.parse_test_output(result.stdout + result.stderr, result.returncode)

                    return TestResult(
                        language=self.config.name,
                        passed=parsed.get('passed', 0),
                        failed=parsed.get('failed', 0),
                        errors=parsed.get('errors', 0),
                        skipped=parsed.get('skipped', 0),
                        total_time=duration,
                        coverage=parsed.get('coverage'),
                        success=result.returncode == 0,
                        output=result.stdout + result.stderr,
                        report_path=parsed.get('report_path')
                    )

                except subprocess.TimeoutExpired:
                    logger.warning(f"Java测试超时: {cmd}")
                    continue
                except Exception as e:
                    logger.warning(f"Java测试命令失败 {cmd}: {e}")
                    continue

            return TestResult(
                language=self.config.name,
                passed=0, failed=0, errors=1, skipped=0,
                total_time=time.time() - start_time,
                coverage=None,
                success=False,
                output="所有Java测试命令都失败",
                report_path=None
            )

        except Exception as e:
            return TestResult(
                language=self.config.name,
                passed=0, failed=0, errors=1, skipped=0,
                total_time=time.time() - start_time,
                coverage=None,
                success=False,
                output=f"Java测试执行异常: {e}",
                report_path=None
            )

    def parse_test_output(self, output: str, returncode: int) -> Dict[str, Any]:
        """解析Java测试输出"""
        result = {
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'skipped': 0,
            'coverage': None,
            'report_path': None
        }

        try:
            lines = output.split('\n')

            # Maven Surefire格式
            for line in lines:
                line = line.strip()

                # Tests run: 5, Failures: 2, Errors: 1, Skipped: 0
                if 'Tests run:' in line:
                    import re
                    run_match = re.search(r'Tests run:\s*(\d+)', line)
                    fail_match = re.search(r'Failures:\s*(\d+)', line)
                    error_match = re.search(r'Errors:\s*(\d+)', line)
                    skip_match = re.search(r'Skipped:\s*(\d+)', line)

                    if run_match:
                        total = int(run_match.group(1))
                        failed = int(fail_match.group(1)) if fail_match else 0
                        errors = int(error_match.group(1)) if error_match else 0
                        skipped = int(skip_match.group(1)) if skip_match else 0
                        passed = total - failed - errors - skipped

                        result.update({
                            'passed': passed,
                            'failed': failed,
                            'errors': errors,
                            'skipped': skipped
                        })
                    break

        except Exception as e:
            logger.debug(f"Java输出解析失败: {e}")

        return result


class GoAdapter(LanguageAdapter):
    """Go适配器"""

    def __init__(self):
        config = LanguageConfig(
            name="Go",
            extensions=[".go"],
            test_commands=[
                "go test ./...",
                "go test -v ./..."
            ],
            coverage_commands=[
                "go test -cover ./...",
                "go test -coverprofile=coverage.out ./..."
            ],
            report_formats=["text", "json", "html"],
            package_managers=["go mod"],
            runtime_requirements=["command:go version"]
        )
        super().__init__(config)

    def detect_projects(self) -> List[Path]:
        """检测Go项目"""
        projects = []

        # 查找go.mod文件
        for go_mod in self.project_root.rglob("go.mod"):
            projects.append(go_mod.parent)

        # 查找包含*_test.go文件的目录
        for test_file in self.project_root.rglob("*_test.go"):
            project_dir = test_file.parent
            if project_dir not in projects:
                projects.append(project_dir)

        return projects

    def setup_environment(self, project_path: Path) -> bool:
        """设置Go环境"""
        try:
            # 下载依赖
            subprocess.run(["go", "mod", "download"], cwd=project_path, timeout=300)
            return True
        except Exception as e:
            logger.error(f"Go环境设置失败: {e}")
            return False

    def run_tests(self, project_path: Path, coverage: bool = False) -> TestResult:
        """运行Go测试"""
        import time
        start_time = time.time()

        try:
            commands = self.config.coverage_commands if coverage else self.config.test_commands

            for cmd in commands:
                try:
                    logger.info(f"运行Go测试: {cmd}")
                    result = subprocess.run(
                        cmd.split(),
                        cwd=project_path,
                        capture_output=True,
                        text=True,
                        timeout=600  # 10分钟超时
                    )

                    duration = time.time() - start_time
                    parsed = self.parse_test_output(result.stdout + result.stderr, result.returncode)

                    return TestResult(
                        language=self.config.name,
                        passed=parsed.get('passed', 0),
                        failed=parsed.get('failed', 0),
                        errors=parsed.get('errors', 0),
                        skipped=parsed.get('skipped', 0),
                        total_time=duration,
                        coverage=parsed.get('coverage'),
                        success=result.returncode == 0,
                        output=result.stdout + result.stderr,
                        report_path=parsed.get('report_path')
                    )

                except subprocess.TimeoutExpired:
                    logger.warning(f"Go测试超时: {cmd}")
                    continue
                except Exception as e:
                    logger.warning(f"Go测试命令失败 {cmd}: {e}")
                    continue

            return TestResult(
                language=self.config.name,
                passed=0, failed=0, errors=1, skipped=0,
                total_time=time.time() - start_time,
                coverage=None,
                success=False,
                output="所有Go测试命令都失败",
                report_path=None
            )

        except Exception as e:
            return TestResult(
                language=self.config.name,
                passed=0, failed=0, errors=1, skipped=0,
                total_time=time.time() - start_time,
                coverage=None,
                success=False,
                output=f"Go测试执行异常: {e}",
                report_path=None
            )

    def parse_test_output(self, output: str, returncode: int) -> Dict[str, Any]:
        """解析Go测试输出"""
        result = {
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'skipped': 0,
            'coverage': None,
            'report_path': None
        }

        try:
            lines = output.split('\n')

            # 统计PASS/FAIL/SKIP
            passed = 0
            failed = 0
            skipped = 0

            for line in lines:
                line = line.strip()
                if line.startswith('PASS'):
                    passed += 1
                elif line.startswith('FAIL'):
                    failed += 1
                elif line.startswith('SKIP'):
                    skipped += 1
                elif 'coverage:' in line:
                    # coverage: 85.2% of statements
                    import re
                    cov_match = re.search(r'coverage:\s*(\d+(?:\.\d+)?)%', line)
                    if cov_match:
                        result['coverage'] = float(cov_match.group(1))

            result.update({
                'passed': passed,
                'failed': failed,
                'skipped': skipped
            })

        except Exception as e:
            logger.debug(f"Go输出解析失败: {e}")

        return result


class MultiLanguageTestRunner:
    """多语言测试运行器"""

    def __init__(self):
        self.adapters = {
            'javascript': JavaScriptAdapter(),
            'java': JavaAdapter(),
            'go': GoAdapter()
        }

        # 检测可用的语言
        self.available_languages = []
        for lang, adapter in self.adapters.items():
            if adapter.is_available():
                self.available_languages.append(lang)
            else:
                logger.info(f"语言 {lang} 不可用，跳过")

        logger.info(f"多语言测试运行器初始化，支持语言: {self.available_languages}")

    def discover_all_projects(self) -> Dict[str, List[Path]]:
        """发现所有语言的项目"""
        projects = {}

        for lang, adapter in self.adapters.items():
            if lang in self.available_languages:
                try:
                    lang_projects = adapter.detect_projects()
                    if lang_projects:
                        projects[lang] = lang_projects
                        logger.info(f"发现 {lang} 项目 {len(lang_projects)} 个")
                except Exception as e:
                    logger.error(f"发现 {lang} 项目失败: {e}")

        return projects

    def run_all_tests(self, coverage: bool = False) -> Dict[str, List[TestResult]]:
        """运行所有语言的测试"""
        logger.info("开始运行多语言测试...")

        results = {}
        projects = self.discover_all_projects()

        for lang, lang_projects in projects.items():
            if lang not in self.available_languages:
                continue

            adapter = self.adapters[lang]
            lang_results = []

            for project_path in lang_projects:
                try:
                    logger.info(f"测试 {lang} 项目: {project_path}")

                    # 设置环境
                    if not adapter.setup_environment(project_path):
                        logger.warning(f"{lang} 项目环境设置失败: {project_path}")
                        continue

                    # 运行测试
                    result = adapter.run_tests(project_path, coverage)
                    lang_results.append(result)

                    status = "✅" if result.success else "❌"
                    logger.info(".2")
                except Exception as e:
                    logger.error(f"测试 {lang} 项目异常 {project_path}: {e}")

            if lang_results:
                results[lang] = lang_results

        logger.info("多语言测试执行完成")
        return results

    def generate_unified_report(self, results: Dict[str, List[TestResult]]):
        """生成统一报告"""
        report_path = Path("test_logs/multilang_test_report.md")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 多语言测试报告\n\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            total_projects = 0
            total_tests = 0
            total_passed = 0
            total_failed = 0
            total_time = 0.0

            f.write("## 📊 各语言测试结果\n\n")
            f.write("| 语言 | 项目数 | 通过 | 失败 | 错误 | 跳过 | 总时间 | 成功率 |\n")
            f.write("|------|--------|------|------|------|------|--------|--------|\n")

            for lang, lang_results in results.items():
                lang_projects = len(lang_results)
                lang_passed = sum(r.passed for r in lang_results)
                lang_failed = sum(r.failed for r in lang_results)
                lang_errors = sum(r.errors for r in lang_results)
                lang_skipped = sum(r.skipped for r in lang_results)
                lang_time = sum(r.total_time for r in lang_results)
                lang_total = lang_passed + lang_failed + lang_errors + lang_skipped
                lang_success_rate = (lang_passed / lang_total * 100) if lang_total > 0 else 0

                f.write(f"| {lang} | {lang_projects} | {lang_passed} | {lang_failed} | {lang_errors} | {lang_skipped} | {lang_time:.2f}s | {lang_success_rate:.1f}% |\n")

                total_projects += lang_projects
                total_passed += lang_passed
                total_failed += lang_failed
                total_time += lang_time

            total_tests = total_passed + total_failed
            overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

            f.write("\n## 📈 总体统计\n\n")
            f.write(f"- **支持语言数**: {len(results)}\n")
            f.write(f"- **总项目数**: {total_projects}\n")
            f.write(f"- **总测试数**: {total_tests}\n")
            f.write(".2")
            f.write(".1")
            f.write("## 🚀 多语言测试价值\n\n")
            f.write("### 对全栈开发团队的价值\n")
            f.write("1. **统一测试体验**: 无论使用什么语言，都有一致的测试运行和报告体验\n")
            f.write("2. **质量保证**: 多语言项目的整体质量通过统一标准进行保障\n")
            f.write("3. **CI/CD集成**: 支持多语言项目的自动化测试流水线\n")
            f.write("4. **开发效率**: 减少语言切换带来的测试环境配置开销\n")
            f.write("\n### 对企业架构的价值\n")
            f.write("1. **技术栈灵活性**: 支持多种技术栈并存的微服务架构\n")
            f.write("2. **质量标准化**: 跨语言的质量标准和测试规范\n")
            f.write("3. **风险控制**: 及早发现多语言组件间的集成问题\n")
            f.write("4. **维护效率**: 统一的测试基础设施降低维护复杂度\n")

        logger.info(f"统一测试报告已生成: {report_path}")

    def run_multilang_test_cycle(self) -> Dict[str, Any]:
        """运行完整的多语言测试周期"""
        logger.info("开始多语言测试周期...")

        start_time = time.time()

        # 运行所有测试
        results = self.run_all_tests(coverage=True)

        # 生成统一报告
        self.generate_unified_report(results)

        # 汇总统计
        summary = {
            'languages_tested': len(results),
            'total_projects': sum(len(lang_results) for lang_results in results.values()),
            'total_tests': sum(sum(r.passed + r.failed + r.errors + r.skipped for r in lang_results) for lang_results in results.values()),
            'total_passed': sum(sum(r.passed for r in lang_results) for lang_results in results.values()),
            'total_failed': sum(sum(r.failed for r in lang_results) for lang_results in results.values()),
            'total_time': sum(sum(r.total_time for r in lang_results) for lang_results in results.values()),
            'execution_time': time.time() - start_time
        }

        logger.info("多语言测试周期完成")
        return summary


def main():
    """主函数"""
    runner = MultiLanguageTestRunner()

    print("🌐 多语言测试适配器启动")
    print(f"🎯 支持语言: {', '.join(runner.available_languages)}")

    if not runner.available_languages:
        print("⚠️ 未检测到任何可用语言运行时")
        return

    # 运行多语言测试周期
    summary = runner.run_multilang_test_cycle()

    print("\n📊 多语言测试结果:")
    print(f"  🌍 测试语言: {summary['languages_tested']} 种")
    print(f"  📦 总项目数: {summary['total_projects']}")
    print(f"  🧪 总测试数: {summary['total_tests']}")
    print(f"  ✅ 通过测试: {summary['total_passed']}")
    print(f"  ❌ 失败测试: {summary['total_failed']}")
    print(".2")
    print(".2")
    print("\n📄 详细报告已保存到: test_logs/multilang_test_report.md")
    print("\n✅ 多语言测试适配器运行完成")


if __name__ == "__main__":
    import time
    main()
