#!/usr/bin/env python3
"""
增量测试执行器 - Phase 14.2 增量测试策略实现

基于git diff分析变更文件，实现智能增量测试执行
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import List, Set, Dict
from dataclasses import dataclass


@dataclass
class TestMapping:
    """测试映射配置"""
    file_pattern: str
    test_paths: List[str]
    priority: int = 1


class IncrementalTestRunner:
    """增量测试执行器"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.test_mappings = self._load_test_mappings()

    def _load_test_mappings(self) -> List[TestMapping]:
        """加载测试映射配置"""
        mappings = [
            # 核心业务逻辑 -> 对应测试
            TestMapping("src/**/*.py", [
                "tests/unit/", "tests/integration/"
            ], priority=5),

            # 数据层 -> 数据相关测试
            TestMapping("src/data/**/*.py", [
                "tests/unit/data/", "tests/integration/test_data_*.py"
            ], priority=4),

            # 基础设施层 -> 基础设施测试
            TestMapping("src/infrastructure/**/*.py", [
                "tests/unit/infrastructure/", "tests/integration/infrastructure/"
            ], priority=3),

            # 配置相关 -> 配置测试
            TestMapping("src/infrastructure/config/**/*.py", [
                "tests/unit/infrastructure/config/", "tests/integration/test_config_*.py"
            ], priority=4),

            # 缓存相关 -> 缓存测试
            TestMapping("src/infrastructure/cache/**/*.py", [
                "tests/unit/infrastructure/cache/", "tests/integration/cache/"
            ], priority=4),

            # ML层 -> ML测试
            TestMapping("src/ml/**/*.py", [
                "tests/unit/ml/", "tests/integration/test_ml_*.py"
            ], priority=4),

            # 交易层 -> 交易测试
            TestMapping("src/trading/**/*.py", [
                "tests/unit/trading/", "tests/integration/trading/"
            ], priority=5),

            # 风险控制层 -> 风险测试
            TestMapping("src/risk/**/*.py", [
                "tests/unit/risk/", "tests/integration/test_risk_*.py"
            ], priority=5),

            # 策略层 -> 策略测试
            TestMapping("src/strategy/**/*.py", [
                "tests/unit/strategy/", "tests/integration/test_strategy_*.py"
            ], priority=4),

            # 测试文件变更 -> 对应测试
            TestMapping("tests/**/*.py", [
                "tests/unit/test_smoke.py"  # 冒烟测试验证
            ], priority=2),
        ]
        return mappings

    def get_changed_files(self, base_ref: str = "HEAD~1") -> List[str]:
        """获取变更的文件列表"""
        try:
            # 获取git diff结果
            result = subprocess.run(
                ["git", "diff", "--name-only", base_ref],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            changed_files = result.stdout.strip().split('\n')
            return [f for f in changed_files if f.strip()]
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to get git diff: {e}")
            return []

    def get_staged_files(self) -> List[str]:
        """获取暂存区的文件"""
        try:
            result = subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            staged_files = result.stdout.strip().split('\n')
            return [f for f in staged_files if f.strip()]
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to get staged files: {e}")
            return []

    def map_files_to_tests(self, changed_files: List[str]) -> Set[str]:
        """将变更文件映射到测试路径"""
        test_paths = set()
        import fnmatch

        for changed_file in changed_files:
            if not changed_file.endswith('.py'):
                continue

            # 检查文件是否匹配映射模式
            for mapping in sorted(self.test_mappings, key=lambda x: x.priority, reverse=True):
                if fnmatch.fnmatch(changed_file, mapping.file_pattern):
                    test_paths.update(mapping.test_paths)
                    break

        return test_paths

    def run_incremental_tests(self, test_paths: Set[str], parallel: bool = True) -> bool:
        """运行增量测试"""
        if not test_paths:
            print("No tests to run based on changes")
            return True

        # 构建pytest命令
        cmd = [sys.executable, "-m", "pytest"]

        # 添加并行执行
        if parallel:
            cmd.extend(["-n", "auto", "--dist=loadscope"])

        # 添加测试路径
        for test_path in sorted(test_paths):
            if Path(self.project_root / test_path).exists():
                cmd.append(test_path)

        # 添加其他选项
        cmd.extend([
            "--tb=short",
            "--durations=10",
            "-q",
            "--maxfail=5"
        ])

        print(f"Running incremental tests: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, cwd=self.project_root)
            return result.returncode == 0
        except Exception as e:
            print(f"Error running tests: {e}")
            return False

    def run_regression_tests(self) -> bool:
        """运行回归测试"""
        print("Running regression tests...")

        cmd = [
            sys.executable, "-m", "pytest",
            "tests/unit/test_smoke.py",  # 冒烟测试
            "-v", "--tb=short"
        ]

        try:
            result = subprocess.run(cmd, cwd=self.project_root)
            return result.returncode == 0
        except Exception as e:
            print(f"Error running regression tests: {e}")
            return False

    def generate_report(self, changed_files: List[str], test_paths: Set[str],
                       test_result: bool) -> Dict:
        """生成执行报告"""
        return {
            "changed_files": changed_files,
            "mapped_test_paths": list(test_paths),
            "test_result": test_result,
            "timestamp": subprocess.run(
                ["date", "+%Y-%m-%d %H:%M:%S"],
                capture_output=True, text=True
            ).stdout.strip()
        }


def main():
    """主函数"""
    project_root = Path(__file__).parent.parent

    # 创建增量测试执行器
    runner = IncrementalTestRunner(project_root)

    # 获取变更文件
    changed_files = runner.get_changed_files()
    staged_files = runner.get_staged_files()
    all_changed = list(set(changed_files + staged_files))

    print(f"📊 检测到 {len(all_changed)} 个变更文件:")
    for file in all_changed[:10]:  # 只显示前10个
        print(f"  - {file}")
    if len(all_changed) > 10:
        print(f"  ... 还有 {len(all_changed) - 10} 个文件")

    # 映射到测试路径
    test_paths = runner.map_files_to_tests(all_changed)
    print(f"🎯 映射到 {len(test_paths)} 个测试路径:")
    for path in sorted(test_paths):
        print(f"  - {path}")

    # 运行增量测试
    if test_paths:
        print("\n🚀 开始增量测试执行...")
        test_result = runner.run_incremental_tests(test_paths)
    else:
        print("\n⚠️  未检测到需要运行的测试")
        test_result = True

    # 运行回归测试
    print("\n🔄 运行回归测试...")
    regression_result = runner.run_regression_tests()

    # 生成报告
    report = runner.generate_report(all_changed, test_paths, test_result and regression_result)

    # 保存报告
    report_file = project_root / "test_logs" / "incremental_test_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n📋 测试报告已保存到: {report_file}")

    # 返回结果
    overall_result = test_result and regression_result
    print(f"\n{'✅' if overall_result else '❌'} 增量测试执行{'成功' if overall_result else '失败'}")

    return 0 if overall_result else 1


if __name__ == "__main__":
    sys.exit(main())
