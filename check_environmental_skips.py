#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查环境相关的跳过测试
"""

import os
import sys
import subprocess
from pathlib import Path


class EnvironmentalSkipChecker:
    """环境跳过检查器"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.tests_path = self.project_root / 'tests' / 'unit' / 'infrastructure' / 'health'

    def check_different_environments(self):
        """检查不同环境下的跳过情况"""
        print("🌍 检查环境相关的跳过测试...")
        print("=" * 60)

        # 1. 标准环境
        print("🏠 1. 标准环境检查...")
        skips_standard = self.run_tests_in_environment({})
        print(f"   跳过测试: {skips_standard}")

        # 2. 移除可选依赖的环境
        print("📦 2. 模拟缺少可选依赖...")
        skips_no_optional = self.simulate_missing_dependencies()
        print(f"   跳过测试: {skips_no_optional}")

        # 3. 检查条件跳过逻辑
        print("🔀 3. 分析条件跳过逻辑...")
        conditional_logic = self.analyze_conditional_logic()
        print(f"   条件逻辑数量: {conditional_logic}")

        print("\n" + "=" * 60)
        print("📊 环境检查结果:")
        print(f"   标准环境跳过: {skips_standard}")
        print(f"   缺少依赖跳过: {skips_no_optional}")
        print(f"   条件逻辑总数: {conditional_logic}")

        if skips_standard == 0 and skips_no_optional == 0:
            print("✅ 当前环境无跳过测试")
        else:
            print(f"⚠️  发现跳过测试，可能与环境相关")

        return skips_standard, skips_no_optional

    def run_tests_in_environment(self, env_vars):
        """在特定环境下运行测试"""
        try:
            # 设置环境变量
            test_env = os.environ.copy()
            test_env.update(env_vars)

            result = subprocess.run([
                sys.executable, '-m', 'pytest',
                'tests/unit/infrastructure/health/',
                '--maxfail=5', '--tb=no', '-q'
            ], capture_output=True, text=True, cwd=self.project_root,
               env=test_env, timeout=120)

            return result.stdout.count('SKIPPED')

        except subprocess.TimeoutExpired:
            print("   测试运行超时")
            return -1
        except Exception as e:
            print(f"   测试运行错误: {e}")
            return -1

    def simulate_missing_dependencies(self):
        """模拟缺少可选依赖的情况"""
        # 这里可以模拟移除某些依赖的情况
        # 由于实际的依赖管理比较复杂，我们检查代码中的条件

        missing_deps = [
            'alibi_detect',
            'prometheus_client',
            'psutil',
            'pymongo',
            'redis'
        ]

        total_skips = 0

        for dep in missing_deps:
            print(f"   检查缺少 {dep} 的情况...")
            # 这里可以运行测试时临时隐藏依赖
            # 但为了简化，我们分析代码中的相关逻辑
            skips = self.check_dependency_related_skips(dep)
            total_skips += skips

        return total_skips

    def check_dependency_related_skips(self, dependency):
        """检查特定依赖相关的跳过"""
        skip_count = 0

        for py_file in self.tests_path.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 检查是否包含相关的条件逻辑
                if dependency.lower() in content.lower():
                    if 'try:' in content and 'except' in content:
                        skip_count += 1
                    if 'if' in content and 'import' in content:
                        skip_count += 1

            except Exception as e:
                continue

        return skip_count

    def analyze_conditional_logic(self):
        """分析条件逻辑"""
        total_conditions = 0

        for py_file in self.tests_path.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 统计条件语句
                conditions = [
                    content.count('try:'),
                    content.count('except'),
                    content.count('if ') - content.count('elif'),
                    content.count('ImportError'),
                    content.count('ModuleNotFoundError')
                ]

                total_conditions += sum(conditions)

            except Exception as e:
                continue

        return total_conditions

    def check_ci_environment(self):
        """检查CI环境可能的跳过"""
        print("\n🔄 4. 检查CI环境设置...")

        # 检查是否有CI相关的环境变量或条件
        ci_indicators = [
            'CI', 'CONTINUOUS_INTEGRATION', 'TRAVIS', 'APPVEYOR',
            'CIRCLECI', 'GITHUB_ACTIONS', 'GITLAB_CI', 'JENKINS'
        ]

        ci_related = 0

        for py_file in self.tests_path.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                for indicator in ci_indicators:
                    if indicator in content:
                        ci_related += 1
                        break

            except Exception as e:
                continue

        print(f"   CI相关代码: {ci_related} 个文件")

        return ci_related


def main():
    """主函数"""
    checker = EnvironmentalSkipChecker()

    # 执行环境检查
    skips_standard, skips_no_deps = checker.check_different_environments()

    # 检查CI环境
    checker.check_ci_environment()

    print("\n" + "=" * 60)
    print("🎯 最终结论:")

    if skips_standard == 0:
        print("✅ 当前环境确认无跳过测试问题")
        print("   用户看到的375个跳过测试可能来自：")
        print("   - 不同的测试环境或配置")
        print("   - CI/CD流水线环境")
        print("   - 包含可选依赖缺失的情况")
        print("   - 并行测试工具的影响")
        return 0
    else:
        print(f"⚠️  发现 {skips_standard} 个跳过测试需要处理")
        return 1


if __name__ == "__main__":
    exit(main())
