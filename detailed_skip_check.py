#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
详细检查跳过测试的脚本
"""

import os
import re
import subprocess
import sys
from pathlib import Path


class DetailedSkipChecker:
    """详细跳过检查器"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.tests_path = self.project_root / 'tests' / 'unit' / 'infrastructure' / 'health'

    def check_all_possible_skips(self):
        """检查所有可能的跳过情况"""
        print("🔍 开始详细跳过检查...")
        print("=" * 60)

        # 1. 检查静态跳过调用
        print("📋 1. 检查静态跳过调用...")
        static_skips = self.check_static_skips()
        print(f"   静态跳过调用: {static_skips} 个")

        # 2. 检查运行时跳过
        print("🏃 2. 检查运行时跳过...")
        runtime_skips = self.check_runtime_skips()
        print(f"   运行时跳过测试: {runtime_skips} 个")

        # 3. 检查条件跳过逻辑
        print("🔀 3. 检查条件跳过逻辑...")
        conditional_skips = self.check_conditional_skips()
        print(f"   条件跳过逻辑: {conditional_skips} 个")

        # 4. 检查导入相关跳过
        print("📦 4. 检查导入相关跳过...")
        import_skips = self.check_import_related_skips()
        print(f"   导入相关跳过: {import_skips} 个")

        # 5. 检查测试标记
        print("🏷️  5. 检查测试标记...")
        markers = self.check_test_markers()
        print(f"   跳过标记: {markers} 个")

        total_potential_skips = static_skips + conditional_skips + import_skips + markers

        print("\n" + "=" * 60)
        print("📊 详细检查结果:")
        print(f"   总潜在跳过: {total_potential_skips}")
        print(f"   实际运行跳过: {runtime_skips}")

        if runtime_skips == 0 and total_potential_skips == 0:
            print("✅ 确认无跳过测试问题")
        elif runtime_skips == 0 and total_potential_skips > 0:
            print("⚠️  有潜在跳过逻辑但未触发")
        else:
            print(f"⚠️  发现 {runtime_skips} 个运行时跳过测试")

        return runtime_skips, total_potential_skips

    def check_static_skips(self):
        """检查静态跳过调用"""
        count = 0
        skip_patterns = [
            r'pytest\.skip\s*\(',
            r'@pytest\.mark\.skip',
            r'@pytest\.mark\.skipif',
            r'unittest\.skip',
            r'self\.skipTest\s*\('
        ]

        for py_file in self.tests_path.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                for pattern in skip_patterns:
                    matches = re.findall(pattern, content)
                    count += len(matches)

            except Exception as e:
                print(f"   错误读取文件 {py_file}: {e}")

        return count

    def check_runtime_skips(self):
        """检查运行时跳过"""
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pytest',
                'tests/unit/infrastructure/health/',
                '--maxfail=10', '--tb=no', '-q'
            ], capture_output=True, text=True, cwd=self.project_root, timeout=120)

            return result.stdout.count('SKIPPED')
        except subprocess.TimeoutExpired:
            print("   运行时检查超时")
            return -1
        except Exception as e:
            print(f"   运行时检查错误: {e}")
            return -1

    def check_conditional_skips(self):
        """检查条件跳过逻辑"""
        count = 0
        conditional_patterns = [
            r'if.*:\s*return',
            r'if.*:\s*pass',
            r'if.*:\s*continue',
            r'raise.*SkipTest',
            r'skip.*if.*not'
        ]

        for py_file in self.tests_path.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                for pattern in conditional_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    count += len(matches)

            except Exception as e:
                print(f"   错误读取文件 {py_file}: {e}")

        return count

    def check_import_related_skips(self):
        """检查导入相关跳过"""
        count = 0
        import_patterns = [
            r'except\s+ImportError',
            r'except.*ImportError',
            r'try:\s*import',
            r'if.*import.*failed',
            r'skip.*import'
        ]

        for py_file in self.tests_path.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                for pattern in import_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    count += len(matches)

            except Exception as e:
                print(f"   错误读取文件 {py_file}: {e}")

        return count

    def check_test_markers(self):
        """检查测试标记"""
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pytest',
                'tests/unit/infrastructure/health/',
                '--collect-only', '--markers'
            ], capture_output=True, text=True, cwd=self.project_root, timeout=30)

            skip_markers = 0
            for line in result.stdout.split('\n'):
                if 'skip' in line.lower():
                    skip_markers += 1

            return skip_markers
        except:
            return 0

    def analyze_skip_sources(self):
        """分析跳过来源"""
        print("\n🔍 分析跳过来源...")

        # 检查是否有特定的跳过模式
        skip_sources = {
            'alibi_detect依赖': 0,
            'prometheus依赖': 0,
            '数据库依赖': 0,
            '网络依赖': 0,
            '其他条件': 0
        }

        for py_file in self.tests_path.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                if 'alibi' in content.lower():
                    skip_sources['alibi_detect依赖'] += 1
                if 'prometheus' in content.lower():
                    skip_sources['prometheus依赖'] += 1
                if 'database' in content.lower() or 'db' in content.lower():
                    skip_sources['数据库依赖'] += 1
                if 'network' in content.lower() or 'http' in content.lower():
                    skip_sources['网络依赖'] += 1

            except Exception as e:
                print(f"   错误读取文件 {py_file}: {e}")

        print("   潜在跳过来源:")
        for source, count in skip_sources.items():
            if count > 0:
                print(f"     {source}: {count} 个文件")

        return skip_sources


def main():
    """主函数"""
    checker = DetailedSkipChecker()

    # 执行详细检查
    runtime_skips, potential_skips = checker.check_all_possible_skips()

    # 分析跳过来源
    checker.analyze_skip_sources()

    print("\n" + "=" * 60)
    if runtime_skips == 0:
        print("🎉 确认：当前无跳过测试问题！")
        print("   如果您看到跳过测试，可能是在不同的环境或时间点。")
        return 0
    else:
        print(f"⚠️  发现 {runtime_skips} 个跳过测试，需要修复")
        return 1


if __name__ == "__main__":
    exit(main())
