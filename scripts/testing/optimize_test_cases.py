#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试用例优化脚本
用于修复测试用例逻辑问题并提升测试质量
"""

import sys
import os
import subprocess
import json
from typing import List, Dict, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCaseOptimizer:
    """测试用例优化器"""

    def __init__(self):
        self.test_results = {}
        self.optimization_log = []
        self.failed_tests = []

    def analyze_test_coverage(self) -> Dict[str, Any]:
        """分析测试覆盖率"""
        print("🔍 分析测试覆盖率...")

        try:
            # 运行覆盖率测试
            result = subprocess.run([
                "conda", "run", "-n", "test", "python", "-m", "pytest",
                "--cov=src/data", "--cov-report=json", "--cov-report=term-missing",
                "tests/unit/data/", "-v"
            ], capture_output=True, text=True, cwd=os.getcwd())

            if result.returncode == 0:
                # 解析覆盖率报告
                coverage_data = json.loads(result.stdout)
                return {
                    "total_coverage": coverage_data.get("totals", {}).get("percent_covered", 0),
                    "missing_lines": coverage_data.get("missing_lines", []),
                    "covered_files": list(coverage_data.get("files", {}).keys())
                }
            else:
                print(f"⚠️ 覆盖率测试失败: {result.stderr}")
                return {"total_coverage": 0, "missing_lines": [], "covered_files": []}

        except Exception as e:
            print(f"❌ 覆盖率分析失败: {e}")
            return {"total_coverage": 0, "missing_lines": [], "covered_files": []}

    def identify_failing_tests(self) -> List[str]:
        """识别失败的测试用例"""
        print("🔍 识别失败的测试用例...")

        try:
            # 运行测试并捕获失败信息
            result = subprocess.run([
                "conda", "run", "-n", "test", "python", "-m", "pytest",
                "tests/unit/data/", "--tb=short", "-v"
            ], capture_output=True, text=True, cwd=os.getcwd())

            failed_tests = []
            if result.returncode != 0:
                # 解析失败信息
                lines = result.stdout.split('\n')
                for line in lines:
                    if "FAILED" in line and "test_" in line:
                        test_name = line.split()[0]
                        failed_tests.append(test_name)

            return failed_tests

        except Exception as e:
            print(f"❌ 测试识别失败: {e}")
            return []

    def optimize_test_logic(self, test_file: str) -> bool:
        """优化测试用例逻辑"""
        print(f"🔧 优化测试用例: {test_file}")

        try:
            # 读取测试文件
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 应用优化规则
            optimized_content = self._apply_optimization_rules(content)

            # 写回文件
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(optimized_content)

            self.optimization_log.append(f"优化完成: {test_file}")
            return True

        except Exception as e:
            print(f"❌ 优化失败 {test_file}: {e}")
            return False

    def _apply_optimization_rules(self, content: str) -> str:
        """应用优化规则"""
        # 规则1: 修复DataFrame布尔判断
        content = content.replace(
            "assert df.empty == True",
            "assert df.empty is True"
        )
        content = content.replace(
            "assert df.empty == False",
            "assert df.empty is False"
        )

        # 规则2: 修复None比较
        content = content.replace(
            "assert result == None",
            "assert result is None"
        )
        content = content.replace(
            "assert result != None",
            "assert result is not None"
        )

        # 规则3: 修复字符串比较
        content = content.replace(
            "assert str(result) == 'None'",
            "assert result is None"
        )

        # 规则4: 添加更好的错误信息
        content = content.replace(
            "assert condition",
            "assert condition, f'Assertion failed: {condition}'"
        )

        return content

    def add_boundary_tests(self, test_file: str) -> bool:
        """添加边界条件测试"""
        print(f"➕ 为 {test_file} 添加边界条件测试...")

        try:
            # 读取现有测试
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 添加边界条件测试
            boundary_tests = self._generate_boundary_tests(test_file)

            # 在文件末尾添加边界测试
            if boundary_tests:
                content += "\n\n# 边界条件测试\n"
                content += boundary_tests

            # 写回文件
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(content)

            return True

        except Exception as e:
            print(f"❌ 添加边界测试失败 {test_file}: {e}")
            return False

    def _generate_boundary_tests(self, test_file: str) -> str:
        """生成边界条件测试"""
        if "validator" in test_file:
            return """
def test_validator_boundary_conditions():
    \"\"\"测试验证器边界条件\"\"\"
    from src.data.validator import DataValidator
    import pandas as pd
    import numpy as np
    
    validator = DataValidator()
    
    # 测试空DataFrame
    df_empty = pd.DataFrame()
    result = validator.validate_data(df_empty)
    assert result.is_valid is False, "空DataFrame应该验证失败"
    
    # 测试全为None的DataFrame
    df_none = pd.DataFrame({'a': [None, None], 'b': [None, None]})
    result = validator.validate_data(df_none)
    assert result.is_valid is False, "全为None的DataFrame应该验证失败"
    
    # 测试单列DataFrame
    df_single = pd.DataFrame({'a': [1, 2, 3]})
    result = validator.validate_data(df_single)
    assert result.is_valid is True, "单列DataFrame应该验证通过"
    
    # 测试大数据量DataFrame
    df_large = pd.DataFrame({
        'a': range(10000),
        'b': range(10000)
    })
    result = validator.validate_data(df_large)
    assert result.is_valid is True, "大数据量DataFrame应该验证通过"
"""

        elif "loader" in test_file:
            return """
def test_loader_boundary_conditions():
    \"\"\"测试加载器边界条件\"\"\"
    from src.data.loader.news_loader import FinancialNewsLoader
    
    # 测试空配置
    loader = FinancialNewsLoader()
    assert loader.adapter_type == "financial_news"
    
    # 测试无效配置
    loader = FinancialNewsLoader({})
    assert loader.adapter_type == "financial_news"
    
    # 测试完整配置
    config = {
        'source': 'test_source',
        'connection_params': {'host': 'localhost'},
        'validation_rules': {'rule1': 'value1'}
    }
    loader = FinancialNewsLoader(config)
    assert loader.adapter_type == "financial_news"
"""

        elif "cache" in test_file:
            return """
def test_cache_boundary_conditions():
    \"\"\"测试缓存边界条件\"\"\"
    from src.data.cache.redis_cache_adapter import RedisCacheAdapter, RedisCacheConfig
    
    # 测试空配置
    config = RedisCacheConfig()
    adapter = RedisCacheAdapter(config)
    assert adapter is not None
    
    # 测试无效配置
    config = RedisCacheConfig(host="invalid_host", port=9999)
    adapter = RedisCacheAdapter(config)
    assert adapter is not None
"""

        return ""

    def optimize_test_performance(self, test_file: str) -> bool:
        """优化测试性能"""
        print(f"⚡ 优化测试性能: {test_file}")

        try:
            # 读取测试文件
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 应用性能优化
            optimized_content = self._apply_performance_optimizations(content)

            # 写回文件
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(optimized_content)

            return True

        except Exception as e:
            print(f"❌ 性能优化失败 {test_file}: {e}")
            return False

    def _apply_performance_optimizations(self, content: str) -> str:
        """应用性能优化"""
        # 优化1: 使用pytest.fixture减少重复初始化
        if "import pytest" not in content:
            content = content.replace(
                "import sys",
                "import sys\nimport pytest"
            )

        # 优化2: 添加类级别的setup和teardown
        if "class Test" in content and "def setup_class" not in content:
            content = content.replace(
                "class Test",
                """class Test:
    @classmethod
    def setup_class(cls):
        \"\"\"类级别设置\"\"\"
        pass
    
    @classmethod
    def teardown_class(cls):
        \"\"\"类级别清理\"\"\"
        pass
"""
            )

        # 优化3: 使用更高效的断言
        content = content.replace(
            "assert len(result) > 0",
            "assert result"
        )

        return content

    def run_optimization_suite(self) -> Dict[str, Any]:
        """运行完整的优化套件"""
        print("🚀 开始测试用例优化套件...")
        print("=" * 50)

        # 1. 分析测试覆盖率
        coverage = self.analyze_test_coverage()
        print(f"📊 当前测试覆盖率: {coverage['total_coverage']:.2f}%")

        # 2. 识别失败的测试
        failed_tests = self.identify_failing_tests()
        print(f"❌ 失败的测试数量: {len(failed_tests)}")

        # 3. 优化测试用例
        test_files = [
            "tests/unit/data/test_validator.py",
            "tests/unit/data/test_loader.py",
            "tests/unit/data/cache/test_redis_cache_adapter.py",
            "tests/unit/data/adapters/test_china_adapter.py"
        ]

        optimized_count = 0
        for test_file in test_files:
            if os.path.exists(test_file):
                if self.optimize_test_logic(test_file):
                    optimized_count += 1
                if self.add_boundary_tests(test_file):
                    optimized_count += 1
                if self.optimize_test_performance(test_file):
                    optimized_count += 1

        # 4. 重新运行测试
        print("\n🔄 重新运行测试...")
        result = subprocess.run([
            "conda", "run", "-n", "test", "python", "-m", "pytest",
            "tests/unit/data/", "--tb=short", "-v"
        ], capture_output=True, text=True, cwd=os.getcwd())

        # 5. 生成优化报告
        optimization_report = {
            "coverage_before": coverage['total_coverage'],
            "failed_tests_before": len(failed_tests),
            "optimized_files": optimized_count,
            "test_result_after": result.returncode == 0,
            "optimization_log": self.optimization_log
        }

        print("=" * 50)
        print("📋 优化报告:")
        print(f"  - 测试覆盖率: {optimization_report['coverage_before']:.2f}%")
        print(f"  - 优化文件数: {optimization_report['optimized_files']}")
        print(f"  - 测试通过: {'✅' if optimization_report['test_result_after'] else '❌'}")

        return optimization_report


def main():
    """主函数"""
    optimizer = TestCaseOptimizer()
    report = optimizer.run_optimization_suite()

    # 保存报告
    with open("test_optimization_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n📄 优化报告已保存到: test_optimization_report.json")


if __name__ == "__main__":
    main()
