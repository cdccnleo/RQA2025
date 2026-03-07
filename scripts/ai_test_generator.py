#!/usr/bin/env python3
"""
AI辅助测试用例生成器
使用大语言模型自动生成测试用例，提高测试覆盖率
"""

import os
import sys
import json
import ast
import inspect
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse


class AITestGenerator:
    """AI辅助测试用例生成器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.templates_dir = self.project_root / "test_templates"
        self.templates_dir.mkdir(exist_ok=True)

        # 测试用例模板
        self.test_templates = {
            "unit_test": """
import pytest
import sys
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from src.{module_path} import {class_name}


class Test{class_name}:
    \"\"\"{class_name} 单元测试\"\"\"

    def setup_method(self):
        \"\"\"测试前准备\"\"\"
        self.instance = {class_name}()

    {test_methods}
""",
            "integration_test": """
import pytest
import sys
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from src.{module_path} import {class_name}


class Test{class_name}Integration:
    \"\"\"{class_name} 集成测试\"\"\"

    def setup_method(self):
        \"\"\"测试前准备\"\"\"
        self.instance = {class_name}()

    {test_methods}
""",
            "edge_case_test": """
import pytest
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from src.{module_path} import {class_name}


class Test{class_name}EdgeCases:
    \"\"\"{class_name} 边界条件测试\"\"\"

    def setup_method(self):
        \"\"\"测试前准备\"\"\"
        self.instance = {class_name}()

    {test_methods}
"""
        }

    def analyze_source_code(self, source_file: str) -> Dict[str, Any]:
        """分析源代码结构"""
        with open(source_file, 'r', encoding='utf-8') as f:
            source_code = f.read()

        try:
            tree = ast.parse(source_code)
            analysis = {
                "classes": [],
                "functions": [],
                "imports": [],
                "complexity": self._calculate_complexity(tree)
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    analysis["classes"].append({
                        "name": node.name,
                        "methods": [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                        "bases": [base.id if isinstance(base, ast.Name) else str(base) for base in node.bases]
                    })
                elif isinstance(node, ast.FunctionDef):
                    analysis["functions"].append({
                        "name": node.name,
                        "args": [arg.arg for arg in node.args.args],
                        "has_return": any(isinstance(n, ast.Return) for n in ast.walk(node))
                    })
                elif isinstance(node, ast.Import):
                    analysis["imports"].extend([alias.name for alias in node.names])
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    analysis["imports"].extend([f"{module}.{alias.name}" for alias in node.names])

            return analysis
        except SyntaxError as e:
            return {"error": f"语法错误: {e}"}

    def _calculate_complexity(self, tree: ast.AST) -> int:
        """计算代码复杂度"""
        complexity = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.With)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        return complexity

    def generate_test_cases(self, analysis: Dict[str, Any], class_name: str) -> str:
        """生成测试用例"""
        test_methods = []

        if "classes" in analysis:
            for cls in analysis["classes"]:
                if cls["name"] == class_name:
                    # 生成构造函数测试
                    test_methods.append(f"""
    def test_init(self):
        \"\"\"测试{class_name}初始化\"\"\"
        assert self.instance is not None
        assert isinstance(self.instance, {class_name})""")

                    # 为每个方法生成测试
                    for method in cls["methods"]:
                        if not method.startswith("_"):  # 跳过私有方法
                            test_methods.append(self._generate_method_test(method, class_name))

                    # 生成边界条件测试
                    test_methods.append(self._generate_edge_case_tests(class_name))

        return "\n".join(test_methods)

    def _generate_method_test(self, method_name: str, class_name: str) -> str:
        """生成方法测试"""
        return f"""
    def test_{method_name}(self):
        \"\"\"测试{class_name}.{method_name}方法\"\"\"
        # 测试正常情况
        result = self.instance.{method_name}()
        assert result is not None

        # 测试异常情况（如果适用）
        # with pytest.raises(Exception):
        #     self.instance.{method_name}(invalid_param)"""

    def _generate_edge_case_tests(self, class_name: str) -> str:
        """生成边界条件测试"""
        return f"""
    def test_edge_cases(self):
        \"\"\"测试{class_name}边界条件\"\"\"
        # 测试None输入
        # with pytest.raises((TypeError, ValueError)):
        #     self.instance.some_method(None)

        # 测试空输入
        # result = self.instance.some_method([])
        # assert result == expected_empty_result

        # 测试大数据输入
        # large_data = list(range(10000))
        # result = self.instance.some_method(large_data)
        # assert len(result) > 0"""

    def generate_test_file(self, source_file: str, test_type: str = "unit_test") -> Optional[str]:
        """生成完整的测试文件"""
        source_path = Path(source_file)
        if not source_path.exists():
            print(f"❌ 源文件不存在: {source_file}")
            return None

        # 分析源代码
        analysis = self.analyze_source_code(source_file)
        if "error" in analysis:
            print(f"❌ 分析源代码失败: {analysis['error']}")
            return None

        # 提取模块路径和类名
        relative_path = source_path.relative_to(self.project_root / "src")
        module_path = str(relative_path).replace(".py", "").replace(os.sep, ".")

        # 找到主要的类
        main_class = None
        if analysis["classes"]:
            # 选择第一个非测试类
            for cls in analysis["classes"]:
                if not cls["name"].endswith("Test"):
                    main_class = cls["name"]
                    break

        if not main_class:
            print(f"❌ 未找到合适的测试类: {source_file}")
            return None

        # 生成测试用例
        test_methods = self.generate_test_cases(analysis, main_class)

        # 生成测试文件内容
        template = self.test_templates.get(test_type, self.test_templates["unit_test"])
        test_content = template.format(
            module_path=module_path,
            class_name=main_class,
            test_methods=test_methods
        )

        # 保存测试文件
        test_dir = self.project_root / "tests" / "ai_generated"
        test_dir.mkdir(exist_ok=True)

        test_filename = f"test_{main_class.lower()}_ai.py"
        test_file_path = test_dir / test_filename

        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write(test_content)

        print(f"✅ AI生成测试文件: {test_file_path}")
        return str(test_file_path)

    def analyze_coverage_gaps(self, coverage_file: str = "coverage.json") -> List[str]:
        """分析覆盖率缺口"""
        coverage_path = self.project_root / coverage_file
        if not coverage_path.exists():
            return []

        try:
            with open(coverage_path, 'r', encoding='utf-8') as f:
                coverage_data = json.load(f)

            missing_lines = []
            if "files" in coverage_data:
                for file_path, file_data in coverage_data["files"].items():
                    if "missing_lines" in file_data:
                        missing_lines.extend(file_data["missing_lines"])

            return missing_lines[:20]  # 返回前20个缺失的行
        except Exception as e:
            print(f"❌ 分析覆盖率文件失败: {e}")
            return []

    def optimize_test_suite(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """优化测试套件"""
        recommendations = {
            "add_tests": [],
            "remove_tests": [],
            "modify_tests": [],
            "performance_improvements": []
        }

        # 基于测试结果的简单分析
        if "failed" in test_results and test_results["failed"] > 0:
            recommendations["add_tests"].append("增加异常处理测试用例")

        if "duration" in test_results and test_results["duration"] > 300:
            recommendations["performance_improvements"].append("优化慢测试用例")

        return recommendations


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RQA2025 AI测试用例生成器')
    parser.add_argument('--source-file', required=True, help='源代码文件路径')
    parser.add_argument('--test-type', choices=['unit_test', 'integration_test', 'edge_case_test'],
                       default='unit_test', help='测试类型')
    parser.add_argument('--output-dir', default='tests/ai_generated', help='输出目录')
    parser.add_argument('--analyze-coverage', action='store_true', help='分析覆盖率缺口')

    args = parser.parse_args()

    # 初始化生成器
    generator = AITestGenerator(".")

    if args.analyze_coverage:
        # 分析覆盖率缺口
        missing_lines = generator.analyze_coverage_gaps()
        print(f"🔍 发现 {len(missing_lines)} 个覆盖率缺口")
        if missing_lines:
            print("前10个缺失行:", missing_lines[:10])
    else:
        # 生成测试文件
        test_file = generator.generate_test_file(args.source_file, args.test_type)
        if test_file:
            print(f"✅ 测试文件生成成功: {test_file}")
            print("💡 提示: 请检查生成的文件并根据需要进行调整")
        else:
            print("❌ 测试文件生成失败")
            sys.exit(1)


if __name__ == "__main__":
    main()
