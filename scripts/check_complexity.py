#!/usr/bin/env python3
"""
RQA2025 复杂度检查脚本

检查Python代码的圈复杂度，确保不超过阈值
基于重构标准：函数/方法复杂度不超过8
"""

import os
import sys
import ast
from typing import Dict, List


class ComplexityChecker:
    """复杂度检查器"""

    def __init__(self, max_complexity: int = 8):
        self.max_complexity = max_complexity
        self.violations: List[Dict] = []

    def check_file(self, filepath: str) -> bool:
        """检查单个文件"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            print(f"⚠️  跳过文件 (编码问题): {filepath}")
            return True

        tree = ast.parse(content, filepath)
        self._check_node_complexity(tree, filepath)
        return len([v for v in self.violations if v['file'] == filepath]) == 0

    def check_directory(self, directory: str) -> bool:
        """检查整个目录"""
        success = True
        for root, dirs, files in os.walk(directory):
            # 跳过隐藏目录和__pycache__
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']

            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    if not self.check_file(filepath):
                        success = False

        return success

    def _check_node_complexity(self, node: ast.AST, filepath: str, context: str = ""):
        """递归检查节点的复杂度"""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            complexity = self._calculate_complexity(node)
            if complexity > self.max_complexity:
                self.violations.append({
                    'file': filepath,
                    'function': f"{context}.{node.name}" if context else node.name,
                    'complexity': complexity,
                    'max_allowed': self.max_complexity,
                    'line': node.lineno
                })
        elif isinstance(node, ast.ClassDef):
            # 检查类中的方法
            for item in node.body:
                self._check_node_complexity(item, filepath, node.name)

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """计算函数的圈复杂度"""
        complexity = 1  # 基础复杂度

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.IfExp, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                # 布尔操作增加复杂度
                complexity += 1
            elif isinstance(child, ast.Try):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1

        return complexity

    def print_report(self):
        """打印检查报告"""
        if not self.violations:
            print("✅ 复杂度检查通过 - 所有函数复杂度均在允许范围内")
            return

        print(f"❌ 发现 {len(self.violations)} 个复杂度违规:")
        print("-" * 60)

        for violation in self.violations:
            print(f"📁 {violation['file']}:{violation['line']}")
            print(f"   函数: {violation['function']}")
            print(f"   复杂度: {violation['complexity']} (允许最大值: {violation['max_allowed']})")
            print()


def main():
    """主函数"""
    print("🔍 RQA2025 复杂度检查")
    print("=" * 40)

    # 检查参数
    if len(sys.argv) < 2:
        print("用法: python check_complexity.py <目录路径>")
        print("示例: python check_complexity.py src")
        sys.exit(1)

    target_path = sys.argv[1]

    if not os.path.exists(target_path):
        print(f"❌ 路径不存在: {target_path}")
        sys.exit(1)

    # 执行检查
    checker = ComplexityChecker(max_complexity=8)  # 基于重构标准

    success = False
    if os.path.isfile(target_path):
        success = checker.check_file(target_path)
    else:
        success = checker.check_directory(target_path)

    # 输出报告
    checker.print_report()

    if success:
        print("✅ 复杂度检查完成 - 符合重构标准")
        sys.exit(0)
    else:
        print("❌ 复杂度检查失败 - 需要重构高复杂度函数")
        sys.exit(1)


if __name__ == "__main__":
    main()
