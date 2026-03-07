#!/usr/bin/env python3
"""
预提交架构检查工具

提供更严格的预提交检查机制，确保代码变更符合架构规范
"""

import os
import sys
import re
import ast
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional


class PreCommitArchitectureChecker:
    """预提交架构检查器"""

    def __init__(self):
        self.violations = []
        self.warnings = []
        self.errors = []
        self.layer_mapping = self._load_layer_mapping()

    def _load_layer_mapping(self) -> Dict[str, str]:
        """加载层级映射配置"""
        return {
            'src/core': 'core',
            'src/infrastructure': 'infrastructure',
            'src/data': 'data',
            'src/gateway': 'gateway',
            'src/features': 'features',
            'src/ml': 'ml',
            'src/backtest': 'backtest',
            'src/risk': 'risk',
            'src/trading': 'trading',
            'src/engine': 'engine'
        }

    def check_staged_files(self) -> bool:
        """检查暂存的文件"""
        try:
            # 获取暂存的文件列表
            import subprocess
            result = subprocess.run(['git', 'diff', '--cached', '--name-only'],
                                  capture_output=True, text=True, cwd='.')

            if result.returncode != 0:
                print("❌ 无法获取暂存文件列表")
                return False

            staged_files = result.stdout.strip().split('\n')
            python_files = [f for f in staged_files if f.endswith('.py') and os.path.exists(f)]

            if not python_files:
                print("✅ 无Python文件变更")
                return True

            print(f"📋 检查 {len(python_files)} 个Python文件")

            success = True
            for file_path in python_files:
                if not self.check_single_file(file_path):
                    success = False

            return success

        except Exception as e:
            print(f"❌ 检查暂存文件时出错: {e}")
            return False

    def check_single_file(self, file_path: str) -> bool:
        """检查单个文件"""
        print(f"🔍 检查文件: {file_path}")

        violations = []
        warnings = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 1. 检查架构层级合规性
            layer_violations = self.check_architecture_compliance(file_path, content)
            violations.extend(layer_violations)

            # 2. 检查导入依赖
            import_violations = self.check_import_dependencies(file_path, content)
            violations.extend(import_violations)

            # 3. 检查命名规范
            naming_violations = self.check_naming_conventions(file_path, content)
            violations.extend(naming_violations)

            # 4. 检查组件工厂规范
            if file_path.endswith('_components.py'):
                component_violations = self.check_component_factory_compliance(file_path, content)
                violations.extend(component_violations)

            # 5. 检查业务概念使用
            concept_violations = self.check_business_concepts(file_path, content)
            violations.extend(concept_violations)

            # 6. 检查代码质量
            quality_warnings = self.check_code_quality(file_path, content)
            warnings.extend(quality_warnings)

            # 输出结果
            if violations:
                print(f"   ❌ 发现 {len(violations)} 个违规")
                for violation in violations:
                    print(f"      - {violation}")
                return False
            elif warnings:
                print(f"   ⚠️ 发现 {len(warnings)} 个警告")
                for warning in warnings:
                    print(f"      - {warning}")
                return True
            else:
                print("   ✅ 通过检查")
                return True

        except Exception as e:
            print(f"   ❌ 检查文件时出错: {e}")
            return False

    def check_architecture_compliance(self, file_path: str, content: str) -> List[str]:
        """检查架构层级合规性"""
        violations = []

        # 确定文件所属层级
        file_layer = None
        for prefix, layer in self.layer_mapping.items():
            if file_path.startswith(prefix):
                file_layer = layer
                break

        if not file_layer:
            violations.append(f"文件 {file_path} 不属于任何已知架构层级")
            return violations

        # 检查文件是否在正确的目录位置
        if not self.is_correct_layer_location(file_path, file_layer):
            violations.append(f"文件 {file_path} 位置不符合架构层级规范")

        return violations

    def is_correct_layer_location(self, file_path: str, layer: str) -> bool:
        """检查文件是否在正确的层级位置"""
        expected_prefix = f"src/{layer}"
        return file_path.startswith(expected_prefix)

    def check_import_dependencies(self, file_path: str, content: str) -> List[str]:
        """检查导入依赖"""
        violations = []

        # 确定当前文件层级
        current_layer = None
        for prefix, layer in self.layer_mapping.items():
            if file_path.startswith(prefix):
                current_layer = layer
                break

        if not current_layer:
            return violations

        # 定义允许的依赖关系
        allowed_dependencies = {
            'core': [],  # 核心层不依赖其他层
            'infrastructure': ['core'],
            'data': ['infrastructure', 'core'],
            'gateway': ['infrastructure', 'core'],
            'features': ['data', 'infrastructure', 'core'],
            'ml': ['features', 'infrastructure', 'core'],
            'backtest': ['ml', 'features', 'data', 'infrastructure', 'core'],
            'risk': ['backtest', 'infrastructure', 'core'],
            'trading': ['risk', 'backtest', 'infrastructure', 'core'],
            'engine': ['trading', 'risk', 'backtest', 'ml', 'features', 'data', 'infrastructure', 'core']
        }

        allowed_layers = allowed_dependencies.get(current_layer, [])

        # 检查from导入
        from_imports = re.findall(r'from\s+src\.(\w+)\s+import', content)
        for imported_layer in from_imports:
            if imported_layer not in allowed_layers and imported_layer != current_layer:
                violations.append(f"禁止的from导入: {current_layer} -> {imported_layer}")

        # 检查直接导入
        direct_imports = re.findall(r'import\s+src\.(\w+)', content)
        for imported_layer in direct_imports:
            if imported_layer not in allowed_layers and imported_layer != current_layer:
                violations.append(f"禁止的直接导入: {current_layer} -> {imported_layer}")

        return violations

    def check_naming_conventions(self, file_path: str, content: str) -> List[str]:
        """检查命名规范"""
        violations = []

        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                # 检查类名
                if isinstance(node, ast.ClassDef):
                    if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
                        violations.append(f"类名不符合规范: {node.name}")

                # 检查函数名
                elif isinstance(node, ast.FunctionDef):
                    if not re.match(r'^[a-z_][a-z0-9_]*$', node.name):
                        violations.append(f"函数名不符合规范: {node.name}")

                # 检查变量名
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    if not re.match(r'^[a-z_][a-z0-9_]*$', node.id):
                        violations.append(f"变量名不符合规范: {node.id}")

        except:
            # 如果AST解析失败，使用正则表达式检查
            pass

        return violations

    def check_component_factory_compliance(self, file_path: str, content: str) -> List[str]:
        """检查组件工厂合规性"""
        violations = []

        # 检查必需的接口定义
        if 'class IComponent' not in content:
            violations.append("缺少标准IComponent接口定义")

        if 'class ComponentFactory' not in content:
            violations.append("缺少标准ComponentFactory类定义")

        # 检查create_component方法
        if 'def create_component' not in content:
            violations.append("缺少create_component方法")

        # 检查错误处理
        if 'try:' not in content or 'except' not in content:
            violations.append("缺少错误处理机制")

        return violations

    def check_business_concepts(self, file_path: str, content: str) -> List[str]:
        """检查业务概念使用"""
        violations = []

        # 确定当前文件层级
        current_layer = None
        for prefix, layer in self.layer_mapping.items():
            if file_path.startswith(prefix):
                current_layer = layer
                break

        if not current_layer:
            return violations

        # 定义各层级禁止的业务概念
        forbidden_concepts = {
            'data': ['trading', 'strategy', 'execution', 'model', 'risk', 'order'],
            'features': ['trading', 'order', 'execution'],
            'ml': ['trading', 'order', 'execution'],
            'core': ['trading', 'strategy', 'execution', 'model', 'risk', 'order'],
            'infrastructure': ['trading', 'strategy', 'execution']
        }

        forbidden_in_layer = forbidden_concepts.get(current_layer, [])

        for concept in forbidden_in_layer:
            if re.search(r'\b' + re.escape(concept) + r'\b', content, re.IGNORECASE):
                violations.append(f"禁止在{current_layer}层使用业务概念: {concept}")

        return violations

    def check_code_quality(self, file_path: str, content: str) -> List[str]:
        """检查代码质量"""
        warnings = []

        lines = content.split('\n')

        # 检查行长度
        for i, line in enumerate(lines, 1):
            if len(line) > 100:
                warnings.append(f"第{i}行过长: {len(line)}字符")

        # 检查函数长度
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_lines = len(content.split('\n')[node.lineno-1:node.body[-1].lineno])
                    if func_lines > 50:
                        warnings.append(f"函数过长: {node.name} ({func_lines}行)")
        except:
            pass

        # 检查异常处理
        if 'except:' in content and 'Exception' not in content:
            warnings.append("使用裸except子句")

        return warnings

    def generate_report(self):
        """生成检查报告"""
        report = []

        report.append("# 预提交架构检查报告")
        report.append("")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        if not self.violations and not self.warnings:
            report.append("## ✅ 检查结果")
            report.append("所有文件均通过架构检查！")
        else:
            if self.violations:
                report.append("## ❌ 违规项")
                for violation in self.violations:
                    report.append(f"- {violation}")

            if self.warnings:
                report.append("## ⚠️ 警告项")
                for warning in self.warnings:
                    report.append(f"- {warning}")

        with open('reports/PRE_COMMIT_CHECK_REPORT.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

    def run_check(self) -> bool:
        """运行检查"""
        print("🚀 开始预提交架构检查...")
        print("="*50)

        try:
            success = self.check_staged_files()
            self.generate_report()

            if success:
                print("
📋 检查报告已保存: reports/PRE_COMMIT_CHECK_REPORT.md"                print("🎉 预提交检查通过！"
            else:
                print("
📋 检查报告已保存: reports/PRE_COMMIT_CHECK_REPORT.md"                print("❌ 预提交检查失败，请修复上述问题！"
            return success

        except Exception as e:
            print(f"\n❌ 预提交检查过程中出错: {e}")
            return False


def main():
    """主函数"""
    checker=PreCommitArchitectureChecker()
    success=checker.run_check()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
