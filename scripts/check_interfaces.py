#!/usr/bin/env python3
"""
RQA2025 接口一致性检查脚本

检查所有组件是否遵循统一接口标准
基于重构成果：已建立统一接口模板
"""

import os
import sys
import ast
from typing import Dict, Set, List


class InterfaceChecker:
    """接口一致性检查器"""

    def __init__(self):
        self.violations: List[Dict] = []
        self.standard_protocols = {
            'IStatusProvider': ['get_status_info'],
            'IHealthCheckable': ['health_check'],
            'ILifecycleManageable': ['initialize', 'shutdown'],
            'IServiceProvider': ['get_service', 'register_service']
        }

    def check_directory(self, directory: str) -> bool:
        """检查整个目录的接口一致性"""
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

    def check_file(self, filepath: str) -> bool:
        """检查单个文件的接口一致性"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            return True

        try:
            tree = ast.parse(content, filepath)
            self._check_class_interfaces(tree, filepath)
        except SyntaxError:
            pass  # 跳过语法错误的文

        return len([v for v in self.violations if v['file'] == filepath]) == 0

    def _check_class_interfaces(self, tree: ast.AST, filepath: str):
        """检查类接口实现"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                self._analyze_class_interface(node, filepath)

    def _analyze_class_interface(self, class_node: ast.ClassDef, filepath: str):
        """分析类的接口实现"""
        class_name = class_node.name
        base_classes = [base.id if isinstance(base, ast.Name) else str(base)
                        for base in class_node.bases]

        # 检查是否继承标准组件
        is_standard_component = any('StandardComponent' in base or 'BaseComponent' in base
                                    for base in base_classes)

        if not is_standard_component:
            # 检查是否实现了标准协议
            methods = self._extract_methods(class_node)

            # 检查核心接口方法
            required_methods = {'get_status_info', 'health_check'}

            missing_methods = required_methods - methods
            if missing_methods:
                self.violations.append({
                    'file': filepath,
                    'class': class_name,
                    'type': 'missing_interface_methods',
                    'missing_methods': list(missing_methods),
                    'description': f"类 {class_name} 缺少标准接口方法: {', '.join(missing_methods)}"
                })

    def _extract_methods(self, class_node: ast.ClassDef) -> Set[str]:
        """提取类中的所有方法名"""
        methods = set()

        for item in class_node.body:
            if isinstance(item, ast.FunctionDef):
                methods.add(item.name)
            elif isinstance(item, ast.AsyncFunctionDef):
                methods.add(item.name)

        return methods

    def print_report(self):
        """打印检查报告"""
        if not self.violations:
            print("✅ 接口一致性检查通过 - 所有组件遵循统一标准")
            return

        print(f"❌ 发现 {len(self.violations)} 个接口不一致问题:")
        print("-" * 60)

        for violation in self.violations:
            print(f"📁 {violation['file']}")
            print(f"   类: {violation['class']}")
            print(f"   问题: {violation['description']}")
            if 'missing_methods' in violation:
                print(f"   缺少方法: {', '.join(violation['missing_methods'])}")
            print()


def main():
    """主函数"""
    print("🔌 RQA2025 接口一致性检查")
    print("=" * 40)

    # 检查参数
    if len(sys.argv) < 2:
        print("用法: python check_interfaces.py <目录路径>")
        print("示例: python check_interfaces.py src")
        sys.exit(1)

    target_path = sys.argv[1]

    if not os.path.exists(target_path):
        print(f"❌ 路径不存在: {target_path}")
        sys.exit(1)

    if not os.path.isdir(target_path):
        print(f"❌ 路径不是目录: {target_path}")
        sys.exit(1)

    # 执行检查
    checker = InterfaceChecker()
    success = checker.check_directory(target_path)

    # 输出报告
    checker.print_report()

    if success:
        print("✅ 接口一致性检查完成 - 符合重构标准")
        sys.exit(0)
    else:
        print("❌ 接口一致性检查失败 - 存在接口不一致问题")
        sys.exit(1)


if __name__ == "__main__":
    main()
