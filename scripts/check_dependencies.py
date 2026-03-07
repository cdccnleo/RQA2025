#!/usr/bin/env python3
"""
RQA2025 依赖关系检查脚本

检查Python模块间的依赖关系，确保没有循环依赖
基于重构成果：已消除所有循环依赖
"""

import os
import sys
import ast
from typing import Dict, Set, List
from collections import defaultdict


class DependencyChecker:
    """依赖关系检查器"""

    def __init__(self):
        self.dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.violations: List[Dict] = []
        self.checked_modules: Set[str] = set()

    def check_directory(self, directory: str) -> bool:
        """检查整个目录的依赖关系"""
        # 第一遍：收集所有依赖关系
        self._collect_dependencies(directory)

        # 第二遍：检测循环依赖
        self._detect_cycles()

        return len(self.violations) == 0

    def _collect_dependencies(self, directory: str):
        """收集依赖关系"""
        for root, dirs, files in os.walk(directory):
            # 跳过隐藏目录和__pycache__
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']

            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    self._analyze_file_dependencies(filepath)

    def _analyze_file_dependencies(self, filepath: str):
        """分析单个文件的依赖关系"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            return

        # 提取模块名
        module_name = self._get_module_name(filepath)

        try:
            tree = ast.parse(content, filepath)
            imports = self._extract_imports(tree)

            # 处理相对导入
            absolute_imports = set()
            for imp in imports:
                if imp.startswith('.'):
                    # 转换为绝对导入
                    abs_import = self._resolve_relative_import(module_name, imp)
                    if abs_import:
                        absolute_imports.add(abs_import)
                else:
                    absolute_imports.add(imp)

            self.dependencies[module_name].update(absolute_imports)

        except SyntaxError:
            pass  # 跳过语法错误的文

    def _get_module_name(self, filepath: str) -> str:
        """从文件路径获取模块名"""
        # 假设src目录是根目录
        if 'src' in filepath:
            rel_path = filepath.split('src', 1)[1]
            if rel_path.startswith(os.sep):
                rel_path = rel_path[1:]
            module_name = rel_path.replace(os.sep, '.').replace('.py', '')
        else:
            # 对于其他目录，直接使用相对路径
            module_name = filepath.replace(os.sep, '.').replace('.py', '')

        return module_name

    def _extract_imports(self, tree: ast.AST) -> Set[str]:
        """提取AST中的所有导入"""
        imports = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])

        return imports

    def _resolve_relative_import(self, current_module: str, relative_import: str) -> str:
        """解析相对导入为绝对导入"""
        try:
            current_parts = current_module.split('.')
            relative_parts = relative_import.split('.')

            # 计算上级目录层数
            up_levels = len([p for p in relative_parts if p == ''])

            # 移除空字符串（点）
            clean_relative = [p for p in relative_parts if p != '']

            # 计算绝对路径
            base_parts = current_parts[:-up_levels] if up_levels > 0 else current_parts[:-1]
            absolute_parts = base_parts + clean_relative

            return '.'.join(absolute_parts)
        except:
            return None

    def _detect_cycles(self):
        """检测循环依赖"""
        visited = set()
        rec_stack = set()

        def dfs(node: str, path: List[str]):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for dependency in self.dependencies.get(node, set()):
                if dependency not in visited:
                    if dfs(dependency, path):
                        return True
                elif dependency in rec_stack:
                    # 发现循环
                    cycle_start = path.index(dependency)
                    cycle = path[cycle_start:] + [dependency]
                    self.violations.append({
                        'type': 'circular_dependency',
                        'cycle': cycle,
                        'description': f"循环依赖: {' -> '.join(cycle)}"
                    })
                    return True

            path.pop()
            rec_stack.remove(node)
            return False

        for module in self.dependencies:
            if module not in visited:
                dfs(module, [])

    def print_report(self):
        """打印检查报告"""
        if not self.violations:
            print("✅ 依赖关系检查通过 - 未发现循环依赖")
            print(f"   检查了 {len(self.dependencies)} 个模块")
            return

        print(f"❌ 发现 {len(self.violations)} 个依赖关系问题:")
        print("-" * 60)

        for violation in self.violations:
            if violation['type'] == 'circular_dependency':
                print(f"🔄 循环依赖:")
                print(f"   路径: {' -> '.join(violation['cycle'])}")
                print()


def main():
    """主函数"""
    print("🔗 RQA2025 依赖关系检查")
    print("=" * 40)

    # 检查参数
    if len(sys.argv) < 2:
        print("用法: python check_dependencies.py <目录路径>")
        print("示例: python check_dependencies.py src")
        sys.exit(1)

    target_path = sys.argv[1]

    if not os.path.exists(target_path):
        print(f"❌ 路径不存在: {target_path}")
        sys.exit(1)

    if not os.path.isdir(target_path):
        print(f"❌ 路径不是目录: {target_path}")
        sys.exit(1)

    # 执行检查
    checker = DependencyChecker()
    success = checker.check_directory(target_path)

    # 输出报告
    checker.print_report()

    if success:
        print("✅ 依赖关系检查完成 - 符合重构标准")
        sys.exit(0)
    else:
        print("❌ 依赖关系检查失败 - 存在循环依赖")
        sys.exit(1)


if __name__ == "__main__":
    main()
