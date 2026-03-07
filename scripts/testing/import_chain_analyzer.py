#!/usr/bin/env python3
"""
import链分析脚本 - 递归分析测试文件的import依赖
"""
import ast
import sys
from pathlib import Path
from typing import Set, Dict, List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
SRC_ROOT = PROJECT_ROOT / 'src'


class ImportChainAnalyzer:
    def __init__(self, project_root: Path, src_root: Path):
        self.project_root = project_root
        self.src_root = src_root
        self.visited: Set[Path] = set()
        self.import_graph: Dict[str, List[str]] = {}

    def analyze_file(self, file_path: Path, depth=0, parent=None):
        if not file_path.exists() or file_path in self.visited:
            return
        self.visited.add(file_path)
        try:
            rel_path = str(file_path.relative_to(self.project_root))
        except Exception:
            rel_path = str(file_path)
        logger.info(f"{'  '*depth}分析: {rel_path}")
        imports = self.extract_imports(file_path)
        self.import_graph[rel_path] = imports
        for imp in imports:
            imp_path = self.resolve_import_to_path(imp)
            if imp_path and imp_path.exists() and self.project_root in imp_path.parents:
                self.analyze_file(imp_path, depth+1, rel_path)

    def extract_imports(self, file_path: Path) -> List[str]:
        imports = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=str(file_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
        except Exception as e:
            logger.warning(f"无法解析imports: {file_path}: {e}")
        return imports

    def resolve_import_to_path(self, import_name: str) -> Path:
        """尝试将import名映射到本地py文件路径"""
        parts = import_name.split('.')
        # 优先src/下查找
        candidate = self.src_root.joinpath(*parts)
        if candidate.with_suffix('.py').exists():
            return candidate.with_suffix('.py')
        if candidate.is_dir() and (candidate / '__init__.py').exists():
            return candidate / '__init__.py'
        # 其次项目根下查找
        candidate = self.project_root.joinpath(*parts)
        if candidate.with_suffix('.py').exists():
            return candidate.with_suffix('.py')
        if candidate.is_dir() and (candidate / '__init__.py').exists():
            return candidate / '__init__.py'
        return Path('')

    def print_graph(self):
        logger.info("\n=== import依赖图 ===")
        for k, v in self.import_graph.items():
            logger.info(f"{k} imports: {v}")

    def find_cycles(self):
        """检测import循环"""
        logger.info("\n=== import循环检测 ===")
        path_stack = []
        visited = set()

        def dfs(node):
            if node in path_stack:
                logger.warning(f"⚠️ 循环import: {' -> '.join(path_stack + [node])}")
                return
            path_stack.append(node)
            for imp in self.import_graph.get(node, []):
                imp_path = self.resolve_import_to_path(imp)
                if not (imp_path and imp_path.exists() and self.project_root in imp_path.parents):
                    continue
                try:
                    rel_imp = str(imp_path.relative_to(self.project_root))
                except Exception:
                    rel_imp = str(imp_path)
                if rel_imp in self.import_graph:
                    dfs(rel_imp)
            path_stack.pop()
        for node in self.import_graph:
            if node not in visited:
                dfs(node)
                visited.add(node)


def main():
    if len(sys.argv) < 2:
        print(f"用法: python {sys.argv[0]} <测试文件路径>")
        sys.exit(1)
    test_file = Path(sys.argv[1]).resolve()
    if not test_file.exists():
        print(f"文件不存在: {test_file}")
        sys.exit(1)
    analyzer = ImportChainAnalyzer(PROJECT_ROOT, SRC_ROOT)
    analyzer.analyze_file(test_file)
    analyzer.print_graph()
    analyzer.find_cycles()


if __name__ == '__main__':
    main()
