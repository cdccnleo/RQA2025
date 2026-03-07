#!/usr/bin/env python3
"""
RQA2025 架构文档自动生成工具

分析代码结构，自动生成详细的架构设计文档

使用方法：
python scripts/generate_architecture_docs.py src/infrastructure/resource/
"""

import os
import ast
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ArchitectureMetrics:
    """架构指标"""
    total_modules: int = 0
    total_classes: int = 0
    total_functions: int = 0
    design_patterns: Dict[str, List[str]] = field(default_factory=dict)


class ArchitectureAnalyzer:
    """架构分析器"""

    def __init__(self):
        self.metrics = ArchitectureMetrics()
        self.modules = {}

    def analyze_directory(self, directory_path: Path) -> Dict[str, Any]:
        """分析目录结构"""
        print(f"🔍 分析目录: {directory_path}")

        python_files = []
        for root, dirs, files in os.walk(directory_path):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)

        print(f"📁 发现 {len(python_files)} 个Python文件")

        for file_path in python_files:
            self._analyze_file(file_path, directory_path)

        self._calculate_metrics()
        return self._generate_report()

    def _analyze_file(self, file_path: Path, directory_path: Path) -> None:
        """分析单个文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content, filename=str(file_path))

            # 简化模块名称生成
            try:
                module_name = str(file_path.relative_to(Path(directory_path))).replace(
                    '/', '.').replace('\\', '.').replace('.py', '')
            except ValueError:
                module_name = file_path.stem

            classes = []
            functions = []

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                    functions.append(node.name)

            self.modules[module_name] = {
                'path': str(file_path),
                'classes': classes,
                'functions': functions
            }

        except Exception as e:
            print(f"⚠️ 分析文件失败 {file_path}: {e}")

    def _calculate_metrics(self) -> None:
        """计算架构指标"""
        self.metrics.total_modules = len(self.modules)
        self.metrics.total_classes = sum(len(m['classes']) for m in self.modules.values())
        self.metrics.total_functions = sum(len(m['functions']) for m in self.modules.values())

        # 识别设计模式
        patterns = defaultdict(list)
        for module_name, info in self.modules.items():
            if 'facade' in module_name.lower():
                patterns['Facade'].append(module_name)
            if 'factory' in module_name.lower():
                patterns['Factory'].append(module_name)
            if 'adapter' in module_name.lower():
                patterns['Adapter'].append(module_name)

        self.metrics.design_patterns = dict(patterns)

    def _generate_report(self) -> Dict[str, Any]:
        """生成分析报告"""
        return {
            'generated_at': datetime.now().isoformat(),
            'metrics': {
                'total_modules': self.metrics.total_modules,
                'total_classes': self.metrics.total_classes,
                'total_functions': self.metrics.total_functions,
            },
            'modules': self.modules,
            'design_patterns': self.metrics.design_patterns
        }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RQA2025 架构文档生成工具")
    parser.add_argument("path", help="要分析的目录路径")
    parser.add_argument("--output", "-o", help="输出文件",
                        default="docs/architecture/generated/architecture_analysis.json")

    args = parser.parse_args()

    analyzer = ArchitectureAnalyzer()
    report = analyzer.analyze_directory(Path(args.path))

    # 确保输出目录存在
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"📄 架构分析报告已生成: {output_file}")

    # 打印摘要
    metrics = report['metrics']
    print("\n📊 分析摘要:")
    print(f"  • 模块数: {metrics['total_modules']}")
    print(f"  • 类数: {metrics['total_classes']}")
    print(f"  • 函数数: {metrics['total_functions']}")
    print(f"  • 识别模式: {len(report['design_patterns'])} 种")


if __name__ == "__main__":
    main()
