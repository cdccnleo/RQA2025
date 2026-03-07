#!/usr/bin/env python3
"""
Phase 14.3: 智能测试分组和依赖管理
根据测试特征智能分组，提高并行执行效率
"""

import os
import json
import ast
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict
import networkx as nx


class TestGroupingOptimizer:
    """测试分组优化器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.test_files = []
        self.test_dependencies = {}
        self.test_metadata = {}

    def scan_test_files(self) -> List[Path]:
        """扫描所有测试文件"""
        print("🔍 扫描测试文件...")

        test_files = []
        for root, dirs, files in os.walk(self.project_root / 'tests'):
            for file in files:
                if file.endswith('_test.py') or file.startswith('test_'):
                    test_files.append(Path(root) / file)

        self.test_files = test_files
        print(f"  📊 发现 {len(test_files)} 个测试文件")
        return test_files

    def analyze_test_file(self, test_file: Path) -> Dict[str, Any]:
        """分析单个测试文件"""
        metadata = {
            'file_path': str(test_file),
            'relative_path': str(test_file.relative_to(self.project_root)),
            'test_classes': [],
            'test_functions': [],
            'imports': [],
            'markers': [],
            'estimated_duration': 'medium',  # fast, medium, slow
            'dependencies': [],
            'complexity': 'simple'  # simple, medium, complex
        }

        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 解析AST
            tree = ast.parse(content)

            # 提取测试类和函数
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name.startswith('Test'):
                    metadata['test_classes'].append(node.name)

                    # 估算复杂度
                    method_count = len([n for n in node.body if isinstance(n, ast.FunctionDef)])
                    if method_count > 10:
                        metadata['complexity'] = 'complex'
                    elif method_count > 5:
                        metadata['complexity'] = 'medium'

                elif isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                    metadata['test_functions'].append(node.name)

                    # 检查是否有slow标记或长执行时间特征
                    if any(isinstance(d, ast.Name) and d.id == 'slow' for d in node.decorator_list):
                        metadata['estimated_duration'] = 'slow'

            # 提取导入
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    metadata['imports'].extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    metadata['imports'].extend(f"{module}.{alias.name}" for alias in node.names)

            # 基于内容特征估算执行时间
            if 'e2e' in str(test_file).lower():
                metadata['estimated_duration'] = 'slow'
            elif 'integration' in str(test_file).lower():
                metadata['estimated_duration'] = 'medium'
            elif len(metadata['test_functions']) > 20:
                metadata['estimated_duration'] = 'slow'
            elif len(metadata['test_functions']) < 5:
                metadata['estimated_duration'] = 'fast'

        except Exception as e:
            metadata['parse_error'] = str(e)

        return metadata

    def analyze_dependencies(self) -> Dict[str, Set[str]]:
        """分析测试依赖关系"""
        print("🔍 分析测试依赖关系...")

        dependencies = defaultdict(set)

        for test_file in self.test_files:
            metadata = self.analyze_test_file(test_file)
            self.test_metadata[str(test_file)] = metadata

            # 基于导入分析依赖
            for imp in metadata['imports']:
                if imp.startswith('src.'):
                    # 依赖src模块
                    dependencies[str(test_file)].add(imp.split('.')[1])

            # 基于文件路径分析层级依赖
            path_parts = test_file.relative_to(self.project_root / 'tests').parts
            if len(path_parts) > 1:
                layer = path_parts[0]
                if layer in ['integration', 'e2e']:
                    # 集成和E2E测试依赖单元测试
                    unit_layer = f"tests/unit/{'/'.join(path_parts[1:])}"
                    if (self.project_root / unit_layer).exists():
                        dependencies[str(test_file)].add(unit_layer)

        self.test_dependencies = dict(dependencies)
        print(f"  📊 识别出 {len(dependencies)} 个依赖关系")
        return dict(dependencies)

    def create_test_groups(self) -> Dict[str, List[str]]:
        """创建智能测试分组"""
        print("🎯 创建智能测试分组...")

        groups = {
            'fast_unit': [],      # 快速单元测试
            'medium_unit': [],    # 中等单元测试
            'slow_unit': [],      # 慢速单元测试
            'integration': [],    # 集成测试
            'e2e': [],           # 端到端测试
            'isolated': []       # 独立测试（无依赖）
        }

        # 按特征分组
        for file_path, metadata in self.test_metadata.items():
            relative_path = metadata['relative_path']

            # 按测试类型分组
            if 'e2e' in relative_path:
                groups['e2e'].append(file_path)
            elif 'integration' in relative_path:
                groups['integration'].append(file_path)
            elif 'unit' in relative_path:
                # 按执行时间细分
                if metadata['estimated_duration'] == 'fast':
                    groups['fast_unit'].append(file_path)
                elif metadata['estimated_duration'] == 'slow':
                    groups['slow_unit'].append(file_path)
                else:
                    groups['medium_unit'].append(file_path)
            else:
                groups['isolated'].append(file_path)

        # 优化分组平衡
        optimized_groups = self.optimize_group_balance(groups)

        print("  📊 分组结果:"        for group_name, files in optimized_groups.items():
            print(f"    {group_name}: {len(files)} 个文件")

        return optimized_groups

    def optimize_group_balance(self, groups: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """优化分组平衡"""
        print("⚖️ 优化分组平衡...")

        # 计算每个文件的权重（基于复杂度）
        file_weights = {}
        for file_path, metadata in self.test_metadata.items():
            weight = 1
            if metadata['complexity'] == 'complex':
                weight = 3
            elif metadata['complexity'] == 'medium':
                weight = 2

            if metadata['estimated_duration'] == 'slow':
                weight *= 2
            elif metadata['estimated_duration'] == 'fast':
                weight *= 0.5

            file_weights[file_path] = weight

        # 为并行执行重新分组
        parallel_groups = {
            'group_1_fast': [],    # 最快的测试
            'group_2_fast': [],    # 次快的测试
            'group_3_medium': [],  # 中等测试
            'group_4_slow': [],    # 慢速测试
            'group_5_isolated': [] # 独立测试
        }

        # 按权重排序并分配到不同组
        sorted_files = sorted(file_weights.items(), key=lambda x: x[1])

        for i, (file_path, weight) in enumerate(sorted_files):
            group_index = i % 4  # 分配到4个并行组
            if weight < 1:
                parallel_groups['group_1_fast'].append(file_path)
            elif weight < 2:
                parallel_groups['group_2_fast'].append(file_path)
            elif weight < 3:
                parallel_groups[f'group_{group_index + 1}_medium'].append(file_path)
            else:
                parallel_groups['group_4_slow'].append(file_path)

        # 保留原始分组逻辑的独立测试
        parallel_groups['group_5_isolated'] = groups.get('isolated', [])

        return parallel_groups

    def detect_dependency_conflicts(self) -> Dict[str, List[str]]:
        """检测依赖冲突"""
        print("🔍 检测依赖冲突...")

        conflicts = {
            'circular_dependencies': [],
            'shared_resource_conflicts': [],
            'module_import_conflicts': []
        }

        # 构建依赖图
        G = nx.DiGraph()
        for test_file, deps in self.test_dependencies.items():
            G.add_node(test_file)
            for dep in deps:
                G.add_edge(test_file, dep)

        # 检测循环依赖
        try:
            cycles = list(nx.simple_cycles(G))
            conflicts['circular_dependencies'] = cycles
        except:
            pass

        # 检测共享资源冲突（简化版）
        shared_resources = defaultdict(list)
        for test_file, metadata in self.test_metadata.items():
            # 基于文件名模式检测可能的资源冲突
            if 'cache' in test_file.lower():
                shared_resources['cache'].append(test_file)
            if 'database' in test_file.lower() or 'db' in test_file.lower():
                shared_resources['database'].append(test_file)
            if 'file' in test_file.lower() or 'io' in test_file.lower():
                shared_resources['file_system'].append(test_file)

        for resource, files in shared_resources.items():
            if len(files) > 3:  # 多个测试使用同一资源
                conflicts['shared_resource_conflicts'].append({
                    'resource': resource,
                    'conflicting_tests': files
                })

        print(f"  ⚠️ 发现 {len(conflicts['circular_dependencies'])} 个循环依赖")
        print(f"  ⚠️ 发现 {len(conflicts['shared_resource_conflicts'])} 个资源冲突")

        return conflicts

    def generate_grouping_strategy(self) -> Dict[str, Any]:
        """生成分组策略"""
        print("🎯 生成分组策略...")

        strategy = {
            'parallel_groups': {},
            'execution_order': [],
            'dependency_warnings': [],
            'optimization_recommendations': []
        }

        # 1. 扫描和分析
        self.scan_test_files()
        self.analyze_dependencies()

        # 2. 创建分组
        groups = self.create_test_groups()
        strategy['parallel_groups'] = groups

        # 3. 确定执行顺序
        execution_order = [
            'group_1_fast',      # 最快
            'group_2_fast',      # 次快
            'group_5_isolated',  # 独立
            'group_3_medium',    # 中等
            'group_4_slow'       # 最慢
        ]
        strategy['execution_order'] = execution_order

        # 4. 检测冲突
        conflicts = self.detect_dependency_conflicts()
        strategy['dependency_warnings'] = conflicts

        # 5. 生成优化建议
        recommendations = []

        if len(groups.get('group_4_slow', [])) > 10:
            recommendations.append('慢速测试较多，建议增加worker数量或单独执行')

        if conflicts['shared_resource_conflicts']:
            recommendations.append('存在资源冲突，建议使用--dist=loadscope或减少并发')

        if len(groups.get('group_5_isolated', [])) > 20:
            recommendations.append('独立测试较多，可以优先并行执行')

        strategy['optimization_recommendations'] = recommendations

        return strategy

    def save_strategy(self, strategy: Dict[str, Any]):
        """保存分组策略"""
        strategy_file = self.project_root / 'test_logs' / 'phase14_test_grouping_strategy.json'

        with open(strategy_file, 'w', encoding='utf-8') as f:
            json.dump(strategy, f, indent=2, ensure_ascii=False)

        print(f"📄 分组策略已保存: {strategy_file}")

    def run_optimization(self) -> Dict[str, Any]:
        """运行完整的分组优化"""
        print("🚀 Phase 14.3: 智能测试分组和依赖管理")
        print("=" * 60)

        strategy = self.generate_grouping_strategy()
        self.save_strategy(strategy)

        print("\n" + "=" * 60)
        print("✅ Phase 14.3 分组优化完成")
        print("=" * 60)

        return strategy


def main():
    """主函数"""
    project_root = Path(__file__).parent.parent
    optimizer = TestGroupingOptimizer(project_root)
    strategy = optimizer.run_optimization()

    # 打印摘要
    if 'parallel_groups' in strategy:
        groups = strategy['parallel_groups']
        print("
📊 分组策略摘要:"        for group_name, files in groups.items():
            print(f"  {group_name}: {len(files)} 个测试文件")

    if 'optimization_recommendations' in strategy:
        print(f"\n💡 优化建议 ({len(strategy['optimization_recommendations'])} 项):")
        for rec in strategy['optimization_recommendations']:
            print(f"  • {rec}")


if __name__ == '__main__':
    main()
