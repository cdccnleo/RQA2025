#!/usr/bin/env python3
"""
RQA2025 资源管理代码组织结构分析器
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict


class CodeOrganizationAnalyzer:
    """代码组织结构分析器"""

    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.analysis_results = {}

    def analyze_structure(self) -> Dict:
        """分析整体代码结构"""
        print("🔍 分析资源管理代码组织结构...")

        # 基本统计
        files = self._get_python_files()
        total_files = len(files)

        print(f"📊 发现 {total_files} 个Python文件")

        # 分类分析
        categories = self._categorize_files(files)
        self._print_categories(categories)

        # 依赖分析
        dependencies = self._analyze_dependencies(files)
        self._print_dependencies(dependencies)

        # 问题识别
        issues = self._identify_issues(files, categories)
        self._print_issues(issues)

        # 建议改进
        recommendations = self._generate_recommendations(categories, dependencies, issues)

        return {
            'total_files': total_files,
            'categories': categories,
            'dependencies': dependencies,
            'issues': issues,
            'recommendations': recommendations
        }

    def _get_python_files(self) -> List[Path]:
        """获取所有Python文件"""
        return list(self.root_path.glob('*.py'))

    def _categorize_files(self, files: List[Path]) -> Dict[str, List[str]]:
        """根据功能分类文件"""
        categories = defaultdict(list)

        for file_path in files:
            filename = file_path.name

            # 跳过备份文件
            if '.backup' in filename:
                categories['备份文件'].append(filename)
                continue

            # 根据命名模式分类
            if 'config' in filename or 'validator' in filename:
                categories['配置和验证'].append(filename)
            elif 'monitor' in filename or 'health' in filename or 'performance' in filename:
                categories['监控和健康检查'].append(filename)
            elif 'scheduler' in filename or 'task' in filename:
                categories['任务调度'].append(filename)
            elif 'api' in filename:
                categories['API接口'].append(filename)
            elif 'manager' in filename:
                categories['资源管理器'].append(filename)
            elif 'component' in filename or 'pool' in filename or 'quota' in filename:
                categories['组件模块'].append(filename)
            elif 'interface' in filename or 'base' in filename:
                categories['基础和接口'].append(filename)
            elif 'decorator' in filename:
                categories['装饰器和工具'].append(filename)
            elif 'dashboard' in filename:
                categories['用户界面'].append(filename)
            elif 'metric' in filename:
                categories['业务指标'].append(filename)
            elif 'optimization' in filename:
                categories['资源优化'].append(filename)
            elif 'shared' in filename:
                categories['共享工具'].append(filename)
            else:
                categories['其他'].append(filename)

        return dict(categories)

    def _analyze_dependencies(self, files: List[Path]) -> Dict[str, Set[str]]:
        """分析文件间的依赖关系"""
        dependencies = defaultdict(set)

        for file_path in files:
            if '.backup' in str(file_path):
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 解析导入语句
                tree = ast.parse(content)
                imports = []

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imports.extend(alias.name for alias in node.names)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.append(node.module)

                filename = file_path.name
                dependencies[filename] = set(imports)

            except Exception as e:
                print(f"⚠️  解析文件 {file_path.name} 时出错: {e}")
                dependencies[file_path.name] = set()

        return dict(dependencies)

    def _identify_issues(self, files: List[Path], categories: Dict[str, List[str]]) -> List[str]:
        """识别代码组织问题"""
        issues = []

        # 检查备份文件
        backup_files = categories.get('备份文件', [])
        if backup_files:
            issues.append(f"发现 {len(backup_files)} 个备份文件: {', '.join(backup_files)}")

        # 检查命名不一致
        naming_issues = self._check_naming_consistency(files)
        issues.extend(naming_issues)

        # 检查循环依赖
        circular_deps = self._check_circular_dependencies()
        if circular_deps:
            issues.extend(circular_deps)

        # 检查文件大小
        large_files = self._check_file_sizes(files)
        if large_files:
            issues.extend(large_files)

        # 检查重复功能
        duplicate_features = self._check_duplicate_features(categories)
        if duplicate_features:
            issues.extend(duplicate_features)

        return issues

    def _check_naming_consistency(self, files: List[Path]) -> List[str]:
        """检查命名一致性"""
        issues = []

        # 检查命名模式
        component_files = [f for f in files if 'component' in f.name]
        non_component_files = [
            f for f in files if 'component' not in f.name and not f.name.endswith('.backup')]

        if len(component_files) > 0 and len(non_component_files) > 0:
            issues.append("文件命名不一致: 部分文件使用'component'后缀，部分不使用")

        # 检查是否有重复的类名
        class_names = defaultdict(list)
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 简单提取类名
                classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
                for class_name in classes:
                    class_names[class_name].append(file_path.name)

            except Exception:
                continue

        duplicates = {name: files for name, files in class_names.items() if len(files) > 1}
        if duplicates:
            issues.append(f"发现重复的类名: {duplicates}")

        return issues

    def _check_circular_dependencies(self) -> List[str]:
        """检查循环依赖"""
        # 简化版本：检查明显的循环依赖
        issues = []

        # 这里可以实现更复杂的循环依赖检测
        # 暂时返回空列表，因为需要更详细的依赖图分析

        return issues

    def _check_file_sizes(self, files: List[Path]) -> List[str]:
        """检查文件大小"""
        issues = []

        for file_path in files:
            if '.backup' in str(file_path):
                continue

            try:
                size = file_path.stat().st_size
                if size > 100 * 1024:  # 100KB
                    issues.append(f"文件过大: {file_path.name} ({size/1024:.1f}KB)")
            except Exception:
                continue

        return issues

    def _check_duplicate_features(self, categories: Dict[str, List[str]]) -> List[str]:
        """检查重复功能"""
        issues = []

        # 检查是否有多个监控相关的文件
        monitor_files = categories.get('监控和健康检查', [])
        if len(monitor_files) > 3:
            issues.append(f"监控功能文件过多 ({len(monitor_files)} 个): {', '.join(monitor_files[:3])}...")

        # 检查是否有多个API文件
        api_files = categories.get('API接口', [])
        if len(api_files) > 1:
            issues.append(f"发现多个API文件: {', '.join(api_files)}")

        return issues

    def _generate_recommendations(self, categories: Dict[str, List[str]],
                                  dependencies: Dict[str, Set[str]],
                                  issues: List[str]) -> List[str]:
        """生成改进建议"""
        recommendations = []

        # 基于问题生成建议
        if any('备份文件' in issue for issue in issues):
            recommendations.append("清理备份文件: 删除不必要的.backup文件")

        if any('命名不一致' in issue for issue in issues):
            recommendations.append("统一命名规范: 建立一致的文件和类命名规范")

        if any('文件过大' in issue for issue in issues):
            recommendations.append("拆分大文件: 将过大的文件拆分为更小的专用模块")

        if any('重复的类名' in issue for issue in issues):
            recommendations.append("重命名重复类: 避免同名的类定义")

        # 基于类别分析的建议
        total_files = sum(len(files) for files in categories.values())
        if total_files > 20:
            recommendations.append("考虑子目录组织: 将相关文件分组到子目录中")

        # 监控文件过多建议
        monitor_files = categories.get('监控和健康检查', [])
        if len(monitor_files) > 4:
            recommendations.append("监控模块重构: 考虑将监控功能合并到统一的监控模块中")

        # 建议的目录结构
        recommendations.append("建议目录结构:\n" +
                               "  ├── core/           # 核心组件\n" +
                               "  ├── monitoring/     # 监控相关\n" +
                               "  ├── scheduling/     # 调度相关\n" +
                               "  ├── api/           # API接口\n" +
                               "  ├── config/        # 配置相关\n" +
                               "  └── utils/         # 工具函数\n")

        return recommendations

    def _print_categories(self, categories: Dict[str, List[str]]):
        """打印分类结果"""
        print("\n📁 文件分类统计:")
        for category, files in categories.items():
            print(f"  {category}: {len(files)} 个文件")
            if len(files) <= 5:
                for file in files:
                    print(f"    - {file}")
            else:
                for file in files[:3]:
                    print(f"    - {file}")
                print(f"    ... 和其他 {len(files)-3} 个文件")

    def _print_dependencies(self, dependencies: Dict[str, Set[str]]):
        """打印依赖分析"""
        print("\n🔗 主要依赖关系:")
        # 只显示有较多依赖的文件
        for filename, deps in dependencies.items():
            if len(deps) > 3:
                print(f"  {filename}: 依赖 {len(deps)} 个模块")

    def _print_issues(self, issues: List[str]):
        """打印问题列表"""
        if issues:
            print("\n⚠️ 发现的问题:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
        else:
            print("\n✅ 未发现明显问题")


def main():
    """主函数"""
    analyzer = CodeOrganizationAnalyzer('src/infrastructure/resource')
    results = analyzer.analyze_structure()

    print("\n🎯 改进建议:")
    for i, rec in enumerate(results['recommendations'], 1):
        print(f"  {i}. {rec}")

    print(f"\n📊 总结: 共 {results['total_files']} 个文件，{len(results['issues'])} 个问题需要关注")


if __name__ == '__main__':
    main()
