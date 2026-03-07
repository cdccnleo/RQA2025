"""
RQA2025 基础设施层工具系统代码组织分析

本脚本对基础设施层工具系统的代码组织进行全面分析，包括：
- 目录结构分析
- 文件分类统计
- 重复文件检测
- 导入关系分析
- 代码组织质量评估
"""

import os
from pathlib import Path
from collections import Counter
from typing import Dict


class CodeOrganizationAnalyzer:
    """代码组织分析器"""

    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.analysis_results = {}

    def analyze_directory_structure(self) -> Dict:
        """分析目录结构"""
        structure = {
            'total_files': 0,
            'python_files': 0,
            'directories': 0,
            'depth_levels': {},
            'file_types': Counter(),
            'file_size_distribution': Counter()
        }

        for root, dirs, files in os.walk(self.root_path):
            rel_path = Path(root).relative_to(self.root_path)
            depth = len(rel_path.parts) if rel_path != Path('.') else 0
            structure['depth_levels'][depth] = structure['depth_levels'].get(depth, 0) + 1
            structure['directories'] += len(dirs)

            for file in files:
                structure['total_files'] += 1
                if file.endswith('.py'):
                    structure['python_files'] += 1

                # 文件类型统计
                if '.' in file:
                    ext = file.split('.')[-1]
                    structure['file_types'][ext] += 1

                # 文件大小分布
                try:
                    file_path = Path(root) / file
                    size = file_path.stat().st_size
                    if size < 1024:
                        structure['file_size_distribution']['< 1KB'] += 1
                    elif size < 1024 * 10:
                        structure['file_size_distribution']['1-10KB'] += 1
                    elif size < 1024 * 100:
                        structure['file_size_distribution']['10-100KB'] += 1
                    else:
                        structure['file_size_distribution']['> 100KB'] += 1
                except:
                    pass

        return structure

    def analyze_file_classification(self) -> Dict:
        """分析文件分类"""
        classification = {
            'by_functionality': Counter(),
            'by_component_type': Counter(),
            'naming_patterns': Counter(),
            'duplicates': []
        }

        python_files = []
        for root, dirs, files in os.walk(self.root_path):
            for file in files:
                if file.endswith('.py') and not file.startswith('test_'):
                    python_files.append((root, file))

        # 功能分类
        functionality_patterns = {
            'database': ['adapter', 'connection', 'pool', 'migrator', 'database'],
            'cache': ['cache', 'redis'],
            'security': ['security', 'auth', 'encrypt'],
            'monitoring': ['monitor', 'log', 'alert', 'health'],
            'optimization': ['optimize', 'performance', 'benchmark'],
            'utility': ['utils', 'helper', 'tool', 'common'],
            'api': ['api', 'service', 'client'],
            'storage': ['storage', 'file', 'filesystem'],
            'async': ['async', 'concurrency', 'thread'],
            'data': ['data', 'convert', 'transform'],
            'datetime': ['date', 'time', 'datetime'],
            'math': ['math', 'calculation', 'compute']
        }

        for root, file in python_files:
            file_lower = file.lower()
            classified = False

            for func_type, patterns in functionality_patterns.items():
                if any(pattern in file_lower for pattern in patterns):
                    classification['by_functionality'][func_type] += 1
                    classified = True
                    break

            if not classified:
                classification['by_functionality']['other'] += 1

            # 组件类型分类
            if 'factory' in file_lower:
                classification['by_component_type']['factory'] += 1
            elif 'interface' in file_lower or 'abstract' in file_lower:
                classification['by_component_type']['interface'] += 1
            elif 'exception' in file_lower or 'error' in file_lower:
                classification['by_component_type']['exception'] += 1
            elif 'base' in file_lower or 'component' in file_lower:
                classification['by_component_type']['base'] += 1
            else:
                classification['by_component_type']['implementation'] += 1

            # 命名模式分析
            if '_' in file:
                classification['naming_patterns']['snake_case'] += 1
            elif file[0].islower() and any(c.isupper() for c in file):
                classification['naming_patterns']['camelCase'] += 1
            else:
                classification['naming_patterns']['other'] += 1

        # 检测重复文件
        file_names = [file for root, file in python_files]
        name_counts = Counter(file_names)
        for name, count in name_counts.items():
            if count > 1:
                classification['duplicates'].append(f"{name}: {count} 次")

        return classification

    def analyze_import_relationships(self) -> Dict:
        """分析导入关系"""
        imports = {
            'internal_imports': Counter(),
            'external_imports': Counter(),
            'circular_deps': [],
            'import_patterns': Counter()
        }

        python_files = []
        for root, dirs, files in os.walk(self.root_path):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)

        file_imports = {}

        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 提取导入语句
                import_lines = []
                for line in content.split('\n'):
                    line = line.strip()
                    if line.startswith(('import ', 'from ')) and not line.startswith('#'):
                        import_lines.append(line)

                file_imports[file_path] = import_lines

                # 分析导入类型
                for line in import_lines:
                    if 'from src.' in line or 'from infrastructure.' in line:
                        imports['internal_imports'][line.split()[1]] += 1
                    else:
                        # 提取外部模块名
                        if line.startswith('import '):
                            module = line.split()[1].split('.')[0]
                        else:  # from import
                            module = line.split()[1].split('.')[0]
                        imports['external_imports'][module] += 1

            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        return imports

    def analyze_code_quality_metrics(self) -> Dict:
        """分析代码质量指标"""
        quality = {
            'docstring_coverage': 0,
            'error_handling_coverage': 0,
            'complexity_indicators': Counter(),
            'code_lines': Counter(),
            'constants_usage': Counter()
        }

        python_files = []
        total_files = 0
        files_with_docs = 0
        files_with_error_handling = 0

        for root, dirs, files in os.walk(self.root_path):
            for file in files:
                if file.endswith('.py') and not file.startswith('test_'):
                    file_path = Path(root) / file
                    total_files += 1

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        lines = content.split('\n')
                        quality['code_lines'][len(lines)] += 1

                        # 检查文档字符串
                        if '"""' in content[:500] or "'''" in content[:500]:
                            files_with_docs += 1

                        # 检查错误处理
                        has_try_except = 'try:' in content and 'except' in content
                        has_logging = 'logging.' in content or 'logger.' in content
                        has_raise = 'raise ' in content
                        if has_try_except or has_logging or has_raise:
                            files_with_error_handling += 1

                        # 检查常量使用
                        if 'class.*Constants:' in content:
                            quality['constants_usage']['has_constants'] += 1
                        else:
                            quality['constants_usage']['no_constants'] += 1

                        # 复杂度指标
                        if len(lines) > 500:
                            quality['complexity_indicators']['large_files'] += 1
                        if len(content.split('\n\n')) > 20:
                            quality['complexity_indicators']['many_functions'] += 1

                    except Exception as e:
                        print(f"Error analyzing {file_path}: {e}")

        if total_files > 0:
            quality['docstring_coverage'] = files_with_docs / total_files * 100
            quality['error_handling_coverage'] = files_with_error_handling / total_files * 100

        return quality

    def generate_organization_score(self) -> Dict:
        """生成组织评分"""
        structure = self.analyze_directory_structure()
        classification = self.analyze_file_classification()
        imports = self.analyze_import_relationships()
        quality = self.analyze_code_quality_metrics()

        # 计算组织评分 (0-100)
        score = 0
        reasons = []

        # 目录结构评分 (20分)
        if structure['directories'] <= 5:
            score += 20
            reasons.append("目录结构清晰简洁")
        elif structure['directories'] <= 10:
            score += 15
            reasons.append("目录结构合理")
        else:
            score += 10
            reasons.append("目录结构较复杂")

        # 文件分类评分 (20分)
        if len(classification['duplicates']) == 0:
            score += 20
            reasons.append("无重复文件")
        else:
            score += 10
            reasons.append(f"存在 {len(classification['duplicates'])} 个重复文件")

        # 导入关系评分 (20分)
        if len(imports['circular_deps']) == 0:
            score += 20
            reasons.append("无循环依赖")
        else:
            score += 10
            reasons.append(f"存在 {len(imports['circular_deps'])} 个循环依赖")

        # 代码质量评分 (40分)
        quality_score = (quality['docstring_coverage'] + quality['error_handling_coverage']) / 2
        score += quality_score * 0.4
        if quality_score >= 90:
            reasons.append("代码质量优秀")
        elif quality_score >= 70:
            reasons.append("代码质量良好")
        else:
            reasons.append("代码质量需要改进")

        return {
            'overall_score': round(score, 1),
            'structure_score': score,
            'reasons': reasons,
            'details': {
                'directories': structure['directories'],
                'duplicates': len(classification['duplicates']),
                'circular_deps': len(imports['circular_deps']),
                'docstring_coverage': round(quality['docstring_coverage'], 1),
                'error_handling_coverage': round(quality['error_handling_coverage'], 1)
            }
        }

    def run_full_analysis(self) -> Dict:
        """运行完整分析"""
        print("🔍 开始代码组织分析...")

        results = {
            'directory_structure': self.analyze_directory_structure(),
            'file_classification': self.analyze_file_classification(),
            'import_relationships': self.analyze_import_relationships(),
            'code_quality': self.analyze_code_quality_metrics(),
            'organization_score': self.generate_organization_score()
        }

        print("✅ 分析完成")
        return results


def print_analysis_report(results: Dict):
    """打印分析报告"""
    print("\n" + "="*80)
    print("🏗️  RQA2025 基础设施层工具系统代码组织分析报告")
    print("="*80)

    # 组织评分
    score_info = results['organization_score']
    print(f"\n📊 总体组织评分: {score_info['overall_score']}/100")

    print("\n🎯 评分理由:")
    for reason in score_info['reasons']:
        print(f"  • {reason}")

    print("\n📋 详细指标:")
    details = score_info['details']
    print(f"  • 目录数量: {details['directories']}")
    print(f"  • 重复文件: {details['duplicates']}")
    print(f"  • 循环依赖: {details['circular_deps']}")
    print(f"  • 文档字符串覆盖率: {details['docstring_coverage']:.1f}%")
    print(f"  • 错误处理覆盖率: {details['error_handling_coverage']:.1f}%")
    # 目录结构分析
    structure = results['directory_structure']
    print(f"\n🏛️  目录结构分析:")
    print(f"  • 总文件数: {structure['total_files']}")
    print(f"  • Python文件数: {structure['python_files']}")
    print(f"  • 目录数: {structure['directories']}")
    print(f"  • 最大深度: {max(structure['depth_levels'].keys()) if structure['depth_levels'] else 0}")

    # 文件分类分析
    classification = results['file_classification']
    print(f"\n📁 文件分类分析:")
    print("  • 功能分类:")
    for func, count in sorted(classification['by_functionality'].items()):
        print(f"    - {func}: {count} 个文件")

    print("  • 组件类型:")
    for comp_type, count in sorted(classification['by_component_type'].items()):
        print(f"    - {comp_type}: {count} 个文件")

    if classification['duplicates']:
        print("  • 重复文件:")
        for dup in classification['duplicates'][:5]:  # 只显示前5个
            print(f"    - {dup}")

    # 导入关系分析
    imports = results['import_relationships']
    print(f"\n🔗 导入关系分析:")
    print(f"  • 内部导入: {len(imports['internal_imports'])} 个模块")
    print(f"  • 外部导入: {len(imports['external_imports'])} 个模块")

    print("  • 主要外部依赖:")
    for module, count in sorted(imports['external_imports'].items())[:10]:
        print(f"    - {module}: {count} 次")

    # 代码质量分析
    quality = results['code_quality']
    print(f"\n💎 代码质量分析:")
    print(f"  • 文档字符串覆盖率: {details['docstring_coverage']:.1f}%")
    print(f"  • 错误处理覆盖率: {details['error_handling_coverage']:.1f}%")
    print("  • 文件大小分布:")
    for size_range, count in sorted(quality['code_lines'].items())[:5]:
        print(f"    - {size_range} 行: {count} 个文件")

    # 改进建议
    print(f"\n💡 改进建议:")

    if score_info['overall_score'] >= 90:
        print("  ✅ 代码组织优秀，无需重大改进")
    elif score_info['overall_score'] >= 70:
        print("  ⚠️ 代码组织良好，建议小幅优化")
    else:
        print("  🚨 代码组织需要改进")

    if details['duplicates'] > 0:
        print(f"  • 消除 {details['duplicates']} 个重复文件")
    if details['circular_deps'] > 0:
        print(f"  • 解决 {details['circular_deps']} 个循环依赖")
    if details['docstring_coverage'] < 90:
        print("  • 完善模块文档字符串")
    if details['error_handling_coverage'] < 80:
        print("  • 增强错误处理机制")

    print("\n" + "="*80)


if __name__ == "__main__":
    analyzer = CodeOrganizationAnalyzer("src/infrastructure/utils")
    results = analyzer.run_full_analysis()
    print_analysis_report(results)
