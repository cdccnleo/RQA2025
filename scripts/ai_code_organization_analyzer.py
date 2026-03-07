#!/usr/bin/env python3
"""
AI智能化代码组织分析器

专门用于分析代码组织结构和架构合理性的模块
可作为AI代码分析器的子模块使用
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class OrganizationIssue:
    """组织问题"""
    severity: str  # 'high', 'medium', 'low'
    category: str  # 'naming', 'structure', 'dependencies', 'duplication'
    description: str
    files_affected: List[str]
    recommendation: str
    confidence: float = 1.0


@dataclass
class OrganizationMetrics:
    """组织指标"""
    total_files: int = 0
    total_lines: int = 0
    avg_file_size: float = 0
    max_file_size: int = 0
    largest_file: str = ""
    categories_count: Dict[str, int] = field(default_factory=dict)
    dependency_complexity: float = 0
    naming_consistency_score: float = 1.0
    structure_score: float = 1.0


@dataclass
class OrganizationAnalysisResult:
    """组织分析结果"""
    metrics: OrganizationMetrics
    issues: List[OrganizationIssue]
    recommendations: List[str]
    category_distribution: Dict[str, List[str]]
    dependency_graph: Dict[str, Set[str]]
    quality_score: float = 0.0


class AICodeOrganizationAnalyzer:
    """AI智能化代码组织分析器"""

    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.python_files = self._discover_python_files()

    def _discover_python_files(self) -> List[Path]:
        """发现所有Python文件"""
        files = []
        for pattern in ['*.py', '**/*.py']:
            files.extend(self.root_path.glob(pattern))

        # 去重并过滤
        files = list(set(files))
        files = [f for f in files if not any(skip in str(f) for skip in [
            '__pycache__', '.git', 'node_modules', 'build', 'dist',
            '.pytest_cache', '.coverage', 'htmlcov'
        ])]

        return sorted(files)

    def analyze_organization(self) -> OrganizationAnalysisResult:
        """执行完整的组织分析"""
        print("🤖 AI代码组织分析器启动...")

        # 1. 收集基本指标
        metrics = self._collect_basic_metrics()

        # 2. 分析文件分类
        categories = self._analyze_file_categories()

        # 3. 分析依赖关系
        dependencies = self._analyze_dependencies()

        # 4. 识别组织问题
        issues = self._identify_issues(metrics, categories, dependencies)

        # 5. 生成改进建议
        recommendations = self._generate_recommendations(issues, categories)

        # 6. 计算整体质量评分
        quality_score = self._calculate_quality_score(metrics, issues, categories)

        result = OrganizationAnalysisResult(
            metrics=metrics,
            issues=issues,
            recommendations=recommendations,
            category_distribution=categories,
            dependency_graph=dependencies,
            quality_score=quality_score
        )

        print("✅ AI代码组织分析完成")
        return result

    def _collect_basic_metrics(self) -> OrganizationMetrics:
        """收集基本指标"""
        metrics = OrganizationMetrics()
        metrics.total_files = len(self.python_files)

        total_lines = 0
        max_size = 0
        largest_file = ""

        for file_path in self.python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = len(content.split('\n'))
                    total_lines += lines

                    if lines > max_size:
                        max_size = lines
                        largest_file = file_path.name

            except Exception:
                continue

        metrics.total_lines = total_lines
        metrics.avg_file_size = total_lines / max(metrics.total_files, 1)
        metrics.max_file_size = max_size
        metrics.largest_file = largest_file

        return metrics

    def _analyze_file_categories(self) -> Dict[str, List[str]]:
        """分析文件分类"""
        categories = defaultdict(list)

        # 基于路径的关键字映射
        path_keywords = {
            'infrastructure': ['infrastructure', 'load_balancer'],
            'integration': ['integration', 'adapters', 'data', 'deployment', 'discovery'],
            'optimization': ['optimization', 'optimizer', 'long_term', 'medium_term', 'short_term'],
            'orchestration': ['orchestration', 'business_process'],
            'patterns': ['patterns'],
            'exceptions': ['exceptions'],
            'examples': ['examples'],
            'event_bus': ['event_bus'],
            'business_process': ['business_process'],
            'security': ['security'],
            'interfaces': ['interfaces'],
            'config': ['config'],
            'utils': ['utils'],
            'services': ['services']
        }

        category_patterns = {
            'core': ['manager', 'core', 'main', 'engine', 'component', 'coordinator', 'handler', 'bus', 'interface', 'shared', 'analyzer', 'reporter', 'orchestrator', 'process', 'integration', 'container', 'base', 'standard'],
            'api': ['api', 'endpoint', 'service', 'controller', 'gateway', 'authentication', 'web_management', 'api_gateway', 'database_service'],
            'config': ['config', 'setting', 'option', 'classes', 'constants', 'process_config'],
            'model': ['model', 'entity', 'dto', 'schema', 'dataclass', 'enum', 'dataclasses'],
            'util': ['util', 'helper', 'tool', 'common', 'validator', 'detector', 'generator', 'decorator', 'async_processor', 'intelligent_decision', 'visualization'],
            'test': ['test', 'spec', 'testing'],
            'monitoring': ['monitor', 'health', 'metric', 'log', 'alert', 'performance', 'business', 'unified', 'status', 'ai_performance', 'high_concurrency'],
            'scheduling': ['scheduler', 'task', 'job', 'queue'],
            'ui': ['ui', 'view', 'template', 'dashboard'],
            'data': ['repository', 'dao', 'storage', 'db', 'event', 'data_adapter', 'models_adapter'],
            'security': ['security', 'access_control', 'audit', 'auth', 'encrypt', 'policy', 'filter', 'auditor', 'factory'],
            'optimization': ['optimization', 'optimizer', 'long_term', 'medium_term', 'short_term', 'documentation_enhancer', 'testing_enhancer'],
            'patterns': ['pattern', 'template', 'interfaces', 'layer_interfaces'],
            'examples': ['example', 'demo', 'architecture_layers', 'refactor'],
            'exceptions': ['exception', 'core_exceptions', 'unified_exceptions'],
        }

        for file_path in self.python_files:
            filename = file_path.name.lower()
            path_str = str(file_path).lower()

            # 特殊文件处理
            if filename == '__init__.py':
                categories['core'].append(file_path.name)
                continue
            elif filename == 'base.py':
                categories['core'].append(file_path.name)
                continue

            # 首先基于路径进行分类
            path_matched = False
            for category, keywords in path_keywords.items():
                if any(keyword in path_str for keyword in keywords):
                    categories[category].append(file_path.name)
                    path_matched = True
                    break

            if path_matched:
                continue

            # 如果路径匹配失败，则基于文件名进行分类
            matched = False
            for category, patterns in category_patterns.items():
                if any(pattern in filename for pattern in patterns):
                    categories[category].append(file_path.name)
                    matched = True
                    break

            if not matched:
                categories['other'].append(file_path.name)

        return dict(categories)

    def _is_standard_library_module(self, module_name: str) -> bool:
        """检查是否为标准库模块"""
        standard_libs = {
            'typing', 'threading', 'asyncio', 'weakref', 'time', 'datetime', 'logging',
            'queue', 'concurrent', 'collections', 'functools', 'itertools', 'operator',
            'pathlib', 'os', 'sys', 'json', 'csv', 're', 'math', 'random', 'statistics',
            'dataclasses', 'enum', 'abc', 'contextlib', 'tempfile', 'shutil', 'glob',
            'linecache', 'pickle', 'copyreg', 'copy', 'pprint', 'reprlib', 'enum',
            'numbers', 'cmath', 'decimal', 'fractions', 'tracemalloc', 'inspect',
            'site', 'warnings', 'contextvars', 'concurrent.futures'
        }

        # 检查是否以标准库模块开头
        return any(module_name.startswith(lib) for lib in standard_libs)

    def _analyze_dependencies(self) -> Dict[str, Set[str]]:
        """分析依赖关系"""
        dependencies = defaultdict(set)

        for file_path in self.python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 解析导入语句
                tree = ast.parse(content)
                imports = set()

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.add(alias.name.split('.')[0])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.add(node.module.split('.')[0])

                # 过滤标准库和第三方库
                stdlib_modules = {
                    'os', 'sys', 'json', 'time', 'datetime', 'threading',
                    'pathlib', 'typing', 'collections', 'functools', 'itertools',
                    're', 'ast', 'inspect', 'logging', 'unittest', 'pytest',
                    'asyncio', 'weakref', 'dataclasses', 'enum', 'abc', 'queue',
                    'concurrent', 'operator', 'contextlib', 'tempfile', 'shutil',
                    'glob', 'linecache', 'pickle', 'copyreg', 'copy', 'pprint',
                    'reprlib', 'numbers', 'cmath', 'decimal', 'fractions',
                    'tracemalloc', 'site', 'warnings', 'contextvars'
                }

                local_imports = {
                    imp for imp in imports if imp not in stdlib_modules and not imp.startswith('.')}
                dependencies[file_path.name] = local_imports

            except Exception as e:
                print(f"⚠️ 解析文件 {file_path.name} 时出错: {e}")
                dependencies[file_path.name] = set()

        return dict(dependencies)

    def _identify_issues(self, metrics: OrganizationMetrics,
                         categories: Dict[str, List[str]],
                         dependencies: Dict[str, Set[str]]) -> List[OrganizationIssue]:
        """识别组织问题"""
        issues = []

        # 1. 文件大小问题
        if metrics.max_file_size > 500:
            issues.append(OrganizationIssue(
                severity='high',
                category='structure',
                description=f'最大文件过大: {metrics.largest_file} ({metrics.max_file_size}行)',
                files_affected=[metrics.largest_file],
                recommendation='将大文件拆分为多个专用模块'
            ))

        # 2. 命名一致性问题
        naming_issues = self._check_naming_consistency()
        issues.extend(naming_issues)

        # 3. 依赖复杂度问题
        complex_deps = [(file, deps) for file, deps in dependencies.items() if len(deps) > 5]
        for file, deps in complex_deps:
            issues.append(OrganizationIssue(
                severity='medium',
                category='dependencies',
                description=f'文件依赖过多: {file} 依赖 {len(deps)} 个模块',
                files_affected=[file],
                recommendation='考虑减少模块间的耦合度或拆分模块'
            ))

        # 4. 类别分布问题
        if len(categories.get('other', [])) > len(self.python_files) * 0.3:
            issues.append(OrganizationIssue(
                severity='medium',
                category='structure',
                description='未分类文件过多，表明命名或组织不够清晰',
                files_affected=categories.get('other', []),
                recommendation='改进文件命名规范，使其更容易分类'
            ))

        # 5. 重复功能问题
        duplicate_issues = self._check_duplicate_features(categories)
        issues.extend(duplicate_issues)

        return issues

    def _check_naming_consistency(self) -> List[OrganizationIssue]:
        """检查命名一致性"""
        issues = []

        # 检查类名重复
        class_names = defaultdict(list)
        for file_path in self.python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
                for class_name in classes:
                    class_names[class_name].append(file_path.name)

            except Exception:
                continue

        duplicates = {name: files for name, files in class_names.items() if len(files) > 1}
        if duplicates:
            issues.append(OrganizationIssue(
                severity='medium',
                category='naming',
                description=f'发现重复类名: {list(duplicates.keys())}',
                files_affected=list(set().union(*duplicates.values())),
                recommendation='重命名重复的类，避免命名冲突'
            ))

        return issues

    def _check_duplicate_features(self, categories: Dict[str, List[str]]) -> List[OrganizationIssue]:
        """检查重复功能"""
        issues = []

        # 检查是否有多个相似的文件
        if len(categories.get('monitoring', [])) > 3:
            issues.append(OrganizationIssue(
                severity='low',
                category='duplication',
                description='监控相关文件过多，可能存在功能重复',
                files_affected=categories['monitoring'],
                recommendation='考虑合并相似的监控功能到一个统一的模块中'
            ))

        if len(categories.get('api', [])) > 2:
            issues.append(OrganizationIssue(
                severity='low',
                category='duplication',
                description='API相关文件过多',
                files_affected=categories['api'],
                recommendation='评估是否需要合并API功能'
            ))

        return issues

    def _generate_recommendations(self, issues: List[OrganizationIssue],
                                  categories: Dict[str, List[str]]) -> List[str]:
        """生成改进建议"""
        recommendations = []

        # 基于问题生成建议
        severity_count = defaultdict(int)
        for issue in issues:
            severity_count[issue.severity] += 1

        if severity_count['high'] > 0:
            recommendations.append('🔴 优先解决高严重性问题，这些问题影响代码质量和可维护性')

        if severity_count['medium'] > 0:
            recommendations.append('🟡 关注中等严重性问题，及时改进可以提升开发效率')

        # 结构建议
        total_files = len(self.python_files)
        if total_files > 20:
            recommendations.append('📁 考虑引入子目录结构，将相关功能的文件组织在一起')

        # 命名建议
        if any(issue.category == 'naming' for issue in issues):
            recommendations.append('🏷️ 建立统一的命名规范，包括文件命名、类命名和方法命名')

        # 依赖建议
        if any(issue.category == 'dependencies' for issue in issues):
            recommendations.append('🔗 优化模块依赖关系，减少不必要的耦合')

        # 架构建议
        recommendations.extend([
            '🏗️ 推荐的分层架构:',
            '  ├── core/          # 核心业务逻辑',
            '  ├── api/           # API接口层',
            '  ├── services/      # 业务服务层',
            '  ├── repositories/  # 数据访问层',
            '  ├── models/        # 数据模型',
            '  ├── utils/         # 工具函数',
            '  ├── config/        # 配置管理',
            '  └── tests/         # 测试代码'
        ])

        return recommendations

    def _calculate_quality_score(self, metrics: OrganizationMetrics,
                                 issues: List[OrganizationIssue],
                                 categories: Dict[str, List[str]]) -> float:
        """计算组织质量评分"""
        base_score = 0.8  # 从0.8开始，更合理的基准

        # 文件大小评估（更宽松的标准）
        if metrics.max_file_size > 1000:
            base_score -= 0.2
        elif metrics.max_file_size > 600:
            base_score -= 0.1

        # 分类分布奖励（鼓励良好的分类）
        other_count = len(categories.get('other', []))
        total_files = sum(len(files) for files in categories.values())

        if other_count == 0:
            base_score += 0.3  # 完美分类奖励
        elif other_count / total_files < 0.1:
            base_score += 0.2  # 优秀分类奖励
        elif other_count / total_files < 0.2:
            base_score += 0.1  # 良好分类奖励

        # 问题严重性惩罚（更合理的权重）
        severity_weights = {
            'high': 0.08,
            'medium': 0.04,
            'low': 0.01
        }

        total_penalty = 0
        for issue in issues:
            penalty = severity_weights.get(issue.severity, 0)
            total_penalty += penalty

        # 限制最大惩罚不超过0.3
        base_score -= min(total_penalty, 0.3)

        # 文件数量合理性奖励
        if 20 <= metrics.total_files <= 80:
            base_score += 0.1
        elif metrics.total_files > 120:
            base_score -= 0.05

        # 分类均衡性奖励（避免某个分类过于集中）
        category_counts = [len(files) for files in categories.values() if files]
        if category_counts:
            max_count = max(category_counts)
            total_count = sum(category_counts)
            concentration_ratio = max_count / total_count

            if concentration_ratio < 0.4:
                base_score += 0.05  # 分类均衡奖励

        # 确保评分在合理范围内
        final_score = max(0.1, min(1.0, base_score))

        print(
            f"  质量评分计算: 基础分={base_score:.3f}, 问题数={len(issues)}, other文件={other_count}, 最终评分={final_score:.3f}")

        return final_score

    def print_analysis_report(self, result: OrganizationAnalysisResult):
        """打印分析报告"""
        print("\n" + "="*80)
        print("📊 AI代码组织分析报告")
        print("="*80)

        # 基本指标
        print("\n📈 基本指标:")
        print(f"  • 总文件数: {result.metrics.total_files}")
        print(f"  • 总代码行: {result.metrics.total_lines:,}")
        print(f"  • 平均文件大小: {result.metrics.avg_file_size:.1f}行")
        print(f"  • 最大文件: {result.metrics.largest_file} ({result.metrics.max_file_size}行)")
        print(f"  • 组织质量评分: {result.quality_score:.3f}")

        # 分类分布
        print("\n📁 文件分类分布:")
        for category, files in result.category_distribution.items():
            print(f"  • {category}: {len(files)} 个文件")
            if len(files) <= 3:
                for file in files:
                    print(f"    - {file}")
            else:
                for file in files[:2]:
                    print(f"    - {file}")
                print(f"    ... 和其他 {len(files)-2} 个文件")

        # 问题列表
        if result.issues:
            print("\n⚠️ 发现的问题:")
            severity_icons = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
            for i, issue in enumerate(result.issues, 1):
                icon = severity_icons.get(issue.severity, '⚪')
                print(f"  {i}. {icon} [{issue.category}] {issue.description}")
                if len(issue.files_affected) <= 3:
                    print(f"     影响文件: {', '.join(issue.files_affected)}")
                else:
                    print(f"     影响文件: {len(issue.files_affected)} 个文件")
        else:
            print("\n✅ 未发现明显组织问题")

        # 改进建议
        print("\n🎯 改进建议:")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"  {i}. {rec}")

        print(f"\n🏆 组织质量评分: {result.quality_score:.3f}/1.0")
        print("="*80)


def integrate_with_ai_analyzer():
    """
    将组织分析器集成到AI代码分析器的建议方案

    建议的集成方式：
    1. 在AI代码分析器中添加组织分析模块
    2. 扩展AnalysisResult类包含组织分析结果
    3. 在主分析流程中调用组织分析
    4. 在报告生成中包含组织分析结果
    """

    integration_code = '''
# 在ai_intelligent_code_analyzer.py中添加：

from scripts.ai_code_organization_analyzer import AICodeOrganizationAnalyzer

class EnhancedAnalysisResult:
    """增强的分析结果，包含组织分析"""
    def __init__(self):
        self.code_quality = None
        self.organization_quality = None  # 新增组织质量分析

def analyze_organization(target_path: str) -> OrganizationAnalysisResult:
    """组织分析函数"""
    analyzer = AICodeOrganizationAnalyzer(target_path)
    return analyzer.analyze_organization()

# 在主分析函数中集成：
def perform_comprehensive_analysis(target_path: str) -> EnhancedAnalysisResult:
    result = EnhancedAnalysisResult()

    # 原有的代码质量分析
    result.code_quality = analyze_code_quality(target_path)

    # 新增的组织质量分析
    result.organization_quality = analyze_organization(target_path)

    # 计算综合评分
    code_score = result.code_quality.quality_score
    org_score = result.organization_quality.quality_score
    result.overall_score = (code_score + org_score) / 2

    return result
'''

    print("🔧 组织分析器集成方案:")
    print("="*60)
    print("建议将AICodeOrganizationAnalyzer作为AI代码分析器的子模块")
    print("集成后的分析器将同时提供代码质量和组织质量评估")
    print("\n集成要点:")
    print("1. ✅ 扩展AnalysisResult包含组织分析结果")
    print("2. ✅ 在主分析流程中调用组织分析")
    print("3. ✅ 在质量评分中结合组织因素")
    print("4. ✅ 在报告中展示组织分析结果")
    print("\n示例集成代码:")
    print(integration_code)


if __name__ == '__main__':
    # 示例用法
    analyzer = AICodeOrganizationAnalyzer('src/infrastructure/resource')
    result = analyzer.analyze_organization()
    analyzer.print_analysis_report(result)

    print("\n" + "="*60)
    integrate_with_ai_analyzer()
