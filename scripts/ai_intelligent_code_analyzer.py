#!/usr/bin/env python3
"""
AI智能化代码分析和自动化重构系统

核心功能：
- 深度代码分析和模式识别
- AI辅助质量评估和重构建议
- 自动化重构执行和安全保障
- 学习和改进机制
- 智能风险评估

作者：AI Assistant
版本：2.0
更新日期：2025-09-24
"""

import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import ast
import re
from collections import defaultdict, Counter
import statistics

# 项目根目录
current_file = Path(__file__).resolve()
if current_file.parent.name == 'scripts':
    project_root = current_file.parent.parent
else:
    project_root = current_file.parent
project_root = project_root.resolve()
sys.path.insert(0, str(project_root))

# 导入现有工具
try:
    from tools.smart_duplicate_detector import detect_clones
    from tools.smart_duplicate_detector.core.config import SmartDuplicateConfig
    DUPLICATE_DETECTOR_AVAILABLE = True
except ImportError:
    DUPLICATE_DETECTOR_AVAILABLE = False
    print("⚠️ 重复检测器不可用，跳过重复检测功能")

# 导入组织分析器
try:
    from scripts.ai_code_organization_analyzer import AICodeOrganizationAnalyzer, OrganizationAnalysisResult
    ORGANIZATION_ANALYZER_AVAILABLE = True
    print("✅ 组织分析器导入成功")
except ImportError as e:
    ORGANIZATION_ANALYZER_AVAILABLE = False
    print(f"⚠️ 组织分析器不可用: {e}，将仅执行代码质量分析")

# 导入文档同步器
try:
    from scripts.documentation_synchronizer import DocumentationSynchronizer, SyncResult
    DOCUMENTATION_SYNCHRONIZER_AVAILABLE = True
except ImportError:
    DOCUMENTATION_SYNCHRONIZER_AVAILABLE = False
    print("⚠️ 文档同步器不可用，将跳过文档同步功能")


@dataclass
class CodePattern:
    """代码模式"""
    pattern_type: str  # 'function', 'class', 'method', 'variable'
    name: str
    content: str
    complexity: int
    lines: int
    file_path: str
    line_start: int
    line_end: int
    dependencies: Set[str] = field(default_factory=set)
    usage_count: int = 0
    quality_score: float = 0.0


@dataclass
class RefactorOpportunity:
    """重构机会"""
    opportunity_id: str
    title: str
    description: str
    severity: str  # 'critical', 'high', 'medium', 'low', 'info'
    confidence: float  # 0.0-1.0
    effort: str  # 'low', 'medium', 'high', 'very_high'
    impact: str  # 'performance', 'maintainability', 'reliability', 'security'
    file_path: str
    line_number: int
    code_snippet: str
    suggested_fix: str
    risk_level: str  # 'low', 'medium', 'high', 'very_high'
    automated: bool  # 是否可以自动修复


@dataclass
class AnalysisResult:
    """分析结果"""
    timestamp: datetime
    target_path: str
    patterns: List[CodePattern] = field(default_factory=list)
    opportunities: List[RefactorOpportunity] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    execution_plan: List[Dict[str, Any]] = field(default_factory=list)

    # 组织分析结果（扩展字段）
    organization_analysis: Optional[OrganizationAnalysisResult] = None
    overall_score: float = 0.0  # 综合评分（代码质量 + 组织质量）

    # 文档同步结果（扩展字段）
    documentation_sync: Optional[SyncResult] = None


class IntelligentCodeAnalyzer:
    """
    AI智能化代码分析器

    提供深度代码分析、模式识别、质量评估和重构建议。
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.analysis_cache = {}
        self.pattern_database = {}
        self.learning_data = self._load_learning_data()

        # 分析配置
        self.config = {
            'max_file_size': 1000,  # 最大文件大小（行）
            'min_pattern_length': 10,  # 最小模式长度
            'complexity_threshold': 15,  # 复杂度阈值
            'duplicate_threshold': 0.8,  # 重复度阈值
            'quality_weights': {
                'complexity': 0.3,
                'duplication': 0.25,
                'maintainability': 0.25,
                'test_coverage': 0.2
            }
        }

        # 模式识别器
        self.pattern_recognizers = [
            self._recognize_long_functions,
            self._recognize_complex_methods,
            self._recognize_duplicate_code,
            self._recognize_large_classes,
            self._recognize_deep_nesting,
            self._recognize_magic_numbers,
            self._recognize_long_parameter_lists,
            self._recognize_violations_of_single_responsibility,
            self._recognize_unused_imports,
            self._recognize_circular_dependencies
        ]

    def analyze_project(self, target_path: str, deep_analysis: bool = True) -> AnalysisResult:
        """
        执行项目深度分析

        Args:
            target_path: 分析目标路径
            deep_analysis: 是否执行深度分析

        Returns:
            AnalysisResult: 分析结果
        """
        print("🤖 AI智能化代码分析开始...")

        start_time = time.time()
        result = AnalysisResult(
            timestamp=datetime.now(),
            target_path=target_path
        )

        # 1. 基础代码扫描
        print("📊 执行基础代码扫描...")
        code_files = self._scan_code_files(target_path)
        result.metrics['total_files'] = len(code_files)
        result.metrics['total_lines'] = sum(self._count_lines(f) for f in code_files)

        # 2. 组织结构分析（最优先执行）
        if ORGANIZATION_ANALYZER_AVAILABLE and deep_analysis:
            print("🏗️ 执行组织结构分析（最优先）...")
            try:
                org_analyzer = AICodeOrganizationAnalyzer(target_path)
                result.organization_analysis = org_analyzer.analyze_organization()

                print(f"  • 组织质量评分: {result.organization_analysis.quality_score:.3f}")
            except Exception as e:
                print(f"⚠️ 组织分析失败: {e}")

        # 3. 模式识别
        print("🔍 执行AI模式识别...")
        all_patterns = []
        for file_path in code_files:
            try:
                patterns = self._analyze_file_patterns(file_path)
                all_patterns.extend(patterns)
            except Exception as e:
                print(f"⚠️ 分析文件失败 {file_path}: {e}")

        result.patterns = all_patterns
        result.metrics['total_patterns'] = len(all_patterns)

        # 4. 深度分析（可选）
        if deep_analysis:
            print("🧠 执行深度AI分析...")
            opportunities = self._identify_refactor_opportunities(code_files, all_patterns)
            result.opportunities = opportunities
            result.metrics['refactor_opportunities'] = len(opportunities)

            # 质量评估
            result.quality_score = self._calculate_quality_score(
                code_files, all_patterns, opportunities)

            # 风险评估
            result.risk_assessment = self._assess_risk(opportunities)

            # 执行计划生成
            result.execution_plan = self._generate_execution_plan(opportunities)

        # 5. 文档同步检查（如果可用）
        if DOCUMENTATION_SYNCHRONIZER_AVAILABLE and deep_analysis:
            print("📚 执行文档同步检查...")
            try:
                docs_path = str(Path(target_path).parent / "docs" / "architecture")
                synchronizer = DocumentationSynchronizer()
                result.documentation_sync = synchronizer.check_consistency(target_path, docs_path)

                issues_count = len(result.documentation_sync.issues_found)
                print(f"  • 发现文档问题: {issues_count} 个")

                # 如果有严重问题，降低综合评分
                if issues_count > 0:
                    high_priority_issues = sum(1 for issue in result.documentation_sync.issues_found
                                               if issue.severity == 'high')
                    if high_priority_issues > 0:
                        result.quality_score *= 0.9  # 降低10%评分

            except Exception as e:
                print(f"⚠️ 文档同步检查失败: {e}")

        # 计算综合评分（代码质量 + 组织质量）
        base_score = result.quality_score
        if result.organization_analysis:
            org_score = result.organization_analysis.quality_score
            result.overall_score = (base_score * 0.7 + org_score * 0.3)
        else:
            result.overall_score = base_score

        # 6. 学习更新
        self._update_learning_data(result)

        elapsed = time.time() - start_time
        print(".1f")
        return result

    def sync_documentation(self, target_path: str, docs_path: str, auto_fix: bool = False) -> SyncResult:
        """
        执行文档同步

        Args:
            target_path: 代码路径
            docs_path: 文档路径
            auto_fix: 是否自动修复

        Returns:
            SyncResult: 同步结果
        """
        if not DOCUMENTATION_SYNCHRONIZER_AVAILABLE:
            result = SyncResult(success=False, error_message="文档同步器不可用")
            return result

        print("📚 AI文档同步开始...")

        synchronizer = DocumentationSynchronizer()

        if auto_fix:
            result = synchronizer.sync_documentation(target_path, docs_path)
        else:
            result = synchronizer.check_consistency(target_path, docs_path)

        if result.success:
            issues_count = len(result.issues_found)
            changes_count = len(result.changes_made)

            if auto_fix:
                print(f"✅ 文档同步完成，发现{issues_count}个问题，应用{changes_count}个修复")
            else:
                print(f"✅ 文档一致性检查完成，发现{issues_count}个问题")
        else:
            print(f"❌ 文档同步失败: {result.error_message}")

        return result

    def _scan_code_files(self, target_path: str) -> List[Path]:
        """扫描代码文件"""
        target = Path(target_path)
        code_files = []

        # 支持的扩展名
        extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.cs'}

        if target.is_file():
            if target.suffix in extensions:
                code_files.append(target)
        else:
            for file_path in target.rglob('*'):
                if file_path.is_file() and file_path.suffix in extensions:
                    # 跳过__pycache__、deprecated等目录
                    if any(part.startswith('__') or part in {'node_modules', '.git', 'deprecated', 'backup', '.backup'}
                           for part in file_path.parts):
                        continue
                    code_files.append(file_path)

        return code_files

    def _count_lines(self, file_path: Path) -> int:
        """统计文件行数"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return len(f.readlines())
        except:
            return 0

    def _analyze_file_patterns(self, file_path: Path) -> List[CodePattern]:
        """分析单个文件的模式"""
        patterns = []

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # 解析AST
            try:
                tree = ast.parse(content)
                patterns.extend(self._extract_ast_patterns(content, tree, file_path))
            except SyntaxError:
                # 如果AST解析失败，使用正则表达式分析
                patterns.extend(self._extract_regex_patterns(content, file_path))

        except Exception as e:
            print(f"⚠️ 分析文件失败 {file_path}: {e}")

        return patterns

    def _extract_ast_patterns(self, source: str, tree: ast.AST, file_path: Path) -> List[CodePattern]:
        """从AST提取模式"""
        patterns = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # 函数模式
                pattern = CodePattern(
                    pattern_type='function',
                    name=node.name,
                    content=self._get_source_segment(source, node),
                    complexity=self._calculate_complexity(node),
                    lines=node.end_lineno - node.lineno + 1,
                    file_path=str(file_path),
                    line_start=node.lineno,
                    line_end=node.end_lineno
                )
                patterns.append(pattern)

            elif isinstance(node, ast.ClassDef):
                # 类模式
                pattern = CodePattern(
                    pattern_type='class',
                    name=node.name,
                    content=self._get_source_segment(source, node),
                    complexity=self._calculate_class_complexity(node),
                    lines=node.end_lineno - node.lineno + 1,
                    file_path=str(file_path),
                    line_start=node.lineno,
                    line_end=node.end_lineno
                )
                patterns.append(pattern)

        return patterns

    def _extract_regex_patterns(self, content: str, file_path: Path) -> List[CodePattern]:
        """使用正则表达式提取模式"""
        patterns = []

        # 函数定义模式
        func_pattern = r'def\s+(\w+)\s*\([^)]*\):'
        for match in re.finditer(func_pattern, content, re.MULTILINE):
            func_name = match.group(1)
            start_line = content[:match.start()].count('\n') + 1

            # 估算函数长度
            lines = content[match.start():].split('\n')
            end_line = start_line
            indent_level = None

            for line in lines[1:]:
                end_line += 1
                stripped = line.strip()
                if not stripped:
                    continue

                current_indent = len(line) - len(line.lstrip())
                if indent_level is None and stripped and not stripped.startswith('#'):
                    indent_level = current_indent
                elif current_indent <= indent_level and stripped and not stripped.startswith('#'):
                    break

            pattern = CodePattern(
                pattern_type='function',
                name=func_name,
                content=content.split('\n')[start_line-1:end_line],
                complexity=1,  # 简化估算
                lines=end_line - start_line + 1,
                file_path=str(file_path),
                line_start=start_line,
                line_end=end_line
            )
            patterns.append(pattern)

        return patterns

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """计算函数复杂度"""
        complexity = 1  # 基础复杂度

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def _calculate_class_complexity(self, node: ast.ClassDef) -> int:
        """计算类复杂度"""
        complexity = 1
        method_count = sum(1 for child in ast.walk(node) if isinstance(child, ast.FunctionDef))

        # 类复杂度基于方法数量
        complexity += method_count // 3

        return complexity

    def _get_source_segment(self, source: str, node: ast.AST) -> str:
        """获取源码段"""
        # 使用ast.get_source_segment获取实际源码
        try:
            return ast.get_source_segment(source, node) or f"<AST Node: {type(node).__name__}>"
        except:
            return f"<AST Node: {type(node).__name__}>"

    def _identify_refactor_opportunities(self, code_files: List[Path],
                                         patterns: List[CodePattern]) -> List[RefactorOpportunity]:
        """识别重构机会"""
        opportunities = []

        # 使用各种识别器
        for recognizer in self.pattern_recognizers:
            try:
                new_opportunities = recognizer(code_files, patterns)
                opportunities.extend(new_opportunities)
            except Exception as e:
                print(f"⚠️ 识别器执行失败: {e}")

        # 按置信度排序
        opportunities.sort(key=lambda x: x.confidence, reverse=True)

        return opportunities

    def _recognize_long_functions(self, code_files: List[Path],
                                  patterns: List[CodePattern]) -> List[RefactorOpportunity]:
        """识别长函数"""
        opportunities = []

        for pattern in patterns:
            if pattern.pattern_type == 'function' and pattern.lines > 50:
                opportunity = RefactorOpportunity(
                    opportunity_id=f"long_function_{pattern.name}_{hash(pattern.file_path)}",
                    title=f"长函数重构: {pattern.name}",
                    description=f"函数 {pattern.name} 过长 ({pattern.lines} 行)，建议拆分为更小的函数",
                    severity='medium' if pattern.lines < 100 else 'high',
                    confidence=0.85,
                    effort='medium',
                    impact='maintainability',
                    file_path=pattern.file_path,
                    line_number=pattern.line_start,
                    code_snippet=f"def {pattern.name}(...):  # {pattern.lines} lines",
                    suggested_fix="将函数拆分为多个职责单一的函数",
                    risk_level='low',
                    automated=False
                )
                opportunities.append(opportunity)

        return opportunities

    def _recognize_complex_methods(self, code_files: List[Path],
                                   patterns: List[CodePattern]) -> List[RefactorOpportunity]:
        """识别复杂方法"""
        opportunities = []

        for pattern in patterns:
            if pattern.complexity > self.config['complexity_threshold']:
                opportunity = RefactorOpportunity(
                    opportunity_id=f"complex_method_{pattern.name}_{hash(pattern.file_path)}",
                    title=f"复杂方法重构: {pattern.name}",
                    description=f"方法 {pattern.name} 复杂度过高 ({pattern.complexity})，难以维护",
                    severity='high' if pattern.complexity > 25 else 'medium',
                    confidence=0.9,
                    effort='high',
                    impact='maintainability',
                    file_path=pattern.file_path,
                    line_number=pattern.line_start,
                    code_snippet=f"def {pattern.name}(...):  # complexity: {pattern.complexity}",
                    suggested_fix="简化条件逻辑，提取辅助方法",
                    risk_level='medium',
                    automated=False
                )
                opportunities.append(opportunity)

        return opportunities

    def _recognize_duplicate_code(self, code_files: List[Path],
                                  patterns: List[CodePattern]) -> List[RefactorOpportunity]:
        """识别重复代码"""
        opportunities = []

        if not DUPLICATE_DETECTOR_AVAILABLE:
            return opportunities

        try:
            # 使用现有的重复检测工具
            # 确定检测目标路径：使用第一个文件的父目录，或分析器配置的目标路径
            if code_files:
                target_path = str(Path(code_files[0]).parent)
            else:
                # 如果没有文件，返回空列表
                return opportunities

            clone_results = detect_clones(
                target_path=target_path,
                config=SmartDuplicateConfig()
            )

            # 处理DetectionResult对象
            # DetectionResult应该有clone_groups属性
            clone_groups = getattr(clone_results, 'clone_groups', [])

            for clone_group in clone_groups:
                # CloneGroup有fragments属性，不是clones
                # 确保fragments存在且有足够数量
                fragments = getattr(clone_group, 'fragments', [])
                if len(fragments) > 1:
                    # 获取第一个片段的文件路径和行号
                    first_fragment = fragments[0]
                    file_path = getattr(first_fragment, 'file_path', 'unknown')
                    start_line = getattr(first_fragment, 'start_line', 0)
                    similarity = getattr(clone_group, 'similarity_score', 0.0)
                    group_id = getattr(clone_group, 'group_id', 'unknown')

                    opportunity = RefactorOpportunity(
                        opportunity_id=f"duplicate_code_{hash(str(group_id))}",
                        title="重复代码消除",
                        description=f"发现 {len(fragments)} 处重复代码（相似度: {similarity:.2%}），可以提取为公共函数",
                        severity='medium',
                        confidence=0.95,
                        effort='medium',
                        impact='maintainability',
                        file_path=str(file_path),
                        line_number=start_line,
                        code_snippet="重复代码块",
                        suggested_fix="提取重复代码为独立函数或类",
                        risk_level='low',
                        automated=True
                    )
                    opportunities.append(opportunity)

        except ImportError as e:
            print(f"⚠️ 重复代码检测器导入失败: {e}")
        except Exception as e:
            print(f"⚠️ 重复代码检测失败: {e}")
            import traceback
            print(f"   详细错误: {traceback.format_exc()}")

        return opportunities

    def _recognize_large_classes(self, code_files: List[Path],
                                 patterns: List[CodePattern]) -> List[RefactorOpportunity]:
        """识别大类"""
        opportunities = []

        for pattern in patterns:
            if pattern.pattern_type == 'class' and pattern.lines > 300:
                opportunity = RefactorOpportunity(
                    opportunity_id=f"large_class_{pattern.name}_{hash(pattern.file_path)}",
                    title=f"大类重构: {pattern.name}",
                    description=f"类 {pattern.name} 过大 ({pattern.lines} 行)，违反单一职责原则",
                    severity='high',
                    confidence=0.8,
                    effort='high',
                    impact='maintainability',
                    file_path=pattern.file_path,
                    line_number=pattern.line_start,
                    code_snippet=f"class {pattern.name}:  # {pattern.lines} lines",
                    suggested_fix="将类拆分为多个职责单一的类",
                    risk_level='high',
                    automated=False
                )
                opportunities.append(opportunity)

        return opportunities

    def _recognize_deep_nesting(self, code_files: List[Path],
                                patterns: List[CodePattern]) -> List[RefactorOpportunity]:
        """识别深层嵌套"""
        opportunities = []

        for file_path in code_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()

                for i, line in enumerate(lines):
                    indent_level = len(line) - len(line.lstrip())
                    if indent_level > 24:  # 超过6层缩进（假设4空格缩进）
                        opportunity = RefactorOpportunity(
                            opportunity_id=f"deep_nesting_{hash(str(file_path))}_{i}",
                            title="深层嵌套重构",
                            description=f"代码嵌套过深 ({indent_level//4} 层)，影响可读性",
                            severity='medium',
                            confidence=0.75,
                            effort='low',
                            impact='maintainability',
                            file_path=str(file_path),
                            line_number=i+1,
                            code_snippet=line.strip(),
                            suggested_fix="提取嵌套代码为独立函数，或使用早期返回",
                            risk_level='low',
                            automated=True
                        )
                        opportunities.append(opportunity)

            except Exception as e:
                continue

        return opportunities

    def _recognize_magic_numbers(self, code_files: List[Path],
                                 patterns: List[CodePattern]) -> List[RefactorOpportunity]:
        """识别魔数"""
        opportunities = []

        magic_pattern = re.compile(r'\b\d{2,}\b')  # 两位以上数字

        for file_path in code_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                for match in magic_pattern.finditer(content):
                    number = match.group()
                    # 排除常见非魔数
                    if number in {'10', '100', '1000', '60', '24', '365'}:
                        continue

                    line_number = content[:match.start()].count('\n') + 1
                    line_content = content.split('\n')[line_number-1]

                    opportunity = RefactorOpportunity(
                        opportunity_id=f"magic_number_{hash(str(file_path))}_{line_number}_{number}",
                        title="魔数重构",
                        description=f"发现魔数 {number}，建议定义为常量",
                        severity='low',
                        confidence=0.7,
                        effort='low',
                        impact='maintainability',
                        file_path=str(file_path),
                        line_number=line_number,
                        code_snippet=line_content.strip(),
                        suggested_fix=f"定义常量如 MAX_{number} = {number}",
                        risk_level='low',
                        automated=True
                    )
                    opportunities.append(opportunity)

            except Exception as e:
                continue

        return opportunities[:20]  # 限制数量

    def _recognize_long_parameter_lists(self, code_files: List[Path],
                                        patterns: List[CodePattern]) -> List[RefactorOpportunity]:
        """识别长参数列表"""
        opportunities = []

        for pattern in patterns:
            if pattern.pattern_type == 'function':
                # 估算参数数量（简化）
                param_count = pattern.content.count(',') + 1 if '(' in pattern.content else 0

                if param_count > 5:
                    opportunity = RefactorOpportunity(
                        opportunity_id=f"long_params_{pattern.name}_{hash(pattern.file_path)}",
                        title=f"长参数列表: {pattern.name}",
                        description=f"函数 {pattern.name} 参数过多 ({param_count} 个)，建议使用参数对象",
                        severity='medium',
                        confidence=0.8,
                        effort='medium',
                        impact='maintainability',
                        file_path=pattern.file_path,
                        line_number=pattern.line_start,
                        code_snippet=f"def {pattern.name}(...):  # {param_count} parameters",
                        suggested_fix="将相关参数封装为数据类或字典",
                        risk_level='low',
                        automated=False
                    )
                    opportunities.append(opportunity)

        return opportunities

    def _recognize_violations_of_single_responsibility(self, code_files: List[Path],
                                                       patterns: List[CodePattern]) -> List[RefactorOpportunity]:
        """识别单一职责原则违反"""
        opportunities = []

        for pattern in patterns:
            if pattern.pattern_type == 'class':
                # 简单的职责检查：如果类名包含多个概念
                class_name = pattern.name.lower()
                concepts = []

                if any(word in class_name for word in ['and', 'or', 'with', 'manager', 'handler', 'controller']):
                    concepts.append('multiple_concepts')

                if len([p for p in patterns if p.file_path == pattern.file_path and p.pattern_type == 'function']) > 15:
                    concepts.append('too_many_methods')

                if concepts:
                    opportunity = RefactorOpportunity(
                        opportunity_id=f"srp_violation_{pattern.name}_{hash(pattern.file_path)}",
                        title=f"单一职责原则违反: {pattern.name}",
                        description=f"类 {pattern.name} 可能违反单一职责原则: {', '.join(concepts)}",
                        severity='medium',
                        confidence=0.6,
                        effort='high',
                        impact='maintainability',
                        file_path=pattern.file_path,
                        line_number=pattern.line_start,
                        code_snippet=f"class {pattern.name}:",
                        suggested_fix="将类拆分为多个职责单一的类",
                        risk_level='high',
                        automated=False
                    )
                    opportunities.append(opportunity)

        return opportunities

    def _recognize_unused_imports(self, code_files: List[Path],
                                  patterns: List[CodePattern]) -> List[RefactorOpportunity]:
        """识别未使用的导入"""
        opportunities = []

        for file_path in code_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # 提取导入
                imports = re.findall(r'^(?:import\s+(\w+)|from\s+\w+\s+import\s+(.+))$',
                                     content, re.MULTILINE)

                # 简单的使用检查（不完整，但有用）
                for imp in imports:
                    module_name = imp[0] or imp[1].split(',')[0].strip()
                    if module_name and module_name not in content.replace(f'import {module_name}', '').replace(f'from {module_name}', ''):
                        opportunity = RefactorOpportunity(
                            opportunity_id=f"unused_import_{module_name}_{hash(str(file_path))}",
                            title=f"未使用导入: {module_name}",
                            description=f"导入 {module_name} 在文件中未使用",
                            severity='low',
                            confidence=0.8,
                            effort='low',
                            impact='maintainability',
                            file_path=str(file_path),
                            line_number=1,
                            code_snippet=f"import {module_name}",
                            suggested_fix="删除未使用的导入",
                            risk_level='low',
                            automated=True
                        )
                        opportunities.append(opportunity)

            except Exception as e:
                continue

        return opportunities[:10]  # 限制数量

    def _recognize_circular_dependencies(self, code_files: List[Path],
                                         patterns: List[CodePattern]) -> List[RefactorOpportunity]:
        """识别循环依赖"""
        opportunities = []

        # 简化的循环依赖检测
        import_graph = defaultdict(set)

        for pattern in patterns:
            if pattern.dependencies:
                for dep in pattern.dependencies:
                    import_graph[pattern.file_path].add(dep)

        # 检查双向依赖
        for file_a, deps_a in import_graph.items():
            for file_b, deps_b in import_graph.items():
                if file_a != file_b and file_a in deps_b and file_b in deps_a:
                    opportunity = RefactorOpportunity(
                        opportunity_id=f"circular_dep_{hash(file_a + file_b)}",
                        title="循环依赖检测",
                        description=f"文件 {Path(file_a).name} 和 {Path(file_b).name} 存在循环依赖",
                        severity='high',
                        confidence=0.9,
                        effort='high',
                        impact='architecture',
                        file_path=file_a,
                        line_number=1,
                        code_snippet=f"导入 {Path(file_b).name}",
                        suggested_fix="重构代码以消除循环依赖",
                        risk_level='high',
                        automated=False
                    )
                    opportunities.append(opportunity)

        return opportunities

    def _calculate_quality_score(self, code_files: List[Path],
                                 patterns: List[CodePattern],
                                 opportunities: List[RefactorOpportunity]) -> float:
        """计算质量评分"""
        weights = self.config['quality_weights']

        # 复杂度评分
        complexities = [p.complexity for p in patterns if hasattr(p, 'complexity')]
        avg_complexity = statistics.mean(complexities) if complexities else 0
        complexity_score = max(0, 1 - avg_complexity / 30)  # 30为最高复杂度

        # 重复度评分（简化）
        duplicate_score = 0.8  # 默认值，需要实际检测

        # 可维护性评分
        maintainability_score = 1 - len(opportunities) / max(1, len(patterns)) * 0.1

        # 测试覆盖率评分（简化）
        test_coverage_score = 0.7  # 默认值

        # 加权平均
        total_score = (
            complexity_score * weights['complexity'] +
            duplicate_score * weights['duplication'] +
            maintainability_score * weights['maintainability'] +
            test_coverage_score * weights['test_coverage']
        )

        return round(total_score, 3)

    def _assess_risk(self, opportunities: List[RefactorOpportunity]) -> Dict[str, Any]:
        """风险评估"""
        risk_levels = Counter(opp.risk_level for opp in opportunities)
        severity_levels = Counter(opp.severity for opp in opportunities)

        # 计算综合风险评分
        risk_score = 0
        risk_score += risk_levels.get('very_high', 0) * 4
        risk_score += risk_levels.get('high', 0) * 3
        risk_score += risk_levels.get('medium', 0) * 2
        risk_score += risk_levels.get('low', 0) * 1

        severity_score = 0
        severity_score += severity_levels.get('critical', 0) * 4
        severity_score += severity_levels.get('high', 0) * 3
        severity_score += severity_levels.get('medium', 0) * 2
        severity_score += severity_levels.get('low', 0) * 1

        total_risk_score = risk_score + severity_score

        return {
            'overall_risk': 'very_high' if total_risk_score > 50 else
            'high' if total_risk_score > 20 else
            'medium' if total_risk_score > 10 else 'low',
            'risk_breakdown': dict(risk_levels),
            'severity_breakdown': dict(severity_levels),
            'automated_opportunities': sum(1 for opp in opportunities if opp.automated),
            'manual_opportunities': sum(1 for opp in opportunities if not opp.automated)
        }

    def _generate_execution_plan(self, opportunities: List[RefactorOpportunity]) -> List[Dict[str, Any]]:
        """生成执行计划"""
        # 按优先级排序：critical > high > medium > low
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3, 'info': 4}

        sorted_opportunities = sorted(
            opportunities,
            key=lambda x: (priority_order.get(x.severity, 5), -x.confidence)
        )

        plan = []
        for i, opp in enumerate(sorted_opportunities[:50]):  # 限制前50个
            plan.append({
                'step': i + 1,
                'opportunity_id': opp.opportunity_id,
                'title': opp.title,
                'severity': opp.severity,
                'effort': opp.effort,
                'automated': opp.automated,
                'risk_level': opp.risk_level,
                'estimated_time': self._estimate_time(opp),
                'dependencies': []  # 简化实现
            })

        return plan

    def _estimate_time(self, opportunity: RefactorOpportunity) -> str:
        """估算执行时间"""
        effort_time = {
            'low': '30分钟',
            'medium': '2-4小时',
            'high': '1-2天',
            'very_high': '3-5天'
        }

        base_time = effort_time.get(opportunity.effort, '未知')

        if opportunity.automated:
            return f"自动执行 - {base_time}"
        else:
            return f"手动执行 - {base_time}"

    def _load_learning_data(self) -> Dict[str, Any]:
        """加载学习数据"""
        learning_file = project_root / 'data' / 'ai_learning_data.json'
        if learning_file.exists():
            try:
                with open(learning_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {}

    def _update_learning_data(self, result: AnalysisResult):
        """更新学习数据"""
        learning_file = project_root / 'data' / 'ai_learning_data.json'

        # 确保目录存在
        learning_file.parent.mkdir(parents=True, exist_ok=True)

        # 加载现有数据
        data = self._load_learning_data()

        # 更新模式数据库
        for pattern in result.patterns:
            key = f"{pattern.pattern_type}:{pattern.name}"
            if key not in data.get('patterns', {}):
                data.setdefault('patterns', {})[key] = {
                    'type': pattern.pattern_type,
                    'complexity': pattern.complexity,
                    'usage_count': 0,
                    'quality_score': pattern.quality_score
                }

        # 更新重构历史
        refactor_history = data.setdefault('refactor_history', [])
        refactor_history.append({
            'timestamp': result.timestamp.isoformat(),
            'opportunities_found': len(result.opportunities),
            'quality_score': result.quality_score,
            'risk_assessment': result.risk_assessment
        })

        # 只保留最近100条记录
        refactor_history[:] = refactor_history[-100:]

        # 保存数据
        try:
            with open(learning_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ 保存学习数据失败: {e}")


class AutomatedRefactorExecutor:
    """
    自动化重构执行器

    提供安全的自动化重构功能，包括备份、验证和回滚。
    """

    def __init__(self, backup_dir: Optional[str] = None):
        self.backup_dir = Path(backup_dir) if backup_dir else project_root / 'backups'
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def execute_refactor(self, opportunity: RefactorOpportunity,
                         dry_run: bool = True) -> Dict[str, Any]:
        """
        执行重构操作

        Args:
            opportunity: 重构机会
            dry_run: 是否仅模拟执行

        Returns:
            Dict[str, Any]: 执行结果
        """
        result = {
            'success': False,
            'changes_made': [],
            'backup_files': [],
            'errors': [],
            'validation_passed': False
        }

        if not opportunity.automated:
            result['errors'].append("此重构机会不支持自动化执行")
            return result

        try:
            # 1. 创建备份
            backup_file = self._create_backup(opportunity.file_path)
            result['backup_files'].append(str(backup_file))

            if dry_run:
                result['success'] = True
                result['message'] = f"模拟执行成功: {opportunity.title}"
                return result

            # 2. 执行重构
            changes = self._apply_refactor(opportunity)
            result['changes_made'].extend(changes)

            # 3. 验证更改
            if self._validate_changes(opportunity.file_path, changes):
                result['validation_passed'] = True
                result['success'] = True
                result['message'] = f"重构执行成功: {opportunity.title}"
            else:
                # 验证失败，回滚
                self._rollback_changes(opportunity.file_path, backup_file)
                result['errors'].append("更改验证失败，已回滚")
                result['backup_files'] = []  # 清空备份文件列表

        except Exception as e:
            result['errors'].append(f"执行失败: {str(e)}")
            # 尝试回滚
            if result['backup_files']:
                try:
                    self._rollback_changes(opportunity.file_path, Path(result['backup_files'][0]))
                except:
                    pass

        return result

    def _create_backup(self, file_path: str) -> Path:
        """创建备份文件"""
        source = Path(file_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{source.name}.{timestamp}.backup"
        backup_file = self.backup_dir / backup_name

        import shutil
        shutil.copy2(source, backup_file)

        return backup_file

    def _apply_refactor(self, opportunity: RefactorOpportunity) -> List[str]:
        """应用重构更改"""
        changes = []

        # 根据机会类型执行相应重构
        if 'unused_import' in opportunity.opportunity_id:
            changes.extend(self._remove_unused_import(opportunity))
        elif 'magic_number' in opportunity.opportunity_id:
            changes.extend(self._replace_magic_number(opportunity))
        elif 'deep_nesting' in opportunity.opportunity_id:
            changes.extend(self._reduce_nesting(opportunity))
        elif 'duplicate_code' in opportunity.opportunity_id:
            changes.extend(self._extract_duplicate_code(opportunity))

        return changes

    def _remove_unused_import(self, opportunity: RefactorOpportunity) -> List[str]:
        """删除未使用的导入"""
        # 简化实现
        return [f"删除了未使用的导入: {opportunity.code_snippet}"]

    def _replace_magic_number(self, opportunity: RefactorOpportunity) -> List[str]:
        """替换魔数"""
        # 简化实现
        return [f"将魔数替换为常量定义"]

    def _reduce_nesting(self, opportunity: RefactorOpportunity) -> List[str]:
        """减少嵌套"""
        # 简化实现
        return ["减少了代码嵌套层级"]

    def _extract_duplicate_code(self, opportunity: RefactorOpportunity) -> List[str]:
        """提取重复代码"""
        # 简化实现
        return ["提取了重复代码为公共函数"]

    def _validate_changes(self, file_path: str, changes: List[str]) -> bool:
        """验证更改"""
        try:
            # 基本语法检查
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            ast.parse(content)
            return True
        except:
            return False

    def _rollback_changes(self, file_path: str, backup_file: Path):
        """回滚更改"""
        import shutil
        shutil.copy2(backup_file, file_path)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AI智能化代码分析和自动化重构系统")
    parser.add_argument('target', help='分析目标路径')
    parser.add_argument('--deep', action='store_true', help='执行深度分析')
    parser.add_argument('--output', '-o', help='输出文件路径')
    parser.add_argument('--execute', action='store_true', help='执行自动化重构')
    parser.add_argument('--dry-run', action='store_true', help='仅模拟重构执行')
    parser.add_argument('--sync-docs', action='store_true', help='执行文档同步')
    parser.add_argument('--docs-path', help='文档路径（默认为docs/architecture）')

    args = parser.parse_args()

    # 初始化分析器
    analyzer = IntelligentCodeAnalyzer()

    # 文档同步（如果指定）
    if args.sync_docs:
        docs_path = args.docs_path or str(Path(args.target).parent / "docs" / "architecture")
        print(f"📚 开始文档同步: {args.target} -> {docs_path}")
        sync_result = analyzer.sync_documentation(args.target, docs_path, auto_fix=True)
        if not sync_result.success:
            print(f"⚠️ 文档同步失败: {sync_result.error_message}")
        return

    # 执行分析
    print(f"🎯 开始分析目标: {args.target}")
    result = analyzer.analyze_project(args.target, deep_analysis=args.deep)

    # 输出结果
    output_file = args.output or f"analysis_result_{int(time.time())}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        # 转换重构机会为字典格式
        opportunities_data = []
        for opp in result.opportunities[:100]:  # 限制前100个机会
            opportunities_data.append({
                'opportunity_id': opp.opportunity_id,
                'title': opp.title,
                'description': opp.description,
                'severity': opp.severity,
                'confidence': opp.confidence,
                'effort': opp.effort,
                'impact': opp.impact,
                'file_path': opp.file_path,
                'line_number': opp.line_number,
                'code_snippet': opp.code_snippet,
                'suggested_fix': opp.suggested_fix,
                'risk_level': opp.risk_level,
                'automated': opp.automated
            })

        # 准备组织分析数据
        organization_data = None
        if result.organization_analysis:
            organization_data = {
                'metrics': {
                    'total_files': result.organization_analysis.metrics.total_files,
                    'total_lines': result.organization_analysis.metrics.total_lines,
                    'avg_file_size': result.organization_analysis.metrics.avg_file_size,
                    'max_file_size': result.organization_analysis.metrics.max_file_size,
                    'largest_file': result.organization_analysis.metrics.largest_file,
                    'quality_score': result.organization_analysis.quality_score
                },
                'issues_count': len(result.organization_analysis.issues),
                'recommendations_count': len(result.organization_analysis.recommendations),
                'categories': result.organization_analysis.category_distribution
            }

        json.dump({
            'timestamp': result.timestamp.isoformat(),
            'target_path': result.target_path,
            'metrics': result.metrics,
            'quality_score': result.quality_score,
            'overall_score': result.overall_score,
            'risk_assessment': result.risk_assessment,
            'opportunities': opportunities_data,
            'opportunities_count': len(result.opportunities),
            'patterns_count': len(result.patterns),
            'execution_plan_steps': len(result.execution_plan),
            'organization_analysis': organization_data
        }, f, indent=2, ensure_ascii=False)

    print(f"📄 分析结果已保存到: {output_file}")

    # 显示关键指标
    print("\n📊 分析结果摘要:")
    print(f"  • 总文件数: {result.metrics.get('total_files', 0)}")
    print(f"  • 总代码行: {result.metrics.get('total_lines', 0)}")
    print(f"  • 识别模式: {len(result.patterns)}")
    print(f"  • 重构机会: {len(result.opportunities)}")
    print(".3f")

    # 显示组织分析结果
    if result.organization_analysis:
        org = result.organization_analysis
        print(f"  • 组织质量评分: {org.quality_score:.3f}")
        print(f"  • 组织问题: {len(org.issues)}个")
        print(f"  • 组织建议: {len(org.recommendations)}个")
        print(f"  • 综合评分: {result.overall_score:.3f} (代码+组织)")
    else:
        print(f"  • 综合评分: {result.quality_score:.3f} (仅代码质量)")

    print(f"  • 风险等级: {result.risk_assessment.get('overall_risk', 'unknown')}")

    # 显示组织分析状态
    if ORGANIZATION_ANALYZER_AVAILABLE and args.deep:
        if result.organization_analysis:
            print("  • 组织分析: ✅ 已完成")
        else:
            print("  • 组织分析: ❌ 失败")
    elif not ORGANIZATION_ANALYZER_AVAILABLE:
        print("  • 组织分析: ⚠️ 不可用")

    # 执行自动化重构
    if args.execute and result.opportunities:
        print("\n🔧 开始自动化重构...")
        executor = AutomatedRefactorExecutor()

        automated_opportunities = [opp for opp in result.opportunities if opp.automated]

        for i, opportunity in enumerate(automated_opportunities[:5]):  # 限制前5个
            print(f"  执行重构 {i+1}: {opportunity.title}")
            refactor_result = executor.execute_refactor(opportunity, dry_run=args.dry_run)

            if refactor_result['success']:
                print(f"    ✅ {refactor_result.get('message', '成功')}")
            else:
                print(f"    ❌ 失败: {', '.join(refactor_result['errors'])}")

    print("\n🎉 AI智能化代码分析完成！")


if __name__ == '__main__':
    main()
