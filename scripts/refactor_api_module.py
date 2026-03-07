#!/usr/bin/env python3
"""
API模块自动化重构工具

用于辅助API模块大类的重构工作
"""

import sys
import ast
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field

# 项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class ClassAnalysis:
    """类分析结果"""
    name: str
    file_path: str
    line_count: int
    methods: List[Dict[str, Any]] = field(default_factory=list)
    responsibilities: List[str] = field(default_factory=list)
    suggested_splits: List[Dict[str, Any]] = field(default_factory=list)


class APIModuleRefactorAssistant:
    """API模块重构助手"""
    
    def __init__(self, api_module_path: str = "src/infrastructure/api"):
        self.api_module_path = Path(api_module_path)
        self.analysis_results: Dict[str, ClassAnalysis] = {}
    
    def analyze_class(self, file_path: Path) -> ClassAnalysis:
        """分析单个类文件"""
        print(f"\n📊 分析文件: {file_path.name}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
        
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            print(f"⚠️  语法错误: {e}")
            return None
        
        # 找到主类（最大的那个）
        main_class = None
        max_lines = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_lines = node.end_lineno - node.lineno + 1
                if class_lines > max_lines and class_lines > 100:  # 至少100行才算大类
                    max_lines = class_lines
                    main_class = node
        
        if not main_class:
            return None
        
        # 分析方法
        methods = []
        for node in main_class.body:
            if isinstance(node, ast.FunctionDef):
                method_lines = node.end_lineno - node.lineno + 1
                methods.append({
                    'name': node.name,
                    'start_line': node.lineno,
                    'end_line': node.end_lineno,
                    'lines': method_lines,
                    'is_private': node.name.startswith('_'),
                    'complexity': self._estimate_complexity(node)
                })
        
        class_lines = main_class.end_lineno - main_class.lineno + 1
        
        analysis = ClassAnalysis(
            name=main_class.name,
            file_path=str(file_path),
            line_count=class_lines,
            methods=methods
        )
        
        # 分析职责
        analysis.responsibilities = self._analyze_responsibilities(methods)
        
        # 生成拆分建议
        analysis.suggested_splits = self._suggest_splits(analysis)
        
        return analysis
    
    def _estimate_complexity(self, node: ast.FunctionDef) -> int:
        """估算函数复杂度"""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity
    
    def _analyze_responsibilities(self, methods: List[Dict]) -> List[str]:
        """分析类的职责"""
        responsibilities = set()
        
        for method in methods:
            name = method['name']
            
            # 根据方法名推断职责
            if 'template' in name.lower():
                responsibilities.add('模板管理')
            elif 'export' in name.lower():
                responsibilities.add('导出功能')
            elif 'statistic' in name.lower() or 'stats' in name.lower():
                responsibilities.add('统计分析')
            elif 'create' in name.lower() and 'test' in name.lower():
                responsibilities.add('测试生成')
            elif 'data' in name.lower():
                responsibilities.add('数据服务测试')
            elif 'feature' in name.lower():
                responsibilities.add('特征服务测试')
            elif 'trading' in name.lower():
                responsibilities.add('交易服务测试')
            elif 'monitoring' in name.lower():
                responsibilities.add('监控服务测试')
        
        return list(responsibilities)
    
    def _suggest_splits(self, analysis: ClassAnalysis) -> List[Dict[str, Any]]:
        """建议如何拆分类"""
        suggestions = []
        
        # 按职责分组方法
        responsibility_groups = {}
        for method in analysis.methods:
            for resp in analysis.responsibilities:
                if self._method_belongs_to_responsibility(method['name'], resp):
                    if resp not in responsibility_groups:
                        responsibility_groups[resp] = []
                    responsibility_groups[resp].append(method)
        
        # 为每个职责创建新类建议
        for responsibility, methods in responsibility_groups.items():
            total_lines = sum(m['lines'] for m in methods)
            class_name = self._generate_class_name(responsibility)
            
            suggestions.append({
                'responsibility': responsibility,
                'suggested_class_name': class_name,
                'methods': [m['name'] for m in methods],
                'estimated_lines': total_lines,
                'priority': 'high' if total_lines > 100 else 'medium'
            })
        
        return suggestions
    
    def _method_belongs_to_responsibility(self, method_name: str, responsibility: str) -> bool:
        """判断方法是否属于某个职责"""
        method_lower = method_name.lower()
        resp_lower = responsibility.lower()
        
        # 简单的关键词匹配
        keywords = {
            '模板管理': ['template', 'load'],
            '导出功能': ['export', 'save'],
            '统计分析': ['statistic', 'stats', 'calculate'],
            '测试生成': ['create', 'generate'],
            '数据服务测试': ['data', 'service'],
            '特征服务测试': ['feature'],
            '交易服务测试': ['trading', 'trade'],
            '监控服务测试': ['monitoring', 'monitor']
        }
        
        if resp_lower in keywords:
            return any(kw in method_lower for kw in keywords[resp_lower])
        
        return False
    
    def _generate_class_name(self, responsibility: str) -> str:
        """根据职责生成类名"""
        name_mapping = {
            '模板管理': 'TestTemplateManager',
            '导出功能': 'TestSuiteExporter',
            '统计分析': 'TestStatisticsCollector',
            '测试生成': 'TestCaseBuilder',
            '数据服务测试': 'DataServiceTestGenerator',
            '特征服务测试': 'FeatureServiceTestGenerator',
            '交易服务测试': 'TradingServiceTestGenerator',
            '监控服务测试': 'MonitoringServiceTestGenerator'
        }
        return name_mapping.get(responsibility, f'{responsibility}Manager')
    
    def analyze_all_large_classes(self) -> Dict[str, ClassAnalysis]:
        """分析所有大类"""
        print("=" * 80)
        print("API模块大类分析")
        print("=" * 80)
        
        # 需要分析的文件
        target_files = [
            'api_test_case_generator.py',
            'openapi_generator.py',
            'api_flow_diagram_generator.py',
            'api_documentation_enhancer.py',
            'api_documentation_search.py'
        ]
        
        results = {}
        for filename in target_files:
            file_path = self.api_module_path / filename
            if file_path.exists():
                analysis = self.analyze_class(file_path)
                if analysis:
                    results[analysis.name] = analysis
                    self._print_analysis_summary(analysis)
        
        return results
    
    def _print_analysis_summary(self, analysis: ClassAnalysis):
        """打印分析摘要"""
        print(f"\n类名: {analysis.name}")
        print(f"总行数: {analysis.line_count}")
        print(f"方法数量: {len(analysis.methods)}")
        print(f"\n识别的职责:")
        for resp in analysis.responsibilities:
            print(f"  • {resp}")
        
        print(f"\n拆分建议 ({len(analysis.suggested_splits)}个新类):")
        for i, suggestion in enumerate(analysis.suggested_splits, 1):
            print(f"\n  {i}. {suggestion['suggested_class_name']} ({suggestion['estimated_lines']}行)")
            print(f"     职责: {suggestion['responsibility']}")
            print(f"     优先级: {suggestion['priority']}")
            print(f"     包含方法: {', '.join(suggestion['methods'][:3])}" + 
                  (f" ... (共{len(suggestion['methods'])}个)" if len(suggestion['methods']) > 3 else ""))
    
    def generate_refactoring_report(self, output_file: str = "analysis_reports/api_refactoring_analysis.json"):
        """生成重构报告"""
        results = self.analyze_all_large_classes()
        
        # 转换为可序列化的格式
        report = {}
        for class_name, analysis in results.items():
            report[class_name] = {
                'file_path': analysis.file_path,
                'line_count': analysis.line_count,
                'method_count': len(analysis.methods),
                'responsibilities': analysis.responsibilities,
                'suggested_splits': analysis.suggested_splits,
                'long_methods': [
                    {'name': m['name'], 'lines': m['lines']}
                    for m in analysis.methods if m['lines'] > 50
                ],
                'complex_methods': [
                    {'name': m['name'], 'complexity': m['complexity']}
                    for m in analysis.methods if m['complexity'] > 10
                ]
            }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n\n📄 详细报告已保存到: {output_path}")
        
        return report
    
    def generate_refactoring_templates(self, output_dir: str = "src/infrastructure/api/refactored"):
        """生成重构模板文件"""
        print("\n" + "=" * 80)
        print("生成重构模板文件")
        print("=" * 80)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 为APITestCaseGenerator生成模板
        self._generate_test_case_generator_templates(output_path)
        
        print(f"\n✅ 模板文件已生成到: {output_path}")
    
    def _generate_test_case_generator_templates(self, output_dir: Path):
        """生成测试用例生成器的重构模板"""
        
        # 1. 基础配置对象
        config_template = '''"""测试用例配置对象"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path


@dataclass
class TestCaseConfig:
    """测试用例配置"""
    title: str
    description: str
    priority: str = "medium"  # high, medium, low
    category: str = "functional"  # functional, integration, performance, security
    preconditions: List[str] = field(default_factory=list)
    test_steps: List[Dict[str, Any]] = field(default_factory=list)
    expected_results: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class ScenarioConfig:
    """测试场景配置"""
    name: str
    description: str
    endpoint: str
    method: str
    setup_steps: List[str] = field(default_factory=list)
    teardown_steps: List[str] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExportConfig:
    """导出配置"""
    format_type: str = "json"
    output_dir: Path = Path("docs/api/tests")
    include_timestamps: bool = True
    include_statistics: bool = True
    pretty_print: bool = True
'''
        
        with open(output_dir / 'test_configs.py', 'w', encoding='utf-8') as f:
            f.write(config_template)
        
        print("  ✅ 创建: test_configs.py")
        
        # 2. 模板管理器
        template_manager = '''"""测试模板管理器"""

from typing import Dict, Any


class TestTemplateManager:
    """负责加载和管理测试模板"""
    
    def __init__(self):
        self.templates: Dict[str, Dict[str, Any]] = self.load_templates()
    
    def load_templates(self) -> Dict[str, Dict[str, Any]]:
        """加载所有测试模板"""
        # TODO: 从原APITestCaseGenerator._load_templates()迁移逻辑
        return {}
    
    def get_template(self, template_type: str, template_name: str) -> Dict[str, Any]:
        """获取指定类型的模板"""
        return self.templates.get(template_type, {}).get(template_name, {})
    
    def get_all_templates(self, template_type: str) -> Dict[str, Any]:
        """获取某类型的所有模板"""
        return self.templates.get(template_type, {})
'''
        
        with open(output_dir / 'template_manager.py', 'w', encoding='utf-8') as f:
            f.write(template_manager)
        
        print("  ✅ 创建: template_manager.py")
        
        # 3. 测试用例构建器基类
        builder_template = '''"""测试用例构建器基类"""

from typing import List
from .test_configs import TestCaseConfig, ScenarioConfig
from .template_manager import TestTemplateManager
# 假设这些类在原模块中定义
# from ..api_test_case_generator import TestCase, TestScenario


class TestCaseBuilder:
    """测试用例构建基类"""
    
    def __init__(self, template_manager: TestTemplateManager):
        self.template_manager = template_manager
    
    def create_test_case(self, config: TestCaseConfig):  # -> TestCase:
        """创建单个测试用例"""
        # TODO: 实现测试用例创建逻辑
        pass
    
    def create_scenario(self, config: ScenarioConfig):  # -> TestScenario:
        """创建测试场景"""
        # TODO: 实现测试场景创建逻辑
        pass
    
    def create_authentication_tests(self) -> List:  # List[TestCase]:
        """创建认证测试"""
        pass
    
    def create_validation_tests(self) -> List:  # List[TestCase]:
        """创建验证测试"""
        pass
'''
        
        with open(output_dir / 'test_case_builder.py', 'w', encoding='utf-8') as f:
            f.write(builder_template)
        
        print("  ✅ 创建: test_case_builder.py")
        
        # 4. 服务测试生成器示例
        service_generator = '''"""数据服务测试生成器"""

from typing import List
from .test_case_builder import TestCaseBuilder
# from ..api_test_case_generator import TestSuite


class DataServiceTestGenerator(TestCaseBuilder):
    """生成数据服务的测试用例"""
    
    def create_test_suite(self):  # -> TestSuite:
        """创建数据服务测试套件"""
        # TODO: 从原create_data_service_test_suite()迁移逻辑
        pass
    
    def _create_data_validation_tests(self) -> List:  # List[TestCase]:
        """创建数据验证测试"""
        pass
    
    def _create_query_tests(self) -> List:  # List[TestCase]:
        """创建查询测试"""
        pass
    
    def _create_cache_tests(self) -> List:  # List[TestCase]:
        """创建缓存测试"""
        pass
'''
        
        with open(output_dir / 'data_service_test_generator.py', 'w', encoding='utf-8') as f:
            f.write(service_generator)
        
        print("  ✅ 创建: data_service_test_generator.py")
        
        # 5. README
        readme = '''# API模块重构文件

本目录包含重构后的API模块新结构文件。

## 文件说明

- `test_configs.py` - 测试配置对象
- `template_manager.py` - 测试模板管理器
- `test_case_builder.py` - 测试用例构建器基类
- `data_service_test_generator.py` - 数据服务测试生成器示例

## 重构步骤

1. 完善各个模板文件中的TODO部分
2. 从原APITestCaseGenerator迁移具体实现
3. 创建其他服务测试生成器（feature, trading, monitoring）
4. 创建TestSuiteExporter和TestStatisticsCollector
5. 创建APITestSuiteCoordinator协调器
6. 编写单元测试
7. 更新导入引用
8. 删除或标记为deprecated原有的大类

## 迁移注意事项

- 保持向后兼容性
- 充分测试每个迁移步骤
- 更新相关文档
'''
        
        with open(output_dir / 'README.md', 'w', encoding='utf-8') as f:
            f.write(readme)
        
        print("  ✅ 创建: README.md")


def main():
    """主函数"""
    print("\n🚀 API模块重构助手")
    print("=" * 80)
    
    assistant = APIModuleRefactorAssistant()
    
    # 1. 分析所有大类
    print("\n步骤 1: 分析大类")
    report = assistant.generate_refactoring_report()
    
    # 2. 生成重构模板
    print("\n步骤 2: 生成重构模板")
    assistant.generate_refactoring_templates()
    
    # 3. 显示总结
    print("\n" + "=" * 80)
    print("📊 重构分析总结")
    print("=" * 80)
    
    total_classes = len(report)
    total_lines = sum(c['line_count'] for c in report.values())
    total_methods = sum(c['method_count'] for c in report.values())
    total_new_classes = sum(len(c['suggested_splits']) for c in report.values())
    
    print(f"\n需要重构的大类: {total_classes}个")
    print(f"总代码行数: {total_lines}行")
    print(f"总方法数: {total_methods}个")
    print(f"建议拆分为: {total_new_classes}个新类")
    
    if total_classes > 0 and total_new_classes > 0:
        print(f"\n平均每个大类将被拆分为: {total_new_classes / total_classes:.1f}个新类")
        print(f"预计重构后平均类大小: ~{total_lines / total_new_classes:.0f}行")
    else:
        print("\n⚠️  未找到符合条件的大类(>100行)或分析失败")
    
    print("\n" + "=" * 80)
    print("下一步:")
    print("1. 查看生成的模板文件: src/infrastructure/api/refactored/")
    print("2. 查看详细分析报告: analysis_reports/api_refactoring_analysis.json")
    print("3. 按照analysis_reports/api_module_refactoring_plan.md执行重构")
    print("=" * 80)


if __name__ == '__main__':
    main()

