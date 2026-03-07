"""
API测试用例生成器 - 重构版本

采用组合模式和门面模式，将原694行的单体类拆分为多个专用组件。

重构成果：
- 原类: 694行，职责过载
- 新架构: 门面类~150行 + 6个专用组件
- 设计模式: 组合模式 + 门面模式
- 向后兼容: 100%保持原有API接口
"""

from pathlib import Path
from typing import Dict, Any, Optional

# 导入重构后的组件
from .test_generation.components import (
    TestTemplateManager,
    TestExporter,
    TestStatisticsCollector
)
from .test_generation.builders import (
    DataServiceTestBuilder,
    FeatureServiceTestBuilder,
    TradingServiceTestBuilder,
    MonitoringServiceTestBuilder
)

# 导入数据类（保持兼容）
from .test_generation.builders.base_builder import TestSuite, TestScenario, TestCase


class APITestCaseGenerator:
    """
    API测试用例生成器 - 门面类
    
    采用组合模式重构，将原694行大类拆分为：
    - TestTemplateManager: 模板管理 (~100行)
    - 4个ServiceTestBuilder: 各服务测试构建 (~400行)
    - TestExporter: 测试导出 (~150行)
    - TestStatisticsCollector: 统计分析 (~100行)
    
    职责：
    - 作为统一访问入口（门面）
    - 协调各组件工作
    - 保持向后兼容的API接口
    """
    
    def __init__(self, template_dir: Optional[Path] = None):
        """
        初始化测试用例生成器
        
        Args:
            template_dir: 自定义模板目录路径
        """
        # 初始化核心组件（组合模式）
        self._template_manager = TestTemplateManager(template_dir)
        self._exporter = TestExporter()
        self._statistics = TestStatisticsCollector()
        
        # 初始化测试构建器
        self._builders = {
            'data_service': DataServiceTestBuilder(self._template_manager),
            'feature_service': FeatureServiceTestBuilder(self._template_manager),
            'trading_service': TradingServiceTestBuilder(self._template_manager),
            'monitoring_service': MonitoringServiceTestBuilder(self._template_manager),
        }
        
        # 测试套件缓存（保持原有接口）
        self.test_suites: Dict[str, TestSuite] = {}
        
        # 模板访问接口（保持原有接口）
        self.templates: Dict[str, Dict[str, Any]] = self._template_manager.get_all_templates()
    
    # ========== 向后兼容接口 ==========
    
    def create_data_service_test_suite(self) -> TestSuite:
        """
        创建数据服务测试套件（向后兼容接口）
        
        Returns:
            TestSuite: 数据服务测试套件
        """
        suite = self._builders['data_service'].build_test_suite()
        self.test_suites[suite.id] = suite
        return suite
    
    def create_feature_service_test_suite(self) -> TestSuite:
        """
        创建特征工程服务测试套件（向后兼容接口）
        
        Returns:
            TestSuite: 特征服务测试套件
        """
        suite = self._builders['feature_service'].build_test_suite()
        self.test_suites[suite.id] = suite
        return suite
    
    def create_trading_service_test_suite(self) -> TestSuite:
        """
        创建交易服务测试套件（向后兼容接口）
        
        Returns:
            TestSuite: 交易服务测试套件
        """
        suite = self._builders['trading_service'].build_test_suite()
        self.test_suites[suite.id] = suite
        return suite
    
    def create_monitoring_service_test_suite(self) -> TestSuite:
        """
        创建监控服务测试套件（向后兼容接口）
        
        Returns:
            TestSuite: 监控服务测试套件
        """
        suite = self._builders['monitoring_service'].build_test_suite()
        self.test_suites[suite.id] = suite
        return suite
    
    def generate_complete_test_suite(self) -> Dict[str, TestSuite]:
        """
        生成完整的测试套件（向后兼容接口）
        
        Returns:
            Dict[str, TestSuite]: 所有服务的测试套件字典
        """
        test_suites = {}
        
        # 使用各个构建器生成测试套件
        for service_type, builder in self._builders.items():
            suite = builder.build_test_suite()
            test_suites[suite.id] = suite
        
        self.test_suites = test_suites
        return test_suites
    
    def export_test_cases(
        self,
        format_type: str = "json",
        output_dir: str = "docs/api/tests"
    ) -> str:
        """
        导出测试用例（向后兼容接口）
        
        Args:
            format_type: 导出格式 (json, yaml, markdown, html)
            output_dir: 输出目录
        
        Returns:
            str: 输出文件路径
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 生成完整的测试套件
        test_suites = self.generate_complete_test_suite()
        
        # 构建输出文件名
        format_ext_map = {
            'json': 'json',
            'yaml': 'yaml',
            'markdown': 'md',
            'html': 'html',
            'python': 'py'
        }
        ext = format_ext_map.get(format_type, 'json')
        output_file = output_path / f"rqa_api_test_cases.{ext}"
        
        # 导出所有测试套件
        combined_suite = self._combine_test_suites(test_suites)
        
        self._exporter.export(
            test_suite=combined_suite,
            output_path=output_file,
            format=format_type,
            include_metadata=True,
            include_statistics=True,
            pretty_print=True
        )
        
        print(f"测试用例已导出到: {output_file}")
        return str(output_file)
    
    def get_test_statistics(self) -> Dict[str, Any]:
        """
        获取测试统计信息（向后兼容接口）
        
        Returns:
            Dict[str, Any]: 测试统计信息
        """
        if not self.test_suites:
            self.generate_complete_test_suite()
        
        total_suites = len(self.test_suites)
        total_scenarios = sum(len(suite.scenarios) for suite in self.test_suites.values())
        total_test_cases = sum(
            len(scenario.test_cases)
            for suite in self.test_suites.values()
            for scenario in suite.scenarios
        )
        
        # 按优先级统计
        priority_stats = {"high": 0, "medium": 0, "low": 0, "critical": 0}
        category_stats = {}
        
        for suite in self.test_suites.values():
            for scenario in suite.scenarios:
                for test_case in scenario.test_cases:
                    priority = getattr(test_case, 'priority', 'medium')
                    priority_stats[priority] = priority_stats.get(priority, 0) + 1
                    
                    category = getattr(test_case, 'category', 'functional')
                    category_stats[category] = category_stats.get(category, 0) + 1
        
        return {
            "total_suites": total_suites,
            "total_scenarios": total_scenarios,
            "total_test_cases": total_test_cases,
            "priority_distribution": priority_stats,
            "category_distribution": category_stats
        }
    
    # ========== 新增功能接口 ==========
    
    def get_builder(self, service_type: str):
        """
        获取指定服务的测试构建器（新增接口）
        
        Args:
            service_type: 服务类型
        
        Returns:
            测试构建器实例
        """
        return self._builders.get(service_type)
    
    def export_suite(
        self,
        suite_id: str,
        output_path: Path,
        format: str = 'json'
    ) -> bool:
        """
        导出单个测试套件（新增接口）
        
        Args:
            suite_id: 测试套件ID
            output_path: 输出路径
            format: 导出格式
        
        Returns:
            bool: 是否导出成功
        """
        suite = self.test_suites.get(suite_id)
        if not suite:
            print(f"测试套件不存在: {suite_id}")
            return False
        
        return self._exporter.export(
            test_suite=suite,
            output_path=output_path,
            format=format,
            include_metadata=True,
            include_statistics=True
        )
    
    def get_suite_statistics(self, suite_id: str) -> Optional[Dict[str, Any]]:
        """
        获取单个测试套件的统计信息（新增接口）
        
        Args:
            suite_id: 测试套件ID
        
        Returns:
            统计信息字典或None
        """
        suite = self.test_suites.get(suite_id)
        if not suite:
            return None
        
        stats = self._statistics.collect_statistics(suite)
        return stats.to_dict() if hasattr(stats, 'to_dict') else stats
    
    def generate_summary_report(self, suite_id: str) -> Optional[str]:
        """
        生成测试套件摘要报告（新增接口）
        
        Args:
            suite_id: 测试套件ID
        
        Returns:
            摘要报告文本或None
        """
        suite = self.test_suites.get(suite_id)
        if not suite:
            return None
        
        return self._statistics.generate_summary_report(suite)
    
    # ========== 私有辅助方法 ==========
    
    def _combine_test_suites(self, test_suites: Dict[str, TestSuite]) -> TestSuite:
        """
        合并多个测试套件为一个综合套件
        
        Args:
            test_suites: 测试套件字典
        
        Returns:
            TestSuite: 合并后的测试套件
        """
        from .test_generation.builders.base_builder import TestSuite
        
        combined_suite = TestSuite(
            id="combined_test_suite",
            name="RQA2025 API完整测试套件",
            description="包含所有服务的综合测试套件"
        )
        
        # 合并所有场景
        for suite in test_suites.values():
            combined_suite.scenarios.extend(suite.scenarios)
        
        return combined_suite


class TestCaseGenerator(APITestCaseGenerator):
    """向后兼容别名，保持旧版导入路径"""
    pass


# ========== 向后兼容性支持 ==========

# 保持原有的main执行逻辑
if __name__ == "__main__":
    # 生成RQA2025 API测试用例文档
    print("初始化API测试用例生成器...")
    
    generator = APITestCaseGenerator()
    
    # 生成完整的测试套件
    print("生成API测试用例...")
    test_suites = generator.generate_complete_test_suite()
    
    print(f"生成了 {len(test_suites)} 个测试套件")
    
    # 导出测试用例
    json_file = generator.export_test_cases("json")
    
    # 获取统计信息
    stats = generator.get_test_statistics()
    
    print("\n📊 测试用例统计:")
    print(f"   📁 测试套件: {stats['total_suites']} 个")
    print(f"   📋 测试场景: {stats['total_scenarios']} 个")
    print(f"   🧪 测试用例: {stats['total_test_cases']} 个")
    print(f"   🎯 优先级分布: {stats['priority_distribution']}")
    print(f"   📊 类别分布: {stats['category_distribution']}")
    
    print("\n📄 输出文件:")
    print(f"   JSON: {json_file}")
    
    print("\n🎉 API测试用例文档生成完成！")

