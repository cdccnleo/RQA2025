"""
API测试套件协调器

职责: 协调各个测试生成器，提供统一的外观接口(Facade Pattern)
这是重构后的主要入口类，提供与原APITestCaseGenerator兼容的接口
"""

from typing import Dict
from pathlib import Path
from .models import TestSuite
from .template_manager import TestTemplateManager
from .generators import (
    DataServiceTestGenerator,
    FeatureServiceTestGenerator,
    TradingServiceTestGenerator,
    MonitoringServiceTestGenerator
)
from .exporter import TestSuiteExporter
from .statistics import TestStatisticsCollector


class APITestSuiteCoordinator:
    """
    API测试套件协调器
    
    这是重构后的主要接口，提供与原APITestCaseGenerator相同的功能
    但内部使用多个专门的类来实现，符合单一职责原则
    """
    
    def __init__(self):
        """初始化协调器和所有组件"""
        # 核心组件
        self.template_manager = TestTemplateManager()
        self.exporter = TestSuiteExporter()
        self.statistics = TestStatisticsCollector()
        
        # 服务测试生成器
        self.data_generator = DataServiceTestGenerator(self.template_manager)
        self.feature_generator = FeatureServiceTestGenerator(self.template_manager)
        self.trading_generator = TradingServiceTestGenerator(self.template_manager)
        self.monitoring_generator = MonitoringServiceTestGenerator(self.template_manager)
        
        # 缓存生成的测试套件
        self.test_suites: Dict[str, TestSuite] = {}
    
    def generate_complete_test_suite(self) -> Dict[str, TestSuite]:
        """
        生成完整的测试套件（所有服务）
        
        Returns:
            包含所有服务测试套件的字典
        """
        print("🔄 生成完整API测试套件...")
        
        self.test_suites = {
            'data_service': self.data_generator.create_test_suite(),
            'feature_service': self.feature_generator.create_test_suite(),
            'trading_service': self.trading_generator.create_test_suite(),
            'monitoring_service': self.monitoring_generator.create_test_suite()
        }
        
        print(f"✅ 生成完成，共{len(self.test_suites)}个测试套件")
        
        return self.test_suites
    
    def create_data_service_test_suite(self) -> TestSuite:
        """创建数据服务测试套件（向后兼容方法）"""
        return self.data_generator.create_test_suite()
    
    def create_feature_service_test_suite(self) -> TestSuite:
        """创建特征服务测试套件（向后兼容方法）"""
        return self.feature_generator.create_test_suite()
    
    def create_trading_service_test_suite(self) -> TestSuite:
        """创建交易服务测试套件（向后兼容方法）"""
        return self.trading_generator.create_test_suite()
    
    def create_monitoring_service_test_suite(self) -> TestSuite:
        """创建监控服务测试套件（向后兼容方法）"""
        return self.monitoring_generator.create_test_suite()
    
    def export_test_cases(
        self,
        format_type: str = "json",
        output_dir: str = "docs/api/tests"
    ):
        """
        导出测试用例（向后兼容方法）
        
        Args:
            format_type: 导出格式
            output_dir: 输出目录
        """
        # 如果还没有生成测试套件，先生成
        if not self.test_suites:
            self.generate_complete_test_suite()
        
        # 导出
        self.exporter.export(self.test_suites, format_type, output_dir)
    
    def get_test_statistics(self) -> Dict:
        """
        获取测试统计信息（向后兼容方法）
        
        Returns:
            统计信息字典
        """
        # 如果还没有生成测试套件，先生成
        if not self.test_suites:
            self.generate_complete_test_suite()
        
        return self.statistics.get_statistics_summary(self.test_suites)
    
    def get_detailed_statistics(self) -> Dict:
        """
        获取详细统计信息
        
        Returns:
            详细统计信息字典
        """
        if not self.test_suites:
            self.generate_complete_test_suite()
        
        return self.statistics.get_detailed_statistics(self.test_suites)


# 向后兼容: 提供与原APITestCaseGenerator相同的接口
class APITestCaseGenerator(APITestSuiteCoordinator):
    """
    API测试用例生成器 - 向后兼容类
    
    这个类继承自APITestSuiteCoordinator，保持与原有代码的兼容性
    新代码应该直接使用APITestSuiteCoordinator或具体的生成器类
    """
    
    def __init__(self):
        """初始化（向后兼容）"""
        super().__init__()
        print("ℹ️  APITestCaseGenerator已重构为模块化架构")
        print("   建议使用APITestSuiteCoordinator或具体的生成器类")

