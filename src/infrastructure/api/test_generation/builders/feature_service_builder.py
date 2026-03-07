"""
特征服务测试构建器

负责构建特征工程服务相关的测试套件
"""

from typing import List
from .base_builder import BaseTestBuilder
from ..models import TestSuite, TestScenario, TestCase


class FeatureServiceTestBuilder(BaseTestBuilder):
    """
    特征服务测试构建器
    
    职责：
    - 构建特征工程API的测试场景
    - 生成特征提取、特征计算、特征存储等测试用例
    """
    
    def build_test_suite(self) -> TestSuite:
        """构建特征服务测试套件"""
        suite = TestSuite(
            id="feature_service_tests",
            name="特征服务API测试",
            description="RQA2025特征工程服务的完整API测试套件"
        )
        
        # 添加各类测试场景
        suite.scenarios.append(self._build_feature_extraction_scenario())
        suite.scenarios.append(self._build_feature_calculation_scenario())
        suite.scenarios.append(self._build_feature_storage_scenario())
        
        return suite
    
    def _build_feature_extraction_scenario(self) -> TestScenario:
        """构建特征提取测试场景"""
        scenario = self._create_test_scenario(
            scenario_id="feature_extraction",
            name="特征提取",
            description="测试从原始数据中提取特征的功能",
            endpoint="/api/v1/features/extract",
            method="POST",
            variables={
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "feature_types": ["technical", "statistical", "pattern"]
            }
        )
        
        # 正常场景
        scenario.test_cases.append(self._create_test_case(
            case_id="feature_extraction_normal",
            title="正常提取技术指标特征",
            description="从K线数据中提取技术指标特征",
            priority="high",
            category="functional",
            expected_results=[
                "返回状态码200",
                "返回提取的特征列表",
                "包含MACD、RSI、BOLL等技术指标"
            ],
            tags=["feature", "extraction", "technical"]
        ))
        
        # 批量提取场景
        scenario.test_cases.append(self._create_test_case(
            case_id="feature_extraction_batch",
            title="批量提取多个交易对特征",
            description="测试批量提取多个交易对的特征",
            priority="medium",
            category="performance",
            expected_results=[
                "返回状态码200",
                "所有交易对的特征都成功提取",
                "批量处理时间在合理范围内"
            ],
            tags=["feature", "extraction", "batch"]
        ))
        
        return scenario
    
    def _build_feature_calculation_scenario(self) -> TestScenario:
        """构建特征计算测试场景"""
        scenario = self._create_test_scenario(
            scenario_id="feature_calculation",
            name="特征计算",
            description="测试特征值的实时计算功能",
            endpoint="/api/v1/features/calculate",
            method="POST",
            variables={
                "symbol": "BTC/USDT",
                "indicators": ["macd", "rsi", "bollinger"]
            }
        )
        
        scenario.test_cases.append(self._create_test_case(
            case_id="feature_calculation_macd",
            title="MACD指标计算",
            description="测试MACD技术指标的计算准确性",
            priority="high",
            category="functional",
            expected_results=[
                "返回状态码200",
                "MACD值计算正确",
                "包含DIF、DEA、MACD三个值"
            ],
            tags=["feature", "calculation", "macd"]
        ))
        
        scenario.test_cases.append(self._create_test_case(
            case_id="feature_calculation_rsi",
            title="RSI指标计算",
            description="测试RSI相对强弱指标的计算准确性",
            priority="high",
            category="functional",
            expected_results=[
                "返回状态码200",
                "RSI值在0-100之间",
                "计算结果与预期一致"
            ],
            tags=["feature", "calculation", "rsi"]
        ))
        
        return scenario
    
    def _build_feature_storage_scenario(self) -> TestScenario:
        """构建特征存储测试场景"""
        scenario = self._create_test_scenario(
            scenario_id="feature_storage",
            name="特征存储",
            description="测试特征的持久化存储功能",
            endpoint="/api/v1/features/store",
            method="POST",
            variables={
                "feature_set_id": "test_feature_set_001",
                "features": {"macd": 0.5, "rsi": 65.2}
            }
        )
        
        scenario.test_cases.append(self._create_test_case(
            case_id="feature_storage_save",
            title="保存特征到数据库",
            description="测试特征数据的持久化存储",
            priority="high",
            category="functional",
            expected_results=[
                "返回状态码201",
                "特征成功保存到数据库",
                "返回特征集ID"
            ],
            tags=["feature", "storage", "save"]
        ))
        
        scenario.test_cases.append(self._create_test_case(
            case_id="feature_storage_retrieve",
            title="从数据库检索特征",
            description="测试从数据库检索已保存的特征",
            priority="high",
            category="functional",
            expected_results=[
                "返回状态码200",
                "成功检索到特征数据",
                "数据完整性校验通过"
            ],
            tags=["feature", "storage", "retrieve"]
        ))
        
        return scenario

