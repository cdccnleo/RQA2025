#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版风控合规层验证脚本
验证风控合规层的基本功能
"""

import sys
import os
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_risk_controller_import():
    """测试风控控制器导入"""
    try:
        from src.trading.risk.risk_controller import RiskController
        logger.info("✅ 风控控制器导入成功")
        return True
    except Exception as e:
        logger.error(f"❌ 风控控制器导入失败: {e}")
        return False

def test_china_risk_controller_import():
    """测试中国风控控制器导入"""
    try:
        from src.trading.risk.china.risk_controller import ChinaRiskController
        logger.info("✅ 中国风控控制器导入成功")
        return True
    except Exception as e:
        logger.error(f"❌ 中国风控控制器导入失败: {e}")
        return False

def test_risk_config_import():
    """测试风控配置导入"""
    try:
        from src.trading.risk.risk_controller import RiskConfig
        logger.info("✅ 风控配置导入成功")
        return True
    except Exception as e:
        logger.error(f"❌ 风控配置导入失败: {e}")
        return False

def test_business_process_orchestrator_import():
    """测试业务流程编排器导入"""
    try:
        from src.core.business_process_orchestrator import BusinessProcessOrchestrator
        logger.info("✅ 业务流程编排器导入成功")
        return True
    except Exception as e:
        logger.error(f"❌ 业务流程编排器导入失败: {e}")
        return False

def test_event_bus_import():
    """测试事件总线导入"""
    try:
        from src.core.event_bus import EventBus
        logger.info("✅ 事件总线导入成功")
        return True
    except Exception as e:
        logger.error(f"❌ 事件总线导入失败: {e}")
        return False

def test_container_import():
    """测试依赖注入容器导入"""
    try:
        from src.core.container import DependencyContainer
        logger.info("✅ 依赖注入容器导入成功")
        return True
    except Exception as e:
        logger.error(f"❌ 依赖注入容器导入失败: {e}")
        return False

def test_architecture_layers_import():
    """测试架构层导入"""
    try:
        from src.core.architecture_layers import (
            CoreServicesLayer,
            InfrastructureLayer,
            DataManagementLayer,
            FeatureProcessingLayer,
            ModelInferenceLayer,
            StrategyDecisionLayer,
            RiskComplianceLayer,
            TradingExecutionLayer,
            MonitoringFeedbackLayer
        )
        logger.info("✅ 架构层导入成功")
        return True
    except Exception as e:
        logger.error(f"❌ 架构层导入失败: {e}")
        return False

def test_feature_engineer_import():
    """测试特征工程器导入"""
    try:
        from src.features.feature_engineer import FeatureEngineer
        logger.info("✅ 特征工程器导入成功")
        return True
    except Exception as e:
        logger.error(f"❌ 特征工程器导入失败: {e}")
        return False

def test_risk_controller_instantiation():
    """测试风控控制器实例化"""
    try:
        from src.trading.risk.risk_controller import RiskController
        controller = RiskController()
        logger.info("✅ 风控控制器实例化成功")
        return True
    except Exception as e:
        logger.error(f"❌ 风控控制器实例化失败: {e}")
        return False

def test_china_risk_controller_instantiation():
    """测试中国风控控制器实例化"""
    try:
        from src.trading.risk.china.risk_controller import ChinaRiskController
        controller = ChinaRiskController()
        logger.info("✅ 中国风控控制器实例化成功")
        return True
    except Exception as e:
        logger.error(f"❌ 中国风控控制器实例化失败: {e}")
        return False

def test_business_process_orchestrator_instantiation():
    """测试业务流程编排器实例化"""
    try:
        from src.core.business_process_orchestrator import BusinessProcessOrchestrator
        orchestrator = BusinessProcessOrchestrator()
        logger.info("✅ 业务流程编排器实例化成功")
        return True
    except Exception as e:
        logger.error(f"❌ 业务流程编排器实例化失败: {e}")
        return False

def test_event_bus_instantiation():
    """测试事件总线实例化"""
    try:
        from src.core.event_bus import EventBus
        event_bus = EventBus()
        logger.info("✅ 事件总线实例化成功")
        return True
    except Exception as e:
        logger.error(f"❌ 事件总线实例化失败: {e}")
        return False

def test_container_instantiation():
    """测试依赖注入容器实例化"""
    try:
        from src.core.container import DependencyContainer
        container = DependencyContainer()
        logger.info("✅ 依赖注入容器实例化成功")
        return True
    except Exception as e:
        logger.error(f"❌ 依赖注入容器实例化失败: {e}")
        return False

def test_architecture_layers_instantiation():
    """测试架构层实例化"""
    try:
        from src.core.architecture_layers import CoreServicesLayer
        core_services = CoreServicesLayer()
        logger.info("✅ 核心服务层实例化成功")
        return True
    except Exception as e:
        logger.error(f"❌ 核心服务层实例化失败: {e}")
        return False

def test_feature_engineer_instantiation():
    """测试特征工程器实例化"""
    try:
        from src.features.feature_engineer import FeatureEngineer
        feature_engine = FeatureEngineer()
        logger.info("✅ 特征工程器实例化成功")
        return True
    except Exception as e:
        logger.error(f"❌ 特征工程器实例化失败: {e}")
        return False

def run_all_tests():
    """运行所有测试"""
    logger.info("🚀 开始运行风控合规层验证测试...")
    
    test_results = {}
    
    # 导入测试
    tests = [
        ("风控控制器导入", test_risk_controller_import),
        ("中国风控控制器导入", test_china_risk_controller_import),
        ("风控配置导入", test_risk_config_import),
        ("业务流程编排器导入", test_business_process_orchestrator_import),
        ("事件总线导入", test_event_bus_import),
        ("依赖注入容器导入", test_container_import),
        ("架构层导入", test_architecture_layers_import),
        ("特征工程器导入", test_feature_engineer_import)
    ]
    
    for test_name, test_func in tests:
        try:
            logger.info(f"运行测试: {test_name}")
            result = test_func()
            test_results[test_name] = result
            
            if result:
                logger.info(f"✅ {test_name} 测试通过")
            else:
                logger.error(f"❌ {test_name} 测试失败")
                
        except Exception as e:
            logger.error(f"❌ {test_name} 测试异常: {e}")
            test_results[test_name] = False
    
    # 实例化测试
    instantiation_tests = [
        ("风控控制器实例化", test_risk_controller_instantiation),
        ("中国风控控制器实例化", test_china_risk_controller_instantiation),
        ("业务流程编排器实例化", test_business_process_orchestrator_instantiation),
        ("事件总线实例化", test_event_bus_instantiation),
        ("依赖注入容器实例化", test_container_instantiation),
        ("架构层实例化", test_architecture_layers_instantiation),
        ("特征工程器实例化", test_feature_engineer_instantiation)
    ]
    
    for test_name, test_func in instantiation_tests:
        try:
            logger.info(f"运行测试: {test_name}")
            result = test_func()
            test_results[test_name] = result
            
            if result:
                logger.info(f"✅ {test_name} 测试通过")
            else:
                logger.error(f"❌ {test_name} 测试失败")
                
        except Exception as e:
            logger.error(f"❌ {test_name} 测试异常: {e}")
            test_results[test_name] = False
    
    # 生成测试报告
    generate_test_report(test_results)
    
    return test_results

def generate_test_report(test_results):
    """生成测试报告"""
    logger.info("生成测试报告...")
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    failed_tests = total_tests - passed_tests
    
    logger.info("=" * 60)
    logger.info("风控合规层验证测试报告")
    logger.info("=" * 60)
    logger.info(f"总测试数: {total_tests}")
    logger.info(f"通过测试数: {passed_tests}")
    logger.info(f"失败测试数: {failed_tests}")
    logger.info(f"通过率: {passed_tests/total_tests*100:.1f}%")
    logger.info("=" * 60)
    
    # 详细结果
    for test_name, result in test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"{test_name}: {status}")
    
    logger.info("=" * 60)
    
    if failed_tests == 0:
        logger.info("🎉 所有测试均通过！风控合规层验证成功！")
    else:
        logger.warning(f"⚠️  有 {failed_tests} 项测试失败，请检查相关功能")

def main():
    """主函数"""
    print("🚀 启动风控合规层验证测试...")
    
    try:
        # 运行所有测试
        results = run_all_tests()
        
        # 检查整体结果
        all_passed = all(results.values())
        
        if all_passed:
            print("\n🎉 风控合规层验证测试全部通过！")
            print("✅ 所有核心组件导入和实例化成功")
            print("✅ 风控合规层基础架构完整")
            return 0
        else:
            print("\n⚠️  风控合规层验证测试部分失败")
            print("请检查失败的测试项并修复相关问题")
            return 1
            
    except Exception as e:
        print(f"\n❌ 测试过程中发生异常: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
