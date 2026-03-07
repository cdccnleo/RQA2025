#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 功能验证测试

验证核心功能模块是否正常工作
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def test_event_system():
    """测试事件系统功能"""
    print("🔍 测试事件系统功能...")
    try:
        from src.core import EventBus
        event_bus = EventBus()
        print("✅ 事件系统初始化成功")
        return {'status': 'PASSED', 'message': 'Event system functional'}
    except Exception as e:
        print(f"❌ 事件系统测试失败: {e}")
        return {'status': 'FAILED', 'error': str(e)}

def test_data_processing():
    """测试数据处理功能"""
    print("🔍 测试数据处理功能...")
    try:
        from src.data import DataValidator
        validator = DataValidator()
        result = validator.validate_data_quality({"test": "data"})
        print(f"✅ 数据验证器工作正常: {result}")
        return {'status': 'PASSED', 'message': 'Data processing functional'}
    except Exception as e:
        print(f"❌ 数据处理测试失败: {e}")
        return {'status': 'FAILED', 'error': str(e)}

def test_model_inference():
    """测试模型推理功能"""
    print("🔍 测试模型推理功能...")
    try:
        from src.ml import ModelEnsemble
        model = ModelEnsemble()
        result = model.predict({"test": "data"})
        print(f"✅ 模型推理工作正常: {result}")
        return {'status': 'PASSED', 'message': 'Model inference functional'}
    except Exception as e:
        print(f"❌ 模型推理测试失败: {e}")
        return {'status': 'FAILED', 'error': str(e)}

def test_trading_engine():
    """测试交易引擎功能"""
    print("🔍 测试交易引擎功能...")
    try:
        from src.trading import TradingEngine
        engine = TradingEngine()
        print(f"✅ 交易引擎初始化成功: {engine.name}")
        return {'status': 'PASSED', 'message': 'Trading engine functional'}
    except Exception as e:
        print(f"❌ 交易引擎测试失败: {e}")
        return {'status': 'FAILED', 'error': str(e)}

def test_risk_management():
    """测试风险管理功能"""
    print("🔍 测试风险管理功能...")
    try:
        from src.risk import RiskManager
        manager = RiskManager()
        print(f"✅ 风险管理器初始化成功: {manager.name}")
        return {'status': 'PASSED', 'message': 'Risk management functional'}
    except Exception as e:
        print(f"❌ 风险管理测试失败: {e}")
        return {'status': 'FAILED', 'error': str(e)}

def test_api_gateway():
    """测试API网关功能"""
    print("🔍 测试API网关功能...")
    try:
        from src.gateway import APIGateway
        gateway = APIGateway()
        result = gateway.route_request({"test": "request"})
        print(f"✅ API网关工作正常: {result}")
        return {'status': 'PASSED', 'message': 'API gateway functional'}
    except Exception as e:
        print(f"❌ API网关测试失败: {e}")
        return {'status': 'FAILED', 'error': str(e)}

def main():
    """主函数"""
    print("🧪 RQA2025 功能验证测试")
    print("=" * 50)

    test_functions = [
        test_event_system,
        test_data_processing,
        test_model_inference,
        test_trading_engine,
        test_risk_management,
        test_api_gateway
    ]

    test_results = []
    passed_count = 0

    for test_func in test_functions:
        try:
            result = test_func()
            test_results.append(result)
            if result.get('status') == 'PASSED':
                passed_count += 1
            print(f"{'✅' if result.get('status') == 'PASSED' else '❌'} {test_func.__name__}")
        except Exception as e:
            test_results.append({
                'status': 'ERROR',
                'error': str(e),
                'test_name': test_func.__name__
            })
            print(f"💥 {test_func.__name__}: {e}")

    print(f"\n{'=' * 50}")
    print("📊 功能验证测试结果")
    print(f"{'=' * 50}")

    print(f"总测试数: {len(test_functions)}")
    print(f"通过测试: {passed_count}")
    print(f"失败测试: {len(test_functions) - passed_count}")
    print(f"成功率: {passed_count / len(test_functions) * 100:.1f}%")

    if passed_count == len(test_functions):
        print("\n🎉 所有功能验证测试通过！")
        return 0
    else:
        print("\n⚠️ 部分功能验证测试失败，需要进一步修复")
        return 1

if __name__ == "__main__":
    exit(main())
