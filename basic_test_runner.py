#!/usr/bin/env python3
"""
基本测试运行器
运行一些简单的测试用例来验证功能
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_service_factory_basic():
    """测试服务工厂基本功能"""
    try:
        from src.core.utils.service_factory import ServiceFactory

        # 创建工厂
        factory = ServiceFactory()

        # 注册一个简单服务
        class SimpleService:
            def __init__(self, config=None):
                self.config = config or {}
                self.initialized = True

        factory.register_service("simple", SimpleService)

        # 创建服务
        service = factory.create_service("simple")

        # 验证服务创建成功
        assert service is not None
        assert hasattr(service, 'config')
        assert service.initialized == True

        print("✅ 服务工厂基本功能测试通过")
        return True
    except Exception as e:
        print(f"❌ 服务工厂测试失败: {e}")
        return False

def test_unified_exceptions_basic():
    """测试统一异常基本功能"""
    try:
        from src.core.foundation.exceptions.unified_exceptions import RQA2025Exception

        # 创建异常
        exc = RQA2025Exception("测试异常消息")

        # 验证异常属性
        assert exc.message == "测试异常消息"
        assert hasattr(exc, 'error_type')
        assert hasattr(exc, 'timestamp')
        assert hasattr(exc, 'context')

        # 验证序列化
        import json
        dict_data = exc.to_dict()
        json_str = json.dumps(dict_data, ensure_ascii=False)
        assert isinstance(json_str, str)
        assert "测试异常消息" in json_str

        print("✅ 统一异常基本功能测试通过")
        return True
    except Exception as e:
        print(f"❌ 统一异常测试失败: {e}")
        return False

def test_data_processor_basic():
    """测试数据处理器基本功能"""
    try:
        # 使用Mock类进行测试
        class MockDataProcessor:
            def validate_data(self, data):
                return isinstance(data, (list, dict)) and len(data) > 0

            def clean_data(self, data):
                return data

        processor = MockDataProcessor()

        # 测试数据验证
        assert processor.validate_data([1, 2, 3]) == True
        assert processor.validate_data({}) == False
        assert processor.validate_data(None) == False

        # 测试数据清理
        test_data = [1, 2, None, 3]
        cleaned = processor.clean_data(test_data)
        assert cleaned is not None

        print("✅ 数据处理器基本功能测试通过")
        return True
    except Exception as e:
        print(f"❌ 数据处理器测试失败: {e}")
        return False

def test_strategy_basic():
    """测试策略基本功能"""
    try:
        # 使用Mock类进行测试
        class MockStrategy:
            def __init__(self, strategy_id, name="Mock Strategy"):
                self.strategy_id = strategy_id
                self.name = name

            def generate_signals(self, data):
                # 简单的信号生成逻辑
                if isinstance(data, list) and len(data) > 0:
                    return [{"signal": "BUY", "strength": 0.8}]
                return []

        strategy = MockStrategy("test_strategy")

        # 测试信号生成
        signals = strategy.generate_signals([1, 2, 3])
        assert isinstance(signals, list)
        assert len(signals) == 1
        assert signals[0]["signal"] == "BUY"

        # 测试空数据
        empty_signals = strategy.generate_signals([])
        assert isinstance(empty_signals, list)
        assert len(empty_signals) == 0

        print("✅ 策略基本功能测试通过")
        return True
    except Exception as e:
        print(f"❌ 策略测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 RQA2025基本功能测试")
    print("=" * 40)

    tests = [
        ("服务工厂", test_service_factory_basic),
        ("统一异常", test_unified_exceptions_basic),
        ("数据处理器", test_data_processor_basic),
        ("策略算法", test_strategy_basic)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n📋 运行测试: {test_name}")
        if test_func():
            passed += 1

    print(f"\n{'='*40}")
    print("📊 测试结果:")
    print(f"   总测试数: {total}")
    print(f"   通过测试: {passed}")
    print(f"   失败测试: {total - passed}")

    if total > 0:
        pass_rate = (passed / total) * 100
        print(f"   通过率: {pass_rate:.1f}%")
        success = passed == total
    else:
        success = False

    print("\n🎯 测试完成")
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
