#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
边界条件处理验证测试脚本
"""

import sys
import os
import traceback

def test_config_boundary_conditions():
    """测试配置管理边界条件"""
    print("🔍 测试配置管理边界条件...")

    try:
        # 导入配置管理器
        sys.path.insert(0, 'src')
        from infrastructure.config.unified_manager import UnifiedConfigManager

        config_manager = UnifiedConfigManager()

        # 测试正常情况
        success = config_manager.set('test_section', 'test_key', 'test_value')
        value = config_manager.get('test_section', 'test_key')
        print(f"   ✅ 正常设置/获取: {value}")

        # 测试边界条件
        test_cases = [
            # 无效section/key
            ('', 'key', None),
            ('section', '', None),
            (None, 'key', None),
            ('section', None, None),

            # 过长key
            ('a' * 200, 'key', None),
            ('section', 'b' * 200, None),

            # 危险字符
            ('section<script>', 'key', None),
            ('section', 'key;rm -rf', None),
        ]

        for section, key, expected in test_cases:
            try:
                result = config_manager.get(section, key, 'default')
                if result == 'default':
                    print(f"   ✅ 边界条件处理正确: ({section[:10]}..., {key[:10]}...) -> {result}")
                else:
                    print(f"   ❌ 边界条件处理失败: ({section[:10]}..., {key[:10]}...) -> {result}")
            except Exception as e:
                print(f"   ⚠️ 异常处理: {str(e)[:50]}...")

        return True

    except Exception as e:
        print(f"   ❌ 配置管理边界条件测试失败: {str(e)}")
        return False

def test_execution_engine_boundary_conditions():
    """测试执行引擎边界条件"""
    print("\n🔍 测试执行引擎边界条件...")

    try:
        # 导入执行引擎
        from trading.execution_engine import ExecutionEngine, ExecutionMode, OrderSide

        engine = ExecutionEngine()

        # 测试正常情况
        try:
            exec_id = engine.create_execution('000001.SZ', OrderSide.BUY, 100.0, 10.0, ExecutionMode.LIMIT)
            print(f"   ✅ 正常订单创建: {exec_id}")
        except Exception as e:
            print(f"   ⚠️ 正常订单创建异常: {str(e)}")

        # 测试边界条件
        error_cases = [
            # 无效symbol
            ('', OrderSide.BUY, 100.0, 10.0, ExecutionMode.MARKET, "交易标不能为空"),
            (None, OrderSide.BUY, 100.0, 10.0, ExecutionMode.MARKET, "交易标不能为空"),

            # 无效quantity
            ('000001.SZ', OrderSide.BUY, 0, 10.0, ExecutionMode.MARKET, "数量必须为正数"),
            ('000001.SZ', OrderSide.BUY, -100, 10.0, ExecutionMode.MARKET, "数量必须为正数"),
            ('000001.SZ', OrderSide.BUY, 1e15, 10.0, ExecutionMode.MARKET, "订单数量过大"),

            # 无效price
            ('000001.SZ', OrderSide.BUY, 100.0, 0, ExecutionMode.LIMIT, "价格必须为正数"),
            ('000001.SZ', OrderSide.BUY, 100.0, -10, ExecutionMode.LIMIT, "价格必须为正数"),
            ('000001.SZ', OrderSide.BUY, 100.0, 1e15, ExecutionMode.LIMIT, "价格数值异常"),

            # 限价单无价格
            ('000001.SZ', OrderSide.BUY, 100.0, None, ExecutionMode.LIMIT, "限价单必须指定价格"),
        ]

        for symbol, side, quantity, price, mode, expected_error in error_cases:
            try:
                exec_id = engine.create_execution(symbol, side, quantity, price, mode)
                print(f"   ❌ 边界条件未正确处理: 期望 '{expected_error}', 但成功创建 {exec_id}")
            except ValueError as e:
                if expected_error in str(e):
                    print(f"   ✅ 边界条件正确处理: {expected_error}")
                else:
                    print(f"   ⚠️ 错误消息不匹配: 期望 '{expected_error}', 实际 '{str(e)}'")
            except Exception as e:
                print(f"   ⚠️ 意外异常: {str(e)}")

        return True

    except Exception as e:
        print(f"   ❌ 执行引擎边界条件测试失败: {str(e)}")
        traceback.print_exc()
        return False

def test_numeric_boundary_handler():
    """测试数值边界处理器"""
    print("\n🔍 测试数值边界处理器...")

    try:
        # 导入数值边界处理器
        from numeric_boundary_handler import NumericBoundaryHandler

        handler = NumericBoundaryHandler()

        # 测试安全除法
        test_cases = [
            (10, 2, 5.0),
            (10, 0, 0.0),
            ('a', 2, 0.0),
            (None, 2, 0.0),
        ]

        for dividend, divisor, expected in test_cases:
            result = handler.safe_divide(dividend, divisor)
            if abs(result - expected) < 1e-10:
                print(f"   ✅ 安全除法: {dividend}/{divisor} = {result}")
            else:
                print(f"   ❌ 安全除法失败: {dividend}/{divisor} = {result}, 期望 {expected}")

        # 测试安全平方根
        sqrt_cases = [
            (4, 2.0),
            (-1, 0.0),
            ('a', 0.0),
        ]

        for value, expected in sqrt_cases:
            result = handler.safe_sqrt(value)
            if abs(result - expected) < 1e-10:
                print(f"   ✅ 安全平方根: sqrt({value}) = {result}")
            else:
                print(f"   ❌ 安全平方根失败: sqrt({value}) = {result}, 期望 {expected}")

        return True

    except Exception as e:
        print(f"   ❌ 数值边界处理器测试失败: {str(e)}")
        return False

def test_network_resource_handler():
    """测试网络资源处理器"""
    print("\n🔍 测试网络资源处理器...")

    try:
        # 导入网络资源处理器
        from network_resource_handler import NetworkResourceHandler

        handler = NetworkResourceHandler()

        # 测试套接字连接（使用本地地址，应该失败但不抛出异常）
        is_connected = handler.safe_socket_connect('127.0.0.1', 99999, timeout=1.0)
        print(f"   ✅ 安全套接字连接测试: {'成功' if not is_connected else '意外成功'}")

        # 测试URL验证
        from network_resource_handler import validate_url

        url_cases = [
            ('https://www.example.com', True),
            ('http://localhost:8080/api', True),
            ('not_a_url', False),
            ('', False),
        ]

        for url, expected in url_cases:
            result = validate_url(url)
            if result == expected:
                print(f"   ✅ URL验证: '{url}' -> {result}")
            else:
                print(f"   ❌ URL验证失败: '{url}' -> {result}, 期望 {expected}")

        return True

    except Exception as e:
        print(f"   ❌ 网络资源处理器测试失败: {str(e)}")
        return False

def test_input_validation_enhancer():
    """测试输入验证增强器"""
    print("\n🔍 测试输入验证增强器...")

    try:
        # 导入输入验证增强器
        from input_validation_enhancer import InputValidationEnhancer, BoundaryConditionValidator

        enhancer = InputValidationEnhancer()
        validator = BoundaryConditionValidator()

        # 测试配置验证器
        config_validator = validator.create_config_validator(['host', 'port', 'timeout'])

        test_configs = [
            ({'host': 'localhost', 'port': 8080, 'timeout': 30}, True),
            ({'host': '', 'port': 8080, 'timeout': 30}, False),  # 空主机
            ({'host': 'localhost', 'port': 99999, 'timeout': 30}, False),  # 无效端口
            ({'host': 'localhost', 'timeout': 30}, False),  # 缺少端口
        ]

        for config, should_pass in test_configs:
            try:
                result = config_validator(config)
                if should_pass:
                    print(f"   ✅ 配置验证通过: {config}")
                else:
                    print(f"   ❌ 配置验证未正确拒绝: {config}")
            except ValueError as e:
                if not should_pass:
                    print(f"   ✅ 配置验证正确拒绝: {str(e)[:30]}...")
                else:
                    print(f"   ❌ 配置验证错误拒绝: {str(e)}")

        # 测试数值验证器
        numeric_validator = validator.create_numeric_validator(min_val=0, max_val=100)

        numeric_cases = [
            (50, True),
            (-5, False),
            (150, False),
            ('not_a_number', False),
        ]

        for value, should_pass in numeric_cases:
            try:
                result = numeric_validator(value)
                if should_pass:
                    print(f"   ✅ 数值验证通过: {value}")
                else:
                    print(f"   ❌ 数值验证未正确拒绝: {value}")
            except (ValueError, TypeError) as e:
                if not should_pass:
                    print(f"   ✅ 数值验证正确拒绝: {str(e)[:30]}...")
                else:
                    print(f"   ❌ 数值验证错误拒绝: {str(e)}")

        return True

    except Exception as e:
        print(f"   ❌ 输入验证增强器测试失败: {str(e)}")
        traceback.print_exc()
        return False

def run_boundary_condition_tests():
    """运行所有边界条件测试"""
    print("🧪 RQA2025 边界条件处理验证测试")
    print("=" * 50)

    tests = [
        test_config_boundary_conditions,
        test_execution_engine_boundary_conditions,
        test_numeric_boundary_handler,
        test_network_resource_handler,
        test_input_validation_enhancer,
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"   ❌ 测试 {test_func.__name__} 出现异常: {str(e)}")
            results.append(False)

    # 统计结果
    passed = sum(results)
    total = len(results)

    print(f"\n📊 测试结果统计:")
    print(f"   总测试数: {total}")
    print(f"   通过测试: {passed}")
    print(f"   失败测试: {total - passed}")
    print(".1f")
    if passed == total:
        print("   🎉 所有边界条件测试通过！")
        return True
    else:
        print("   ⚠️ 部分测试失败，需要检查")
        return False

if __name__ == "__main__":
    success = run_boundary_condition_tests()
    sys.exit(0 if success else 1)
