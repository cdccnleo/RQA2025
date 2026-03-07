#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 多股票支持测试

测试内容：
1. 策略配置解析器
2. 股票代码映射服务
3. 多股票数据管理器

作者: AI Assistant
创建日期: 2026-02-21
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_strategy_config_parser():
    """测试策略配置解析器"""
    print("\n" + "="*60)
    print("测试策略配置解析器")
    print("="*60)
    
    from src.data.strategy_config_parser import get_strategy_config_parser
    
    parser = get_strategy_config_parser()
    
    # 测试1: 加载默认策略配置
    print("\n1. 加载默认策略配置...")
    config = parser.get_config("default_strategy")
    
    if config:
        print(f"  ✓ 成功加载配置: {config.strategy_id}")
        print(f"  ✓ 策略名称: {config.strategy_name}")
        print(f"  ✓ 股票数量: {len(config.symbols)}")
        print(f"  ✓ 股票代码: {config.symbols}")
    else:
        print("  ✗ 未找到默认策略配置")
        return False
    
    # 测试2: 验证配置
    print("\n2. 验证配置...")
    is_valid, errors = parser.validate_config(config)
    if is_valid:
        print("  ✓ 配置验证通过")
    else:
        print("  ✗ 配置验证失败:")
        for error in errors:
            print(f"    - {error}")
        return False
    
    # 测试3: 加载所有配置
    print("\n3. 加载所有配置...")
    all_configs = parser.load_all_configs()
    print(f"  ✓ 加载了 {len(all_configs)} 个配置")
    
    return True


def test_symbol_mapping_service():
    """测试股票代码映射服务"""
    print("\n" + "="*60)
    print("测试股票代码映射服务")
    print("="*60)
    
    from src.data.symbol_mapping_service import get_symbol_mapping_service
    
    service = get_symbol_mapping_service()
    
    # 测试1: 注册映射
    print("\n1. 注册策略映射...")
    result = service.register_mapping(
        strategy_id="test_strategy",
        symbols=["000001", "000002", "000858"],
        priority=1,
        weight=1.0
    )
    if result:
        print("  ✓ 映射注册成功")
    else:
        print("  ✗ 映射注册失败")
        return False
    
    # 测试2: 获取策略的股票代码
    print("\n2. 获取策略的股票代码...")
    symbols = service.get_symbols_for_strategy("test_strategy")
    print(f"  ✓ 获取到 {len(symbols)} 只股票: {symbols}")
    
    # 测试3: 获取股票的策略
    print("\n3. 获取股票的策略...")
    strategies = service.get_strategies_for_symbol("000001")
    print(f"  ✓ 股票000001属于策略: {strategies}")
    
    # 测试4: 获取统计信息
    print("\n4. 获取统计信息...")
    stats = service.get_stats()
    print(f"  ✓ 总策略数: {stats['total_strategies']}")
    print(f"  ✓ 总股票数: {stats['total_symbols']}")
    
    return True


def test_multi_stock_data_manager():
    """测试多股票数据管理器"""
    print("\n" + "="*60)
    print("测试多股票数据管理器")
    print("="*60)
    
    from src.data.multi_stock_data_manager import get_multi_stock_data_manager
    
    manager = get_multi_stock_data_manager()
    
    # 测试1: 从策略获取股票代码
    print("\n1. 从策略获取股票代码...")
    symbols = manager.get_symbols_from_strategy("default_strategy")
    if symbols:
        print(f"  ✓ 获取到 {len(symbols)} 只股票: {symbols}")
    else:
        print("  ! 策略没有配置股票代码（这是正常的，如果没有配置）")
    
    # 测试2: 注册策略映射
    print("\n2. 注册策略映射...")
    result = manager.register_strategy_mapping(
        strategy_id="test_multi_strategy",
        symbols=["000001", "000002"],
        priority=1
    )
    if result:
        print("  ✓ 策略映射注册成功")
    else:
        print("  ✗ 策略映射注册失败")
        return False
    
    # 测试3: 获取策略映射后的股票代码
    print("\n3. 获取策略映射后的股票代码...")
    symbols = manager.get_symbols_from_strategy("test_multi_strategy")
    print(f"  ✓ 获取到 {len(symbols)} 只股票: {symbols}")
    
    # 测试4: 缓存统计
    print("\n4. 缓存统计...")
    stats = manager.get_cache_stats()
    print(f"  ✓ 缓存键总数: {stats['total_keys']}")
    print(f"  ✓ 有效缓存: {stats['valid_keys']}")
    
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("Phase 2 多股票支持测试")
    print("="*60)
    
    tests = [
        ("策略配置解析器", test_strategy_config_parser),
        ("股票代码映射服务", test_symbol_mapping_service),
        ("多股票数据管理器", test_multi_stock_data_manager),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n✗ {name} 测试失败: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print("测试结果")
    print("="*60)
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
