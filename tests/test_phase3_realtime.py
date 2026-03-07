#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3 实时数据集成测试

测试内容：
1. 实时数据路由器
2. 实时信号集成
3. WebSocket发布器

作者: AI Assistant
创建日期: 2026-02-21
"""

import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_realtime_data_router():
    """测试实时数据路由器"""
    print("\n" + "="*60)
    print("测试实时数据路由器")
    print("="*60)
    
    from src.data.realtime_data_router import (
        get_realtime_data_router,
        RealtimeMarketData
    )
    
    router = get_realtime_data_router()
    
    # 测试1: 注册数据源
    print("\n1. 注册数据源...")
    def mock_data_source():
        pass
    
    result = router.register_data_source(
        name="test_source",
        handler=mock_data_source,
        priority=1
    )
    if result:
        print("  ✓ 数据源注册成功")
    else:
        print("  ✗ 数据源注册失败")
        return False
    
    # 测试2: 订阅数据
    print("\n2. 订阅数据...")
    received_data = []
    def data_callback(data):
        received_data.append(data)
    
    router.subscribe("000001", data_callback)
    print("  ✓ 订阅成功")
    
    # 测试3: 路由数据
    print("\n3. 路由数据...")
    test_data = RealtimeMarketData(
        symbol="000001",
        timestamp=datetime.now(),
        open=10.0,
        high=11.0,
        low=9.5,
        close=10.5,
        volume=10000,
        amount=105000.0,
        source="test_source"
    )
    
    result = router.route_data(test_data)
    if result:
        print("  ✓ 数据路由成功")
    else:
        print("  ! 数据被去重或路由失败")
    
    # 测试4: 获取最新数据
    print("\n4. 获取最新数据...")
    latest = router.get_latest_data("000001")
    if latest:
        print(f"  ✓ 获取到最新数据: {latest.symbol} @ {latest.close}")
    else:
        print("  ✗ 未获取到最新数据")
    
    # 测试5: 获取统计信息
    print("\n5. 获取统计信息...")
    stats = router.get_stats()
    print(f"  ✓ 总消息数: {stats['total_messages']}")
    print(f"  ✓ 去重消息数: {stats['deduplicated_messages']}")
    print(f"  ✓ 订阅股票数: {stats['subscribed_symbols']}")
    
    return True


def test_realtime_signal_integration():
    """测试实时信号集成"""
    print("\n" + "="*60)
    print("测试实时信号集成")
    print("="*60)
    
    from src.data.realtime_signal_integration import (
        get_realtime_signal_integration,
        simple_momentum_signal_generator,
        RealtimeMarketData
    )
    from src.data.realtime_data_router import get_realtime_data_router
    
    integration = get_realtime_signal_integration()
    router = get_realtime_data_router()
    
    # 测试1: 启动服务
    print("\n1. 启动服务...")
    integration.start()
    print("  ✓ 服务已启动")
    
    # 测试2: 注册信号生成器
    print("\n2. 注册信号生成器...")
    integration.register_signal_generator(
        "momentum",
        simple_momentum_signal_generator
    )
    print("  ✓ 信号生成器注册成功")
    
    # 测试3: 注册WebSocket回调
    print("\n3. 注册WebSocket回调...")
    received_signals = []
    def signal_callback(signal):
        received_signals.append(signal)
        print(f"    收到信号: {signal.symbol} - {signal.signal_type}")
    
    integration.register_websocket_callback(signal_callback)
    print("  ✓ WebSocket回调注册成功")
    
    # 测试4: 模拟市场数据并生成信号
    print("\n4. 模拟市场数据并生成信号...")
    test_data = RealtimeMarketData(
        symbol="000001",
        timestamp=datetime.now(),
        open=10.0,
        high=11.0,
        low=9.5,
        close=10.5,  # 5%涨幅，应该生成买入信号
        volume=10000,
        amount=105000.0,
        source="test"
    )
    
    # 路由数据，触发信号生成
    router.route_data(test_data)
    print(f"  ✓ 数据已路由，生成 {len(received_signals)} 个信号")
    
    # 测试5: 获取统计信息
    print("\n5. 获取统计信息...")
    stats = integration.get_stats()
    print(f"  ✓ 信号生成器数: {stats['signal_generators']}")
    print(f"  ✓ WebSocket回调数: {stats['websocket_callbacks']}")
    print(f"  ✓ 生成的信号数: {stats['signals_generated']}")
    
    # 停止服务
    integration.stop()
    
    return True


def test_websocket_publisher():
    """测试WebSocket发布器"""
    print("\n" + "="*60)
    print("测试WebSocket发布器")
    print("="*60)
    
    from src.gateway.web.websocket_publisher import get_websocket_publisher
    
    publisher = get_websocket_publisher()
    
    # 测试1: 启动发布器
    print("\n1. 启动发布器...")
    publisher.start()
    print("  ✓ 发布器已启动")
    
    # 测试2: 模拟客户端连接
    print("\n2. 模拟客户端连接...")
    publisher.on_connect("test_sid_1", {})
    print("  ✓ 客户端连接成功")
    
    # 测试3: 订阅股票
    print("\n3. 订阅股票...")
    publisher.subscribe_symbol("test_sid_1", "000001")
    publisher.subscribe_symbol("test_sid_1", "000002")
    print("  ✓ 股票订阅成功")
    
    # 测试4: 获取客户端信息
    print("\n4. 获取客户端信息...")
    client_info = publisher.get_client_info("test_sid_1")
    if client_info:
        print(f"  ✓ 客户端ID: {client_info['sid']}")
        print(f"  ✓ 订阅股票: {client_info['subscribed_symbols']}")
    
    # 测试5: 获取统计信息
    print("\n5. 获取统计信息...")
    stats = publisher.get_stats()
    print(f"  ✓ 总连接数: {stats['total_connections']}")
    print(f"  ✓ 活跃连接数: {stats['active_connections']}")
    print(f"  ✓ 股票订阅数: {stats['symbol_subscriptions']}")
    
    # 测试6: 模拟客户端断开
    print("\n6. 模拟客户端断开...")
    publisher.on_disconnect("test_sid_1")
    print("  ✓ 客户端断开成功")
    
    # 停止发布器
    publisher.stop()
    
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("Phase 3 实时数据集成测试")
    print("="*60)
    
    tests = [
        ("实时数据路由器", test_realtime_data_router),
        ("实时信号集成", test_realtime_signal_integration),
        ("WebSocket发布器", test_websocket_publisher),
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
