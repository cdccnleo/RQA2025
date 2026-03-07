#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4 信号验证监控测试

测试内容：
1. 信号验证引擎
2. 信号过滤器
3. 信号监控器

作者: AI Assistant
创建日期: 2026-02-21
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_signal_validation_engine():
    """测试信号验证引擎"""
    print("\n" + "="*60)
    print("测试信号验证引擎")
    print("="*60)
    
    from src.trading.signal.signal_validation_engine import get_signal_validation_engine
    
    engine = get_signal_validation_engine()
    
    # 测试1: 验证信号
    print("\n1. 验证信号...")
    
    # 创建测试信号
    signal = {
        'symbol': '000001',
        'signal_type': 'buy',
        'confidence': 0.85,
        'timestamp': datetime.now()
    }
    
    # 创建测试市场数据
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    market_data = pd.DataFrame({
        'date': dates,
        'open': [10.0 + i*0.1 for i in range(30)],
        'high': [11.0 + i*0.1 for i in range(30)],
        'low': [9.5 + i*0.1 for i in range(30)],
        'close': [10.5 + i*0.1 for i in range(30)],
        'volume': [1000000 + i*10000 for i in range(30)],
        'amount': [10000000 + i*100000 for i in range(30)]
    })
    
    result = engine.validate_signal(signal, market_data)
    
    print(f"  ✓ 信号ID: {result.signal_id}")
    print(f"  ✓ 综合评分: {result.overall_score:.2f}")
    print(f"  ✓ 质量评分: {result.quality_score:.2f}")
    print(f"  ✓ 风险评分: {result.risk_score:.2f}")
    print(f"  ✓ 回测评分: {result.backtest_score:.2f}")
    print(f"  ✓ 是否有效: {result.is_valid}")
    
    # 测试2: 检查验证详情
    print("\n2. 检查验证详情...")
    if 'quality_details' in result.details:
        print(f"  ✓ 质量详情: {result.details['quality_details']}")
    if 'risk_details' in result.details:
        print(f"  ✓ 风险详情: {result.details['risk_details']}")
    if 'backtest_result' in result.details:
        backtest = result.details['backtest_result']
        print(f"  ✓ 回测胜率: {backtest.get('win_rate', 0):.2%}")
        print(f"  ✓ 夏普比率: {backtest.get('sharpe_ratio', 0):.2f}")
    
    return True


def test_signal_filter():
    """测试信号过滤器"""
    print("\n" + "="*60)
    print("测试信号过滤器")
    print("="*60)
    
    from src.trading.signal.signal_filter import get_signal_filter, FilterConfig
    from src.trading.signal.signal_validation_engine import SignalValidationResult
    
    # 创建过滤器配置
    config = FilterConfig(
        min_overall_score=60.0,
        min_quality_score=60.0,
        max_risk_score=70.0,
        min_confidence=0.5
    )
    
    filter = get_signal_filter(config)
    
    # 测试1: 过滤通过的信号
    print("\n1. 测试通过过滤的信号...")
    signal = {
        'symbol': '000001',
        'signal_type': 'buy',
        'confidence': 0.85
    }
    
    validation_result = SignalValidationResult(
        signal_id='test_001',
        overall_score=75.0,
        quality_score=80.0,
        risk_score=50.0,
        backtest_score=70.0,
        is_valid=True,
        validation_time=datetime.now(),
        details={}
    )
    
    passed, reason = filter.filter_signal(signal, validation_result)
    if passed:
        print(f"  ✓ 信号通过过滤: {reason}")
    else:
        print(f"  ✗ 信号未通过过滤: {reason}")
        return False
    
    # 测试2: 过滤失败的信号（低置信度）
    print("\n2. 测试低置信度信号过滤...")
    low_confidence_signal = {
        'symbol': '000002',
        'signal_type': 'sell',
        'confidence': 0.3  # 低于阈值0.5
    }
    
    passed, reason = filter.filter_signal(low_confidence_signal)
    if not passed:
        print(f"  ✓ 低置信度信号被正确过滤: {reason}")
    else:
        print(f"  ✗ 低置信度信号应该被过滤")
        return False
    
    # 测试3: 获取统计信息
    print("\n3. 获取过滤统计...")
    stats = filter.get_stats()
    print(f"  ✓ 总信号数: {stats['total_signals']}")
    print(f"  ✓ 通过数: {stats['passed_signals']}")
    print(f"  ✓ 过滤数: {stats['filtered_signals']}")
    print(f"  ✓ 通过率: {stats['pass_rate']:.2%}")
    
    return True


def test_signal_monitor():
    """测试信号监控器"""
    print("\n" + "="*60)
    print("测试信号监控器")
    print("="*60)
    
    from src.trading.signal.signal_monitor import get_signal_monitor
    from src.trading.signal.signal_validation_engine import SignalValidationResult
    
    monitor = get_signal_monitor()
    
    # 测试1: 记录信号
    print("\n1. 记录信号...")
    signal = {
        'symbol': '000001',
        'signal_type': 'buy',
        'confidence': 0.85,
        'price': 10.5
    }
    
    validation_result = SignalValidationResult(
        signal_id='test_001',
        overall_score=75.0,
        quality_score=80.0,
        risk_score=50.0,
        backtest_score=70.0,
        is_valid=True,
        validation_time=datetime.now(),
        details={}
    )
    
    monitor.record_signal(signal, validation_result)
    print("  ✓ 信号记录成功")
    
    # 记录更多信号
    for i in range(5):
        signal['symbol'] = f'00000{i}'
        signal['signal_type'] = 'buy' if i % 2 == 0 else 'sell'
        monitor.record_signal(signal, validation_result)
    
    print("  ✓ 批量信号记录成功")
    
    # 测试2: 计算当前指标
    print("\n2. 计算当前指标...")
    metrics = monitor.calculate_current_metrics()
    print(f"  ✓ 总信号数: {metrics.total_signals}")
    print(f"  ✓ 有效信号数: {metrics.valid_signals}")
    print(f"  ✓ 买入信号数: {metrics.buy_signals}")
    print(f"  ✓ 卖出信号数: {metrics.sell_signals}")
    print(f"  ✓ 平均质量评分: {metrics.avg_quality_score:.2f}")
    
    # 测试3: 获取监控面板数据
    print("\n3. 获取监控面板数据...")
    dashboard_data = monitor.get_dashboard_data()
    print(f"  ✓ 时间戳: {dashboard_data['timestamp']}")
    print(f"  ✓ 当前指标: {dashboard_data['current_metrics']}")
    print(f"  ✓ 告警规则数: {len(dashboard_data['alert_rules'])}")
    print(f"  ✓ 历史记录数: {dashboard_data['history_length']}")
    
    # 测试4: 注册告警回调
    print("\n4. 注册告警回调...")
    alerts_received = []
    def alert_callback(alert_data):
        alerts_received.append(alert_data)
        print(f"    收到告警: {alert_data.get('rule_name', 'Unknown')}")
    
    monitor.register_alert_callback(alert_callback)
    print("  ✓ 告警回调注册成功")
    
    # 测试5: 获取统计信息
    print("\n5. 获取统计信息...")
    stats = monitor.get_stats()
    print(f"  ✓ 总信号数: {stats['total_signals']}")
    print(f"  ✓ 有效信号数: {stats['valid_signals']}")
    print(f"  ✓ 历史记录数: {stats['history_length']}")
    print(f"  ✓ 告警规则数: {stats['alert_rules_count']}")
    
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("Phase 4 信号验证监控测试")
    print("="*60)
    
    tests = [
        ("信号验证引擎", test_signal_validation_engine),
        ("信号过滤器", test_signal_filter),
        ("信号监控器", test_signal_monitor),
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
