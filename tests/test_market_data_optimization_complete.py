#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
市场数据获取优化完整测试套件

测试所有四个Phase的组件：
- Phase 1: 数据采集优化
- Phase 2: 多股票支持
- Phase 3: 实时数据集成
- Phase 4: 信号验证监控

作者: AI Assistant
创建日期: 2026-02-21
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class Phase1DataCollectionTests:
    """Phase 1: 数据采集优化测试"""
    
    @staticmethod
    def test_data_collection_orchestrator():
        """测试数据采集协调器"""
        print("\n  [Phase 1.1] 数据采集协调器...")
        
        from src.data.collectors.data_collection_orchestrator import get_data_collection_orchestrator
        
        orchestrator = get_data_collection_orchestrator()
        
        # 注册模拟采集器
        class MockCollector:
            def collect_and_save(self, symbol):
                return True
        
        result = orchestrator.register_collector("mock", MockCollector())
        assert result, "采集器注册失败"
        
        # 调度任务
        task_ids = orchestrator.schedule_collection(["000001"], "mock")
        assert len(task_ids) > 0, "任务调度失败"
        
        # 获取状态
        status = orchestrator.get_status()
        assert "collectors" in status, "状态获取失败"
        
        print("    ✓ 数据采集协调器测试通过")
        return True
    
    @staticmethod
    def test_enhanced_akshare_collector():
        """测试增强版AKShare采集器"""
        print("\n  [Phase 1.2] 增强版AKShare采集器...")
        
        from src.data.collectors.enhanced_akshare_collector import get_enhanced_akshare_collector
        
        collector = get_enhanced_akshare_collector()
        
        # 测试数据质量检查器
        test_data = [
            {"open": 10.0, "high": 11.0, "low": 9.5, "close": 10.5, "volume": 1000, "date": "2024-01-01"}
        ]
        
        passed, errors = collector.quality_checker.check_quality(test_data, "000001")
        assert passed, f"数据质量检查失败: {errors}"
        
        print("    ✓ 增强版AKShare采集器测试通过")
        return True


class Phase2MultiStockTests:
    """Phase 2: 多股票支持测试"""
    
    @staticmethod
    def test_strategy_config_parser():
        """测试策略配置解析器"""
        print("\n  [Phase 2.1] 策略配置解析器...")
        
        from src.data.strategy_config_parser import get_strategy_config_parser
        
        parser = get_strategy_config_parser()
        
        # 加载默认策略配置
        config = parser.get_config("default_strategy")
        assert config is not None, "配置加载失败"
        assert len(config.symbols) > 0, "股票代码列表为空"
        
        print(f"    ✓ 策略配置解析器测试通过 (加载了 {len(config.symbols)} 只股票)")
        return True
    
    @staticmethod
    def test_symbol_mapping_service():
        """测试股票代码映射服务"""
        print("\n  [Phase 2.2] 股票代码映射服务...")
        
        from src.data.symbol_mapping_service import get_symbol_mapping_service
        
        service = get_symbol_mapping_service()
        
        # 注册映射
        result = service.register_mapping("test_strategy", ["000001", "000002"])
        assert result, "映射注册失败"
        
        # 获取股票代码
        symbols = service.get_symbols_for_strategy("test_strategy")
        assert len(symbols) == 2, f"股票代码数量错误: {len(symbols)}"
        
        print("    ✓ 股票代码映射服务测试通过")
        return True
    
    @staticmethod
    def test_multi_stock_data_manager():
        """测试多股票数据管理器"""
        print("\n  [Phase 2.3] 多股票数据管理器...")
        
        from src.data.multi_stock_data_manager import get_multi_stock_data_manager
        
        manager = get_multi_stock_data_manager()
        
        # 获取策略股票代码
        symbols = manager.get_symbols_from_strategy("default_strategy")
        # 注意：如果没有配置可能为空，这是正常的
        
        # 测试缓存统计
        stats = manager.get_cache_stats()
        assert "total_keys" in stats, "缓存统计获取失败"
        
        print("    ✓ 多股票数据管理器测试通过")
        return True


class Phase3RealtimeTests:
    """Phase 3: 实时数据集成测试"""
    
    @staticmethod
    def test_realtime_data_router():
        """测试实时数据路由器"""
        print("\n  [Phase 3.1] 实时数据路由器...")
        
        from src.data.realtime_data_router import get_realtime_data_router, RealtimeMarketData
        
        router = get_realtime_data_router()
        
        # 订阅数据
        received_data = []
        def callback(data):
            received_data.append(data)
        
        router.subscribe("000001", callback)
        
        # 路由测试数据
        test_data = RealtimeMarketData(
            symbol="000001",
            timestamp=datetime.now(),
            open=10.0, high=11.0, low=9.5, close=10.5,
            volume=10000, amount=105000.0, source="test"
        )
        
        router.route_data(test_data)
        
        # 获取最新数据
        latest = router.get_latest_data("000001")
        assert latest is not None, "最新数据获取失败"
        assert latest.close == 10.5, "数据不匹配"
        
        print("    ✓ 实时数据路由器测试通过")
        return True
    
    @staticmethod
    def test_realtime_signal_integration():
        """测试实时信号集成"""
        print("\n  [Phase 3.2] 实时信号集成...")
        
        from src.data.realtime_signal_integration import get_realtime_signal_integration
        
        integration = get_realtime_signal_integration()
        
        # 启动服务
        integration.start()
        
        # 注册信号生成器
        integration.register_signal_generator("test", lambda x: None)
        
        # 获取统计
        stats = integration.get_stats()
        assert "signal_generators" in stats, "统计获取失败"
        
        # 停止服务
        integration.stop()
        
        print("    ✓ 实时信号集成测试通过")
        return True
    
    @staticmethod
    def test_websocket_publisher():
        """测试WebSocket发布器"""
        print("\n  [Phase 3.3] WebSocket发布器...")
        
        from src.gateway.web.websocket_publisher import get_websocket_publisher
        
        publisher = get_websocket_publisher()
        
        # 启动发布器
        publisher.start()
        
        # 模拟客户端连接
        publisher.on_connect("test_sid", {})
        
        # 订阅股票
        publisher.subscribe_symbol("test_sid", "000001")
        
        # 获取统计
        stats = publisher.get_stats()
        assert "total_connections" in stats, "统计获取失败"
        
        # 断开连接
        publisher.on_disconnect("test_sid")
        
        # 停止发布器
        publisher.stop()
        
        print("    ✓ WebSocket发布器测试通过")
        return True


class Phase4SignalMonitoringTests:
    """Phase 4: 信号验证监控测试"""
    
    @staticmethod
    def test_signal_validation_engine():
        """测试信号验证引擎"""
        print("\n  [Phase 4.1] 信号验证引擎...")
        
        from src.trading.signal.signal_validation_engine import get_signal_validation_engine
        
        engine = get_signal_validation_engine()
        
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
        
        # 验证信号
        result = engine.validate_signal(signal, market_data)
        
        assert result.overall_score > 0, "综合评分计算失败"
        assert result.quality_score > 0, "质量评分计算失败"
        assert result.is_valid, "信号验证失败"
        
        print(f"    ✓ 信号验证引擎测试通过 (综合评分: {result.overall_score:.2f})")
        return True
    
    @staticmethod
    def test_signal_filter():
        """测试信号过滤器"""
        print("\n  [Phase 4.2] 信号过滤器...")
        
        from src.trading.signal.signal_filter import get_signal_filter
        from src.trading.signal.signal_validation_engine import SignalValidationResult
        
        filter = get_signal_filter()
        
        # 测试通过的信号
        signal = {'symbol': '000001', 'signal_type': 'buy', 'confidence': 0.85}
        validation = SignalValidationResult(
            signal_id='test', overall_score=75.0, quality_score=80.0,
            risk_score=50.0, backtest_score=70.0, is_valid=True,
            validation_time=datetime.now(), details={}
        )
        
        passed, reason = filter.filter_signal(signal, validation)
        assert passed, f"信号应该通过过滤: {reason}"
        
        # 测试低置信度信号
        low_conf_signal = {'symbol': '000002', 'signal_type': 'sell', 'confidence': 0.3}
        passed, reason = filter.filter_signal(low_conf_signal)
        assert not passed, "低置信度信号应该被过滤"
        
        print("    ✓ 信号过滤器测试通过")
        return True
    
    @staticmethod
    def test_signal_monitor():
        """测试信号监控器"""
        print("\n  [Phase 4.3] 信号监控器...")
        
        from src.trading.signal.signal_monitor import get_signal_monitor
        from src.trading.signal.signal_validation_engine import SignalValidationResult
        
        monitor = get_signal_monitor()
        
        # 记录信号
        signal = {'symbol': '000001', 'signal_type': 'buy', 'confidence': 0.85}
        validation = SignalValidationResult(
            signal_id='test', overall_score=75.0, quality_score=80.0,
            risk_score=50.0, backtest_score=70.0, is_valid=True,
            validation_time=datetime.now(), details={}
        )
        
        monitor.record_signal(signal, validation)
        
        # 计算指标
        metrics = monitor.calculate_current_metrics()
        assert metrics.total_signals > 0, "指标计算失败"
        
        # 获取面板数据
        dashboard = monitor.get_dashboard_data()
        assert "current_metrics" in dashboard, "面板数据获取失败"
        
        print("    ✓ 信号监控器测试通过")
        return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*70)
    print("市场数据获取优化完整测试套件")
    print("="*70)
    
    all_tests = [
        ("Phase 1: 数据采集优化", [
            Phase1DataCollectionTests.test_data_collection_orchestrator,
            Phase1DataCollectionTests.test_enhanced_akshare_collector,
        ]),
        ("Phase 2: 多股票支持", [
            Phase2MultiStockTests.test_strategy_config_parser,
            Phase2MultiStockTests.test_symbol_mapping_service,
            Phase2MultiStockTests.test_multi_stock_data_manager,
        ]),
        ("Phase 3: 实时数据集成", [
            Phase3RealtimeTests.test_realtime_data_router,
            Phase3RealtimeTests.test_realtime_signal_integration,
            Phase3RealtimeTests.test_websocket_publisher,
        ]),
        ("Phase 4: 信号验证监控", [
            Phase4SignalMonitoringTests.test_signal_validation_engine,
            Phase4SignalMonitoringTests.test_signal_filter,
            Phase4SignalMonitoringTests.test_signal_monitor,
        ]),
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for phase_name, tests in all_tests:
        print(f"\n{'='*70}")
        print(f"{phase_name}")
        print(f"{'='*70}")
        
        phase_passed = 0
        phase_failed = 0
        
        for test_func in tests:
            total_tests += 1
            try:
                if test_func():
                    passed_tests += 1
                    phase_passed += 1
                else:
                    failed_tests += 1
                    phase_failed += 1
            except Exception as e:
                print(f"    ✗ 测试失败: {e}")
                import traceback
                traceback.print_exc()
                failed_tests += 1
                phase_failed += 1
        
        print(f"\n  {phase_name} 结果: {phase_passed} 通过, {phase_failed} 失败")
    
    print("\n" + "="*70)
    print("总体测试结果")
    print("="*70)
    print(f"总测试数: {total_tests}")
    print(f"通过: {passed_tests}")
    print(f"失败: {failed_tests}")
    print(f"通过率: {passed_tests/total_tests*100:.1f}%")
    print("="*70 + "\n")
    
    return failed_tests == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
