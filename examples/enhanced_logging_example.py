"""
增强日志系统使用示例
演示关键业务日志强制采样和采样日志与全量日志的关联机制
"""

import time
from datetime import datetime, timedelta

# 导入增强日志系统组件
from src.infrastructure.logging.enhanced_log_sampler import (
    EnhancedLogSampler,
    BusinessLogType,
    LogEntry
)
from src.infrastructure.logging.business_log_manager import (
    BusinessLogManager,
    BusinessLogConfig
)
from src.infrastructure.logging.log_correlation_query import (
    LogCorrelationQuery,
    CorrelationQuery
)


def demo_enhanced_log_sampling():
    """演示增强日志采样功能"""
    print("=== 增强日志采样演示 ===")

    # 初始化增强采样器
    sampler_config = {
        'default_rate': 0.3,
        'critical_business_types': [
            BusinessLogType.ORDER.value,
            BusinessLogType.TRADE.value,
            BusinessLogType.RISK.value,
            BusinessLogType.ACCOUNT.value
        ],
        'level_rates': {
            'DEBUG': 0.1,
            'INFO': 0.5,
            'WARNING': 1.0,
            'ERROR': 1.0,
            'CRITICAL': 1.0
        }
    }

    sampler = EnhancedLogSampler()
    sampler.configure(sampler_config)

    print(f"采样器配置: {sampler.get_config()}")

    # 模拟不同类型的日志
    test_logs = [
        # 订单相关日志 (应该100%采样)
        {
            'level': 'INFO',
            'message': '订单下单成功: order_id=12345, symbol=000001.SZ, price=10.50',
            'business_type': BusinessLogType.ORDER
        },
        {
            'level': 'WARNING',
            'message': '订单部分成交: order_id=12345, filled_qty=100, remaining_qty=200',
            'business_type': BusinessLogType.ORDER
        },
        # 交易相关日志 (应该100%采样)
        {
            'level': 'INFO',
            'message': '交易成交: trade_id=67890, symbol=000001.SZ, qty=100, price=10.52',
            'business_type': BusinessLogType.TRADE
        },
        # 风控相关日志 (应该100%采样)
        {
            'level': 'ERROR',
            'message': '风控检查失败: order_id=12345, reason=资金不足',
            'business_type': BusinessLogType.RISK
        },
        # 调试日志 (应该按采样率采样)
        {
            'level': 'DEBUG',
            'message': '计算技术指标: MA5=10.45, MA10=10.38',
            'business_type': BusinessLogType.DEBUG
        },
        {
            'level': 'DEBUG',
            'message': '更新缓存数据: cache_key=market_data_000001',
            'business_type': BusinessLogType.DEBUG
        }
    ]

    print("\n模拟日志采样结果:")
    for i, log_data in enumerate(test_logs, 1):
        # 创建日志条目
        log_entry = LogEntry(
            timestamp=datetime.now(),
            level=log_data['level'],
            message=log_data['message'],
            logger=f"test.{log_data['business_type'].value}",
            business_type=log_data['business_type'],
            trace_id=f"trace_{i}",
            correlation_id=f"corr_{i}"
        )

        # 检查是否采样
        should_sample = sampler.should_sample(log_entry)
        status = "✅ 采样" if should_sample else "❌ 过滤"

        print(
            f"{i}. [{log_data['business_type'].value}] {log_data['level']}: {log_data['message'][:50]}... {status}")

        if should_sample:
            sampler.record_sampled_log(log_entry)

    print(f"\n采样统计: 共记录 {len(sampler._sampled_logs)} 条采样日志")


def demo_business_log_manager():
    """演示业务日志管理器"""
    print("\n=== 业务日志管理器演示 ===")

    # 初始化业务日志管理器
    config = BusinessLogConfig(
        critical_business_types=[
            BusinessLogType.ORDER,
            BusinessLogType.TRADE,
            BusinessLogType.RISK,
            BusinessLogType.ACCOUNT
        ],
        business_log_rate=1.0,  # 业务日志100%采样
        debug_log_rate=0.1,     # 调试日志10%采样
        enable_correlation=True,
        enable_trace=True
    )

    manager = BusinessLogManager(config)

    # 模拟订单处理流程
    print("\n模拟订单处理流程:")

    # 1. 订单接收
    order_id = "ORDER_20250719_001"
    correlation_id = manager.log_business_operation(
        operation="order_receive",
        business_type=BusinessLogType.ORDER,
        message=f"收到订单: {order_id}, symbol=000001.SZ, price=10.50, qty=1000",
        level="INFO",
        extra={'order_id': order_id, 'symbol': '000001.SZ', 'price': 10.50, 'qty': 1000}
    )

    # 2. 风控检查
    manager.log_business_operation(
        operation="risk_check",
        business_type=BusinessLogType.RISK,
        message=f"风控检查通过: {order_id}",
        level="INFO",
        extra={'order_id': order_id, 'risk_score': 0.85},
        correlation_id=correlation_id
    )

    # 3. 订单执行
    manager.log_business_operation(
        operation="order_execute",
        business_type=BusinessLogType.ORDER,
        message=f"订单执行成功: {order_id}, filled_qty=800, avg_price=10.52",
        level="INFO",
        extra={'order_id': order_id, 'filled_qty': 800, 'avg_price': 10.52},
        correlation_id=correlation_id
    )

    # 4. 交易记录
    manager.log_business_operation(
        operation="trade_record",
        business_type=BusinessLogType.TRADE,
        message=f"交易记录: trade_id=TRADE_001, order_id={order_id}, qty=800, price=10.52",
        level="INFO",
        extra={'trade_id': 'TRADE_001', 'order_id': order_id, 'qty': 800, 'price': 10.52},
        correlation_id=correlation_id
    )

    # 5. 调试信息
    manager.log_debug_operation(
        operation="market_data",
        message=f"市场数据更新: symbol=000001.SZ, bid=10.51, ask=10.53",
        extra={'symbol': '000001.SZ', 'bid': 10.51, 'ask': 10.53},
        trace_id=correlation_id
    )

    # 获取操作追踪
    trace_logs = manager.get_operation_trace(correlation_id)
    print(f"\n操作追踪日志 (correlation_id={correlation_id}):")
    for log in trace_logs:
        print(f"  [{log.timestamp.strftime('%H:%M:%S')}] {log.business_type.value}: {log.message}")

    # 获取关键业务日志
    critical_logs = manager.get_critical_business_logs()
    print(f"\n关键业务日志数量: {len(critical_logs)}")

    # 获取统计信息
    stats = manager.get_statistics()
    print(f"\n业务日志管理器统计: {stats}")


def demo_log_correlation_query():
    """演示日志关联查询"""
    print("\n=== 日志关联查询演示 ===")

    # 初始化关联查询器
    query_manager = LogCorrelationQuery(
        max_query_history=100,
        query_timeout=300
    )

    # 模拟索引一些日志条目
    test_logs = [
        LogEntry(
            timestamp=datetime.now() - timedelta(minutes=5),
            level="INFO",
            message="订单下单: order_id=12345",
            logger="business.order",
            business_type=BusinessLogType.ORDER,
            trace_id="trace_123",
            correlation_id="corr_123"
        ),
        LogEntry(
            timestamp=datetime.now() - timedelta(minutes=4),
            level="INFO",
            message="风控检查: order_id=12345",
            logger="business.risk",
            business_type=BusinessLogType.RISK,
            trace_id="trace_123",
            correlation_id="corr_123"
        ),
        LogEntry(
            timestamp=datetime.now() - timedelta(minutes=3),
            level="INFO",
            message="交易成交: trade_id=67890",
            logger="business.trade",
            business_type=BusinessLogType.TRADE,
            trace_id="trace_123",
            correlation_id="corr_123"
        )
    ]

    # 索引日志条目
    for log_entry in test_logs:
        query_manager.index_log_entry(log_entry)

    # 执行关联查询
    query = CorrelationQuery(
        trace_id="trace_123",
        business_type=BusinessLogType.ORDER,
        time_range=(datetime.now() - timedelta(minutes=10), datetime.now())
    )

    result = query_manager.query_correlation(query)

    print(f"\n关联查询结果:")
    print(f"  查询ID: {result.query_id}")
    print(f"  采样日志数量: {len(result.sampled_logs)}")
    print(f"  关联日志数量: {len(result.related_logs)}")
    print(f"  统计信息: {result.statistics}")

    # 导出查询结果
    json_result = query_manager.export_query_result(result.query_id, "json")
    print(f"\n查询结果导出 (JSON):")
    print(json_result[:500] + "..." if len(json_result) > 500 else json_result)

    # 获取查询历史
    history = query_manager.get_query_history(limit=5)
    print(f"\n查询历史数量: {len(history)}")

    # 获取统计信息
    stats = query_manager.get_statistics()
    print(f"\n关联查询器统计: {stats}")


def demo_integrated_logging_system():
    """演示集成日志系统"""
    print("\n=== 集成日志系统演示 ===")

    # 初始化所有组件
    sampler = EnhancedLogSampler()
    manager = BusinessLogManager()
    query_manager = LogCorrelationQuery()

    # 模拟完整的交易流程
    print("\n模拟完整交易流程:")

    # 1. 开始交易会话
    session_id = f"SESSION_{int(time.time())}"
    trace_id = sampler.generate_trace_id()

    print(f"开始交易会话: {session_id}, trace_id: {trace_id}")

    # 2. 账户检查
    account_corr_id = manager.log_business_operation(
        operation="account_check",
        business_type=BusinessLogType.ACCOUNT,
        message=f"账户检查: 可用资金=1000000",
        level="INFO",
        extra={'session_id': session_id, 'available_balance': 1000000},
        trace_id=trace_id
    )

    # 3. 订单处理
    order_corr_id = manager.log_business_operation(
        operation="order_processing",
        business_type=BusinessLogType.ORDER,
        message=f"订单处理: symbol=000001.SZ, price=10.50, qty=1000",
        level="INFO",
        extra={'session_id': session_id, 'symbol': '000001.SZ', 'price': 10.50, 'qty': 1000},
        trace_id=trace_id
    )

    # 4. 风控检查
    risk_corr_id = manager.log_business_operation(
        operation="risk_validation",
        business_type=BusinessLogType.RISK,
        message=f"风控检查: 风险评分=0.85, 通过",
        level="INFO",
        extra={'session_id': session_id, 'risk_score': 0.85, 'status': 'passed'},
        trace_id=trace_id
    )

    # 5. 交易执行
    trade_corr_id = manager.log_business_operation(
        operation="trade_execution",
        business_type=BusinessLogType.TRADE,
        message=f"交易执行: filled_qty=800, avg_price=10.52",
        level="INFO",
        extra={'session_id': session_id, 'filled_qty': 800, 'avg_price': 10.52},
        trace_id=trace_id
    )

    # 6. 调试信息
    manager.log_debug_operation(
        operation="market_analysis",
        message=f"市场分析: MA5=10.45, MA10=10.38, 趋势=上涨",
        extra={'session_id': session_id, 'ma5': 10.45, 'ma10': 10.38, 'trend': 'up'},
        trace_id=trace_id
    )

    # 查询关联日志
    query = CorrelationQuery(
        trace_id=trace_id,
        time_range=(datetime.now() - timedelta(minutes=10), datetime.now())
    )

    result = query_manager.query_correlation(query)

    print(f"\n完整交易流程关联查询结果:")
    print(f"  总日志数量: {result.statistics.get('total_logs_count', 0)}")
    print(f"  业务类型分布: {result.statistics.get('business_type_distribution', {})}")
    print(f"  日志级别分布: {result.statistics.get('log_level_distribution', {})}")

    # 验证关键业务日志完整性
    critical_logs = manager.get_critical_business_logs()
    print(f"\n关键业务日志验证:")
    print(
        f"  订单相关: {len([log for log in critical_logs if log.business_type == BusinessLogType.ORDER])}")
    print(
        f"  交易相关: {len([log for log in critical_logs if log.business_type == BusinessLogType.TRADE])}")
    print(
        f"  风控相关: {len([log for log in critical_logs if log.business_type == BusinessLogType.RISK])}")
    print(
        f"  账户相关: {len([log for log in critical_logs if log.business_type == BusinessLogType.ACCOUNT])}")


def main():
    """主函数"""
    print("增强日志系统演示")
    print("=" * 50)

    try:
        # 演示增强日志采样
        demo_enhanced_log_sampling()

        # 演示业务日志管理器
        demo_business_log_manager()

        # 演示日志关联查询
        demo_log_correlation_query()

        # 演示集成日志系统
        demo_integrated_logging_system()

        print("\n" + "=" * 50)
        print("✅ 增强日志系统演示完成")
        print("\n主要特性:")
        print("  ✅ 关键业务日志强制采样 (订单、交易、风控、账户)")
        print("  ✅ 采样日志与全量日志的关联机制")
        print("  ✅ 区分业务日志和调试日志")
        print("  ✅ 确保关键操作可完整追溯")
        print("  ✅ 支持跟踪ID和关联ID")
        print("  ✅ 提供统计和导出功能")

    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
