"""
统一日志接口使用示例
展示现有日志系统与增强日志系统的分工协作
"""

import time
from datetime import datetime, timedelta

# 导入统一日志接口
from src.infrastructure.logging.unified_logging_interface import (
    UnifiedLoggingInterface,
    configure_logging,
    log_basic,
    log_business,
    log_debug,
    query_correlation
)
from src.infrastructure.logging.enhanced_log_sampler import BusinessLogType
from src.infrastructure.logging.log_correlation_query import CorrelationQuery


def demo_system_startup_logging():
    """演示系统启动日志 - 使用现有日志系统"""
    print("=== 系统启动日志演示 (现有日志系统) ===")

    # 配置基础日志系统
    basic_config = {
        'basic': {
            'level': 'INFO',
            'handlers': [
                {'type': 'file', 'filename': 'logs/system.log'},
                {'type': 'console'}
            ],
            'sampling': {
                'default_rate': 0.5,
                'level_rates': {
                    'DEBUG': 0.1,
                    'INFO': 0.8,
                    'WARNING': 1.0,
                    'ERROR': 1.0
                }
            }
        }
    }

    interface = UnifiedLoggingInterface(basic_config)

    # 系统启动日志
    interface.log_basic("system.startup", "INFO", "系统启动开始")
    interface.log_basic("system.config", "INFO", "配置文件加载成功")
    interface.log_basic("system.database", "INFO", "数据库连接建立")
    interface.log_basic("system.cache", "INFO", "缓存系统初始化")
    interface.log_basic("system.startup", "INFO", "系统启动完成")

    # 性能监控日志
    interface.log_basic("performance.memory", "INFO", "内存使用: 512MB")
    interface.log_basic("performance.cpu", "INFO", "CPU使用率: 15%")
    interface.log_basic("performance.disk", "INFO", "磁盘使用率: 45%")

    print("✅ 系统启动日志记录完成")


def demo_business_operation_logging():
    """演示业务操作日志 - 使用增强日志系统"""
    print("\n=== 业务操作日志演示 (增强日志系统) ===")

    # 配置增强日志系统
    enhanced_config = {
        'enhanced': {
            'sampling': {
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
                    'ERROR': 1.0
                }
            },
            'business_logging': {
                'business_log_rate': 1.0,
                'debug_log_rate': 0.1,
                'enable_correlation': True,
                'enable_trace': True
            }
        }
    }

    interface = UnifiedLoggingInterface(enhanced_config)

    # 开始订单处理流程
    order_id = "ORDER_20250719_001"
    trace_id = interface.start_trace()
    correlation_id = interface.start_correlation()

    # 设置业务上下文
    interface.set_business_context(BusinessLogType.ORDER, "order_processing")

    # 记录订单处理流程
    interface.log_business(
        operation="order_receive",
        business_type=BusinessLogType.ORDER,
        message=f"收到订单: {order_id}",
        level="INFO",
        order_id=order_id,
        symbol="000001.SZ",
        price=10.50,
        qty=1000
    )

    # 风控检查
    interface.log_business(
        operation="risk_check",
        business_type=BusinessLogType.RISK,
        message=f"风控检查通过: {order_id}",
        level="INFO",
        order_id=order_id,
        risk_score=0.85
    )

    # 订单执行
    interface.log_business(
        operation="order_execute",
        business_type=BusinessLogType.ORDER,
        message=f"订单执行成功: {order_id}",
        level="INFO",
        order_id=order_id,
        filled_qty=800,
        avg_price=10.52
    )

    # 交易记录
    interface.log_business(
        operation="trade_record",
        business_type=BusinessLogType.TRADE,
        message=f"交易记录: trade_id=TRADE_001, order_id={order_id}",
        level="INFO",
        trade_id="TRADE_001",
        order_id=order_id,
        qty=800,
        price=10.52
    )

    # 调试信息
    interface.log_debug(
        operation="market_analysis",
        message=f"市场分析: symbol=000001.SZ, bid=10.51, ask=10.53",
        symbol="000001.SZ",
        bid=10.51,
        ask=10.53
    )

    # 获取操作追踪
    trace_logs = interface.get_operation_trace(correlation_id)
    print(f"\n操作追踪日志 (correlation_id={correlation_id}):")
    for log in trace_logs:
        print(f"  [{log.timestamp.strftime('%H:%M:%S')}] {log.business_type.value}: {log.message}")

    # 获取关键业务日志
    critical_logs = interface.get_critical_business_logs()
    print(f"\n关键业务日志数量: {len(critical_logs)}")

    print("✅ 业务操作日志记录完成")


def demo_correlation_query():
    """演示关联查询功能"""
    print("\n=== 关联查询演示 ===")

    # 配置包含查询功能的系统
    full_config = {
        'enhanced': {
            'sampling': {
                'default_rate': 0.3,
                'critical_business_types': [
                    BusinessLogType.ORDER.value,
                    BusinessLogType.TRADE.value
                ]
            },
            'business_logging': {
                'business_log_rate': 1.0,
                'debug_log_rate': 0.1,
                'enable_correlation': True,
                'enable_trace': True
            },
            'correlation_query': {
                'max_query_history': 100,
                'query_timeout': 300
            }
        }
    }

    interface = UnifiedLoggingInterface(full_config)

    # 创建一些测试日志
    trace_id = interface.start_trace()
    correlation_id = interface.start_correlation()

    # 记录相关日志
    interface.log_business(
        operation="test_operation",
        business_type=BusinessLogType.ORDER,
        message="测试订单操作",
        level="INFO",
        trace_id=trace_id,
        correlation_id=correlation_id
    )

    interface.log_business(
        operation="test_trade",
        business_type=BusinessLogType.TRADE,
        message="测试交易操作",
        level="INFO",
        trace_id=trace_id,
        correlation_id=correlation_id
    )

    # 执行关联查询
    query = CorrelationQuery(
        trace_id=trace_id,
        business_type=BusinessLogType.ORDER,
        time_range=(datetime.now() - timedelta(minutes=10), datetime.now())
    )

    result = interface.query_correlation(query)

    print(f"关联查询结果:")
    print(f"  查询ID: {result.query_id}")
    print(f"  采样日志数量: {len(result.sampled_logs)}")
    print(f"  关联日志数量: {len(result.related_logs)}")
    print(f"  统计信息: {result.statistics}")

    print("✅ 关联查询演示完成")


def demo_integrated_logging():
    """演示集成日志系统"""
    print("\n=== 集成日志系统演示 ===")

    # 完整配置
    full_config = {
        'basic': {
            'level': 'INFO',
            'handlers': [
                {'type': 'file', 'filename': 'logs/integrated.log'},
                {'type': 'console'}
            ],
            'sampling': {
                'default_rate': 0.5,
                'level_rates': {
                    'DEBUG': 0.1,
                    'INFO': 0.8,
                    'ERROR': 1.0
                }
            }
        },
        'enhanced': {
            'sampling': {
                'default_rate': 0.3,
                'critical_business_types': [
                    BusinessLogType.ORDER.value,
                    BusinessLogType.TRADE.value,
                    BusinessLogType.RISK.value,
                    BusinessLogType.ACCOUNT.value
                ]
            },
            'business_logging': {
                'business_log_rate': 1.0,
                'debug_log_rate': 0.1,
                'enable_correlation': True,
                'enable_trace': True
            },
            'correlation_query': {
                'max_query_history': 100,
                'query_timeout': 300
            }
        }
    }

    interface = UnifiedLoggingInterface(full_config)

    # 模拟完整的交易流程
    print("模拟完整交易流程:")

    # 1. 系统级日志 (现有系统)
    interface.log_basic("system.trading", "INFO", "交易系统启动")
    interface.log_basic("system.market_data", "INFO", "行情数据连接建立")

    # 2. 业务级日志 (增强系统)
    session_id = f"SESSION_{int(time.time())}"
    trace_id = interface.start_trace()
    correlation_id = interface.start_correlation()

    # 账户检查
    interface.log_business(
        operation="account_check",
        business_type=BusinessLogType.ACCOUNT,
        message=f"账户检查: 可用资金=1000000",
        level="INFO",
        session_id=session_id,
        available_balance=1000000,
        trace_id=trace_id,
        correlation_id=correlation_id
    )

    # 订单处理
    interface.log_business(
        operation="order_processing",
        business_type=BusinessLogType.ORDER,
        message=f"订单处理: symbol=000001.SZ, price=10.50, qty=1000",
        level="INFO",
        session_id=session_id,
        symbol="000001.SZ",
        price=10.50,
        qty=1000,
        trace_id=trace_id,
        correlation_id=correlation_id
    )

    # 风控检查
    interface.log_business(
        operation="risk_validation",
        business_type=BusinessLogType.RISK,
        message=f"风控检查: 风险评分=0.85, 通过",
        level="INFO",
        session_id=session_id,
        risk_score=0.85,
        status="passed",
        trace_id=trace_id,
        correlation_id=correlation_id
    )

    # 交易执行
    interface.log_business(
        operation="trade_execution",
        business_type=BusinessLogType.TRADE,
        message=f"交易执行: filled_qty=800, avg_price=10.52",
        level="INFO",
        session_id=session_id,
        filled_qty=800,
        avg_price=10.52,
        trace_id=trace_id,
        correlation_id=correlation_id
    )

    # 调试信息
    interface.log_debug(
        operation="market_analysis",
        message=f"市场分析: MA5=10.45, MA10=10.38, 趋势=上涨",
        session_id=session_id,
        ma5=10.45,
        ma10=10.38,
        trend="up",
        trace_id=trace_id
    )

    # 3. 系统级日志 (现有系统)
    interface.log_basic("system.trading", "INFO", "交易流程完成")
    interface.log_basic("system.performance", "INFO", "交易延迟: 15ms")

    # 4. 关联查询
    query = CorrelationQuery(
        trace_id=trace_id,
        time_range=(datetime.now() - timedelta(minutes=10), datetime.now())
    )

    result = interface.query_correlation(query)

    print(f"\n完整交易流程关联查询结果:")
    print(f"  总日志数量: {result.statistics.get('total_logs_count', 0)}")
    print(f"  业务类型分布: {result.statistics.get('business_type_distribution', {})}")
    print(f"  日志级别分布: {result.statistics.get('log_level_distribution', {})}")

    # 5. 统计信息
    stats = interface.get_statistics()
    print(f"\n系统统计信息:")
    print(f"  基础日志数量: {stats['basic_logs']}")
    print(f"  业务日志数量: {stats['business_logs']}")
    print(f"  调试日志数量: {stats['debug_logs']}")
    print(f"  关联查询次数: {stats['correlation_queries']}")

    print("✅ 集成日志系统演示完成")


def demo_global_interface():
    """演示全局日志接口"""
    print("\n=== 全局日志接口演示 ===")

    # 配置全局接口
    global_config = {
        'basic': {
            'level': 'INFO',
            'sampling': {'default_rate': 0.5}
        },
        'enhanced': {
            'sampling': {
                'default_rate': 0.3,
                'critical_business_types': [BusinessLogType.ORDER.value]
            },
            'business_logging': {
                'business_log_rate': 1.0,
                'debug_log_rate': 0.1
            }
        }
    }

    configure_logging(global_config)

    # 使用全局接口
    log_basic("global.test", "INFO", "全局基础日志测试")

    correlation_id = log_business(
        operation="global_business",
        business_type=BusinessLogType.ORDER,
        message="全局业务日志测试",
        level="INFO"
    )

    trace_id = log_debug(
        operation="global_debug",
        message="全局调试日志测试"
    )

    # 全局关联查询
    query = CorrelationQuery(
        business_type=BusinessLogType.ORDER,
        time_range=(datetime.now() - timedelta(minutes=10), datetime.now())
    )

    result = query_correlation(query)

    print(f"全局接口查询结果: {len(result.sampled_logs)} 条采样日志")

    print("✅ 全局日志接口演示完成")


def main():
    """主函数"""
    print("统一日志接口演示")
    print("=" * 60)

    try:
        # 演示系统启动日志 (现有系统)
        demo_system_startup_logging()

        # 演示业务操作日志 (增强系统)
        demo_business_operation_logging()

        # 演示关联查询功能
        demo_correlation_query()

        # 演示集成日志系统
        demo_integrated_logging()

        # 演示全局日志接口
        demo_global_interface()

        print("\n" + "=" * 60)
        print("✅ 统一日志接口演示完成")
        print("\n分工总结:")
        print("  📋 现有日志系统:")
        print("     - 系统启动、配置、性能监控日志")
        print("     - 基础采样、负载调整")
        print("     - 通用错误、异常日志")
        print("     - 结构化日志输出")
        print("  🚀 增强日志系统:")
        print("     - 关键业务操作日志 (订单、交易、风控、账户)")
        print("     - 业务操作完整追溯")
        print("     - 关联查询和问题排查")
        print("     - 合规审计支持")
        print("  🔗 统一接口:")
        print("     - 协调两个系统的使用")
        print("     - 提供统一的配置管理")
        print("     - 支持上下文管理")
        print("     - 提供统计和监控")

    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
