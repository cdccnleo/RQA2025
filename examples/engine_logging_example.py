#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
引擎层统一日志记录器使用示例
展示如何使用统一日志记录器进行各种类型的日志记录
"""

import time
from pathlib import Path

from src.engine.logging import (
    UnifiedEngineLogger, LogContext, get_unified_logger, configure_engine_logging,
    log_operation, log_performance
)


def basic_logging_example():
    """基本日志记录示例"""
    print("=== 基本日志记录示例 ===")

    # 配置日志记录器
    config = {
        'level': 'DEBUG',
        'console_output': True,
        'log_file': 'logs/engine_example.log'
    }

    logger = UnifiedEngineLogger("example_logger", config)

    # 记录不同级别的日志
    logger.debug("这是一条调试信息")
    logger.info("这是一条信息日志")
    logger.warning("这是一条警告信息")
    logger.error("这是一条错误信息")
    logger.critical("这是一条严重错误信息")

    print("基本日志记录完成\n")


def context_logging_example():
    """上下文日志记录示例"""
    print("=== 上下文日志记录示例 ===")

    logger = get_unified_logger("context_logger")

    # 创建日志上下文
    context = LogContext(
        component="market_data_processor",
        operation="process_tick_data",
        correlation_id="req_12345",
        user_id="user_001",
        session_id="session_abc"
    )

    # 记录带上下文的日志
    logger.info("开始处理市场数据", context)

    # 模拟处理过程
    time.sleep(0.1)

    # 记录处理结果
    logger.info("市场数据处理完成", context)

    print("上下文日志记录完成\n")


def performance_logging_example():
    """性能日志记录示例"""
    print("=== 性能日志记录示例 ===")

    logger = get_unified_logger("performance_logger")

    # 记录性能指标
    logger.performance_log("数据库查询", 0.05)
    logger.performance_log("网络请求", 0.12)
    logger.performance_log("数据处理", 0.08)

    print("性能日志记录完成\n")


def business_logging_example():
    """业务日志记录示例"""
    print("=== 业务日志记录示例 ===")

    logger = get_unified_logger("business_logger")

    # 记录订单创建事件
    order_data = {
        "order_id": "ORD_20250127001",
        "symbol": "AAPL",
        "quantity": 100,
        "price": 150.25,
        "side": "BUY"
    }
    logger.business_log("order_created", order_data)

    # 记录交易执行事件
    trade_data = {
        "trade_id": "TRD_20250127001",
        "order_id": "ORD_20250127001",
        "executed_quantity": 100,
        "executed_price": 150.30,
        "commission": 1.50
    }
    logger.business_log("trade_executed", trade_data)

    print("业务日志记录完成\n")


def security_logging_example():
    """安全日志记录示例"""
    print("=== 安全日志记录示例 ===")

    logger = get_unified_logger("security_logger")

    # 记录登录尝试
    login_details = {
        "user_id": "user_001",
        "ip_address": "192.168.1.100",
        "user_agent": "Mozilla/5.0...",
        "timestamp": time.time()
    }
    logger.security_log("login_attempt", login_details)

    # 记录权限检查
    permission_details = {
        "user_id": "user_001",
        "resource": "/api/trading/orders",
        "action": "CREATE",
        "result": "GRANTED"
    }
    logger.security_log("permission_check", permission_details)

    print("安全日志记录完成\n")


@log_operation("process_market_data", "market_processor")
def operation_context_example():
    """操作上下文示例"""
    print("=== 操作上下文示例 ===")

    logger = get_unified_logger("operation_logger")

    # 在操作上下文中执行一些工作
    logger.info("开始处理市场数据")
    time.sleep(0.1)  # 模拟处理时间
    logger.info("市场数据处理完成")

    print("操作上下文示例完成\n")


@log_performance("complex_calculation", "calculator")
def performance_decorator_example():
    """性能装饰器示例"""
    print("=== 性能装饰器示例 ===")

    # 模拟复杂计算
    result = 0
    for i in range(1000000):
        result += i * 0.001

    return result


def configuration_example():
    """配置示例"""
    print("=== 配置示例 ===")

    # 配置多个组件的日志记录
    config = {
        'root': {
            'level': 'INFO'
        },
        'components': {
            'market_processor': {
                'level': 'DEBUG',
                'log_file': 'logs/market_processor.log',
                'console_output': True
            },
            'order_manager': {
                'level': 'INFO',
                'log_file': 'logs/order_manager.log',
                'console_output': False
            },
            'risk_engine': {
                'level': 'WARNING',
                'log_file': 'logs/risk_engine.log',
                'console_output': True
            }
        }
    }

    configure_engine_logging(config)

    # 使用不同组件的日志记录器
    market_logger = get_unified_logger("market_processor")
    order_logger = get_unified_logger("order_manager")
    risk_logger = get_unified_logger("risk_engine")

    market_logger.debug("市场数据处理器调试信息")
    order_logger.info("订单管理器信息")
    risk_logger.warning("风险引擎警告")

    print("配置示例完成\n")


def integration_example():
    """集成示例"""
    print("=== 集成示例 ===")

    logger = get_unified_logger("integration_logger")

    # 模拟一个完整的业务流程
    with logger.operation_context("trading_session", "trading_engine") as context:
        logger.info("开始交易会话", context)

        # 记录业务事件
        logger.business_log("session_started", {
            "session_id": context.correlation_id,
            "user_id": "user_001",
            "start_time": time.time()
        })

        # 记录性能指标
        logger.performance_log("session_initialization", 0.05)

        # 记录安全事件
        logger.security_log("session_authorized", {
            "user_id": "user_001",
            "session_id": context.correlation_id,
            "authorization_level": "FULL_TRADING"
        })

        # 模拟交易操作
        time.sleep(0.1)

        logger.info("交易会话完成", context)

    print("集成示例完成\n")


def main():
    """主函数"""
    print("引擎层统一日志记录器使用示例")
    print("=" * 50)

    # 创建日志目录
    Path("logs").mkdir(exist_ok=True)

    try:
        # 运行各种示例
        basic_logging_example()
        context_logging_example()
        performance_logging_example()
        business_logging_example()
        security_logging_example()
        operation_context_example()
        performance_decorator_example()
        configuration_example()
        integration_example()

        print("所有示例执行完成！")
        print("请查看 logs/ 目录下的日志文件")

    except Exception as e:
        print(f"示例执行出错: {e}")


if __name__ == "__main__":
    main()
