#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradingLogger 使用示例
演示交易系统的日志记录功能
"""

from infrastructure.logging import TradingLogger
import time
import random
import sys
import os
# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def simulate_trading_operations():
    """模拟交易操作日志记录"""

    # 创建交易Logger
    trading_logger = TradingLogger(
        name="trading.engine",
        log_dir="logs/trading"
    )

    print("=== 交易Logger演示 ===")

    # 模拟不同的交易操作
    operations = [
        {
            "type": "BUY",
            "symbol": "AAPL",
            "quantity": 100,
            "price": 150.25,
            "trade_id": "T001"
        },
        {
            "type": "SELL",
            "symbol": "GOOGL",
            "quantity": 50,
            "price": 2800.50,
            "trade_id": "T002"
        },
        {
            "type": "BUY",
            "symbol": "MSFT",
            "quantity": 75,
            "price": 305.80,
            "trade_id": "T003"
        }
    ]

    for op in operations:
        # 模拟交易执行
        success = random.choice([True, True, True, False])  # 75%成功率

        if success:
            trading_logger.info("交易执行成功",
                                trade_id=op["trade_id"],
                                symbol=op["symbol"],
                                side=op["type"],
                                quantity=op["quantity"],
                                price=op["price"],
                                timestamp=time.time(),
                                execution_time=random.uniform(0.001, 0.05)
                                )
            print(f"✓ 交易 {op['trade_id']} 执行成功")
        else:
            trading_logger.error("交易执行失败",
                                 trade_id=op["trade_id"],
                                 symbol=op["symbol"],
                                 side=op["type"],
                                 quantity=op["quantity"],
                                 error_code="INSUFFICIENT_FUNDS",
                                 timestamp=time.time()
                                 )
            print(f"✗ 交易 {op['trade_id']} 执行失败")

        time.sleep(0.1)  # 模拟处理时间

    # 市场数据日志
    trading_logger.info("市场数据更新",
                        symbol="AAPL",
                        bid_price=150.20,
                        ask_price=150.30,
                        volume=1000000,
                        timestamp=time.time()
                        )

    # 订单簿状态
    trading_logger.debug("订单簿快照",
                         symbol="AAPL",
                         bids=[(150.20, 100), (150.15, 200), (150.10, 150)],
                         asks=[(150.30, 150), (150.35, 100), (150.40, 200)],
                         timestamp=time.time()
                         )

    print("\n交易日志记录完成")
    print(f"Logger名称: {trading_logger.name}")
    print(f"日志级别: {trading_logger.level}")
    print(f"日志分类: {trading_logger.category}")
    print(f"日志目录: {trading_logger.log_dir}")


if __name__ == "__main__":
    simulate_trading_operations()
