#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DatabaseLogger 使用示例
演示数据库操作的日志记录功能
"""

from infrastructure.logging import DatabaseLogger
import time
import random
import sys
import os
# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def simulate_database_operations():
    """模拟数据库操作日志记录"""

    # 创建数据库Logger
    db_logger = DatabaseLogger(
        name="database.operations",
        log_dir="logs/database"
    )

    print("=== 数据库Logger演示 ===")

    # 模拟不同的数据库操作
    operations = [
        {
            "type": "SELECT",
            "table": "trades",
            "query": "SELECT * FROM trades WHERE symbol = ? AND date >= ?",
            "params": ["AAPL", "2024-01-01"],
            "expected_rows": 1000
        },
        {
            "type": "INSERT",
            "table": "orders",
            "query": "INSERT INTO orders (symbol, quantity, price, user_id) VALUES (?, ?, ?, ?)",
            "params": ["GOOGL", 100, 2800.50, "user123"],
            "expected_rows": 1
        },
        {
            "type": "UPDATE",
            "table": "portfolio",
            "query": "UPDATE portfolio SET cash_balance = cash_balance - ? WHERE user_id = ?",
            "params": [280050.00, "user123"],
            "expected_rows": 1
        },
        {
            "type": "DELETE",
            "table": "expired_sessions",
            "query": "DELETE FROM sessions WHERE created_at < ?",
            "params": ["2024-01-01"],
            "expected_rows": 25
        },
        {
            "type": "SELECT",
            "table": "market_data",
            "query": "SELECT symbol, price, volume FROM market_data WHERE volume > ? ORDER BY volume DESC",
            "params": [1000000],
            "expected_rows": 50
        }
    ]

    # 连接池配置
    connection_pool = {
        "min_connections": 5,
        "max_connections": 50,
        "current_connections": 15,
        "idle_connections": 8,
        "active_connections": 7
    }

    for op in operations:
        # 模拟查询执行时间
        query_time = random.uniform(0.001, 2.0)
        rows_affected = random.randint(1, op["expected_rows"]) if op["expected_rows"] > 0 else 0

        # 模拟查询结果
        success = random.choice([True, True, True, False])  # 75%成功率

        if success:
            db_logger.info("数据库查询执行成功",
                           operation=op["type"],
                           table=op["table"],
                           query_time=round(query_time, 3),
                           rows_affected=rows_affected,
                           connection_pool_size=connection_pool["current_connections"],
                           slow_query_threshold=1.0,
                           is_slow_query=query_time > 1.0,
                           timestamp=time.time()
                           )

            if op["type"] == "SELECT" and rows_affected > 100:
                db_logger.warning("大数据集查询",
                                  table=op["table"],
                                  rows_returned=rows_affected,
                                  query_time=round(query_time, 3),
                                  optimization_suggestion="考虑添加索引或分页查询"
                                  )

            print(f"✅ {op['type']} {op['table']}: {query_time:.3f}s, {rows_affected} 行")
        else:
            error_code = random.choice(["CONNECTION_TIMEOUT", "DEADLOCK",
                                       "CONSTRAINT_VIOLATION", "LOCK_WAIT"])
            db_logger.error("数据库查询执行失败",
                            operation=op["type"],
                            table=op["table"],
                            error_code=error_code,
                            query_time=round(query_time, 3),
                            connection_pool_size=connection_pool["current_connections"],
                            timestamp=time.time()
                            )
            print(f"❌ {op['type']} {op['table']}: {error_code}")

        time.sleep(0.05)

    # 连接池监控
    db_logger.info("连接池状态",
                   pool_size=connection_pool["current_connections"],
                   active_connections=connection_pool["active_connections"],
                   idle_connections=connection_pool["idle_connections"],
                   max_connections=connection_pool["max_connections"],
                   utilization_rate=round(
                       connection_pool["active_connections"] / connection_pool["max_connections"], 2),
                   timestamp=time.time()
                   )

    # 数据库维护操作
    maintenance_ops = [
        {"type": "VACUUM", "table": "historical_data", "duration": 45.2},
        {"type": "REINDEX", "table": "trades", "duration": 12.8},
        {"type": "ANALYZE", "table": "market_data", "duration": 8.5}
    ]

    for maint in maintenance_ops:
        db_logger.info("数据库维护操作",
                       operation=maint["type"],
                       table=maint["table"],
                       duration=round(maint["duration"], 2),
                       space_reclaimed_mb=round(random.uniform(10, 500), 2),
                       timestamp=time.time()
                       )
        print(f"🔧 {maint['type']} {maint['table']}: {maint['duration']:.1f}s")

    # 备份操作
    db_logger.info("数据库备份完成",
                   backup_type="incremental",
                   backup_size_gb=round(random.uniform(5, 50), 2),
                   duration_seconds=round(random.uniform(300, 1800), 2),
                   compression_ratio=0.7,
                   timestamp=time.time()
                   )

    print("\n数据库日志记录完成")
    print(f"Logger名称: {db_logger.name}")
    print(f"日志级别: {db_logger.level}")
    print(f"日志分类: {db_logger.category}")
    print(f"日志目录: {db_logger.log_dir}")


if __name__ == "__main__":
    simulate_database_operations()
