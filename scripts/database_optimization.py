#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库查询优化脚本
"""

import time
import sqlite3
import threading
import json


class DatabaseOptimizer:
    """数据库优化器"""

    def __init__(self, db_name="test.db"):
        self.db_name = db_name
        self.connection_pool = []
        self.pool_lock = threading.Lock()

    def get_connection(self):
        """获取数据库连接"""
        with self.pool_lock:
            if self.connection_pool:
                return self.connection_pool.pop()
            return sqlite3.connect(self.db_name)

    def return_connection(self, conn):
        """返回数据库连接"""
        with self.pool_lock:
            if len(self.connection_pool) < 10:
                self.connection_pool.append(conn)
            else:
                conn.close()

    def setup_test_database(self):
        """设置测试数据库"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # 创建测试表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategies (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT,
                status TEXT,
                created_at REAL,
                data TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_items (
                id INTEGER PRIMARY KEY,
                strategy_id INTEGER,
                symbol TEXT,
                weight REAL,
                price REAL,
                FOREIGN KEY (strategy_id) REFERENCES strategies (id)
            )
        ''')

        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_strategy_status ON strategies(status)')
        cursor.execute(
            'CREATE INDEX IF NOT EXISTS idx_portfolio_strategy ON portfolio_items(strategy_id)')

        # 插入测试数据
        for i in range(1000):
            cursor.execute(
                'INSERT INTO strategies (name, type, status, created_at, data) VALUES (?, ?, ?, ?, ?)',
                (f'Strategy_{i}', f'Type_{i%5}', 'active' if i %
                 2 == 0 else 'inactive', time.time(), f'Data_{i}' * 10)
            )

        for i in range(5000):
            cursor.execute(
                'INSERT INTO portfolio_items (strategy_id, symbol, weight, price) VALUES (?, ?, ?, ?)',
                (i % 1000 + 1, f'SYMBOL_{i%100}', (i % 100)/100.0, 100 + (i % 50))
            )

        conn.commit()
        self.return_connection(conn)

    def test_query_performance(self, query_type="basic"):
        """测试查询性能"""
        conn = self.get_connection()
        cursor = conn.cursor()

        start_time = time.time()

        if query_type == "basic":
            # 基本查询
            cursor.execute("SELECT * FROM strategies WHERE status = ?", ("active",))
            results = cursor.fetchall()

        elif query_type == "join":
            # 联接查询
            cursor.execute('''
                SELECT s.name, p.symbol, p.weight
                FROM strategies s
                JOIN portfolio_items p ON s.id = p.strategy_id
                WHERE s.status = ?
            ''', ("active",))
            results = cursor.fetchall()

        elif query_type == "aggregate":
            # 聚合查询
            cursor.execute('''
                SELECT symbol, SUM(weight) as total_weight, AVG(price) as avg_price
                FROM portfolio_items
                GROUP BY symbol
                ORDER BY total_weight DESC
                LIMIT 10
            ''')
            results = cursor.fetchall()

        elif query_type == "complex":
            # 复杂查询
            cursor.execute('''
                SELECT s.name, COUNT(p.id) as item_count, SUM(p.weight) as total_weight
                FROM strategies s
                LEFT JOIN portfolio_items p ON s.id = p.strategy_id
                WHERE s.created_at > ?
                GROUP BY s.id, s.name
                HAVING COUNT(p.id) > 3
                ORDER BY total_weight DESC
            ''', (time.time() - 3600,))
            results = cursor.fetchall()

        end_time = time.time()

        self.return_connection(conn)

        return {
            "query_type": query_type,
            "execution_time": end_time - start_time,
            "result_count": len(results),
            "results": results[:5]  # 只返回前5个结果作为示例
        }

    def test_index_effectiveness(self):
        """测试索引有效性"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # 测试有索引的查询
        start_time = time.time()
        cursor.execute("SELECT * FROM strategies WHERE status = ?", ("active",))
        indexed_results = cursor.fetchall()
        indexed_time = time.time() - start_time

        # 测试无索引的查询（模拟）
        cursor.execute("SELECT * FROM strategies WHERE created_at > ?", (time.time() - 3600,))
        non_indexed_results = cursor.fetchall()
        non_indexed_time = time.time() - indexed_time - start_time

        self.return_connection(conn)

        return {
            "indexed_query_time": indexed_time,
            "non_indexed_query_time": non_indexed_time,
            "indexed_result_count": len(indexed_results),
            "non_indexed_result_count": len(non_indexed_results),
            "performance_improvement": (non_indexed_time - indexed_time) / non_indexed_time if non_indexed_time > 0 else 0
        }

    def test_connection_pooling(self):
        """测试连接池性能"""
        def worker(worker_id, results):
            for i in range(50):
                start_time = time.time()
                conn = self.get_connection()

                # 执行查询
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM strategies")
                result = cursor.fetchone()

                self.return_connection(conn)
                end_time = time.time()

                results.append(end_time - start_time)

        start_time = time.time()
        results = []

        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker, args=(i, results))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        end_time = time.time()

        return {
            "pool_size": len(self.connection_pool),
            "total_connections_created": 10 * 50,
            "total_time": end_time - start_time,
            "avg_query_time": sum(results) / len(results),
            "queries_per_second": len(results) / (end_time - start_time)
        }


def test_query_optimizations():
    """测试查询优化"""
    print("测试数据库查询优化...")

    optimizer = DatabaseOptimizer()
    optimizer.setup_test_database()

    optimization_results = {
        "test_time": time.time(),
        "query_tests": [],
        "index_tests": {},
        "connection_pool_tests": {}
    }

    # 测试不同类型的查询
    query_types = ["basic", "join", "aggregate", "complex"]
    for query_type in query_types:
        result = optimizer.test_query_performance(query_type)
        optimization_results["query_tests"].append(result)
        print(f"   {query_type}查询: {result['execution_time']:.4f}秒, 返回{result['result_count']}条记录")

    # 测试索引效果
    print("\n测试索引效果:")
    index_result = optimizer.test_index_effectiveness()
    optimization_results["index_tests"] = index_result
    print(f"   有索引查询: {index_result['indexed_query_time']:.4f}秒")
    print(f"   无索引查询: {index_result['non_indexed_query_time']:.4f}秒")
    print(f"   性能提升: {index_result['performance_improvement']:.2%}")

    # 测试连接池
    print("\n测试连接池:")
    pool_result = optimizer.test_connection_pooling()
    optimization_results["connection_pool_tests"] = pool_result
    print(f"   查询总数: {pool_result['total_connections_created']}")
    print(f"   每秒查询数: {pool_result['queries_per_second']:.1f}")
    print(f"   平均查询时间: {pool_result['avg_query_time']:.4f}秒")

    return optimization_results


def main():
    """主函数"""
    print("开始数据库查询优化测试...")

    try:
        results = test_query_optimizations()

        # 保存结果
        with open('database_optimization_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print("\n数据库查询优化测试完成，结果已保存到 database_optimization_results.json")

        return results

    except Exception as e:
        print(f"测试过程中出错: {e}")
        return None


if __name__ == '__main__':
    main()
