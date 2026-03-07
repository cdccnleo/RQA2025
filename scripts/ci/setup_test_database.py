#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CI/CD 测试数据库设置脚本
"""

import os
import sys
import sqlite3
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import redis


def setup_sqlite_database():
    """设置SQLite测试数据库"""
    print("🔧 设置SQLite测试数据库...")

    db_path = 'test.db'
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 创建测试表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS test_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username VARCHAR(50) UNIQUE NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS test_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            symbol VARCHAR(10) NOT NULL,
            quantity INTEGER NOT NULL,
            price DECIMAL(10,2) NOT NULL,
            order_type VARCHAR(10) NOT NULL,
            status VARCHAR(20) DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES test_users (id)
        )
    ''')

    # 插入测试数据
    cursor.executemany('''
        INSERT INTO test_users (username, email) VALUES (?, ?)
    ''', [
        ('testuser1', 'test1@example.com'),
        ('testuser2', 'test2@example.com'),
        ('testuser3', 'test3@example.com')
    ])

    cursor.executemany('''
        INSERT INTO test_orders (user_id, symbol, quantity, price, order_type) VALUES (?, ?, ?, ?, ?)
    ''', [
        (1, 'AAPL', 100, 150.50, 'buy'),
        (2, 'GOOGL', 50, 2800.00, 'sell'),
        (3, 'MSFT', 75, 300.25, 'buy')
    ])

    conn.commit()
    conn.close()

    print("✅ SQLite测试数据库设置完成")


def setup_postgresql_database():
    """设置PostgreSQL测试数据库"""
    print("🔧 设置PostgreSQL测试数据库...")

    try:
        # 连接到默认数据库
        conn = psycopg2.connect(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=os.getenv('POSTGRES_PORT', '5432'),
            user=os.getenv('POSTGRES_USER', 'postgres'),
            password=os.getenv('POSTGRES_PASSWORD', 'testpassword'),
            database='postgres'
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

        cursor = conn.cursor()

        # 删除并重新创建测试数据库
        cursor.execute("DROP DATABASE IF EXISTS testdb")
        cursor.execute("CREATE DATABASE testdb")

        conn.close()

        # 连接到测试数据库
        conn = psycopg2.connect(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=os.getenv('POSTGRES_PORT', '5432'),
            user=os.getenv('POSTGRES_USER', 'postgres'),
            password=os.getenv('POSTGRES_PASSWORD', 'testpassword'),
            database='testdb'
        )

        cursor = conn.cursor()

        # 创建测试表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_orders (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES test_users(id),
                symbol VARCHAR(10) NOT NULL,
                quantity INTEGER NOT NULL,
                price DECIMAL(10,2) NOT NULL,
                order_type VARCHAR(10) NOT NULL,
                status VARCHAR(20) DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 插入测试数据
        cursor.executemany('''
            INSERT INTO test_users (username, email) VALUES (%s, %s)
        ''', [
            ('testuser1', 'test1@example.com'),
            ('testuser2', 'test2@example.com'),
            ('testuser3', 'test3@example.com')
        ])

        cursor.executemany('''
            INSERT INTO test_orders (user_id, symbol, quantity, price, order_type) VALUES (%s, %s, %s, %s, %s)
        ''', [
            (1, 'AAPL', 100, 150.50, 'buy'),
            (2, 'GOOGL', 50, 2800.00, 'sell'),
            (3, 'MSFT', 75, 300.25, 'buy')
        ])

        conn.commit()
        conn.close()

        print("✅ PostgreSQL测试数据库设置完成")

    except Exception as e:
        print(f"❌ PostgreSQL数据库设置失败: {e}")
        print("ℹ️  跳过PostgreSQL设置，使用SQLite备用方案")
        setup_sqlite_database()


def setup_redis_cache():
    """设置Redis缓存"""
    print("🔧 设置Redis缓存...")

    try:
        redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', '6379')),
            db=1,  # 使用数据库1作为测试数据库
            decode_responses=True
        )

        # 清理测试数据库
        redis_client.flushdb()

        # 设置测试数据
        redis_client.set('test:key1', 'value1', ex=3600)
        redis_client.set('test:key2', 'value2', ex=3600)

        # 设置哈希数据
        redis_client.hset('test:user:1', mapping={
            'name': 'Test User',
            'email': 'test@example.com',
            'role': 'admin'
        })

        # 设置列表数据
        redis_client.lpush('test:queue', 'item1', 'item2', 'item3')

        # 验证连接
        if redis_client.ping():
            print("✅ Redis缓存设置完成")
        else:
            print("❌ Redis连接失败")

    except Exception as e:
        print(f"❌ Redis设置失败: {e}")
        print("ℹ️  跳过Redis设置")


def setup_test_environment():
    """设置完整的测试环境"""
    print("🚀 开始设置测试环境...")

    # 设置数据库
    if os.getenv('USE_POSTGRESQL', 'false').lower() == 'true':
        setup_postgresql_database()
    else:
        setup_sqlite_database()

    # 设置缓存
    if os.getenv('USE_REDIS', 'false').lower() == 'true':
        setup_redis_cache()

    # 创建测试目录
    os.makedirs('test_logs', exist_ok=True)
    os.makedirs('test_reports', exist_ok=True)
    os.makedirs('test_cache', exist_ok=True)

    # 设置环境变量
    os.environ['TESTING'] = 'true'
    os.environ['DATABASE_URL'] = 'sqlite:///test.db'
    os.environ['REDIS_URL'] = 'redis://localhost:6379/1'
    os.environ['LOG_LEVEL'] = 'INFO'

    print("✅ 测试环境设置完成")


def main():
    """主函数"""
    try:
        setup_test_environment()
        print("🎉 CI/CD测试环境准备就绪！")
    except Exception as e:
        print(f"❌ 测试环境设置失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
