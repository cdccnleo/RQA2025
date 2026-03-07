#!/usr/bin/env python3
"""
测试容器中的PostgreSQL连接
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, '/app')

try:
    from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection
    from src.gateway.web.feature_task_persistence import _list_from_postgresql
    print("✅ 成功导入模块")
except Exception as e:
    print(f"❌ 导入模块失败: {e}")
    sys.exit(1)

def test_db_connection():
    """测试数据库连接"""
    print("\n📡 测试数据库连接...")
    conn = None
    try:
        conn = get_db_connection()
        if conn:
            print("✅ 数据库连接成功!")
            print(f"  连接对象: {conn}")
            print(f"  连接状态: {'打开' if not conn.closed else '关闭'}")
            return conn
        else:
            print("❌ 数据库连接失败!")
            return None
    except Exception as e:
        print(f"❌ 连接测试异常: {e}")
        return None
    finally:
        if conn and not conn.closed:
            return_db_connection(conn)
            print("🔄 连接已归还到连接池")

def test_list_tasks():
    """测试从PostgreSQL列出任务"""
    print("\n📋 测试从PostgreSQL列出任务...")
    try:
        tasks = _list_from_postgresql()
        print(f"✅ 从PostgreSQL获取任务数: {len(tasks)}")
        for task in tasks:
            print(f"  - {task['task_id']}: {task['task_type']} - {task['status']}")
        return tasks
    except Exception as e:
        print(f"❌ 列出任务异常: {e}")
        return []

def main():
    """主测试函数"""
    print("🚀 开始测试容器中的PostgreSQL连接")
    print("=" * 60)
    
    # 测试数据库连接
    conn = test_db_connection()
    
    # 测试列出任务
    tasks = test_list_tasks()
    
    print("=" * 60)
    print("📋 测试完成")

if __name__ == "__main__":
    main()
