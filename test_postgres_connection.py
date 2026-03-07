#!/usr/bin/env python3
"""
测试PostgreSQL连接和特征任务保存功能
"""

import json
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection
    from src.gateway.web.feature_task_persistence import _save_to_postgresql
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

def test_save_task():
    """测试保存任务到PostgreSQL"""
    print("\n💾 测试保存任务到PostgreSQL...")
    
    # 创建测试任务
    test_task = {
        "task_id": "test_task_12345",
        "task_type": "测试任务",
        "status": "pending",
        "progress": 0,
        "feature_count": 0,
        "start_time": 1234567890,
        "config": {"test": True, "message": "测试任务"}
    }
    
    try:
        success = _save_to_postgresql(test_task)
        if success:
            print("✅ 任务保存到PostgreSQL成功!")
            print(f"  任务ID: {test_task['task_id']}")
        else:
            print("❌ 任务保存到PostgreSQL失败!")
    except Exception as e:
        print(f"❌ 保存任务异常: {e}")

def main():
    """主测试函数"""
    print("🚀 开始测试PostgreSQL连接和数据保存功能")
    print("=" * 60)
    
    # 测试数据库连接
    conn = test_db_connection()
    
    # 测试保存任务
    test_save_task()
    
    print("=" * 60)
    print("📋 测试完成")

if __name__ == "__main__":
    main()
