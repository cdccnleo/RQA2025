"""
调试数据库连接
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection


def debug_db_connection():
    """调试数据库连接"""
    print("\n" + "="*80)
    print("🔍 调试数据库连接")
    print("="*80)
    
    # 获取数据库连接
    print("\n📋 获取数据库连接:")
    conn = get_db_connection()
    
    if conn:
        print("   ✅ 数据库连接成功")
        
        # 执行简单查询
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM feature_engineering_tasks")
            count = cursor.fetchone()[0]
            print(f"   ✅ 任务表记录数: {count}")
            cursor.close()
        except Exception as e:
            print(f"   ❌ 查询失败: {e}")
        finally:
            return_db_connection(conn)
    else:
        print("   ❌ 数据库连接失败")


if __name__ == "__main__":
    debug_db_connection()
