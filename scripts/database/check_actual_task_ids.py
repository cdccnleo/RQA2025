"""
检查 feature_store 表中实际存储的 task_id
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection


def check_feature_store_task_ids():
    """检查 feature_store 表中的 task_id"""
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            print("❌ 无法连接到数据库")
            return
        
        cursor = conn.cursor()
        
        # 查询所有不同的 task_id
        cursor.execute("""
            SELECT DISTINCT task_id, COUNT(*) as feature_count
            FROM feature_store
            GROUP BY task_id
            ORDER BY task_id
        """)
        
        rows = cursor.fetchall()
        
        print("=" * 80)
        print("📊 feature_store 表中的任务ID")
        print("=" * 80)
        
        if not rows:
            print("\n⚠️ feature_store 表中没有数据")
        else:
            print(f"\n找到 {len(rows)} 个不同的任务ID:\n")
            for row in rows:
                task_id = row[0]
                count = row[1]
                print(f"  Task ID: {task_id}")
                print(f"  特征数量: {count}")
                print()
        
        # 查询具体的特征
        cursor.execute("""
            SELECT task_id, feature_name, feature_type, symbol
            FROM feature_store
            ORDER BY task_id, feature_name
        """)
        
        rows = cursor.fetchall()
        
        if rows:
            print("\n" + "=" * 80)
            print("📋 特征详情")
            print("=" * 80)
            
            current_task = None
            for row in rows:
                task_id = row[0]
                feature_name = row[1]
                feature_type = row[2]
                symbol = row[3]
                
                if task_id != current_task:
                    print(f"\n任务: {task_id}")
                    current_task = task_id
                
                print(f"  - {feature_name} (类型: {feature_type}, 股票: {symbol})")
        
        cursor.close()
        
    except Exception as e:
        print(f"❌ 查询失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if conn:
            return_db_connection(conn)


def check_task_mapping():
    """检查任务ID映射关系"""
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            return
        
        cursor = conn.cursor()
        
        # 查询 feature_engineering_tasks 表
        cursor.execute("""
            SELECT task_id, status, feature_count, created_at
            FROM feature_engineering_tasks
            WHERE task_id LIKE 'feature_task_single_%'
            ORDER BY created_at DESC
            LIMIT 10
        """)
        
        rows = cursor.fetchall()
        
        print("\n" + "=" * 80)
        print("📊 feature_engineering_tasks 表中的任务")
        print("=" * 80)
        
        for row in rows:
            task_id = row[0]
            status = row[1]
            feature_count = row[2]
            created_at = row[3]
            
            print(f"\n任务ID: {task_id}")
            print(f"  状态: {status}")
            print(f"  特征数量: {feature_count}")
            print(f"  创建时间: {created_at}")
        
        cursor.close()
        
    except Exception as e:
        print(f"❌ 查询失败: {e}")
    finally:
        if conn:
            return_db_connection(conn)


if __name__ == "__main__":
    check_feature_store_task_ids()
    check_task_mapping()
