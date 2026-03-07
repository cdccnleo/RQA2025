"""
检查数据库中的任务状态
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection


def check_db():
    """检查数据库"""
    print("\n" + "="*80)
    print("📊 数据库任务状态检查")
    print("="*80)
    
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 查询所有任务
        cursor.execute("""
            SELECT task_id, status, created_at, updated_at
            FROM feature_engineering_tasks
            ORDER BY created_at DESC
        """)
        
        tasks = cursor.fetchall()
        
        print(f"\n   数据库总任务数: {len(tasks)}")
        
        # 按状态统计
        status_counts = {}
        for task in tasks:
            status = task[1]
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print(f"\n   状态分布:")
        for status, count in sorted(status_counts.items()):
            print(f"      - {status}: {count}")
        
        # 显示所有任务
        print(f"\n   所有任务列表:")
        for i, task in enumerate(tasks, 1):
            task_id, status, created_at, updated_at = task
            print(f"      {i}. {task_id} | {status} | {created_at}")
        
        cursor.close()
        
    except Exception as e:
        print(f"❌ 查询失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if conn:
            return_db_connection(conn)


if __name__ == "__main__":
    check_db()
