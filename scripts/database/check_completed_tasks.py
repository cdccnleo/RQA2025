"""
检查completed任务的时间戳
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection


def check_completed_tasks():
    """检查completed任务"""
    print("\n" + "="*80)
    print("📊 检查 completed 任务")
    print("="*80)
    
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 查询所有任务，按created_at排序
        cursor.execute("""
            SELECT task_id, status, created_at, updated_at, start_time, end_time
            FROM feature_engineering_tasks
            ORDER BY created_at DESC
        """)
        
        tasks = cursor.fetchall()
        
        print(f"\n   所有任务（按 created_at 降序）:")
        for i, task in enumerate(tasks, 1):
            task_id, status, created_at, updated_at, start_time, end_time = task
            print(f"      {i}. {task_id}")
            print(f"         status: {status}")
            print(f"         created_at: {created_at}")
            print(f"         updated_at: {updated_at}")
            print(f"         start_time: {start_time}")
            print(f"         end_time: {end_time}")
        
        # 检查completed任务
        cursor.execute("""
            SELECT task_id, status, created_at, start_time
            FROM feature_engineering_tasks
            WHERE status = 'completed'
            ORDER BY created_at DESC
        """)
        
        completed_tasks = cursor.fetchall()
        print(f"\n   Completed 任务:")
        for task in completed_tasks:
            task_id, status, created_at, start_time = task
            print(f"      - {task_id}: created_at={created_at}, start_time={start_time}")
        
        cursor.close()
        
    except Exception as e:
        print(f"❌ 查询失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if conn:
            return_db_connection(conn)


if __name__ == "__main__":
    check_completed_tasks()
