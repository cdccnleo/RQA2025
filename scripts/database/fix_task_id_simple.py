"""
修复 feature_store 表中错误的 task_id 映射
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection


def fix_task_ids():
    """修复任务ID"""
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            print("❌ 无法连接到数据库")
            return False
        
        cursor = conn.cursor()
        
        # 已知的映射关系（根据之前的查询结果）
        mappings = [
            ("26c209c5-dfcb-4c0f-a54a-3093ffb6c476", "feature_task_single_688702_1771756738"),
            ("63c44896-2654-4ada-ba96-664deda3f36a", "feature_task_single_002837_1771756710"),
        ]
        
        for uuid_id, business_id in mappings:
            # 更新 task_id
            cursor.execute("""
                UPDATE feature_store
                SET task_id = %s
                WHERE task_id = %s
            """, (business_id, uuid_id))
            
            updated = cursor.rowcount
            
            # 更新 feature_id
            cursor.execute("""
                UPDATE feature_store
                SET feature_id = REPLACE(feature_id, %s, %s)
                WHERE task_id = %s
            """, (uuid_id, business_id, business_id))
            
            conn.commit()
            print(f"✅ 修复: {uuid_id} -> {business_id} ({updated} 条记录)")
        
        cursor.close()
        print("\n🎉 修复完成！")
        return True
        
    except Exception as e:
        print(f"❌ 修复失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if conn:
            return_db_connection(conn)


if __name__ == "__main__":
    print("🚀 开始修复任务ID映射")
    success = fix_task_ids()
    sys.exit(0 if success else 1)
