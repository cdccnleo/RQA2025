"""
清理特征存储中的基础价格特征
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection


# 基础价格特征列表
BASIC_PRICE_FEATURES = [
    'open', 'high', 'low', 'close', 'volume', 'amount',
    '开盘价', '最高价', '最低价', '收盘价', '成交量', '成交额',
    'Open', 'High', 'Low', 'Close', 'Volume', 'Amount',
    'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'AMOUNT'
]


def cleanup_basic_features():
    """清理基础价格特征"""
    print("\n" + "="*80)
    print("🧹 清理特征存储中的基础价格特征")
    print("="*80)
    
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 查询要删除的基础价格特征
        cursor.execute("""
            SELECT feature_id, task_id, feature_name
            FROM feature_store
            WHERE feature_name = ANY(%s)
        """, (BASIC_PRICE_FEATURES,))
        
        features_to_delete = cursor.fetchall()
        
        if not features_to_delete:
            print("\n✅ 未发现基础价格特征，无需清理")
            return
        
        print(f"\n📊 发现 {len(features_to_delete)} 个基础价格特征需要清理:")
        for feature_id, task_id, feature_name in features_to_delete[:10]:
            print(f"   - {feature_name} (任务: {task_id})")
        if len(features_to_delete) > 10:
            print(f"   ... 还有 {len(features_to_delete) - 10} 个")
        
        # 删除基础价格特征
        cursor.execute("""
            DELETE FROM feature_store
            WHERE feature_name = ANY(%s)
        """, (BASIC_PRICE_FEATURES,))
        
        deleted_count = cursor.rowcount
        conn.commit()
        
        print(f"\n✅ 成功删除 {deleted_count} 个基础价格特征记录")
        
        # 更新相关任务的 feature_count
        cursor.execute("""
            SELECT DISTINCT task_id
            FROM feature_store
        """)
        
        remaining_tasks = cursor.fetchall()
        print(f"\n📊 清理后还有 {len(remaining_tasks)} 个任务包含特征数据")
        
        for task_row in remaining_tasks:
            task_id = task_row[0]
            cursor.execute("""
                SELECT COUNT(*) FROM feature_store WHERE task_id = %s
            """, (task_id,))
            count = cursor.fetchone()[0]
            print(f"   - 任务 {task_id}: {count} 个特征")
        
        cursor.close()
        
    except Exception as e:
        print(f"\n❌ 清理失败: {e}")
        import traceback
        traceback.print_exc()
        if conn:
            conn.rollback()
    finally:
        if conn:
            return_db_connection(conn)


def main():
    print("🚀 开始清理特征存储中的基础价格特征")
    cleanup_basic_features()
    print("\n" + "="*80)
    print("✅ 清理完成")
    print("="*80)


if __name__ == "__main__":
    main()
