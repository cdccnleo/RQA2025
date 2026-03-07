"""
检查最新的特征提取任务
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection


def check_latest_task():
    """检查最新的特征提取任务"""
    print(f"\n{'='*80}")
    print(f"🔍 检查最新的特征提取任务")
    print(f"{'='*80}")
    
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            print("❌ 无法连接到数据库")
            return None
        
        cursor = conn.cursor()
        
        # 查询最新的5个任务
        cursor.execute("""
            SELECT task_id, status, feature_count, config, created_at, updated_at
            FROM feature_engineering_tasks
            ORDER BY created_at DESC
            LIMIT 5
        """)
        
        rows = cursor.fetchall()
        
        if not rows:
            print("\n❌ 没有找到任何任务")
            return None
        
        import json
        
        print(f"\n📋 最新的5个特征提取任务:")
        for i, row in enumerate(rows, 1):
            task_id = row[0]
            status = row[1]
            feature_count = row[2]
            config = row[3]
            created_at = row[4]
            updated_at = row[5]
            
            # 解析 config
            if isinstance(config, str):
                try:
                    config = json.loads(config)
                except:
                    config = {}
            
            symbol = config.get('symbol', 'N/A')
            indicators = config.get('indicators', [])
            task_id_prefix = config.get('task_id_prefix', 'N/A')
            
            print(f"\n   {i}. 任务ID: {task_id}")
            print(f"      状态: {status}")
            print(f"      特征数量: {feature_count}")
            print(f"      股票代码: {symbol}")
            print(f"      配置指标: {indicators}")
            print(f"      任务ID前缀: {task_id_prefix}")
            print(f"      创建时间: {created_at}")
            print(f"      更新时间: {updated_at}")
            
            # 查询特征存储
            cursor.execute("""
                SELECT feature_name, feature_type
                FROM feature_store
                WHERE task_id = %s
                ORDER BY feature_name
            """, (task_id,))
            
            feature_rows = cursor.fetchall()
            
            if feature_rows:
                feature_names = [r[0] for r in feature_rows]
                
                # 检查基础价格特征
                basic_price_features = ['open', 'high', 'low', 'close', 'volume', 'date']
                basic_found = [f for f in feature_names if f in basic_price_features]
                
                print(f"      特征列表 ({len(feature_names)} 个):")
                print(f"         {feature_names}")
                
                if basic_found:
                    print(f"      ⚠️  包含基础价格/日期特征: {basic_found}")
                else:
                    print(f"      ✅ 不包含基础价格/日期特征")
            else:
                print(f"      ❌ 特征存储表中没有数据")
        
        cursor.close()
        
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if conn:
            return_db_connection(conn)


if __name__ == "__main__":
    check_latest_task()
