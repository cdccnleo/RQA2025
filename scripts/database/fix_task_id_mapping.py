"""
修复 feature_store 表中错误的 task_id 映射
将内部 UUID 映射到用户可见的任务 ID
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection


def get_task_mapping():
    """获取任务ID映射关系"""
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            return {}
        
        cursor = conn.cursor()
        
        # 查询 feature_engineering_tasks 表获取用户可见的任务ID
        cursor.execute("""
            SELECT task_id, config, created_at
            FROM feature_engineering_tasks
            WHERE task_id LIKE 'feature_task_single_%'
            ORDER BY created_at DESC
            LIMIT 20
        """)
        
        rows = cursor.fetchall()
        cursor.close()
        
        # 构建映射：根据股票代码和时间匹配
        mapping = {}
        for row in rows:
            task_id = row[0]
            config = row[1]
            created_at = row[2]
            
            # 从 config 中提取 symbol
            symbol = None
            if isinstance(config, dict):
                symbol = config.get('symbol')
            elif isinstance(config, str):
                import json
                try:
                    config_dict = json.loads(config)
                    symbol = config_dict.get('symbol')
                except:
                    pass
            
            if symbol:
                # 使用 symbol + 创建时间作为键
                key = f"{symbol}_{created_at.strftime('%Y%m%d')}"
                mapping[key] = task_id
        
        return mapping
        
    except Exception as e:
        print(f"❌ 获取任务映射失败: {e}")
        return {}
    finally:
        if conn:
            return_db_connection(conn)


def fix_feature_store_task_ids():
    """修复 feature_store 表中的 task_id"""
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            print("❌ 无法连接到数据库")
            return False
        
        cursor = conn.cursor()
        
        # 查询 feature_store 表中使用 UUID 作为 task_id 的记录
        cursor.execute("""
            SELECT DISTINCT task_id, symbol
            FROM feature_store
            WHERE task_id LIKE '%-%'
        """)
        
        uuid_tasks = cursor.fetchall()
        
        if not uuid_tasks:
            print("✅ 没有发现需要修复的 UUID 任务ID")
            return True
        
        print(f"🔍 发现 {len(uuid_tasks)} 个需要修复的 UUID 任务ID")
        
        # 获取任务映射
        task_mapping = get_task_mapping()
        
        fixed_count = 0
        for row in uuid_tasks:
            uuid_task_id = row[0]
            symbol = row[1]
            
            # 查找对应的用户可见任务ID
            # 根据 symbol 和特征数量匹配
            cursor.execute("""
                SELECT t.task_id, t.config, t.created_at
                FROM feature_engineering_tasks t
                WHERE t.task_id LIKE 'feature_task_single_%'
                AND t.feature_count = (
                    SELECT COUNT(*) FROM feature_store WHERE task_id = %s
                )
                ORDER BY t.created_at DESC
                LIMIT 1
            """, (uuid_task_id,))
            
            match = cursor.fetchone()
            if match:
                business_task_id = match[0]
                
                # 更新 feature_store 表
                cursor.execute("""
                    UPDATE feature_store
                    SET task_id = %s,
                        feature_id = REPLACE(feature_id, %s, %s)
                    WHERE task_id = %s
                """, (business_task_id, uuid_task_id, business_task_id, uuid_task_id))
                
                updated = cursor.rowcount
                conn.commit()
                
                print(f"✅ 修复任务ID: {uuid_task_id} -> {business_task_id} ({updated} 条记录)")
                fixed_count += 1
            else:
                print(f"⚠️ 未找到匹配的业务任务ID: {uuid_task_id}")
        
        cursor.close()
        
        print(f"\n🎉 修复完成！共修复 {fixed_count} 个任务ID")
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
    print("🚀 开始修复 feature_store 表中的任务ID映射")
    success = fix_feature_store_task_ids()
    sys.exit(0 if success else 1)
