"""
创建特征质量历史记录表
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection


def create_quality_history_table():
    """创建特征质量历史记录表"""
    print("\n" + "="*80)
    print("📝 创建特征质量历史记录表")
    print("="*80)
    
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 检查表是否已存在
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'feature_quality_history'
            )
        """)
        
        table_exists = cursor.fetchone()[0]
        
        if table_exists:
            print("\n⚠️  feature_quality_history 表已存在")
            
            # 检查表结构
            cursor.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'feature_quality_history'
                ORDER BY ordinal_position
            """)
            
            columns = cursor.fetchall()
            print("\n📊 现有表结构:")
            for col_name, data_type in columns:
                print(f"   - {col_name}: {data_type}")
        else:
            # 创建表
            cursor.execute("""
                CREATE TABLE feature_quality_history (
                    history_id SERIAL PRIMARY KEY,
                    feature_id INTEGER NOT NULL REFERENCES feature_store(feature_id) ON DELETE CASCADE,
                    quality_score FLOAT NOT NULL,
                    data_quality_metrics JSONB,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 创建索引
            cursor.execute("""
                CREATE INDEX idx_quality_history_feature_id 
                ON feature_quality_history(feature_id)
            """)
            
            cursor.execute("""
                CREATE INDEX idx_quality_history_recorded_at 
                ON feature_quality_history(recorded_at)
            """)
            
            cursor.execute("""
                CREATE INDEX idx_quality_history_feature_time 
                ON feature_quality_history(feature_id, recorded_at)
            """)
            
            conn.commit()
            print("\n✅ feature_quality_history 表创建成功")
            print("\n📊 表结构:")
            print("   - history_id: SERIAL PRIMARY KEY")
            print("   - feature_id: INTEGER (外键)")
            print("   - quality_score: FLOAT")
            print("   - data_quality_metrics: JSONB")
            print("   - recorded_at: TIMESTAMP")
            print("   - created_at: TIMESTAMP")
            print("\n📊 索引:")
            print("   - idx_quality_history_feature_id")
            print("   - idx_quality_history_recorded_at")
            print("   - idx_quality_history_feature_time")
        
        cursor.close()
        
    except Exception as e:
        print(f"\n❌ 创建表失败: {e}")
        import traceback
        traceback.print_exc()
        if conn:
            conn.rollback()
    finally:
        if conn:
            return_db_connection(conn)


def migrate_existing_quality_data():
    """迁移现有的质量数据到历史表"""
    print("\n" + "="*80)
    print("📝 迁移现有质量数据到历史表")
    print("="*80)
    
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 检查feature_store表是否有数据
        cursor.execute("SELECT COUNT(*) FROM feature_store")
        count = cursor.fetchone()[0]
        
        if count == 0:
            print("\n⚠️  feature_store 表中没有数据，无需迁移")
            return
        
        # 检查是否已有历史记录
        cursor.execute("SELECT COUNT(*) FROM feature_quality_history")
        history_count = cursor.fetchone()[0]
        
        if history_count > 0:
            print(f"\n⚠️  历史表中已有 {history_count} 条记录，跳过迁移")
            return
        
        # 迁移数据
        cursor.execute("""
            INSERT INTO feature_quality_history (feature_id, quality_score, recorded_at)
            SELECT feature_id, quality_score, NOW()
            FROM feature_store
            WHERE quality_score IS NOT NULL
        """)
        
        migrated_count = cursor.rowcount
        conn.commit()
        
        print(f"\n✅ 已迁移 {migrated_count} 条质量数据到历史表")
        
        cursor.close()
        
    except Exception as e:
        print(f"\n❌ 迁移数据失败: {e}")
        import traceback
        traceback.print_exc()
        if conn:
            conn.rollback()
    finally:
        if conn:
            return_db_connection(conn)


def main():
    print("🚀 开始创建特征质量历史记录表")
    create_quality_history_table()
    migrate_existing_quality_data()
    print("\n" + "="*80)
    print("✅ 完成")
    print("="*80)


if __name__ == "__main__":
    main()
