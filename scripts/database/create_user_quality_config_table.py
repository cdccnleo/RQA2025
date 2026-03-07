"""
创建用户自定义评分配置表
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection


def create_user_quality_config_table():
    """创建用户自定义评分配置表"""
    print("\n" + "="*80)
    print("📝 创建用户自定义评分配置表")
    print("="*80)
    
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 检查表是否已存在
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'user_feature_quality_config'
            )
        """)
        
        table_exists = cursor.fetchone()[0]
        
        if table_exists:
            print("\n⚠️  user_feature_quality_config 表已存在")
            
            # 检查表结构
            cursor.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'user_feature_quality_config'
                ORDER BY ordinal_position
            """)
            
            columns = cursor.fetchall()
            print("\n📊 现有表结构:")
            for col_name, data_type in columns:
                print(f"   - {col_name}: {data_type}")
        else:
            # 创建表
            cursor.execute("""
                CREATE TABLE user_feature_quality_config (
                    config_id SERIAL PRIMARY KEY,
                    user_id VARCHAR(255) NOT NULL,
                    feature_name VARCHAR(255) NOT NULL,
                    custom_score FLOAT NOT NULL CHECK (custom_score >= 0 AND custom_score <= 1),
                    reason TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, feature_name)
                )
            """)
            
            # 创建索引
            cursor.execute("""
                CREATE INDEX idx_user_feature_config_user_id 
                ON user_feature_quality_config(user_id)
            """)
            
            cursor.execute("""
                CREATE INDEX idx_user_feature_config_feature 
                ON user_feature_quality_config(feature_name)
            """)
            
            cursor.execute("""
                CREATE INDEX idx_user_feature_config_user_feature 
                ON user_feature_quality_config(user_id, feature_name)
            """)
            
            conn.commit()
            print("\n✅ user_feature_quality_config 表创建成功")
            print("\n📊 表结构:")
            print("   - config_id: SERIAL PRIMARY KEY")
            print("   - user_id: VARCHAR(255)")
            print("   - feature_name: VARCHAR(255)")
            print("   - custom_score: FLOAT (0-1)")
            print("   - reason: TEXT")
            print("   - is_active: BOOLEAN")
            print("   - created_at: TIMESTAMP")
            print("   - updated_at: TIMESTAMP")
            print("\n📊 索引:")
            print("   - idx_user_feature_config_user_id")
            print("   - idx_user_feature_config_feature")
            print("   - idx_user_feature_config_user_feature")
        
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


def main():
    print("🚀 开始创建用户自定义评分配置表")
    create_user_quality_config_table()
    print("\n" + "="*80)
    print("✅ 完成")
    print("="*80)


if __name__ == "__main__":
    main()
