"""
特征存储表迁移脚本
自动创建 feature_store 表
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection


def migrate_feature_store_table():
    """创建 feature_store 表"""
    conn = None
    try:
        print("🔗 连接到 PostgreSQL 数据库...")
        conn = get_db_connection()
        if not conn:
            print("❌ 无法连接到数据库")
            return False
        
        print("✅ 数据库连接成功")
        cursor = conn.cursor()
        
        # 创建特征存储表
        print("📝 创建 feature_store 表...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_store (
                feature_id VARCHAR(200) PRIMARY KEY,
                task_id VARCHAR(100) NOT NULL,
                feature_name VARCHAR(100) NOT NULL,
                feature_type VARCHAR(50),
                parameters JSONB,
                symbol VARCHAR(20),
                quality_score DECIMAL(5, 4),
                importance DECIMAL(5, 4),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        print("✅ feature_store 表创建成功或已存在")
        
        # 创建索引
        print("📝 创建索引...")
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_feature_store_task_id 
            ON feature_store(task_id);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_feature_store_symbol 
            ON feature_store(symbol);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_feature_store_feature_type 
            ON feature_store(feature_type);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_feature_store_created_at 
            ON feature_store(created_at DESC);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_feature_store_task_feature 
            ON feature_store(task_id, feature_name);
        """)
        print("✅ 索引创建成功")
        
        # 添加表注释
        print("📝 添加表注释...")
        cursor.execute("""
            COMMENT ON TABLE feature_store IS '特征存储表，存储特征工程任务生成的特征元数据';
        """)
        cursor.execute("""
            COMMENT ON COLUMN feature_store.feature_id IS '特征唯一标识，格式：{task_id}_{feature_name}';
        """)
        cursor.execute("""
            COMMENT ON COLUMN feature_store.task_id IS '关联的特征工程任务ID';
        """)
        cursor.execute("""
            COMMENT ON COLUMN feature_store.feature_name IS '特征名称，如 SMA_5, EMA_10';
        """)
        cursor.execute("""
            COMMENT ON COLUMN feature_store.feature_type IS '特征类型，如 SMA, EMA, RSI, MACD';
        """)
        cursor.execute("""
            COMMENT ON COLUMN feature_store.parameters IS '特征参数，JSONB格式，如 {"period": 5}';
        """)
        cursor.execute("""
            COMMENT ON COLUMN feature_store.symbol IS '股票代码';
        """)
        cursor.execute("""
            COMMENT ON COLUMN feature_store.quality_score IS '特征质量评分，0-1之间的小数';
        """)
        cursor.execute("""
            COMMENT ON COLUMN feature_store.importance IS '特征重要性，0-1之间的小数';
        """)
        cursor.execute("""
            COMMENT ON COLUMN feature_store.created_at IS '特征创建时间';
        """)
        cursor.execute("""
            COMMENT ON COLUMN feature_store.updated_at IS '特征更新时间';
        """)
        print("✅ 表注释添加成功")
        
        conn.commit()
        cursor.close()
        
        print("\n🎉 特征存储表迁移完成！")
        return True
        
    except Exception as e:
        print(f"❌ 迁移失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if conn:
            return_db_connection(conn)


if __name__ == "__main__":
    success = migrate_feature_store_table()
    sys.exit(0 if success else 1)
