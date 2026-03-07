"""
修复 feature_store 表外键约束问题
移除外键约束，允许独立存储特征数据
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection


def remove_foreign_key_constraint():
    """移除外键约束"""
    conn = None
    try:
        print("🔗 连接到 PostgreSQL 数据库...")
        conn = get_db_connection()
        if not conn:
            print("❌ 无法连接到数据库")
            return False
        
        print("✅ 数据库连接成功")
        cursor = conn.cursor()
        
        # 检查外键约束是否存在
        print("🔍 检查外键约束...")
        cursor.execute("""
            SELECT conname 
            FROM pg_constraint 
            WHERE conrelid = 'feature_store'::regclass 
            AND contype = 'f';
        """)
        
        fk_constraints = cursor.fetchall()
        
        if fk_constraints:
            print(f"发现 {len(fk_constraints)} 个外键约束:")
            for fk in fk_constraints:
                fk_name = fk[0]
                print(f"  - 删除外键约束: {fk_name}")
                cursor.execute(f"""
                    ALTER TABLE feature_store 
                    DROP CONSTRAINT IF EXISTS {fk_name};
                """)
            print("✅ 外键约束已删除")
        else:
            print("✅ 没有发现外键约束")
        
        conn.commit()
        cursor.close()
        
        print("\n🎉 外键约束修复完成！")
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
    success = remove_foreign_key_constraint()
    sys.exit(0 if success else 1)
