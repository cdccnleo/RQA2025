"""
检查 feature_selection_history 表结构和数据
分析是否与 feature_store 表冗余
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection


def get_table_structure(table_name: str):
    """获取表结构信息"""
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            print(f"❌ 无法连接到数据库")
            return None
        
        cursor = conn.cursor()
        
        # 获取列信息
        cursor.execute("""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position
        """, (table_name,))
        
        columns = cursor.fetchall()
        
        # 获取约束信息
        cursor.execute("""
            SELECT tc.constraint_name, tc.constraint_type
            FROM information_schema.table_constraints tc
            WHERE tc.table_name = %s
        """, (table_name,))
        
        constraints = cursor.fetchall()
        
        # 获取索引信息
        cursor.execute("""
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE tablename = %s
        """, (table_name,))
        
        indexes = cursor.fetchall()
        
        # 获取记录数
        cursor.execute(f"""
            SELECT COUNT(*) FROM {table_name}
        """)
        
        count = cursor.fetchone()[0]
        
        cursor.close()
        
        return {
            "columns": columns,
            "constraints": constraints,
            "indexes": indexes,
            "count": count
        }
        
    except Exception as e:
        print(f"❌ 获取表结构失败: {e}")
        return None
    finally:
        if conn:
            return_db_connection(conn)


def get_sample_data(table_name: str, limit: int = 5):
    """获取样本数据"""
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            return None
        
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT * FROM {table_name} LIMIT %s
        """, (limit,))
        
        rows = cursor.fetchall()
        
        # 获取列名
        cursor.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position
        """, (table_name,))
        
        columns = [col[0] for col in cursor.fetchall()]
        
        cursor.close()
        
        return {"columns": columns, "rows": rows}
        
    except Exception as e:
        print(f"❌ 获取样本数据失败: {e}")
        return None
    finally:
        if conn:
            return_db_connection(conn)


def check_table_usage(table_name: str):
    """检查表在代码中的使用情况"""
    import subprocess
    import os
    
    project_root = os.path.join(os.path.dirname(__file__), '../..')
    
    try:
        # 使用 grep 搜索表名
        result = subprocess.run(
            ['grep', '-r', table_name, '--include=*.py', project_root],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            # 过滤掉脚本本身和 __pycache__
            filtered = [line for line in lines if '__pycache__' not in line and 'check_feature_selection_history.py' not in line]
            return filtered
        return []
    except Exception as e:
        print(f"⚠️ 搜索代码使用情况失败: {e}")
        return []


def main():
    print("=" * 80)
    print("🔍 检查 feature_selection_history 表")
    print("=" * 80)
    
    # 1. 检查表是否存在
    conn = get_db_connection()
    if not conn:
        print("❌ 无法连接到数据库")
        return
    
    cursor = conn.cursor()
    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'feature_selection_history'
        );
    """)
    
    exists = cursor.fetchone()[0]
    cursor.close()
    return_db_connection(conn)
    
    if not exists:
        print("⚠️ feature_selection_history 表不存在")
        return
    
    print("✅ feature_selection_history 表存在")
    
    # 2. 获取表结构
    print("\n" + "=" * 80)
    print("📋 feature_selection_history 表结构")
    print("=" * 80)
    
    structure = get_table_structure('feature_selection_history')
    if structure:
        print(f"\n记录数: {structure['count']}")
        
        print("\n列信息:")
        for col in structure['columns']:
            print(f"  - {col[0]}: {col[1]} (Nullable: {col[2]}, Default: {col[3]})")
        
        print("\n约束:")
        for con in structure['constraints']:
            print(f"  - {con[0]}: {con[1]}")
        
        print("\n索引:")
        for idx in structure['indexes']:
            print(f"  - {idx[0]}")
    
    # 3. 获取样本数据
    print("\n" + "=" * 80)
    print("📊 feature_selection_history 样本数据")
    print("=" * 80)
    
    sample = get_sample_data('feature_selection_history', 3)
    if sample and sample['rows']:
        print(f"\n列: {', '.join(sample['columns'])}")
        for i, row in enumerate(sample['rows'], 1):
            print(f"\n记录 {i}:")
            for col, val in zip(sample['columns'], row):
                print(f"  {col}: {val}")
    else:
        print("\n⚠️ 表中没有数据")
    
    # 4. 对比 feature_store 表
    print("\n" + "=" * 80)
    print("📋 feature_store 表结构（对比）")
    print("=" * 80)
    
    store_structure = get_table_structure('feature_store')
    if store_structure:
        print(f"\n记录数: {store_structure['count']}")
        
        print("\n列信息:")
        for col in store_structure['columns']:
            print(f"  - {col[0]}: {col[1]}")
    
    # 5. 检查代码使用情况
    print("\n" + "=" * 80)
    print("🔍 代码中使用情况")
    print("=" * 80)
    
    usage = check_table_usage('feature_selection_history')
    if usage:
        print(f"\n找到 {len(usage)} 处引用:")
        for line in usage[:10]:  # 只显示前10条
            print(f"  {line}")
        if len(usage) > 10:
            print(f"  ... 还有 {len(usage) - 10} 处引用")
    else:
        print("\n⚠️ 未在代码中找到引用")
    
    # 6. 分析冗余性
    print("\n" + "=" * 80)
    print("📊 冗余性分析")
    print("=" * 80)
    
    print("\nfeature_selection_history 用途分析:")
    print("  - 根据表名推断：记录特征选择历史")
    print("  - 可能包含：特征选择方法、选择时间、选择的特征列表等")
    print("  - 与 feature_store 的区别：")
    print("    * feature_store: 存储特征元数据（名称、类型、参数）")
    print("    * feature_selection_history: 存储特征选择操作历史")
    
    if structure and store_structure:
        print(f"\n数据量对比:")
        print(f"  - feature_selection_history: {structure['count']} 条记录")
        print(f"  - feature_store: {store_structure['count']} 条记录")
    
    print("\n结论:")
    if not usage:
        print("  ⚠️ 该表在代码中未被引用，可能是遗留表或预留表")
    elif structure and structure['count'] == 0:
        print("  ⚠️ 该表为空表，可能是预留表或已废弃")
    else:
        print("  ✅ 该表与 feature_store 功能不同，不冗余")
        print("     feature_store: 存储特征定义")
        print("     feature_selection_history: 存储特征选择历史记录")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
