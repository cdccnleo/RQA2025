"""
检查特征存储中是否包含基础价格特征
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection


# 基础价格特征列表（精确匹配）
BASIC_PRICE_FEATURES = [
    'open', 'high', 'low', 'close', 'volume', 'amount',
    '开盘价', '最高价', '最低价', '收盘价', '成交量', '成交额',
    'Open', 'High', 'Low', 'Close', 'Volume', 'Amount',
    'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'AMOUNT',
    'date', 'datetime', 'timestamp', 'trade_date'
]


def check_feature_store():
    """检查 feature_store 表中是否存在基础价格特征"""
    print("\n" + "="*80)
    print("步骤1: 检查 feature_store 表中的基础价格特征")
    print("="*80)
    
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 查询所有特征名称
        cursor.execute("""
            SELECT DISTINCT feature_name
            FROM feature_store
            ORDER BY feature_name
        """)
        
        features = cursor.fetchall()
        print(f"\n📊 特征存储表中共有 {len(features)} 个不同特征")
        
        # 检查基础价格特征
        basic_features = []
        other_features = []
        
        for row in features:
            feature_name = row[0]
            # 精确匹配基础价格特征
            if feature_name.lower() in [f.lower() for f in BASIC_PRICE_FEATURES]:
                basic_features.append(feature_name)
            else:
                other_features.append(feature_name)
        
        print(f"\n📊 基础价格特征: {len(basic_features)} 个")
        if basic_features:
            for feature in basic_features:
                print(f"   - {feature}")
        else:
            print("   ✅ 未发现基础价格特征")
        
        print(f"\n📊 其他特征: {len(other_features)} 个")
        for feature in other_features[:20]:  # 只显示前20个
            print(f"   - {feature}")
        if len(other_features) > 20:
            print(f"   ... 还有 {len(other_features) - 20} 个特征")
        
        # 统计基础价格特征的数量
        if basic_features:
            cursor.execute("""
                SELECT COUNT(*) as count
                FROM feature_store
                WHERE feature_name = ANY(%s)
            """, (basic_features,))
            
            count = cursor.fetchone()[0]
            print(f"\n⚠️  基础价格特征记录总数: {count}")
        
        cursor.close()
        return basic_features, other_features
        
    except Exception as e:
        print(f"\n❌ 查询失败: {e}")
        import traceback
        traceback.print_exc()
        return [], []
    finally:
        if conn:
            return_db_connection(conn)


def check_feature_by_task():
    """按任务检查特征分布"""
    print("\n" + "="*80)
    print("步骤2: 按任务检查特征分布")
    print("="*80)
    
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 查询每个任务的特征数量
        cursor.execute("""
            SELECT task_id, COUNT(*) as feature_count,
                   COUNT(CASE WHEN feature_name = ANY(%s) THEN 1 END) as basic_count
            FROM feature_store
            GROUP BY task_id
            ORDER BY task_id
        """, (BASIC_PRICE_FEATURES,))
        
        tasks = cursor.fetchall()
        
        print(f"\n📊 共有 {len(tasks)} 个任务包含特征数据")
        print(f"\n{'任务ID':<40} {'总特征数':<10} {'基础价格特征数':<15}")
        print("-" * 70)
        
        for task_id, total_count, basic_count in tasks:
            print(f"{task_id:<40} {total_count:<10} {basic_count:<15}")
        
        cursor.close()
        
    except Exception as e:
        print(f"\n❌ 查询失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if conn:
            return_db_connection(conn)


def main():
    print("🚀 开始检查特征存储中的基础价格特征")
    
    basic_features, other_features = check_feature_store()
    check_feature_by_task()
    
    print("\n" + "="*80)
    if basic_features:
        print("⚠️  检查结果: 发现基础价格特征，需要清理")
    else:
        print("✅ 检查结果: 未发现基础价格特征")
    print("="*80)


if __name__ == "__main__":
    main()
