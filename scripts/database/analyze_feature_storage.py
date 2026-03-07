"""
分析特征存储的重复存储问题
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection


def analyze_feature_storage():
    """分析特征存储情况"""
    print("\n" + "="*80)
    print("📊 特征存储重复存储分析")
    print("="*80)
    
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 1. 统计总体情况
        cursor.execute("""
            SELECT COUNT(DISTINCT task_id) as task_count,
                   COUNT(DISTINCT feature_name) as feature_type_count,
                   COUNT(*) as total_records
            FROM feature_store
        """)
        
        task_count, feature_type_count, total_records = cursor.fetchone()
        print(f"\n📊 总体统计:")
        print(f"   - 任务数量: {task_count}")
        print(f"   - 特征类型数: {feature_type_count}")
        print(f"   - 总记录数: {total_records}")
        
        # 2. 按特征名称统计重复情况
        cursor.execute("""
            SELECT feature_name, COUNT(DISTINCT task_id) as task_count,
                   COUNT(*) as record_count
            FROM feature_store
            GROUP BY feature_name
            ORDER BY task_count DESC
        """)
        
        feature_stats = cursor.fetchall()
        print(f"\n📊 特征重复存储情况:")
        print(f"{'特征名称':<25} {'涉及任务数':<12} {'记录数':<10}")
        print("-" * 50)
        
        duplicated_features = []
        for feature_name, task_count, record_count in feature_stats:
            print(f"{feature_name:<25} {task_count:<12} {record_count:<10}")
            if task_count > 1:
                duplicated_features.append((feature_name, task_count, record_count))
        
        # 3. 分析重复存储的特征
        if duplicated_features:
            print(f"\n⚠️  发现 {len(duplicated_features)} 个特征被重复存储:")
            for feature_name, task_count, record_count in duplicated_features:
                print(f"   - {feature_name}: 被 {task_count} 个任务存储")
                
                # 查看具体任务
                cursor.execute("""
                    SELECT DISTINCT task_id, symbol
                    FROM feature_store
                    WHERE feature_name = %s
                """, (feature_name,))
                
                tasks = cursor.fetchall()
                for task_id, symbol in tasks:
                    print(f"      • 任务: {task_id}, 股票: {symbol}")
        else:
            print(f"\n✅ 未发现重复存储的特征")
        
        # 4. 按任务统计
        cursor.execute("""
            SELECT task_id, symbol, COUNT(*) as feature_count
            FROM feature_store
            GROUP BY task_id, symbol
            ORDER BY task_id
        """)
        
        task_stats = cursor.fetchall()
        print(f"\n📊 按任务统计:")
        print(f"{'任务ID':<45} {'股票代码':<12} {'特征数':<10}")
        print("-" * 70)
        
        for task_id, symbol, feature_count in task_stats:
            print(f"{task_id:<45} {symbol or 'N/A':<12} {feature_count:<10}")
        
        # 5. 分析是否应该共享存储
        print(f"\n📊 存储策略分析:")
        
        # 检查是否有相同股票、相同特征被重复存储
        cursor.execute("""
            SELECT symbol, feature_name, COUNT(DISTINCT task_id) as task_count
            FROM feature_store
            WHERE symbol IS NOT NULL
            GROUP BY symbol, feature_name
            HAVING COUNT(DISTINCT task_id) > 1
            ORDER BY task_count DESC
        """)
        
        shared_features = cursor.fetchall()
        
        if shared_features:
            print(f"\n⚠️  发现相同股票、相同特征被多个任务存储:")
            print(f"{'股票代码':<12} {'特征名称':<25} {'任务数':<10}")
            print("-" * 50)
            for symbol, feature_name, task_count in shared_features[:10]:
                print(f"{symbol:<12} {feature_name:<25} {task_count:<10}")
            if len(shared_features) > 10:
                print(f"   ... 还有 {len(shared_features) - 10} 个")
        else:
            print(f"\n✅ 未发现相同股票、相同特征的重复存储")
        
        cursor.close()
        
    except Exception as e:
        print(f"\n❌ 查询失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if conn:
            return_db_connection(conn)


def propose_optimization():
    """提出优化建议"""
    print("\n" + "="*80)
    print("💡 优化建议")
    print("="*80)
    
    print("\n📋 问题分析:")
    print("   当前特征存储是按任务维度存储的，每个任务独立存储其生成的特征")
    print("   这导致相同股票、相同特征被多个任务重复存储")
    print("   随着任务数增加，存储空间会线性增长")
    
    print("\n📋 优化方案:")
    
    print("\n   方案1: 按股票维度共享存储（推荐）")
    print("   - 优点:")
    print("      • 相同股票、相同特征只存储一次")
    print("      • 节省存储空间")
    print("      • 便于特征复用")
    print("   - 缺点:")
    print("      • 需要重构存储逻辑")
    print("      • 需要处理特征版本管理")
    print("      • 任务与特征的关联关系需要调整")
    
    print("\n   方案2: 保持现状，添加去重机制")
    print("   - 优点:")
    print("      • 实现简单")
    print("      • 不影响现有逻辑")
    print("   - 缺点:")
    print("      • 存储空间浪费")
    print("      • 无法从根本上解决问题")
    
    print("\n   方案3: 混合存储策略")
    print("   - 公共特征: 按股票维度共享存储（如技术指标）")
    print("   - 私有特征: 按任务维度独立存储（如自定义特征）")
    print("   - 优点:")
    print("      • 兼顾空间效率和灵活性")
    print("   - 缺点:")
    print("      • 实现复杂度高")


if __name__ == "__main__":
    print("🚀 开始分析特征存储的重复存储问题")
    analyze_feature_storage()
    propose_optimization()
    print("\n" + "="*80)
    print("✅ 分析完成")
    print("="*80)
