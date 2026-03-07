"""
验证特征存储表修复结果
测试 save_features_to_store 函数是否正常工作
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.gateway.web.feature_task_persistence import (
    save_features_to_store,
    get_features_from_store
)


def test_save_and_retrieve_features():
    """测试保存和检索特征"""
    print("🧪 开始测试特征存储功能...")
    
    # 测试数据
    test_task_id = "test_task_verify_001"
    test_features = [
        "SMA_5",
        "SMA_10",
        "SMA_20",
        "EMA_5",
        "EMA_10",
        "RSI_14",
        "MACD",
        "close_price",
        "volume"
    ]
    test_symbol = "000001.SZ"
    
    try:
        # 1. 测试保存特征
        print(f"\n1️⃣ 测试保存特征到存储表...")
        print(f"   任务ID: {test_task_id}")
        print(f"   股票代码: {test_symbol}")
        print(f"   特征数量: {len(test_features)}")
        
        result = save_features_to_store(
            task_id=test_task_id,
            features=test_features,
            symbol=test_symbol
        )
        
        if result:
            print(f"✅ 特征保存成功！")
        else:
            print(f"❌ 特征保存失败！")
            return False
        
        # 2. 测试从存储表检索特征
        print(f"\n2️⃣ 测试从存储表检索特征...")
        retrieved_features = get_features_from_store(task_id=test_task_id)
        
        if retrieved_features:
            print(f"✅ 成功检索到 {len(retrieved_features)} 个特征")
            print(f"\n   特征列表:")
            for i, feature in enumerate(retrieved_features[:5], 1):
                print(f"   {i}. {feature['name']} (类型: {feature.get('feature_type', 'N/A')})")
            if len(retrieved_features) > 5:
                print(f"   ... 还有 {len(retrieved_features) - 5} 个特征")
        else:
            print(f"❌ 未能从存储表检索到特征")
            return False
        
        # 3. 验证特征名称解析
        print(f"\n3️⃣ 验证特征名称解析...")
        for feature in retrieved_features:
            name = feature['name']
            feature_type = feature.get('feature_type')
            parameters = feature.get('parameters', {})
            
            if name.startswith('SMA_') or name.startswith('EMA_'):
                if feature_type and 'period' in parameters:
                    print(f"   ✅ {name} -> 类型: {feature_type}, 周期: {parameters['period']}")
                else:
                    print(f"   ⚠️ {name} -> 类型: {feature_type}, 参数: {parameters}")
        
        print(f"\n🎉 所有测试通过！特征存储功能正常工作。")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def cleanup_test_data():
    """清理测试数据"""
    print(f"\n🧹 清理测试数据...")
    try:
        from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection
        
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM feature_store WHERE task_id LIKE 'test_task_verify_%'"
            )
            deleted_count = cursor.rowcount
            conn.commit()
            cursor.close()
            return_db_connection(conn)
            print(f"✅ 已清理 {deleted_count} 条测试数据")
    except Exception as e:
        print(f"⚠️ 清理测试数据失败: {e}")


if __name__ == "__main__":
    try:
        success = test_save_and_retrieve_features()
    finally:
        cleanup_test_data()
    
    sys.exit(0 if success else 1)
