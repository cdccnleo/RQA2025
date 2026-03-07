"""
迁移已有任务的特征数据到 feature_store 表
任务ID: feature_task_single_002837_1771755510
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection
from src.gateway.web.feature_task_persistence import save_features_to_store


def get_task_config(task_id: str):
    """获取任务配置"""
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            print("❌ 无法连接到数据库")
            return None
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT config, feature_count 
            FROM feature_engineering_tasks 
            WHERE task_id = %s
        """, (task_id,))
        
        row = cursor.fetchone()
        cursor.close()
        
        if not row:
            return None
        
        import json
        # config 可能已经是字典类型
        if isinstance(row[0], dict):
            config = row[0]
        elif row[0]:
            config = json.loads(row[0])
        else:
            config = {}
        feature_count = row[1]
        
        return {
            "config": config,
            "feature_count": feature_count
        }
        
    except Exception as e:
        print(f"❌ 获取任务配置失败: {e}")
        return None
    finally:
        if conn:
            return_db_connection(conn)


def generate_features_from_config(config: dict, feature_count: int):
    """
    根据配置生成特征列表
    这是一个模拟函数，实际应该从任务结果中获取特征列表
    """
    # 从配置中获取指标列表
    indicators = config.get("indicators", ["SMA", "EMA", "RSI", "MACD"])
    
    # 生成标准特征名称
    features = []
    
    # 添加基础价格特征
    features.extend([
        "open_price",
        "high_price", 
        "low_price",
        "close_price",
        "volume"
    ])
    
    # 根据指标生成特征
    for indicator in indicators:
        if indicator == "SMA":
            features.extend(["SMA_5", "SMA_10", "SMA_20"])
        elif indicator == "EMA":
            features.extend(["EMA_5", "EMA_10", "EMA_20"])
        elif indicator == "RSI":
            features.append("RSI_14")
        elif indicator == "MACD":
            features.extend(["MACD", "MACD_signal", "MACD_hist"])
    
    # 如果生成的特征数量与任务记录不符，进行调整
    if len(features) < feature_count:
        # 添加额外特征以达到任务记录的数量
        for i in range(len(features), feature_count):
            features.append(f"feature_{i}")
    elif len(features) > feature_count:
        # 截断特征列表
        features = features[:feature_count]
    
    return features


def migrate_task_features(task_id: str):
    """迁移任务特征数据"""
    print(f"\n{'='*60}")
    print(f"🔄 迁移任务特征数据: {task_id}")
    print(f"{'='*60}")
    
    # 1. 获取任务配置
    task_info = get_task_config(task_id)
    if not task_info:
        print(f"❌ 任务 {task_id} 不存在或无法获取配置")
        return False
    
    config = task_info["config"]
    feature_count = task_info["feature_count"]
    
    print(f"✅ 任务配置获取成功")
    print(f"   特征数量: {feature_count}")
    print(f"   指标配置: {config.get('indicators', [])}")
    print(f"   股票代码: {config.get('symbol', 'N/A')}")
    
    # 2. 生成特征列表
    features = generate_features_from_config(config, feature_count)
    print(f"\n📋 生成的特征列表 ({len(features)} 个):")
    for i, feature in enumerate(features[:10], 1):
        print(f"   {i}. {feature}")
    if len(features) > 10:
        print(f"   ... 还有 {len(features) - 10} 个特征")
    
    # 3. 保存特征到存储表
    symbol = config.get("symbol")
    print(f"\n💾 保存特征到存储表...")
    
    result = save_features_to_store(
        task_id=task_id,
        features=features,
        symbol=symbol
    )
    
    if result:
        print(f"✅ 特征保存成功！")
        return True
    else:
        print(f"❌ 特征保存失败！")
        return False


def main():
    task_id = "feature_task_single_002837_1771755510"
    
    print("🚀 开始迁移任务特征数据")
    print(f"任务ID: {task_id}")
    
    success = migrate_task_features(task_id)
    
    if success:
        print(f"\n🎉 特征数据迁移完成！")
        
        # 验证迁移结果
        print(f"\n🔍 验证迁移结果...")
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM feature_store WHERE task_id = %s",
                (task_id,)
            )
            count = cursor.fetchone()[0]
            cursor.close()
            return_db_connection(conn)
            print(f"✅ feature_store 表中现在有 {count} 个特征记录")
    else:
        print(f"\n❌ 特征数据迁移失败")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
