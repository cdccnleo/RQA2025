"""
检查新任务的特征存储详情
分析为什么技术指标不在特征存储列表中
任务ID: feature_task_single_002837_1771757472
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection


def check_task_detail(task_id: str):
    """检查任务详情"""
    print(f"\n{'='*80}")
    print(f"🔍 检查任务详情: {task_id}")
    print(f"{'='*80}")
    
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            print("❌ 无法连接到数据库")
            return None
        
        cursor = conn.cursor()
        
        # 查询任务详情
        cursor.execute("""
            SELECT task_id, task_type, status, progress, feature_count,
                   config, data, error_message, created_at, updated_at
            FROM feature_engineering_tasks
            WHERE task_id = %s
        """, (task_id,))
        
        row = cursor.fetchone()
        
        if not row:
            print(f"❌ 任务 {task_id} 不存在")
            return None
        
        import json
        
        task_info = {
            "task_id": row[0],
            "task_type": row[1],
            "status": row[2],
            "progress": row[3],
            "feature_count": row[4],
            "config": row[5],
            "data": row[6],
            "error_message": row[7],
            "created_at": row[8],
            "updated_at": row[9]
        }
        
        print(f"✅ 任务找到")
        print(f"   任务类型: {task_info['task_type']}")
        print(f"   状态: {task_info['status']}")
        print(f"   进度: {task_info['progress']}%")
        print(f"   特征数量: {task_info['feature_count']}")
        
        # 解析 config
        config = task_info['config']
        if isinstance(config, str):
            try:
                config = json.loads(config)
            except:
                config = {}
        elif not isinstance(config, dict):
            config = {}
        
        print(f"\n📋 任务配置:")
        print(f"   股票代码 (symbol): {config.get('symbol', 'N/A')}")
        print(f"   开始日期 (start_date): {config.get('start_date', 'N/A')}")
        print(f"   结束日期 (end_date): {config.get('end_date', 'N/A')}")
        print(f"   技术指标 (indicators): {config.get('indicators', [])}")
        
        # 解析 data
        data = task_info['data']
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except:
                data = {}
        elif not isinstance(data, dict):
            data = {}
        
        if data:
            print(f"\n📋 任务数据:")
            print(f"   {json.dumps(data, indent=2, ensure_ascii=False)[:500]}")
        
        cursor.close()
        return task_info
        
    except Exception as e:
        print(f"❌ 查询失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if conn:
            return_db_connection(conn)


def check_feature_store_detail(task_id: str):
    """检查特征存储详情"""
    print(f"\n{'='*80}")
    print(f"🔍 检查特征存储详情: {task_id}")
    print(f"{'='*80}")
    
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            print("❌ 无法连接到数据库")
            return None
        
        cursor = conn.cursor()
        
        # 查询特征数量
        cursor.execute("""
            SELECT COUNT(*) FROM feature_store WHERE task_id = %s
        """, (task_id,))
        
        count = cursor.fetchone()[0]
        print(f"✅ 特征存储表中有 {count} 个特征记录")
        
        # 查询特征详情
        cursor.execute("""
            SELECT feature_id, feature_name, feature_type, parameters, 
                   symbol, quality_score, importance, created_at
            FROM feature_store
            WHERE task_id = %s
            ORDER BY feature_name
        """, (task_id,))
        
        rows = cursor.fetchall()
        cursor.close()
        
        if not rows:
            print("❌ 没有找到特征记录")
            return []
        
        features = []
        print(f"\n📋 特征列表:")
        for i, row in enumerate(rows, 1):
            feature = {
                "feature_id": row[0],
                "feature_name": row[1],
                "feature_type": row[2],
                "parameters": row[3],
                "symbol": row[4],
                "quality_score": row[5],
                "importance": row[6],
                "created_at": row[7]
            }
            features.append(feature)
            
            params_str = str(feature['parameters']) if feature['parameters'] else "N/A"
            print(f"   {i}. {feature['feature_name']}")
            print(f"      类型: {feature['feature_type'] or 'N/A'}")
            print(f"      参数: {params_str}")
            print(f"      股票: {feature['symbol'] or 'N/A'}")
        
        return features
        
    except Exception as e:
        print(f"❌ 查询失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if conn:
            return_db_connection(conn)


def analyze_missing_indicators(task_info, features):
    """分析缺失的技术指标"""
    print(f"\n{'='*80}")
    print(f"🔍 分析缺失的技术指标")
    print(f"{'='*80}")
    
    import json
    
    # 获取配置的 indicators
    config = task_info.get('config', {})
    if isinstance(config, str):
        try:
            config = json.loads(config)
        except:
            config = {}
    
    indicators = config.get('indicators', [])
    print(f"\n📋 配置的技术指标: {indicators}")
    
    # 获取实际的特征名称
    feature_names = [f['feature_name'] for f in features]
    print(f"\n📋 实际的特征名称: {feature_names}")
    
    # 分析缺失的指标
    print(f"\n🔍 分析结果:")
    
    # 基础价格特征
    basic_features = ['open', 'high', 'low', 'close', 'volume', 'date']
    basic_count = sum(1 for f in feature_names if f in basic_features)
    print(f"   基础价格特征: {basic_count} 个")
    
    # 技术指标特征
    indicator_features = []
    for indicator in indicators:
        indicator_upper = indicator.upper()
        # 检查是否有该指标的特征
        matching = [f for f in feature_names if f.startswith(indicator_upper)]
        if matching:
            indicator_features.extend(matching)
            print(f"   ✅ {indicator}: {matching}")
        else:
            print(f"   ❌ {indicator}: 未找到")
    
    print(f"\n📊 总结:")
    print(f"   配置的技术指标: {len(indicators)} 个")
    print(f"   找到的技术指标特征: {len(indicator_features)} 个")
    print(f"   基础价格特征: {basic_count} 个")
    print(f"   总特征数: {len(features)} 个")
    
    if len(indicator_features) == 0 and len(indicators) > 0:
        print(f"\n⚠️ 问题确认: 配置的技术指标没有生成对应的特征!")
        print(f"   可能原因:")
        print(f"   1. 特征引擎没有正确计算技术指标")
        print(f"   2. 数据量不足以计算技术指标（如需要一定周期）")
        print(f"   3. 特征引擎返回的 DataFrame 不包含技术指标列")


def main():
    task_id = "feature_task_single_002837_1771757472"
    
    print("🚀 开始检查新任务特征存储详情")
    print(f"任务ID: {task_id}")
    
    # 1. 查询任务详情
    task_info = check_task_detail(task_id)
    if not task_info:
        return
    
    # 2. 查询特征存储详情
    features = check_feature_store_detail(task_id)
    if features is None:
        return
    
    # 3. 分析缺失的技术指标
    analyze_missing_indicators(task_info, features)
    
    print(f"\n{'='*80}")
    print(f"📝 检查完成")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
