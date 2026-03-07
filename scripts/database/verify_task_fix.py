"""
验证新任务特征存储修复结果
任务ID: feature_task_single_002837_1771757793
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection


def check_task_and_features(task_id: str):
    """检查任务和特征"""
    print(f"\n{'='*80}")
    print(f"🔍 验证任务: {task_id}")
    print(f"{'='*80}")
    
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            print("❌ 无法连接到数据库")
            return None
        
        cursor = conn.cursor()
        
        # 1. 查询任务详情
        cursor.execute("""
            SELECT task_id, status, feature_count, config
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
            "status": row[1],
            "feature_count": row[2],
            "config": row[3]
        }
        
        print(f"✅ 任务状态: {task_info['status']}")
        print(f"   特征数量: {task_info['feature_count']}")
        
        # 解析 config
        config = task_info['config']
        if isinstance(config, str):
            try:
                config = json.loads(config)
            except:
                config = {}
        
        indicators = config.get('indicators', [])
        print(f"   配置指标: {indicators}")
        
        # 2. 查询特征存储
        cursor.execute("""
            SELECT feature_name, feature_type, parameters
            FROM feature_store
            WHERE task_id = %s
            ORDER BY feature_name
        """, (task_id,))
        
        rows = cursor.fetchall()
        
        if not rows:
            print("\n❌ 特征存储表中没有数据")
            return None
        
        print(f"\n✅ 特征存储表中有 {len(rows)} 个特征")
        
        # 分析特征
        feature_names = [row[0] for row in rows]
        
        # 基础特征
        basic_features = ['open', 'high', 'low', 'close', 'volume', 'date']
        basic_count = sum(1 for f in feature_names if f in basic_features)
        
        # 技术指标特征
        indicator_features = []
        for indicator in indicators:
            indicator_upper = indicator.upper()
            matching = [f for f in feature_names if f.startswith(indicator_upper)]
            if matching:
                indicator_features.extend(matching)
        
        print(f"\n📊 特征分析:")
        print(f"   基础价格特征: {basic_count} 个")
        print(f"   技术指标特征: {len(indicator_features)} 个")
        
        if indicator_features:
            print(f"\n✅ 成功生成的技术指标:")
            for feat in indicator_features[:10]:
                print(f"   - {feat}")
            if len(indicator_features) > 10:
                print(f"   ... 还有 {len(indicator_features) - 10} 个")
        else:
            print(f"\n❌ 未找到技术指标特征!")
        
        print(f"\n📋 所有特征列表:")
        for i, row in enumerate(rows, 1):
            feat_name = row[0]
            feat_type = row[1] or 'N/A'
            params = str(row[2]) if row[2] else 'N/A'
            print(f"   {i}. {feat_name} (类型: {feat_type})")
        
        cursor.close()
        
        # 返回结果
        return {
            "task_info": task_info,
            "total_features": len(rows),
            "basic_features": basic_count,
            "indicator_features": len(indicator_features),
            "indicators_configured": len(indicators),
            "success": len(indicator_features) > 0
        }
        
    except Exception as e:
        print(f"❌ 查询失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if conn:
            return_db_connection(conn)


def main():
    task_id = "feature_task_single_002837_1771757793"
    
    print("🚀 开始验证新任务特征存储修复结果")
    
    result = check_task_and_features(task_id)
    
    print(f"\n{'='*80}")
    print(f"📊 验证总结")
    print(f"{'='*80}")
    
    if result:
        if result["success"]:
            print(f"\n🎉 修复成功！")
            print(f"   ✅ 任务状态: {result['task_info']['status']}")
            print(f"   ✅ 总特征数: {result['total_features']}")
            print(f"   ✅ 技术指标: {result['indicator_features']} 个")
            print(f"\n   新创建的任务正确生成了技术指标特征！")
        else:
            print(f"\n⚠️ 修复可能未生效")
            print(f"   任务状态: {result['task_info']['status']}")
            print(f"   总特征数: {result['total_features']}")
            print(f"   技术指标: {result['indicator_features']} 个 (配置: {result['indicators_configured']} 个)")
            print(f"\n   建议: 检查 Docker 容器是否使用最新代码")
    else:
        print(f"\n❌ 验证失败，无法获取任务信息")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
