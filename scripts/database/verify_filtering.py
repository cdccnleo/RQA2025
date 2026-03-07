"""
验证基础价格特征过滤和date特征评估
任务ID: task_1771762793
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection


def verify_filtering(task_id: str):
    """验证基础价格特征过滤"""
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
        
        # 查询任务详情
        cursor.execute("""
            SELECT task_id, status, feature_count, config, error_message, created_at, updated_at
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
            "config": row[3],
            "error_message": row[4],
            "created_at": row[5],
            "updated_at": row[6]
        }
        
        print(f"\n📋 任务基本信息:")
        print(f"   任务ID: {task_info['task_id']}")
        print(f"   状态: {task_info['status']}")
        print(f"   特征数量: {task_info['feature_count']}")
        print(f"   创建时间: {task_info['created_at']}")
        print(f"   更新时间: {task_info['updated_at']}")
        
        if task_info['error_message']:
            print(f"   ⚠️ 错误信息: {task_info['error_message']}")
        
        # 解析 config
        config = task_info['config']
        if isinstance(config, str):
            try:
                config = json.loads(config)
            except:
                config = {}
        
        indicators = config.get('indicators', [])
        symbol = config.get('symbol', 'N/A')
        
        print(f"\n📊 任务配置:")
        print(f"   股票代码: {symbol}")
        print(f"   配置指标: {indicators}")
        print(f"   指标数量: {len(indicators)}")
        
        # 查询特征存储
        cursor.execute("""
            SELECT feature_name, feature_type, parameters, symbol, quality_score, importance
            FROM feature_store
            WHERE task_id = %s
            ORDER BY feature_name
        """, (task_id,))
        
        rows = cursor.fetchall()
        
        if not rows:
            print("\n❌ 特征存储表中没有数据")
            return None
        
        print(f"\n✅ 特征存储表中有 {len(rows)} 个特征")
        
        features = []
        feature_names = []
        
        print(f"\n📊 特征详细列表:")
        for i, row in enumerate(rows, 1):
            feature = {
                "feature_name": row[0],
                "feature_type": row[1],
                "parameters": row[2],
                "symbol": row[3],
                "quality_score": row[4],
                "importance": row[5]
            }
            features.append(feature)
            feature_names.append(feature['feature_name'])
            
            params_str = str(feature['parameters']) if feature['parameters'] else "N/A"
            quality_str = f"{feature['quality_score']:.2f}" if feature['quality_score'] else "N/A"
            importance_str = f"{feature['importance']:.2f}" if feature['importance'] else "N/A"
            
            print(f"   {i}. {feature['feature_name']}")
            print(f"      类型: {feature['feature_type'] or 'N/A'}")
            print(f"      参数: {params_str}")
            print(f"      质量: {quality_str}")
            print(f"      重要性: {importance_str}")
        
        # 分析特征类型
        basic_price_features = ['open', 'high', 'low', 'close', 'volume']
        basic_features_found = [f for f in feature_names if f in basic_price_features]
        
        # 检查是否包含date特征
        date_features = [f for f in feature_names if 'date' in f.lower()]
        
        # 技术指标特征
        indicator_features = []
        indicator_mapping = {}
        
        for indicator in indicators:
            indicator_lower = indicator.lower()
            matching = [f for f in feature_names if indicator_lower in f.lower()]
            if matching:
                indicator_features.extend(matching)
                indicator_mapping[indicator] = matching
        
        # 其他特征
        other_features = [f for f in feature_names 
                         if f not in basic_features_found and f not in indicator_features and f not in date_features]
        
        print(f"\n{'='*80}")
        print(f"📊 特征分类统计")
        print(f"{'='*80}")
        
        print(f"\n📈 基础价格特征 ({len(basic_features_found)} 个):")
        if basic_features_found:
            for feat in basic_features_found:
                print(f"   ❌ {feat} (不应存在)")
        else:
            print("   ✅ 已正确过滤")
        
        print(f"\n📈 日期特征 ({len(date_features)} 个):")
        if date_features:
            for feat in date_features:
                print(f"   ⚠️  {feat}")
        else:
            print("   ✅ 无日期特征")
        
        print(f"\n📈 技术指标特征 ({len(indicator_features)} 个):")
        if indicator_features:
            for indicator, feats in indicator_mapping.items():
                print(f"   ✅ {indicator}: {feats}")
        else:
            print("   ❌ 未找到技术指标特征")
        
        if other_features:
            print(f"\n📈 其他特征 ({len(other_features)} 个):")
            for feat in other_features:
                print(f"   - {feat}")
        
        # date特征评估
        print(f"\n{'='*80}")
        print(f"📊 date特征评估")
        print(f"{'='*80}")
        
        print(f"\n🤔 date特征是否应作为特征的评估:")
        print(f"   在量化交易系统中，date（日期）通常是:")
        print(f"   ✅ 时间序列索引 - 用于按时间排序和组织数据")
        print(f"   ⚠️  通常不作为模型输入特征 - 因为日期本身是标识符，不是预测信号")
        print(f"   ✅ 可以提取时间特征 - 如星期几、月份、季度等，这些可以作为特征")
        print(f"   ❌ 原始日期字符串 - 不应直接作为数值特征输入模型")
        
        print(f"\n   当前任务的date特征:")
        if date_features:
            print(f"   ⚠️  包含 {len(date_features)} 个日期特征")
            print(f"   建议：")
            print(f"   1. 如果date是原始日期字符串，建议移除")
            print(f"   2. 如果需要时间信息，建议提取为数值特征（如 dayofweek, month, quarter）")
            print(f"   3. date可以作为DataFrame的索引，而不是特征列")
        else:
            print(f"   ✅ 不包含日期特征（符合量化交易系统最佳实践）")
        
        # 合规性评估
        compliance_score = 0.0
        checks = []
        
        # 检查1: 任务状态
        if task_info['status'] == 'completed':
            compliance_score += 0.2
            checks.append("✅ 任务状态: completed")
        else:
            checks.append("❌ 任务状态: 未完成")
        
        # 检查2: 基础价格特征过滤
        if len(basic_features_found) == 0:
            compliance_score += 0.2
            checks.append("✅ 基础价格特征过滤: 通过")
        else:
            checks.append(f"❌ 基础价格特征过滤: 失败，仍有 {len(basic_features_found)} 个")
        
        # 检查3: 技术指标生成
        if len(indicator_features) > 0:
            compliance_score += 0.3
            checks.append(f"✅ 技术指标生成: {len(indicator_features)} 个")
        else:
            checks.append("❌ 技术指标生成: 失败")
        
        # 检查4: 指标覆盖率
        coverage = len(indicator_mapping) / len(indicators) if indicators else 0
        if coverage >= 0.8:
            compliance_score += 0.15
            checks.append(f"✅ 指标覆盖率: {coverage*100:.1f}%")
        elif coverage >= 0.5:
            compliance_score += 0.1
            checks.append(f"⚠️ 指标覆盖率: {coverage*100:.1f}%")
        else:
            checks.append(f"❌ 指标覆盖率: {coverage*100:.1f}%")
        
        # 检查5: 日期特征
        if len(date_features) == 0:
            compliance_score += 0.15
            checks.append("✅ 日期特征: 无")
        else:
            checks.append(f"⚠️ 日期特征: 有 {len(date_features)} 个")
        
        print(f"\n📊 合规性检查:")
        for check in checks:
            print(f"   {check}")
        
        print(f"\n📊 总体合规性评分: {compliance_score*100:.1f}%")
        
        if compliance_score >= 0.8:
            print("🎉 优秀 - 完全符合量化交易系统要求")
        elif compliance_score >= 0.6:
            print("⚠️ 良好 - 基本符合要求，但有改进空间")
        else:
            print("❌ 不合格 - 不符合量化交易系统要求")
        
        # 建议
        print(f"\n{'='*80}")
        print(f"💡 改进建议")
        print(f"{'='*80}")
        
        if len(basic_features_found) > 0:
            print("❌ 严重问题：基础价格特征过滤失败")
            print("   建议：检查过滤逻辑是否正确执行")
        else:
            print("✅ 基础价格特征已正确过滤")
        
        if len(date_features) > 0:
            print("⚠️ 建议：考虑移除或转换日期特征")
            print("   理由：")
            print("   1. 原始日期字符串不应直接作为数值特征")
            print("   2. 日期可以作为DataFrame索引")
            print("   3. 如果需要时间信息，建议提取为数值特征（dayofweek, month等）")
        
        cursor.close()
        
        return {
            "task_info": task_info,
            "total_features": len(rows),
            "basic_features": len(basic_features_found),
            "date_features": len(date_features),
            "indicator_features": len(indicator_features),
            "indicators_configured": len(indicators),
            "indicators_generated": len(indicator_mapping),
            "indicator_mapping": indicator_mapping,
            "compliance_score": compliance_score,
            "success": len(basic_features_found) == 0 and len(indicator_features) > 0
        }
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if conn:
            return_db_connection(conn)


def main():
    task_id = "task_1771762793"
    
    print("🚀 开始验证基础价格特征过滤和date特征评估")
    print(f"任务ID: {task_id}")
    
    # 验证任务
    result = verify_filtering(task_id)
    
    if not result:
        print("\n❌ 验证失败")
        return
    
    # 最终总结
    print(f"\n{'='*80}")
    print(f"🎯 验证总结")
    print(f"{'='*80}")
    
    print(f"\n任务ID: {task_id}")
    print(f"任务状态: {result['task_info']['status']}")
    print(f"总特征数: {result['total_features']}")
    print(f"基础价格特征: {result['basic_features']} 个")
    print(f"日期特征: {result['date_features']} 个")
    print(f"技术指标特征: {result['indicator_features']} 个")
    print(f"配置指标数: {result['indicators_configured']} 个")
    print(f"成功生成指标: {result['indicators_generated']} 个")
    print(f"合规性评分: {result['compliance_score']*100:.1f}%")
    
    if result['success']:
        print(f"\n🎉🎉🎉 基础价格特征过滤成功！🎉🎉🎉")
        print(f"   ✅ 基础价格特征已正确过滤")
        print(f"   ✅ 只保留技术指标特征")
        print(f"   ✅ 符合量化交易系统要求")
    else:
        print(f"\n⚠️ 过滤可能未完全生效")
        if result['basic_features'] > 0:
            print(f"   ❌ 仍有 {result['basic_features']} 个基础价格特征")
        if result['indicator_features'] == 0:
            print(f"   ❌ 未找到技术指标特征")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
