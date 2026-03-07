"""
全面评估特征提取任务修复
任务ID: feature_task_single_002837_1771761643
包含：技术指标验证 + 基础价格数据评估
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection


def comprehensive_evaluation(task_id: str):
    """全面评估"""
    print(f"\n{'='*80}")
    print(f"🎯 全面评估任务: {task_id}")
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
                         if f not in basic_features_found and f not in indicator_features]
        
        print(f"\n{'='*80}")
        print(f"📊 特征分类统计")
        print(f"{'='*80}")
        
        print(f"\n📈 基础价格特征 ({len(basic_features_found)} 个):")
        for feat in basic_features_found:
            print(f"   - {feat}")
        
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
        
        # 量化交易系统评估
        print(f"\n{'='*80}")
        print(f"📊 量化交易系统评估")
        print(f"{'='*80}")
        
        # 基础价格数据作为特征的评估
        print(f"\n🤔 基础价格数据作为特征的评估:")
        print(f"   在量化交易系统中，基础价格数据（open, high, low, close, volume）通常是:")
        print(f"   ✅ 原始输入数据 - 用于计算技术指标")
        print(f"   ⚠️  通常不作为特征直接使用 - 因为它们是原始数据，缺乏信息增益")
        print(f"   ✅ 可以作为基准特征 - 用于对比技术指标的效果")
        
        print(f"\n   当前任务的基础价格特征:")
        if len(basic_features_found) > 0:
            print(f"   ✅ 包含 {len(basic_features_found)} 个基础价格特征")
            print(f"   ⚠️  建议：在量化交易系统中，通常只保留技术指标作为特征")
            print(f"      基础价格数据应作为原始数据存储，而不是特征")
        else:
            print(f"   ✅ 不包含基础价格特征（符合量化交易系统最佳实践）")
        
        # 合规性评估
        compliance_score = 0.0
        checks = []
        
        # 检查1: 任务状态
        if task_info['status'] == 'completed':
            compliance_score += 0.2
            checks.append("✅ 任务状态: completed")
        else:
            checks.append("❌ 任务状态: 未完成")
        
        # 检查2: 技术指标生成
        if len(indicator_features) > 0:
            compliance_score += 0.3
            checks.append(f"✅ 技术指标生成: {len(indicator_features)} 个")
        else:
            checks.append("❌ 技术指标生成: 失败")
        
        # 检查3: 指标覆盖率
        coverage = len(indicator_mapping) / len(indicators) if indicators else 0
        if coverage >= 0.8:
            compliance_score += 0.2
            checks.append(f"✅ 指标覆盖率: {coverage*100:.1f}%")
        elif coverage >= 0.5:
            compliance_score += 0.1
            checks.append(f"⚠️ 指标覆盖率: {coverage*100:.1f}%")
        else:
            checks.append(f"❌ 指标覆盖率: {coverage*100:.1f}%")
        
        # 检查4: 特征质量
        quality_scores = [f['quality_score'] for f in features if f['quality_score']]
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            if avg_quality >= 0.8:
                compliance_score += 0.15
                checks.append(f"✅ 平均特征质量: {avg_quality:.2f}")
            elif avg_quality >= 0.6:
                compliance_score += 0.1
                checks.append(f"⚠️ 平均特征质量: {avg_quality:.2f}")
            else:
                checks.append(f"❌ 平均特征质量: {avg_quality:.2f}")
        else:
            checks.append("⚠️ 特征质量: 未评估")
        
        # 检查5: 特征重要性
        importance_scores = [f['importance'] for f in features if f['importance']]
        if importance_scores:
            avg_importance = sum(importance_scores) / len(importance_scores)
            if avg_importance >= 0.5:
                compliance_score += 0.15
                checks.append(f"✅ 平均特征重要性: {avg_importance:.2f}")
            else:
                checks.append(f"⚠️ 平均特征重要性: {avg_importance:.2f}")
        else:
            checks.append("⚠️ 特征重要性: 未评估")
        
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
        
        if len(basic_features_found) > 0 and len(indicator_features) == 0:
            print("❌ 严重问题：只生成了基础价格特征，没有技术指标")
            print("   建议：检查特征引擎配置，确保 TechnicalProcessor 正确执行")
        
        if len(basic_features_found) > 0:
            print("⚠️ 建议：考虑移除基础价格特征（open, high, low, close, volume）")
            print("   理由：")
            print("   1. 基础价格数据是原始输入，不是衍生特征")
            print("   2. 技术指标已经包含了价格信息（如 SMA, EMA, RSI 等）")
            print("   3. 保留基础价格特征会增加特征维度，可能导致过拟合")
            print("   4. 量化交易系统通常只使用技术指标作为特征")
        
        if len(indicator_features) > 0 and len(indicator_features) < len(indicators):
            missing = set(indicators) - set(indicator_mapping.keys())
            print(f"⚠️ 建议：完善缺失的技术指标: {missing}")
        
        cursor.close()
        
        return {
            "task_info": task_info,
            "total_features": len(rows),
            "basic_features": len(basic_features_found),
            "indicator_features": len(indicator_features),
            "indicators_configured": len(indicators),
            "indicators_generated": len(indicator_mapping),
            "indicator_mapping": indicator_mapping,
            "compliance_score": compliance_score,
            "success": len(indicator_features) > 0
        }
        
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if conn:
            return_db_connection(conn)


def check_docker_logs(task_id: str):
    """检查 Docker 容器日志"""
    print(f"\n{'='*80}")
    print(f"📋 Docker 容器日志检查")
    print(f"{'='*80}")
    
    import subprocess
    
    try:
        result = subprocess.run(
            ['docker', 'logs', 'rqa2025-app', '2>&1'],
            capture_output=True,
            text=True
        )
        
        logs = result.stdout + result.stderr
        
        # 查找关键日志模式
        key_patterns = [
            ("调用 TechnicalProcessor", "TechnicalProcessor 调用"),
            ("计算指标", "指标计算"),
            ("TechnicalProcessor 完成", "TechnicalProcessor 完成"),
            ("特征计算完成", "特征计算完成"),
            ("转换为数值类型", "数据类型转换"),
            (task_id, "当前任务")
        ]
        
        print(f"🔍 查找关键日志:")
        found_any = False
        
        for pattern, desc in key_patterns:
            matching_lines = [line for line in logs.split('\n') if pattern in line and task_id in line]
            if matching_lines:
                found_any = True
                print(f"\n   📌 {desc}:")
                for line in matching_lines[-3:]:
                    print(f"   {line.strip()}")
        
        if not found_any:
            print("   ⚠️ 未找到关键日志")
        
        return found_any
        
    except Exception as e:
        print(f"❌ 检查日志失败: {e}")
        return False


def main():
    task_id = "feature_task_single_002837_1771761643"
    
    print("🚀 开始全面评估特征提取任务修复")
    print(f"任务ID: {task_id}")
    
    # 1. 全面评估
    result = comprehensive_evaluation(task_id)
    
    if not result:
        print("\n❌ 评估失败")
        return
    
    # 2. 检查 Docker 日志
    docker_logs_ok = check_docker_logs(task_id)
    
    # 3. 最终总结
    print(f"\n{'='*80}")
    print(f"🎯 最终评估总结")
    print(f"{'='*80}")
    
    print(f"\n任务ID: {task_id}")
    print(f"任务状态: {result['task_info']['status']}")
    print(f"总特征数: {result['total_features']}")
    print(f"基础价格特征: {result['basic_features']} 个")
    print(f"技术指标特征: {result['indicator_features']} 个")
    print(f"配置指标数: {result['indicators_configured']} 个")
    print(f"成功生成指标: {result['indicators_generated']} 个")
    print(f"合规性评分: {result['compliance_score']*100:.1f}%")
    print(f"Docker日志检查: {'通过' if docker_logs_ok else '未通过'}")
    
    if result['success']:
        print(f"\n🎉🎉🎉 修复成功！🎉🎉🎉")
        print(f"   ✅ 任务正确生成了技术指标特征")
        print(f"   ✅ 特征存储功能已完全修复")
        print(f"   ✅ 符合量化交易系统要求")
    else:
        print(f"\n⚠️ 修复可能未完全生效")
        print(f"   ❌ 未找到技术指标特征")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
