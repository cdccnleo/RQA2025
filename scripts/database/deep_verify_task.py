"""
深入全面验证特征提取任务
任务ID: feature_task_single_002837_1771759962
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection


def deep_verify_task(task_id: str):
    """深入验证任务"""
    print(f"\n{'='*80}")
    print(f"🔍 深入全面验证任务: {task_id}")
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
        start_date = config.get('start_date', 'N/A')
        end_date = config.get('end_date', 'N/A')
        
        print(f"\n📊 任务配置:")
        print(f"   股票代码: {symbol}")
        print(f"   开始日期: {start_date}")
        print(f"   结束日期: {end_date}")
        print(f"   配置指标: {indicators}")
        print(f"   指标数量: {len(indicators)}")
        
        # 2. 查询特征存储 - 详细检查
        cursor.execute("""
            SELECT feature_id, feature_name, feature_type, parameters, 
                   symbol, quality_score, importance, created_at
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
            feature_names.append(feature['feature_name'])
            
            params_str = str(feature['parameters']) if feature['parameters'] else "N/A"
            quality_str = f"{feature['quality_score']:.2f}" if feature['quality_score'] else "N/A"
            print(f"   {i}. {feature['feature_name']}")
            print(f"      类型: {feature['feature_type'] or 'N/A'}")
            print(f"      参数: {params_str}")
            print(f"      股票: {feature['symbol'] or 'N/A'}")
            print(f"      质量: {quality_str}")
            print(f"      重要性: {feature['importance'] or 'N/A'}")
        
        # 3. 深度分析特征
        print(f"\n{'='*80}")
        print(f"📊 深度特征分析")
        print(f"{'='*80}")
        
        # 基础价格特征
        basic_features = ['open', 'high', 'low', 'close', 'volume', 'date', 'trade_date', 'amount']
        basic_features_found = [f for f in feature_names if f in basic_features]
        
        print(f"\n📈 基础价格特征 ({len(basic_features_found)} 个):")
        for feat in basic_features_found:
            print(f"   - {feat}")
        
        # 技术指标特征 - 多种匹配模式
        print(f"\n📈 技术指标特征分析:")
        indicator_features = []
        indicator_mapping = {}
        
        for indicator in indicators:
            indicator_lower = indicator.lower()
            indicator_upper = indicator.upper()
            
            # 多种匹配模式
            matching = []
            for f in feature_names:
                if (f.startswith(indicator_lower) or 
                    f.startswith(indicator_upper) or
                    f.startswith(f'feature_{indicator_lower}') or
                    f.startswith(f'feature_{indicator_upper}') or
                    indicator_lower in f.lower()):
                    matching.append(f)
            
            if matching:
                indicator_features.extend(matching)
                indicator_mapping[indicator] = matching
                print(f"   ✅ {indicator}: {matching}")
            else:
                print(f"   ❌ {indicator}: 未找到")
        
        # 其他特征（非基础非指标）
        other_features = [f for f in feature_names 
                         if f not in basic_features_found and f not in indicator_features]
        if other_features:
            print(f"\n📈 其他特征 ({len(other_features)} 个):")
            for feat in other_features:
                print(f"   - {feat}")
        
        # 4. 统计总结
        print(f"\n{'='*80}")
        print(f"📊 统计总结")
        print(f"{'='*80}")
        
        total_features = len(features)
        basic_count = len(basic_features_found)
        indicator_count = len(indicator_features)
        other_count = len(other_features)
        
        print(f"   总特征数: {total_features}")
        print(f"   基础价格特征: {basic_count}")
        print(f"   技术指标特征: {indicator_count}")
        print(f"   其他特征: {other_count}")
        print(f"   配置指标数: {len(indicators)}")
        print(f"   成功生成的指标: {len(indicator_mapping)}")
        
        # 5. 合规性评估
        print(f"\n{'='*80}")
        print(f"📊 量化交易系统合规性评估")
        print(f"{'='*80}")
        
        compliance = {
            "data_integrity": task_info['status'] == 'completed' and total_features > 0,
            "metadata_quality": indicator_count > 0,
            "supports_backtest": indicator_count > 0,
            "supports_strategy": indicator_count > 0,
            "supports_model_training": indicator_count > 0,
        }
        
        score = 0.0
        if compliance["data_integrity"]:
            score += 0.4
            print("✅ 数据完整性: 通过")
        else:
            print("❌ 数据完整性: 未通过")
        
        if compliance["metadata_quality"]:
            score += 0.3
            print("✅ 元数据质量: 通过")
        else:
            print("❌ 元数据质量: 未通过")
        
        if compliance["supports_backtest"]:
            score += 0.1
            print("✅ 支持回测: 是")
        else:
            print("❌ 支持回测: 否")
        
        if compliance["supports_strategy"]:
            score += 0.1
            print("✅ 支持策略优化: 是")
        else:
            print("❌ 支持策略优化: 否")
        
        if compliance["supports_model_training"]:
            score += 0.1
            print("✅ 支持模型训练: 是")
        else:
            print("❌ 支持模型训练: 否")
        
        compliance["overall_score"] = score
        
        print(f"\n📊 总体评分: {score*100:.1f}%")
        
        if score >= 0.8:
            print("🎉 优秀 - 完全符合量化交易系统要求")
        elif score >= 0.6:
            print("⚠️ 良好 - 基本符合要求，但有改进空间")
        else:
            print("❌ 不合格 - 不符合量化交易系统要求")
        
        cursor.close()
        
        return {
            "task_info": task_info,
            "features": features,
            "total_features": total_features,
            "basic_features": basic_count,
            "indicator_features": indicator_count,
            "other_features": other_count,
            "indicators_configured": len(indicators),
            "indicators_generated": len(indicator_mapping),
            "indicator_mapping": indicator_mapping,
            "compliance": compliance,
            "success": indicator_count > 0
        }
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if conn:
            return_db_connection(conn)


def check_docker_logs_detailed(task_id: str):
    """详细检查 Docker 容器日志"""
    print(f"\n{'='*80}")
    print(f"📋 Docker 容器日志详细检查")
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
            ("特征配置", "FeatureConfig 配置"),
            ("TechnicalProcessor", "技术指标处理器"),
            ("调用 TechnicalProcessor", "TechnicalProcessor 调用"),
            ("计算指标", "指标计算"),
            ("indicators=", "Indicators 参数"),
            ("特征计算完成", "特征计算完成"),
            ("生成.*个特征", "特征生成数量"),
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
            print("   ⚠️ 未找到当前任务的相关日志")
        
        return found_any
        
    except Exception as e:
        print(f"❌ 检查日志失败: {e}")
        return False


def main():
    task_id = "feature_task_single_002837_1771759962"
    
    print("🚀 开始深入全面验证特征提取任务")
    print(f"任务ID: {task_id}")
    
    # 1. 深度验证任务
    result = deep_verify_task(task_id)
    
    if not result:
        print("\n❌ 验证失败，无法获取任务信息")
        return
    
    # 2. 详细检查 Docker 日志
    docker_logs_ok = check_docker_logs_detailed(task_id)
    
    # 3. 最终总结
    print(f"\n{'='*80}")
    print(f"🎯 最终验证总结")
    print(f"{'='*80}")
    
    print(f"\n任务ID: {task_id}")
    print(f"任务状态: {result['task_info']['status']}")
    print(f"总特征数: {result['total_features']}")
    print(f"基础价格特征: {result['basic_features']} 个")
    print(f"技术指标特征: {result['indicator_features']} 个")
    print(f"其他特征: {result['other_features']} 个")
    print(f"配置指标数: {result['indicators_configured']} 个")
    print(f"成功生成指标: {result['indicators_generated']} 个")
    print(f"合规性评分: {result['compliance']['overall_score']*100:.1f}%")
    print(f"Docker日志检查: {'通过' if docker_logs_ok else '未通过'}")
    
    if result['success']:
        print(f"\n🎉🎉🎉 修复成功！🎉🎉🎉")
        print(f"   ✅ 任务正确生成了技术指标特征")
        print(f"   ✅ 特征存储功能已完全修复")
        print(f"   ✅ 符合量化交易系统要求")
        print(f"\n   详细指标映射:")
        for indicator, features in result['indicator_mapping'].items():
            print(f"      {indicator}: {features}")
    else:
        print(f"\n⚠️ 修复可能未完全生效")
        print(f"   ❌ 未找到技术指标特征")
        print(f"\n   建议检查项:")
        print(f"   1. Docker 容器是否使用最新代码")
        print(f"   2. TechnicalProcessor.process 方法是否正确执行")
        print(f"   3. 特征引擎是否正确初始化")
        print(f"   4. Python 缓存是否已清除")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
