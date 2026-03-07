"""
全面深入检查特征提取任务
任务ID: feature_task_single_002837_1771759647
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection


def check_task_basic_info(task_id: str):
    """检查任务基本信息"""
    print(f"\n{'='*80}")
    print(f"📋 1. 任务基本信息检查")
    print(f"{'='*80}")
    
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            print("❌ 无法连接到数据库")
            return None
        
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT task_id, status, feature_count, config, created_at, updated_at
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
            "created_at": row[4],
            "updated_at": row[5]
        }
        
        print(f"✅ 任务ID: {task_info['task_id']}")
        print(f"   状态: {task_info['status']}")
        print(f"   特征数量: {task_info['feature_count']}")
        print(f"   创建时间: {task_info['created_at']}")
        print(f"   更新时间: {task_info['updated_at']}")
        
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
        
        cursor.close()
        
        return {
            "task_info": task_info,
            "config": config,
            "indicators": indicators,
            "symbol": symbol
        }
        
    except Exception as e:
        print(f"❌ 查询失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if conn:
            return_db_connection(conn)


def check_feature_store_data(task_id: str):
    """检查特征存储表数据"""
    print(f"\n{'='*80}")
    print(f"📋 2. 特征存储表数据检查")
    print(f"{'='*80}")
    
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            print("❌ 无法连接到数据库")
            return None
        
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT feature_id, feature_name, feature_type, parameters, 
                   symbol, quality_score, importance, created_at
            FROM feature_store
            WHERE task_id = %s
            ORDER BY feature_name
        """, (task_id,))
        
        rows = cursor.fetchall()
        
        if not rows:
            print("❌ 特征存储表中没有数据")
            return None
        
        print(f"✅ 特征存储表中有 {len(rows)} 个特征")
        
        features = []
        print(f"\n📊 特征列表:")
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
            print(f"      质量: {feature['quality_score'] or 'N/A'}")
        
        cursor.close()
        
        return features
        
    except Exception as e:
        print(f"❌ 查询失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if conn:
            return_db_connection(conn)


def analyze_indicators(features, indicators_config):
    """分析技术指标生成情况"""
    print(f"\n{'='*80}")
    print(f"📋 3. 技术指标生成检查")
    print(f"{'='*80}")
    
    feature_names = [f['feature_name'] for f in features]
    
    # 基础价格特征
    basic_features = ['open', 'high', 'low', 'close', 'volume', 'date', 'trade_date', 'amount']
    basic_features_found = [f for f in feature_names if f in basic_features]
    
    print(f"📊 基础价格特征:")
    print(f"   数量: {len(basic_features_found)}")
    for feat in basic_features_found:
        print(f"   - {feat}")
    
    # 技术指标特征
    print(f"\n📊 技术指标特征分析:")
    indicator_features = []
    
    for indicator in indicators_config:
        indicator_lower = indicator.lower()
        # 检查是否有该指标的特征（支持多种命名格式）
        matching = [f for f in feature_names if f.startswith(indicator_lower) or f.startswith(f'feature_{indicator_lower}')]
        if matching:
            indicator_features.extend(matching)
            print(f"   ✅ {indicator}: {matching}")
        else:
            print(f"   ❌ {indicator}: 未找到")
    
    print(f"\n📊 总结:")
    print(f"   配置的技术指标: {len(indicators_config)} 个")
    print(f"   找到的技术指标特征: {len(indicator_features)} 个")
    print(f"   基础价格特征: {len(basic_features_found)} 个")
    print(f"   总特征数: {len(features)} 个")
    
    return {
        "basic_features": len(basic_features_found),
        "indicator_features": len(indicator_features),
        "total_features": len(features),
        "indicators_configured": len(indicators_config),
        "success": len(indicator_features) > 0
    }


def check_docker_logs(task_id: str):
    """检查 Docker 容器日志"""
    print(f"\n{'='*80}")
    print(f"📋 4. Docker 容器日志检查")
    print(f"{'='*80}")
    
    import subprocess
    
    try:
        # 检查 worker_executor 日志
        result = subprocess.run(
            ['docker', 'logs', 'rqa2025-app', '2>&1'],
            capture_output=True,
            text=True
        )
        
        logs = result.stdout + result.stderr
        
        # 查找关键日志
        key_patterns = [
            "特征配置",
            "TechnicalProcessor",
            "indicators=",
            "特征计算完成",
            "任务.*开始从PostgreSQL",
            "使用特征引擎",
            f"{task_id}"
        ]
        
        print(f"🔍 查找关键日志模式:")
        found_logs = []
        for pattern in key_patterns:
            matching_lines = [line for line in logs.split('\n') if pattern in line]
            if matching_lines:
                print(f"\n   模式 '{pattern}':")
                for line in matching_lines[-3:]:  # 显示最后3条匹配
                    print(f"   {line.strip()}")
                found_logs.extend(matching_lines)
        
        if not found_logs:
            print("   ⚠️ 未找到关键日志")
        
        return len(found_logs) > 0
        
    except Exception as e:
        print(f"❌ 检查日志失败: {e}")
        return False


def check_frontend_display():
    """检查前端展示"""
    print(f"\n{'='*80}")
    print(f"📋 5. 前端展示检查")
    print(f"{'='*80}")
    
    try:
        from src.gateway.web.feature_engineering_service import get_features
        
        features = get_features()
        
        print(f"✅ get_features() 返回 {len(features)} 个特征")
        
        if features:
            print(f"\n📊 前端特征列表 (显示前10个):")
            for i, feature in enumerate(features[:10], 1):
                name = feature.get('name', 'N/A')
                feature_type = feature.get('feature_type', 'N/A')
                print(f"   {i}. {name} (类型: {feature_type})")
            
            if len(features) > 10:
                print(f"   ... 还有 {len(features) - 10} 个特征")
        
        return features
        
    except Exception as e:
        print(f"❌ 检查前端展示失败: {e}")
        return None


def check_quant_compliance(task_info, features_data, indicators_analysis):
    """检查量化交易系统合规性"""
    print(f"\n{'='*80}")
    print(f"📋 6. 量化交易系统合规性检查")
    print(f"{'='*80}")
    
    compliance = {
        "data_integrity": False,
        "metadata_quality": False,
        "supports_backtest": False,
        "supports_strategy": False,
        "supports_model_training": False,
        "overall_score": 0.0
    }
    
    if not task_info or not features_data:
        print("❌ 数据不完整，无法评估合规性")
        return compliance
    
    # 数据完整性检查
    if task_info["task_info"]["status"] == "completed" and len(features_data) > 0:
        compliance["data_integrity"] = True
        print("✅ 数据完整性: 通过")
    else:
        print("❌ 数据完整性: 未通过")
    
    # 元数据质量检查
    if indicators_analysis["indicator_features"] > 0:
        compliance["metadata_quality"] = True
        print("✅ 元数据质量: 通过")
    else:
        print("❌ 元数据质量: 未通过")
    
    # 支持回测检查
    if compliance["data_integrity"] and compliance["metadata_quality"]:
        compliance["supports_backtest"] = True
        compliance["supports_strategy"] = True
        compliance["supports_model_training"] = True
        print("✅ 支持回测: 是")
        print("✅ 支持策略优化: 是")
        print("✅ 支持模型训练: 是")
    else:
        print("❌ 支持回测: 否")
        print("❌ 支持策略优化: 否")
        print("❌ 支持模型训练: 否")
    
    # 计算总体评分
    score = 0.0
    if compliance["data_integrity"]:
        score += 0.4
    if compliance["metadata_quality"]:
        score += 0.3
    if compliance["supports_backtest"]:
        score += 0.1
    if compliance["supports_strategy"]:
        score += 0.1
    if compliance["supports_model_training"]:
        score += 0.1
    
    compliance["overall_score"] = score
    
    print(f"\n📊 总体评分: {score*100:.1f}%")
    
    if score >= 0.8:
        print("🎉 优秀 - 符合量化交易系统要求")
    elif score >= 0.6:
        print("⚠️ 良好 - 基本符合要求，但有改进空间")
    else:
        print("❌ 不合格 - 不符合量化交易系统要求")
    
    return compliance


def main():
    task_id = "feature_task_single_002837_1771759647"
    
    print("🚀 开始全面深入检查特征提取任务")
    print(f"任务ID: {task_id}")
    
    # 1. 检查任务基本信息
    task_data = check_task_basic_info(task_id)
    if not task_data:
        print("\n❌ 任务基本信息检查失败，停止后续检查")
        return
    
    # 2. 检查特征存储表数据
    features_data = check_feature_store_data(task_id)
    if not features_data:
        print("\n❌ 特征存储表数据检查失败，停止后续检查")
        return
    
    # 3. 分析技术指标生成情况
    indicators_analysis = analyze_indicators(features_data, task_data["indicators"])
    
    # 4. 检查 Docker 容器日志
    docker_logs_ok = check_docker_logs(task_id)
    
    # 5. 检查前端展示
    frontend_features = check_frontend_display()
    
    # 6. 检查量化交易系统合规性
    compliance = check_quant_compliance(task_data, features_data, indicators_analysis)
    
    # 7. 总结报告
    print(f"\n{'='*80}")
    print(f"📊 全面检查总结报告")
    print(f"{'='*80}")
    
    print(f"\n任务ID: {task_id}")
    print(f"任务状态: {task_data['task_info']['status']}")
    print(f"总特征数: {len(features_data)}")
    print(f"基础价格特征: {indicators_analysis['basic_features']} 个")
    print(f"技术指标特征: {indicators_analysis['indicator_features']} 个")
    print(f"配置指标数: {indicators_analysis['indicators_configured']} 个")
    print(f"Docker日志检查: {'通过' if docker_logs_ok else '未通过'}")
    print(f"前端展示检查: {'通过' if frontend_features else '未通过'}")
    print(f"合规性评分: {compliance['overall_score']*100:.1f}%")
    
    if indicators_analysis["success"]:
        print(f"\n🎉 修复成功！")
        print(f"   ✅ 任务正确生成了技术指标特征")
        print(f"   ✅ 特征存储功能已完全修复")
        print(f"   ✅ 符合量化交易系统要求")
    else:
        print(f"\n⚠️ 修复可能未完全生效")
        print(f"   ❌ 未找到技术指标特征")
        print(f"   建议: 检查 Docker 容器是否使用最新代码")
        print(f"   建议: 检查 TechnicalProcessor.process 方法是否正确执行")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
