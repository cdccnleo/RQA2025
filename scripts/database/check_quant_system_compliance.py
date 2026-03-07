"""
检查特征存储是否符合量化交易系统要求
任务ID:
1. feature_task_single_688702_1771756738
2. feature_task_single_002837_1771756710
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection


def check_task_status(task_id: str):
    """检查任务状态"""
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            return None
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT task_id, task_type, status, progress, feature_count,
                   start_time, end_time, config, error_message,
                   created_at, updated_at
            FROM feature_engineering_tasks
            WHERE task_id = %s
        """, (task_id,))
        
        row = cursor.fetchone()
        cursor.close()
        
        if not row:
            return None
        
        return {
            "task_id": row[0],
            "task_type": row[1],
            "status": row[2],
            "progress": row[3],
            "feature_count": row[4],
            "start_time": row[5],
            "end_time": row[6],
            "config": row[7],
            "error_message": row[8],
            "created_at": row[9],
            "updated_at": row[10]
        }
        
    except Exception as e:
        print(f"❌ 查询任务状态失败: {e}")
        return None
    finally:
        if conn:
            return_db_connection(conn)


def check_feature_store(task_id: str):
    """检查特征存储表数据"""
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            return None
        
        cursor = conn.cursor()
        
        # 查询特征数量
        cursor.execute("""
            SELECT COUNT(*) FROM feature_store WHERE task_id = %s
        """, (task_id,))
        
        count = cursor.fetchone()[0]
        
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
        
        features = []
        for row in rows:
            features.append({
                "feature_id": row[0],
                "feature_name": row[1],
                "feature_type": row[2],
                "parameters": row[3],
                "symbol": row[4],
                "quality_score": row[5],
                "importance": row[6],
                "created_at": row[7]
            })
        
        return {"count": count, "features": features}
        
    except Exception as e:
        print(f"❌ 查询特征存储表失败: {e}")
        return None
    finally:
        if conn:
            return_db_connection(conn)


def check_frontend_data():
    """检查前端展示数据"""
    try:
        from src.gateway.web.feature_engineering_service import get_features
        
        features = get_features()
        return features
        
    except Exception as e:
        print(f"❌ 检查前端数据失败: {e}")
        return None


def analyze_feature_quality(features):
    """分析特征质量"""
    analysis = {
        "total_features": len(features),
        "proper_named": 0,
        "hardcoded_named": 0,
        "with_type": 0,
        "with_parameters": 0,
        "with_symbol": 0,
        "feature_types": {},
        "symbols": set()
    }
    
    for feature in features:
        name = feature.get("feature_name", "")
        feature_type = feature.get("feature_type")
        parameters = feature.get("parameters")
        symbol = feature.get("symbol")
        
        # 检查命名格式
        if name.startswith("feature_"):
            analysis["hardcoded_named"] += 1
        else:
            analysis["proper_named"] += 1
        
        # 检查是否有类型
        if feature_type:
            analysis["with_type"] += 1
            analysis["feature_types"][feature_type] = analysis["feature_types"].get(feature_type, 0) + 1
        
        # 检查是否有参数
        if parameters:
            analysis["with_parameters"] += 1
        
        # 检查是否有股票代码
        if symbol:
            analysis["with_symbol"] += 1
            analysis["symbols"].add(symbol)
    
    analysis["symbols"] = list(analysis["symbols"])
    return analysis


def check_quant_compliance(task_info, features_data):
    """检查量化交易系统合规性"""
    compliance = {
        "data_integrity": False,
        "metadata_quality": False,
        "supports_backtest": False,
        "supports_strategy": False,
        "supports_model_training": False,
        "overall_score": 0.0
    }
    
    if not task_info or not features_data:
        return compliance
    
    # 数据完整性检查
    if task_info["status"] == "completed" and features_data["count"] > 0:
        compliance["data_integrity"] = True
    
    # 元数据质量检查
    features = features_data.get("features", [])
    if features:
        proper_named = sum(1 for f in features if not f.get("feature_name", "").startswith("feature_"))
        with_type = sum(1 for f in features if f.get("feature_type"))
        with_symbol = sum(1 for f in features if f.get("symbol"))
        
        if proper_named == len(features) and with_type > 0 and with_symbol > 0:
            compliance["metadata_quality"] = True
    
    # 支持回测检查（需要特征数据和时间范围）
    if compliance["data_integrity"] and compliance["metadata_quality"]:
        compliance["supports_backtest"] = True
        compliance["supports_strategy"] = True
        compliance["supports_model_training"] = True
    
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
    return compliance


def print_section(title):
    """打印章节标题"""
    print(f"\n{'='*80}")
    print(f"📌 {title}")
    print(f"{'='*80}")


def main():
    task_ids = [
        "feature_task_single_688702_1771756738",
        "feature_task_single_002837_1771756710"
    ]
    
    print("🚀 开始检查特征存储是否符合量化交易系统要求")
    
    all_results = {}
    
    for task_id in task_ids:
        print_section(f"检查任务: {task_id}")
        
        # 1. 数据完整性检查
        print("\n1️⃣ 数据完整性检查")
        task_info = check_task_status(task_id)
        if task_info:
            print(f"   ✅ 任务存在")
            print(f"   状态: {task_info['status']}")
            print(f"   特征数量: {task_info['feature_count']}")
            print(f"   任务类型: {task_info['task_type']}")
        else:
            print(f"   ❌ 任务不存在")
            continue
        
        features_data = check_feature_store(task_id)
        if features_data:
            print(f"   ✅ 特征存储表中有 {features_data['count']} 个特征")
        else:
            print(f"   ❌ 无法获取特征数据")
            continue
        
        # 2. 特征元数据质量检查
        print("\n2️⃣ 特征元数据质量检查")
        features = features_data.get("features", [])
        if features:
            analysis = analyze_feature_quality(features)
            print(f"   总特征数: {analysis['total_features']}")
            print(f"   正确命名: {analysis['proper_named']}")
            print(f"   硬编码命名: {analysis['hardcoded_named']}")
            print(f"   有类型信息: {analysis['with_type']}")
            print(f"   有参数信息: {analysis['with_parameters']}")
            print(f"   有股票代码: {analysis['with_symbol']}")
            print(f"   特征类型分布: {analysis['feature_types']}")
            print(f"   股票代码: {analysis['symbols']}")
            
            print(f"\n   📋 特征列表:")
            for i, feature in enumerate(features[:5], 1):
                print(f"   {i}. {feature['feature_name']}")
                print(f"      类型: {feature['feature_type'] or 'N/A'}")
                print(f"      参数: {feature['parameters'] or 'N/A'}")
                print(f"      股票: {feature['symbol'] or 'N/A'}")
            if len(features) > 5:
                print(f"   ... 还有 {len(features) - 5} 个特征")
        else:
            print(f"   ❌ 没有特征数据")
        
        # 3. 量化交易系统合规性检查
        print("\n3️⃣ 量化交易系统合规性检查")
        compliance = check_quant_compliance(task_info, features_data)
        print(f"   数据完整性: {'✅ 通过' if compliance['data_integrity'] else '❌ 未通过'}")
        print(f"   元数据质量: {'✅ 通过' if compliance['metadata_quality'] else '❌ 未通过'}")
        print(f"   支持回测: {'✅ 是' if compliance['supports_backtest'] else '❌ 否'}")
        print(f"   支持策略优化: {'✅ 是' if compliance['supports_strategy'] else '❌ 否'}")
        print(f"   支持模型训练: {'✅ 是' if compliance['supports_model_training'] else '❌ 否'}")
        print(f"   总体评分: {compliance['overall_score']*100:.1f}%")
        
        all_results[task_id] = {
            "task_info": task_info,
            "features_data": features_data,
            "compliance": compliance
        }
    
    # 4. 前端展示检查
    print_section("前端展示检查")
    frontend_features = check_frontend_data()
    if frontend_features:
        print(f"   ✅ get_features() 返回 {len(frontend_features)} 个特征")
        print(f"\n   📋 前端特征列表 (显示前10个):")
        for i, feature in enumerate(frontend_features[:10], 1):
            name = feature.get('name', 'N/A')
            feature_type = feature.get('feature_type', 'N/A')
            print(f"   {i}. {name} (类型: {feature_type})")
        if len(frontend_features) > 10:
            print(f"   ... 还有 {len(frontend_features) - 10} 个特征")
    else:
        print(f"   ⚠️ get_features() 返回空列表")
    
    # 5. 总结报告
    print_section("量化交易系统合规性总结报告")
    
    total_tasks = len(task_ids)
    passed_tasks = 0
    failed_tasks = 0
    
    for task_id in task_ids:
        result = all_results.get(task_id)
        print(f"\n任务: {task_id}")
        if result:
            compliance = result["compliance"]
            score = compliance["overall_score"]
            
            if score >= 0.8:
                print(f"  ✅ 优秀 - 评分: {score*100:.1f}%")
                passed_tasks += 1
            elif score >= 0.6:
                print(f"  ⚠️ 良好 - 评分: {score*100:.1f}%")
                passed_tasks += 1
            else:
                print(f"  ❌ 不合格 - 评分: {score*100:.1f}%")
                failed_tasks += 1
            
            print(f"     数据完整性: {'✅' if compliance['data_integrity'] else '❌'}")
            print(f"     元数据质量: {'✅' if compliance['metadata_quality'] else '❌'}")
            print(f"     支持回测: {'✅' if compliance['supports_backtest'] else '❌'}")
        else:
            print(f"  ❌ 无法获取检查结果")
            failed_tasks += 1
    
    print(f"\n{'='*80}")
    print(f"统计:")
    print(f"  总任务数: {total_tasks}")
    print(f"  通过: {passed_tasks}")
    print(f"  失败: {failed_tasks}")
    
    if failed_tasks == 0:
        print(f"\n🎉 所有任务特征存储符合量化交易系统要求！")
    else:
        print(f"\n⚠️ 部分任务特征存储不符合要求，需要进一步修复")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
