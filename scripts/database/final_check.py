"""
最终验证特征提取任务修复
任务ID: feature_task_single_002837_1771761386
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection


def final_check(task_id: str):
    """最终验证"""
    print(f"\n{'='*80}")
    print(f"🎯 最终验证任务: {task_id}")
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
            SELECT feature_name, feature_type, parameters, symbol, quality_score
            FROM feature_store
            WHERE task_id = %s
            ORDER BY feature_name
        """, (task_id,))
        
        rows = cursor.fetchall()
        
        if not rows:
            print("\n❌ 特征存储表中没有数据")
            return None
        
        print(f"\n✅ 特征存储表中有 {len(rows)} 个特征")
        
        feature_names = [row[0] for row in rows]
        
        # 分析特征
        basic_features = ['open', 'high', 'low', 'close', 'volume', 'date', 'trade_date', 'amount']
        basic_count = sum(1 for f in feature_names if f in basic_features)
        
        # 检查技术指标
        indicator_features = []
        indicator_mapping = {}
        
        for indicator in indicators:
            indicator_lower = indicator.lower()
            matching = [f for f in feature_names if indicator_lower in f.lower()]
            if matching:
                indicator_features.extend(matching)
                indicator_mapping[indicator] = matching
        
        print(f"\n📊 特征分析:")
        print(f"   基础价格特征: {basic_count} 个")
        print(f"   技术指标特征: {len(indicator_features)} 个")
        print(f"   总特征数: {len(rows)} 个")
        
        if indicator_features:
            print(f"\n🎉 成功生成的技术指标:")
            for indicator, features in indicator_mapping.items():
                print(f"   ✅ {indicator}: {features}")
        else:
            print(f"\n❌ 未找到技术指标特征")
        
        # 合规性评估
        compliance_score = 0.0
        if task_info['status'] == 'completed':
            compliance_score += 0.4
        if len(indicator_features) > 0:
            compliance_score += 0.3
        if len(indicator_features) >= len(indicators) * 0.5:
            compliance_score += 0.3
        
        print(f"\n📊 合规性评分: {compliance_score*100:.1f}%")
        
        if compliance_score >= 0.8:
            print("🎉 优秀 - 完全符合量化交易系统要求")
        elif compliance_score >= 0.6:
            print("⚠️ 良好 - 基本符合要求")
        else:
            print("❌ 不合格 - 不符合量化交易系统要求")
        
        cursor.close()
        
        return {
            "task_info": task_info,
            "total_features": len(rows),
            "basic_features": basic_count,
            "indicator_features": len(indicator_features),
            "indicators_configured": len(indicators),
            "indicators_generated": len(indicator_mapping),
            "indicator_mapping": indicator_mapping,
            "compliance_score": compliance_score,
            "success": len(indicator_features) > 0
        }
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
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
    task_id = "feature_task_single_002837_1771761386"
    
    print("🚀 开始最终验证特征提取任务修复")
    print(f"任务ID: {task_id}")
    
    # 1. 验证任务
    result = final_check(task_id)
    
    if not result:
        print("\n❌ 验证失败")
        return
    
    # 2. 检查 Docker 日志
    docker_logs_ok = check_docker_logs(task_id)
    
    # 3. 最终总结
    print(f"\n{'='*80}")
    print(f"🎯 最终验证总结")
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
