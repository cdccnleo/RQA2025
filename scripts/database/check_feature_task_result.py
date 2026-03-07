"""
检查特征提取任务存储结果
任务ID: feature_task_single_002837_1771755510
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection


def check_task_status(task_id: str):
    """检查任务状态"""
    print(f"\n{'='*60}")
    print(f"🔍 检查任务状态: {task_id}")
    print(f"{'='*60}")
    
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            print("❌ 无法连接到数据库")
            return None
        
        cursor = conn.cursor()
        
        # 查询任务状态
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
            print(f"❌ 任务 {task_id} 不存在")
            return None
        
        task = {
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
        
        print(f"✅ 任务找到")
        print(f"   任务类型: {task['task_type']}")
        print(f"   状态: {task['status']}")
        print(f"   进度: {task['progress']}%")
        print(f"   特征数量: {task['feature_count']}")
        print(f"   创建时间: {task['created_at']}")
        print(f"   更新时间: {task['updated_at']}")
        
        if task['error_message']:
            print(f"   错误信息: {task['error_message']}")
        
        return task
        
    except Exception as e:
        print(f"❌ 查询任务状态失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if conn:
            return_db_connection(conn)


def check_feature_store(task_id: str):
    """检查特征存储表数据"""
    print(f"\n{'='*60}")
    print(f"🔍 检查特征存储表数据: {task_id}")
    print(f"{'='*60}")
    
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
        print(f"\n📋 特征列表 (显示前10个):")
        for i, row in enumerate(rows[:10], 1):
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
        
        if len(rows) > 10:
            print(f"   ... 还有 {len(rows) - 10} 个特征")
        
        return features
        
    except Exception as e:
        print(f"❌ 查询特征存储表失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if conn:
            return_db_connection(conn)


def check_frontend_data():
    """检查前端展示数据"""
    print(f"\n{'='*60}")
    print(f"🔍 检查前端展示数据 (get_features)")
    print(f"{'='*60}")
    
    try:
        from src.gateway.web.feature_engineering_service import get_features
        
        features = get_features()
        
        if not features:
            print("⚠️ get_features() 返回空列表")
            return []
        
        print(f"✅ get_features() 返回 {len(features)} 个特征")
        print(f"\n📋 前端特征列表 (显示前10个):")
        
        for i, feature in enumerate(features[:10], 1):
            name = feature.get('name', 'N/A')
            display_name = feature.get('display_name', name)
            feature_type = feature.get('feature_type', 'N/A')
            
            print(f"   {i}. {name}")
            if display_name != name:
                print(f"      显示名称: {display_name}")
            print(f"      类型: {feature_type}")
        
        if len(features) > 10:
            print(f"   ... 还有 {len(features) - 10} 个特征")
        
        return features
        
    except Exception as e:
        print(f"❌ 检查前端数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def check_consistency(task_id: str, task_info, features):
    """数据一致性检查"""
    print(f"\n{'='*60}")
    print(f"🔍 数据一致性检查")
    print(f"{'='*60}")
    
    if not task_info:
        print("❌ 任务信息不可用，无法进行检查")
        return False
    
    # 检查任务状态
    if task_info['status'] != 'completed':
        print(f"⚠️ 任务状态不是 completed: {task_info['status']}")
    else:
        print(f"✅ 任务状态为 completed")
    
    # 检查特征数量一致性
    if features is not None:
        stored_count = len(features)
        task_count = task_info.get('feature_count', 0)
        
        if stored_count == task_count:
            print(f"✅ 特征数量一致: 任务记录 {task_count} = 存储表 {stored_count}")
        else:
            print(f"⚠️ 特征数量不一致: 任务记录 {task_count} ≠ 存储表 {stored_count}")
    
    # 检查特征名称格式
    if features:
        print(f"\n📋 特征名称格式检查:")
        hardcoded_count = 0
        proper_name_count = 0
        
        for feature in features:
            name = feature.get('name', '')
            if name.startswith('feature_'):
                hardcoded_count += 1
            else:
                proper_name_count += 1
        
        if hardcoded_count > 0:
            print(f"   ⚠️ 发现 {hardcoded_count} 个硬编码特征名称 (feature_xxx)")
        
        if proper_name_count > 0:
            print(f"   ✅ 发现 {proper_name_count} 个正确格式的特征名称")
    
    return True


def main():
    task_id = "feature_task_single_002837_1771755510"
    
    print("🚀 开始检查特征提取任务存储结果")
    print(f"任务ID: {task_id}")
    
    # 1. 检查任务状态
    task_info = check_task_status(task_id)
    
    # 2. 检查特征存储表
    features = check_feature_store(task_id)
    
    # 3. 检查前端数据
    frontend_features = check_frontend_data()
    
    # 4. 一致性检查
    check_consistency(task_id, task_info, features)
    
    # 总结
    print(f"\n{'='*60}")
    print(f"📊 检查结果总结")
    print(f"{'='*60}")
    
    if task_info and features:
        print(f"✅ 任务和特征数据都存在")
        print(f"   任务状态: {task_info['status']}")
        print(f"   特征数量: {len(features)}")
        
        if task_info['status'] == 'completed' and len(features) > 0:
            print(f"\n🎉 特征存储结果符合预期！")
        else:
            print(f"\n⚠️ 特征存储结果可能不完整")
    else:
        print(f"\n❌ 检查结果异常，请查看详细日志")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
