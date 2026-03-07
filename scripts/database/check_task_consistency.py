"""
检查特征提取任务仪表盘显示记录数与数据库记录一致性
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection


def check_database_tasks():
    """检查数据库中的任务记录"""
    print("\n" + "="*80)
    print("📊 数据库任务记录统计")
    print("="*80)
    
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 统计总任务数
        cursor.execute("SELECT COUNT(*) FROM feature_engineering_tasks")
        total_count = cursor.fetchone()[0]
        print(f"\n   总任务数: {total_count}")
        
        # 按状态统计
        cursor.execute("""
            SELECT status, COUNT(*) 
            FROM feature_engineering_tasks 
            GROUP BY status
            ORDER BY status
        """)
        status_counts = cursor.fetchall()
        print(f"\n   按状态分布:")
        for status, count in status_counts:
            print(f"      - {status}: {count}")
        
        # 查询最近的10个任务
        cursor.execute("""
            SELECT task_id, status, feature_count, config, created_at
            FROM feature_engineering_tasks
            ORDER BY created_at DESC
            LIMIT 10
        """)
        
        recent_tasks = cursor.fetchall()
        print(f"\n   最近10个任务:")
        for i, task in enumerate(recent_tasks, 1):
            task_id, status, feature_count, config, created_at = task
            if isinstance(config, str):
                try:
                    config = json.loads(config)
                except:
                    config = {}
            symbol = config.get('symbol', 'N/A') if isinstance(config, dict) else 'N/A'
            print(f"      {i}. {task_id} | {status} | {symbol} | {feature_count} features | {created_at}")
        
        cursor.close()
        return total_count, status_counts, recent_tasks
        
    except Exception as e:
        print(f"❌ 数据库查询失败: {e}")
        import traceback
        traceback.print_exc()
        return 0, [], []
    finally:
        if conn:
            return_db_connection(conn)


def check_frontend_api():
    """检查前端API返回的数据"""
    print("\n" + "="*80)
    print("📊 前端API数据检查")
    print("="*80)
    
    try:
        # 模拟API调用获取任务列表
        from src.gateway.web.feature_task_persistence import list_feature_tasks
        
        tasks = list_feature_tasks(limit=100)
        print(f"\n   API返回任务数: {len(tasks)}")
        
        # 按状态统计
        status_counts = {}
        for task in tasks:
            status = task.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print(f"\n   API返回任务状态分布:")
        for status, count in sorted(status_counts.items()):
            print(f"      - {status}: {count}")
        
        # 显示最近的5个任务
        print(f"\n   API返回最近5个任务:")
        for i, task in enumerate(tasks[:5], 1):
            task_id = task.get('task_id', 'N/A')
            status = task.get('status', 'N/A')
            symbol = task.get('symbol', 'N/A')
            feature_count = task.get('feature_count', 0)
            print(f"      {i}. {task_id} | {status} | {symbol} | {feature_count} features")
        
        return len(tasks), status_counts, tasks
        
    except Exception as e:
        print(f"❌ API调用失败: {e}")
        import traceback
        traceback.print_exc()
        return 0, {}, []


def analyze_difference(db_total, api_total, db_status, api_status):
    """分析差异原因"""
    print("\n" + "="*80)
    print("🔍 差异分析")
    print("="*80)
    
    print(f"\n   数据库总记录数: {db_total}")
    print(f"   API返回记录数: {api_total}")
    print(f"   差异: {db_total - api_total}")
    
    if db_total != api_total:
        print(f"\n   ⚠️  发现不一致!")
        
        # 分析状态分布差异
        print(f"\n   状态分布对比:")
        db_status_dict = dict(db_status)
        
        all_statuses = set(db_status_dict.keys()) | set(api_status.keys())
        for status in sorted(all_statuses):
            db_count = db_status_dict.get(status, 0)
            api_count = api_status.get(status, 0)
            diff = db_count - api_count
            if diff != 0:
                print(f"      - {status}: DB={db_count}, API={api_count}, 差异={diff}")
            else:
                print(f"      - {status}: DB={db_count}, API={api_count}, ✅一致")
    else:
        print(f"\n   ✅ 记录数一致")


def check_frontend_logic():
    """检查前端展示逻辑"""
    print("\n" + "="*80)
    print("📋 前端展示逻辑分析")
    print("="*80)
    
    print("\n   前端文件: web-static/feature-engineering-monitor.html")
    print("   相关函数:")
    print("      - loadTasks() - 加载任务列表")
    print("      - renderTasks() - 渲染任务列表")
    print("      - filterTasks() - 过滤任务")
    
    print("\n   可能的原因:")
    print("      1. 前端过滤: 可能只显示特定状态的任务")
    print("      2. 分页限制: API有limit参数，可能只返回部分数据")
    print("      3. 状态过滤: 前端可能过滤掉某些状态的任务")
    print("      4. 时间范围: 可能只显示最近某段时间的任务")
    
    print("\n   检查建议:")
    print("      - 查看前端是否有过滤逻辑")
    print("      - 检查API调用的参数")
    print("      - 确认是否有过期任务被隐藏")


def main():
    print("🚀 开始检查特征提取任务记录一致性")
    
    # 检查数据库
    db_total, db_status, db_tasks = check_database_tasks()
    
    # 检查API
    api_total, api_status, api_tasks = check_frontend_api()
    
    # 分析差异
    analyze_difference(db_total, api_total, db_status, api_status)
    
    # 检查前端逻辑
    check_frontend_logic()
    
    print("\n" + "="*80)
    print("✅ 检查完成")
    print("="*80)


if __name__ == "__main__":
    main()
