#!/usr/bin/env python3
"""
同步文件系统中的特征选择任务到PostgreSQL数据库

用途：将历史特征选择任务从文件系统同步到数据库，确保数据一致性
"""

import json
import os
import sys
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def sync_tasks():
    """同步任务"""
    # 数据目录
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "feature_selection_tasks")
    
    if not os.path.exists(data_dir):
        print(f"❌ 目录不存在: {data_dir}")
        return
    
    # 获取所有任务文件
    task_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    print(f"📁 发现 {len(task_files)} 个任务文件")
    
    # 导入PostgreSQL连接
    try:
        from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection
    except ImportError as e:
        print(f"❌ 无法导入PostgreSQL模块: {e}")
        return
    
    # 连接数据库
    conn = get_db_connection()
    if not conn:
        print("❌ 无法连接到PostgreSQL数据库")
        return
    
    try:
        cursor = conn.cursor()
        
        # 确保表存在
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_selection_tasks (
                task_id VARCHAR(100) PRIMARY KEY,
                task_type VARCHAR(50) NOT NULL DEFAULT 'feature_selection',
                status VARCHAR(20) NOT NULL DEFAULT 'pending',
                progress INTEGER DEFAULT 0,
                symbol VARCHAR(20),
                source_task_id VARCHAR(100),
                selection_method VARCHAR(50),
                n_features INTEGER DEFAULT 10,
                auto_execute BOOLEAN DEFAULT TRUE,
                input_features JSONB,
                total_input_features INTEGER DEFAULT 0,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # 创建索引
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_selection_tasks_status 
            ON feature_selection_tasks(status);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_selection_tasks_symbol 
            ON feature_selection_tasks(symbol);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_selection_tasks_created 
            ON feature_selection_tasks(created_at DESC);
        """)
        
        conn.commit()
        
        # 同步每个任务
        synced_count = 0
        skipped_count = 0
        error_count = 0
        
        for filename in task_files:
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    task = json.load(f)
                
                task_id = task.get('task_id')
                if not task_id:
                    print(f"⚠️ 跳过 {filename}: 缺少task_id")
                    skipped_count += 1
                    continue
                
                # 检查任务是否已存在
                cursor.execute(
                    "SELECT task_id FROM feature_selection_tasks WHERE task_id = %s",
                    (task_id,)
                )
                if cursor.fetchone():
                    print(f"⏭️ 跳过 {task_id}: 已存在")
                    skipped_count += 1
                    continue
                
                # 转换时间戳
                created_at = task.get('created_at') or task.get('start_time') or task.get('saved_at')
                if created_at:
                    created_at = datetime.fromtimestamp(created_at)
                else:
                    created_at = datetime.now()
                
                # 插入任务 - 使用实际的表结构
                cursor.execute("""
                    INSERT INTO feature_selection_tasks (
                        task_id, task_type, status, progress, symbol,
                        parent_task_id, selection_method, top_k,
                        start_time, end_time, processing_time,
                        total_input_features, total_selected_features, symbols_processed,
                        results, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    task_id,
                    task.get('task_type', 'feature_selection'),
                    task.get('status', 'completed'),
                    task.get('progress', 100),
                    task.get('symbol'),
                    task.get('parent_task_id') or task.get('source_task_id') or task.get('batch_id'),
                    task.get('selection_method', 'importance'),
                    task.get('top_k') or task.get('n_features', 10),
                    task.get('start_time'),
                    task.get('end_time'),
                    task.get('processing_time'),
                    task.get('total_input_features', 0),
                    task.get('total_selected_features', 0),
                    task.get('symbols_processed', 1),
                    json.dumps(task.get('results', {})),
                    created_at
                ))
                
                print(f"✅ 已同步: {task_id}")
                synced_count += 1
                
            except Exception as e:
                print(f"❌ 同步 {filename} 失败: {e}")
                error_count += 1
        
        conn.commit()
        cursor.close()
        
        print(f"\n📊 同步结果:")
        print(f"  ✅ 成功: {synced_count}")
        print(f"  ⏭️ 跳过: {skipped_count}")
        print(f"  ❌ 错误: {error_count}")
        print(f"  📁 总计: {len(task_files)}")
        
    except Exception as e:
        print(f"❌ 同步过程出错: {e}")
    finally:
        return_db_connection(conn)

if __name__ == "__main__":
    print("🚀 开始同步特征选择任务到PostgreSQL...\n")
    sync_tasks()
    print("\n✨ 同步完成")
