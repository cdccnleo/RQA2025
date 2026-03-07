#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征选择历史数据迁移脚本

将现有的 JSON 文件数据迁移到 PostgreSQL 数据库
"""

import json
import os
import sys
import psycopg2
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def migrate_data():
    """迁移数据从 JSON 文件到 PostgreSQL"""
    
    # 读取 JSON 文件
    json_file = "data/feature_selection_history.json"
    if not os.path.exists(json_file):
        print(f"JSON 文件不存在: {json_file}")
        return False
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not data:
            print("JSON 文件为空，无需迁移")
            return True
        
        print(f"从 JSON 文件读取了 {len(data)} 条记录")
        
        # 连接 PostgreSQL
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "postgres"),
            port=os.getenv("POSTGRES_PORT", "5432"),
            database=os.getenv("POSTGRES_DB", "rqa2025_prod"),
            user=os.getenv("POSTGRES_USER", "rqa2025_admin"),
            password=os.getenv("POSTGRES_PASSWORD", "SecurePass123!")
        )
        
        migrated_count = 0
        
        with conn.cursor() as cur:
            for record in data:
                try:
                    # 转换时间戳
                    timestamp = record.get('timestamp', datetime.now().timestamp())
                    if isinstance(timestamp, (int, float)):
                        dt = datetime.fromtimestamp(timestamp)
                    else:
                        dt = datetime.now()
                    
                    cur.execute("""
                        INSERT INTO feature_selection_history (
                            selection_id, task_id, timestamp, input_features,
                            input_feature_count, selection_method, selection_params,
                            selected_features, selected_feature_count, selection_ratio,
                            evaluation_metrics, processing_time, notes
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (selection_id) DO NOTHING
                    """, (
                        record.get('selection_id'),
                        record.get('task_id'),
                        dt,
                        json.dumps(record.get('input_features', [])),
                        record.get('input_feature_count', 0),
                        record.get('selection_method', ''),
                        json.dumps(record.get('selection_params', {})),
                        json.dumps(record.get('selected_features', [])),
                        record.get('selected_feature_count', 0),
                        record.get('selection_ratio', 0.0),
                        json.dumps(record.get('evaluation_metrics', {})),
                        record.get('processing_time', 0.0),
                        record.get('notes', '')
                    ))
                    migrated_count += 1
                    
                except Exception as e:
                    print(f"迁移记录失败 {record.get('selection_id')}: {e}")
                    continue
        
        conn.commit()
        conn.close()
        
        print(f"成功迁移 {migrated_count}/{len(data)} 条记录到 PostgreSQL")
        return True
        
    except Exception as e:
        print(f"数据迁移失败: {e}")
        return False

if __name__ == "__main__":
    success = migrate_data()
    sys.exit(0 if success else 1)
