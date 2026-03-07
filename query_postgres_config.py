#!/usr/bin/env python3
"""
从PostgreSQL数据库查询生产环境的数据源配置
"""
import psycopg2
import json

# 数据库连接信息
DB_HOST = 'localhost'
DB_PORT = '5432'
DB_NAME = 'rqa2025'
DB_USER = 'rqa2025_admin'
DB_PASSWORD = 'password123!@#'

def get_production_configs():
    """
    从PostgreSQL数据库获取生产环境的数据源配置
    """
    try:
        # 连接数据库
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        
        # 创建游标
        cur = conn.cursor()
        
        # 查询生产环境的数据源配置
        query = """
        SELECT id, source_name, config, rate_limit, enabled, environment, created_at, updated_at 
        FROM data_source_configs 
        WHERE environment = 'production'
        ORDER BY id
        """
        cur.execute(query)
        
        # 获取结果
        rows = cur.fetchall()
        
        # 关闭游标和连接
        cur.close()
        conn.close()
        
        # 处理结果
        configs = []
        for row in rows:
            config_item = {
                "id": row[0],
                "source_name": row[1],
                "config": json.loads(row[2]) if row[2] else {},
                "rate_limit": row[3],
                "enabled": row[4],
                "environment": row[5],
                "created_at": row[6].isoformat() if row[6] else None,
                "updated_at": row[7].isoformat() if row[7] else None
            }
            configs.append(config_item)
        
        return configs
        
    except Exception as e:
        print(f"Error querying database: {e}")
        return []

if __name__ == "__main__":
    # 获取配置
    configs = get_production_configs()
    
    # 输出配置
    print("Production environment data source configs:")
    print(json.dumps(configs, ensure_ascii=False, indent=2))
    
    # 保存到临时文件
    with open('production_configs.json', 'w', encoding='utf-8') as f:
        json.dump(configs, f, ensure_ascii=False, indent=2)
    
    print("\nConfigs saved to production_configs.json")
