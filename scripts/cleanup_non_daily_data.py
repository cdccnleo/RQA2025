#!/usr/bin/env python3
"""
清理PostgreSQL数据库中非daily数据类型的记录
删除akshare_stock_data表中data_type字段非'daily'的所有数据
"""

import logging
import sys
import subprocess
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/db_cleanup.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

def run_psql_command(sql):
    """
    通过docker exec运行psql命令
    """
    try:
        # 使用docker exec直接连接到PostgreSQL容器
        cmd = [
            'docker', 'exec', '-i', 'rqa2025-postgres',
            'psql', '-U', 'rqa2025_admin', '-d', 'rqa2025_prod',
            '-t', '-A', '-c', sql  # 添加-t -A参数去除格式信息
        ]
        
        logger.debug(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env={'PGPASSWORD': 'SecurePass123!'}
        )
        
        if result.returncode != 0:
            logger.error(f"psql命令执行失败: {result.stderr}")
            return False, result.stderr
        
        return True, result.stdout.strip()
        
    except Exception as e:
        logger.error(f"运行psql命令失败: {e}")
        return False, str(e)

def cleanup_non_daily_data():
    """
    清理非daily数据类型的记录
    """
    try:
        # 1. 先查询非daily数据的数量
        success, output = run_psql_command("""
            SELECT COUNT(*)
            FROM akshare_stock_data
            WHERE data_type != 'daily'
        """)
        
        if not success:
            logger.error("查询非daily数据数量失败")
            return False
        
        non_daily_count = int(output.strip())
        logger.info(f"找到 {non_daily_count} 条非daily数据类型的记录")
        
        if non_daily_count == 0:
            logger.info("无需清理，数据库中没有非daily数据类型的记录")
            return True
        
        # 2. 执行删除操作
        success, output = run_psql_command("""
            DELETE FROM akshare_stock_data
            WHERE data_type != 'daily'
        """)
        
        if not success:
            logger.error("删除非daily数据失败")
            return False
        
        logger.info(f"成功删除非daily数据类型的记录")
        
        # 3. 验证删除结果
        success, output = run_psql_command("""
            SELECT COUNT(*)
            FROM akshare_stock_data
            WHERE data_type != 'daily'
        """)
        
        if not success:
            logger.error("验证删除结果失败")
            return False
        
        remaining_count = int(output.strip())
        logger.info(f"删除后剩余 {remaining_count} 条非daily数据类型的记录")
        
        return True
        
    except Exception as e:
        logger.error(f"清理非daily数据失败: {e}")
        return False

def main():
    """
    主函数
    """
    logger.info("开始执行PostgreSQL数据库非daily数据清理任务")
    
    success = cleanup_non_daily_data()
    
    if success:
        logger.info("✅ 数据库清理任务执行成功")
        sys.exit(0)
    else:
        logger.error("❌ 数据库清理任务执行失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
