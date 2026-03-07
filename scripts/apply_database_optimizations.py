#!/usr/bin/env python3
"""
应用数据库优化脚本
"""

import time
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_sql_file(file_path: str) -> str:
    """读取SQL文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"读取SQL文件失败: {e}")
        return ""

def execute_sql_via_docker(sql_content: str):
    """通过Docker执行SQL"""
    import subprocess
    import tempfile
    import os

    try:
        # 创建临时SQL文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sql', delete=False, encoding='utf-8') as f:
            f.write(sql_content)
            temp_sql_file = f.name

        try:
            # 复制文件到容器
            copy_cmd = f"docker cp {temp_sql_file} rqa2025-postgres:/tmp/optimize_schema.sql"
            result = subprocess.run(copy_cmd, shell=True, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"复制文件到容器失败: {result.stderr}")
                return False

            # 在容器中执行SQL
            exec_cmd = "docker exec rqa2025-postgres psql -U rqa2025_admin -d rqa2025_prod -f /tmp/optimize_schema.sql"
            result = subprocess.run(exec_cmd, shell=True, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("数据库优化脚本执行成功")
                logger.info("执行输出:")
                logger.info(result.stdout)
                return True
            else:
                logger.error(f"数据库优化脚本执行失败: {result.stderr}")
                return False

        finally:
            # 清理临时文件
            try:
                os.unlink(temp_sql_file)
            except:
                pass

    except Exception as e:
        logger.error(f"执行SQL失败: {e}")
        return False

def main():
    """主函数"""
    logger.info("开始应用数据库优化...")

    # 检查SQL文件是否存在
    sql_file = Path("scripts/optimize_postgresql_schema.sql")
    if not sql_file.exists():
        logger.error(f"SQL文件不存在: {sql_file}")
        return False

    # 读取SQL内容
    sql_content = read_sql_file(str(sql_file))
    if not sql_content:
        return False

    logger.info(f"读取到SQL内容，长度: {len(sql_content)} 字符")

    # 执行SQL
    success = execute_sql_via_docker(sql_content)

    if success:
        logger.info("✅ 数据库优化应用完成！")
        logger.info("优化的内容包括:")
        logger.info("  - 复合索引优化")
        logger.info("  - 数据完整性约束")
        logger.info("  - 自动更新触发器")
        logger.info("  - 性能统计表")
    else:
        logger.error("❌ 数据库优化应用失败")

    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)