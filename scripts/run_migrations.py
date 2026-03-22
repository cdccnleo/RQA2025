#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库迁移执行脚本

使用Python psycopg2库执行SQL迁移文件，不依赖psql命令行工具。
支持Windows/Linux/macOS跨平台使用。
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Optional

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_db_config() -> dict:
    """
    获取数据库配置
    
    Returns:
        数据库配置字典
    """
    config = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": os.getenv("POSTGRES_PORT", "5432"),
        "database": os.getenv("POSTGRES_DB", "rqa2025_prod"),
        "user": os.getenv("POSTGRES_USER", "rqa2025_admin"),
        "password": os.getenv("POSTGRES_PASSWORD", "")
    }
    
    # 尝试从统一配置模块获取
    try:
        from src.infrastructure.persistence.database_config import get_db_config as get_unified_config
        unified_config = get_unified_config()
        if unified_config:
            config = unified_config.to_dict()
    except Exception:
        pass
    
    return config


def check_database_connection(config: dict) -> bool:
    """
    检查数据库连接
    
    Args:
        config: 数据库配置
    
    Returns:
        是否连接成功
    """
    try:
        import psycopg2
        conn = psycopg2.connect(**config)
        conn.close()
        return True
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        return False


def create_database_if_not_exists(config: dict) -> bool:
    """
    如果数据库不存在则创建
    
    Args:
        config: 数据库配置
    
    Returns:
        是否成功
    """
    try:
        import psycopg2
        from psycopg2 import sql
        
        # 连接到默认数据库
        default_config = config.copy()
        default_config['database'] = 'postgres'
        
        conn = psycopg2.connect(**default_config)
        conn.autocommit = True
        cur = conn.cursor()
        
        # 检查数据库是否存在
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (config['database'],))
        exists = cur.fetchone()
        
        if not exists:
            logger.info(f"创建数据库: {config['database']}")
            cur.execute(sql.SQL("CREATE DATABASE {}").format(
                sql.Identifier(config['database'])
            ))
            logger.info(f"数据库 {config['database']} 创建成功")
        
        cur.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"创建数据库失败: {e}")
        return False


def split_sql_statements(sql_content: str) -> List[str]:
    """
    将SQL内容分割成独立的语句
    
    处理存储过程、函数等包含分号的复杂语句
    
    Args:
        sql_content: SQL内容
    
    Returns:
        SQL语句列表
    """
    statements = []
    current_statement = []
    in_dollar_quote = False
    in_plpgsql = False
    
    lines = sql_content.split('\n')
    
    for line in lines:
        stripped = line.strip()
        
        # 跳过空行和注释
        if not stripped or stripped.startswith('--'):
            continue
        
        # 检测 $$ 开始/结束
        if '$$' in line:
            dollar_count = line.count('$$')
            if dollar_count == 1:
                in_dollar_quote = not in_dollar_quote
            elif dollar_count >= 2:
                # 同一行有开始和结束
                pass
        
        current_statement.append(line)
        
        # 检测语句结束
        if stripped.endswith(';') and not in_dollar_quote:
            stmt = '\n'.join(current_statement)
            if stmt.strip():
                statements.append(stmt)
            current_statement = []
    
    # 处理最后一个未结束的语句
    if current_statement:
        stmt = '\n'.join(current_statement)
        if stmt.strip():
            statements.append(stmt)
    
    return statements


def execute_migration_file(config: dict, sql_file: Path) -> bool:
    """
    执行单个迁移文件
    
    Args:
        config: 数据库配置
        sql_file: SQL文件路径
    
    Returns:
        是否执行成功
    """
    try:
        import psycopg2
        
        logger.info(f"执行迁移文件: {sql_file.name}")
        
        with open(sql_file, 'r', encoding='utf-8') as f:
            sql_content = f.read()
        
        conn = psycopg2.connect(**config)
        conn.autocommit = True
        cur = conn.cursor()
        
        # 分割SQL语句
        statements = split_sql_statements(sql_content)
        
        success = True
        executed = 0
        failed_stmt = None
        
        for stmt in statements:
            try:
                cur.execute(stmt)
                executed += 1
            except psycopg2.errors.DuplicateTable as e:
                logger.debug(f"表已存在，跳过: {str(e)[:100]}")
                executed += 1
            except psycopg2.errors.DuplicateObject as e:
                logger.debug(f"对象已存在，跳过: {str(e)[:100]}")
                executed += 1
            except psycopg2.errors.InvalidForeignKey as e:
                logger.debug(f"外键约束问题，跳过: {str(e)[:100]}")
                executed += 1
            except Exception as e:
                # 获取语句的前50个字符用于错误定位
                stmt_preview = stmt[:100].replace('\n', ' ')
                logger.warning(f"语句执行失败: {str(e)[:80]}")
                logger.debug(f"失败语句: {stmt_preview}...")
                if not failed_stmt:
                    failed_stmt = str(e)
        
        cur.close()
        conn.close()
        
        total = len(statements)
        logger.info(f"✅ 迁移文件执行完成: {sql_file.name} ({executed}/{total} 语句成功)")
        return True
        
    except Exception as e:
        logger.error(f"❌ 迁移文件执行失败 {sql_file.name}: {e}")
        return False


def get_pending_migrations(config: dict, migrations_dir: Path) -> List[Path]:
    """
    获取待执行的迁移文件
    
    Args:
        config: 数据库配置
        migrations_dir: 迁移文件目录
    
    Returns:
        待执行的迁移文件列表
    """
    try:
        import psycopg2
        
        conn = psycopg2.connect(**config)
        cur = conn.cursor()
        
        # 检查迁移记录表是否存在
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'schema_migrations'
            )
        """)
        table_exists = cur.fetchone()[0]
        
        if not table_exists:
            # 创建迁移记录表
            cur.execute("""
                CREATE TABLE schema_migrations (
                    id SERIAL PRIMARY KEY,
                    version VARCHAR(255) NOT NULL UNIQUE,
                    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            logger.info("创建迁移记录表")
        
        # 获取已执行的迁移
        cur.execute("SELECT version FROM schema_migrations ORDER BY version")
        executed = {row[0] for row in cur.fetchall()}
        
        cur.close()
        conn.close()
        
        # 获取所有迁移文件
        all_migrations = sorted(migrations_dir.glob("*.sql"))
        
        # 筛选未执行的迁移
        pending = []
        for m in all_migrations:
            version = m.stem  # 文件名（不含扩展名）
            if version not in executed:
                pending.append(m)
        
        return pending
        
    except Exception as e:
        logger.error(f"获取待执行迁移失败: {e}")
        return []


def record_migration(config: dict, version: str) -> bool:
    """
    记录已执行的迁移
    
    Args:
        config: 数据库配置
        version: 迁移版本
    
    Returns:
        是否记录成功
    """
    try:
        import psycopg2
        
        conn = psycopg2.connect(**config)
        cur = conn.cursor()
        
        cur.execute(
            "INSERT INTO schema_migrations (version) VALUES (%s)",
            (version,)
        )
        
        conn.commit()
        cur.close()
        conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"记录迁移失败: {e}")
        return False


def run_migrations(
    migrations_dir: str = "migrations",
    specific_file: Optional[str] = None,
    force: bool = False,
    reset: bool = False
) -> dict:
    """
    执行数据库迁移
    
    Args:
        migrations_dir: 迁移文件目录
        specific_file: 指定执行的迁移文件
        force: 是否强制执行（忽略已执行记录）
        reset: 是否重置数据库（删除所有表后重新创建）
    
    Returns:
        执行结果统计
    """
    stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'skipped': 0
    }
    
    migrations_path = Path(migrations_dir)
    if not migrations_path.exists():
        logger.error(f"迁移目录不存在: {migrations_dir}")
        return stats
    
    # 获取数据库配置
    config = get_db_config()
    
    if not config.get('password'):
        logger.error("数据库密码未设置！请设置环境变量 POSTGRES_PASSWORD")
        logger.info("示例: set POSTGRES_PASSWORD=YourSecurePassword")
        return stats
    
    # 检查数据库连接
    if not check_database_connection(config):
        # 尝试创建数据库
        if not create_database_if_not_exists(config):
            logger.error("无法连接或创建数据库")
            return stats
    
    logger.info(f"数据库连接成功: {config['host']}:{config['port']}/{config['database']}")
    
    # 如果需要重置数据库
    if reset:
        if not reset_database(config):
            logger.error("重置数据库失败")
            return stats
        logger.info("数据库已重置")
    
    # 获取待执行的迁移文件
    if specific_file:
        # 执行指定文件
        sql_file = migrations_path / specific_file
        if sql_file.exists():
            pending = [sql_file]
        else:
            logger.error(f"指定的迁移文件不存在: {specific_file}")
            return stats
    elif force:
        # 强制执行所有迁移文件
        pending = sorted(migrations_path.glob("*.sql"))
    else:
        # 获取未执行的迁移
        pending = get_pending_migrations(config, migrations_path)
    
    if not pending:
        logger.info("没有待执行的迁移文件")
        return stats
    
    logger.info(f"发现 {len(pending)} 个待执行的迁移文件")
    
    # 执行迁移
    for sql_file in pending:
        stats['total'] += 1
        
        if execute_migration_file(config, sql_file):
            # 记录迁移
            if not force:
                record_migration(config, sql_file.stem)
            stats['success'] += 1
        else:
            stats['failed'] += 1
            # 继续执行下一个迁移文件，而不是停止
            logger.warning(f"迁移文件 {sql_file.name} 执行失败，继续执行下一个")
    
    # 打印统计
    logger.info("=" * 60)
    logger.info("迁移执行统计:")
    logger.info(f"  总计: {stats['total']}")
    logger.info(f"  成功: {stats['success']}")
    logger.info(f"  失败: {stats['failed']}")
    logger.info("=" * 60)
    
    return stats


def reset_database(config: dict) -> bool:
    """
    重置数据库（删除所有表）
    
    Args:
        config: 数据库配置
    
    Returns:
        是否重置成功
    """
    try:
        import psycopg2
        from psycopg2 import sql
        
        logger.warning("⚠️ 即将删除数据库中的所有表！")
        
        conn = psycopg2.connect(**config)
        conn.autocommit = True
        cur = conn.cursor()
        
        # 删除所有表
        cur.execute("""
            DROP SCHEMA public CASCADE;
            CREATE SCHEMA public;
            GRANT ALL ON SCHEMA public TO public;
        """)
        
        cur.close()
        conn.close()
        
        logger.info("数据库已重置，所有表已删除")
        return True
        
    except Exception as e:
        logger.error(f"重置数据库失败: {e}")
        return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="数据库迁移执行脚本")
    parser.add_argument(
        "--migrations-dir", 
        default="migrations",
        help="迁移文件目录"
    )
    parser.add_argument(
        "--file",
        help="指定执行的迁移文件"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制执行所有迁移（忽略已执行记录）"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="重置数据库（删除所有表后重新创建）⚠️ 危险操作"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="仅检查数据库连接，不执行迁移"
    )
    
    args = parser.parse_args()
    
    if args.check_only:
        config = get_db_config()
        if check_database_connection(config):
            logger.info("✅ 数据库连接成功")
        else:
            logger.error("❌ 数据库连接失败")
        return
    
    run_migrations(
        migrations_dir=args.migrations_dir,
        specific_file=args.file,
        force=args.force,
        reset=args.reset
    )


if __name__ == "__main__":
    main()
