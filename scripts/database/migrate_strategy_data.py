"""
策略数据迁移脚本
将现有文件系统中的策略数据导入到PostgreSQL数据库
支持策略构思、策略回测等环节的数据迁移
"""

import os
import json
import logging
import argparse
from typing import List, Dict, Any
import time
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 数据存储目录
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
STRATEGY_CONCEPTIONS_DIR = os.path.join(DATA_DIR, "strategy_conceptions")
BACKTEST_RESULTS_DIR = os.path.join(DATA_DIR, "backtest_results")

# 延迟导入数据库模块
db_available = False
try:
    from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection
    db_available = True
    logger.info("数据库连接模块加载成功")
except ImportError as e:
    logger.error(f"加载数据库连接模块失败: {e}")
    db_available = False


class StrategyDataMigrator:
    """
    策略数据迁移类
    """
    
    def __init__(self):
        self.conn = None
        self.cursor = None
    
    def connect(self):
        """
        连接数据库
        """
        if not db_available:
            logger.error("数据库连接模块不可用")
            return False
        
        try:
            self.conn = get_db_connection()
            if not self.conn:
                logger.error("无法获取数据库连接")
                return False
            
            self.cursor = self.conn.cursor()
            logger.info("数据库连接成功")
            return True
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            return False
    
    def disconnect(self):
        """
        断开数据库连接
        """
        try:
            if self.cursor:
                self.cursor.close()
            if self.conn:
                return_db_connection(self.conn)
            logger.info("数据库连接已关闭")
        except Exception as e:
            logger.error(f"关闭数据库连接失败: {e}")
    
    def create_tables(self):
        """
        创建必要的数据库表
        """
        if not self.conn or not self.cursor:
            logger.error("数据库连接未初始化")
            return False
        
        try:
            # 创建策略构思表
            logger.info("创建策略构思表...")
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategy_conceptions (
                    id VARCHAR(255) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    type VARCHAR(100) NOT NULL,
                    description TEXT,
                    target_market VARCHAR(100),
                    risk_level VARCHAR(50),
                    nodes JSONB NOT NULL,
                    connections JSONB NOT NULL,
                    parameters JSONB,
                    backtest_result JSONB,
                    version INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    saved_locally BOOLEAN DEFAULT FALSE
                );
            """)
            
            # 创建索引
            self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_strategy_conceptions_type ON strategy_conceptions(type);
                CREATE INDEX IF NOT EXISTS idx_strategy_conceptions_created_at ON strategy_conceptions(created_at);
            """)
            
            # 创建回测结果表
            logger.info("创建回测结果表...")
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS backtest_results (
                    backtest_id VARCHAR(100) PRIMARY KEY,
                    strategy_id VARCHAR(100) NOT NULL,
                    status VARCHAR(20) NOT NULL,
                    start_date DATE NOT NULL,
                    end_date DATE NOT NULL,
                    initial_capital DECIMAL(18, 2) NOT NULL,
                    final_capital DECIMAL(18, 2),
                    total_return DECIMAL(10, 4),
                    annualized_return DECIMAL(10, 4),
                    sharpe_ratio DECIMAL(10, 4),
                    max_drawdown DECIMAL(10, 4),
                    win_rate DECIMAL(10, 4),
                    total_trades INTEGER,
                    equity_curve JSONB,
                    trades JSONB,
                    metrics JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # 创建回测结果表索引
            self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_backtest_strategy ON backtest_results(strategy_id);
                CREATE INDEX IF NOT EXISTS idx_backtest_created ON backtest_results(created_at DESC);
            """)
            
            self.conn.commit()
            logger.info("数据库表创建成功")
            return True
        except Exception as e:
            logger.error(f"创建数据库表失败: {e}")
            if self.conn:
                self.conn.rollback()
            return False
    
    def migrate_strategy_conceptions(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        迁移策略构思数据
        
        Args:
            dry_run: 是否仅模拟迁移，不实际执行
        
        Returns:
            迁移结果统计
        """
        stats = {
            "total_files": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "errors": []
        }
        
        if not os.path.exists(STRATEGY_CONCEPTIONS_DIR):
            logger.warning(f"策略构思目录不存在: {STRATEGY_CONCEPTIONS_DIR}")
            return stats
        
        logger.info(f"开始迁移策略构思数据，目录: {STRATEGY_CONCEPTIONS_DIR}")
        
        # 获取所有策略构思文件
        files = []
        for filename in os.listdir(STRATEGY_CONCEPTIONS_DIR):
            if filename.endswith('.json'):
                files.append(filename)
        
        stats["total_files"] = len(files)
        logger.info(f"发现 {len(files)} 个策略构思文件")
        
        for filename in files:
            filepath = os.path.join(STRATEGY_CONCEPTIONS_DIR, filename)
            logger.info(f"处理文件: {filename}")
            
            try:
                # 读取文件
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 验证数据
                if not data.get('id'):
                    logger.warning(f"文件 {filename} 缺少id字段，跳过")
                    stats["skipped"] += 1
                    continue
                
                strategy_id = data.get('id')
                
                # 检查是否已存在
                self.cursor.execute(
                    "SELECT 1 FROM strategy_conceptions WHERE id = %s",
                    (strategy_id,)
                )
                exists = self.cursor.fetchone() is not None
                
                if exists:
                    logger.info(f"策略 {strategy_id} 已存在，跳过")
                    stats["skipped"] += 1
                    continue
                
                # 准备数据
                insert_data = {
                    'id': data.get('id'),
                    'name': data.get('name', ''),
                    'type': data.get('type', ''),
                    'description': data.get('description'),
                    'target_market': data.get('targetMarket'),
                    'risk_level': data.get('riskLevel'),
                    'nodes': json.dumps(data.get('nodes', [])),
                    'connections': json.dumps(data.get('connections', [])),
                    'parameters': json.dumps(data.get('parameters', {})),
                    'backtest_result': json.dumps(data.get('backtest_result', {})),
                    'version': data.get('version', 1),
                    'created_at': datetime.fromtimestamp(data.get('created_at', time.time())),
                    'updated_at': datetime.fromtimestamp(data.get('updated_at', time.time())),
                    'saved_locally': True
                }
                
                if dry_run:
                    logger.info(f"[DRY RUN] 准备导入策略: {strategy_id}")
                    stats["success"] += 1
                else:
                    # 执行插入
                    self.cursor.execute("""
                        INSERT INTO strategy_conceptions (
                            id, name, type, description, target_market, risk_level,
                            nodes, connections, parameters, backtest_result, version,
                            created_at, updated_at, saved_locally
                        ) VALUES (
                            %(id)s, %(name)s, %(type)s, %(description)s, %(target_market)s, %(risk_level)s,
                            %(nodes)s, %(connections)s, %(parameters)s, %(backtest_result)s, %(version)s,
                            %(created_at)s, %(updated_at)s, %(saved_locally)s
                        )
                    """, insert_data)
                    
                    self.conn.commit()
                    logger.info(f"策略 {strategy_id} 导入成功")
                    stats["success"] += 1
                    
            except Exception as e:
                logger.error(f"处理文件 {filename} 失败: {e}")
                stats["failed"] += 1
                stats["errors"].append({
                    "file": filename,
                    "error": str(e)
                })
                if self.conn:
                    self.conn.rollback()
        
        logger.info(f"策略构思数据迁移完成: 总计 {stats['total_files']}, 成功 {stats['success']}, 失败 {stats['failed']}, 跳过 {stats['skipped']}")
        return stats
    
    def migrate_backtest_results(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        迁移回测结果数据
        
        Args:
            dry_run: 是否仅模拟迁移，不实际执行
        
        Returns:
            迁移结果统计
        """
        stats = {
            "total_files": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "errors": []
        }
        
        if not os.path.exists(BACKTEST_RESULTS_DIR):
            logger.warning(f"回测结果目录不存在: {BACKTEST_RESULTS_DIR}")
            return stats
        
        logger.info(f"开始迁移回测结果数据，目录: {BACKTEST_RESULTS_DIR}")
        
        # 获取所有回测结果文件
        files = []
        for filename in os.listdir(BACKTEST_RESULTS_DIR):
            if filename.endswith('.json'):
                files.append(filename)
        
        stats["total_files"] = len(files)
        logger.info(f"发现 {len(files)} 个回测结果文件")
        
        for filename in files:
            filepath = os.path.join(BACKTEST_RESULTS_DIR, filename)
            logger.info(f"处理文件: {filename}")
            
            try:
                # 读取文件
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 验证数据
                backtest_id = data.get('backtest_id')
                if not backtest_id:
                    logger.warning(f"文件 {filename} 缺少backtest_id字段，跳过")
                    stats["skipped"] += 1
                    continue
                
                # 检查是否已存在
                self.cursor.execute(
                    "SELECT 1 FROM backtest_results WHERE backtest_id = %s",
                    (backtest_id,)
                )
                exists = self.cursor.fetchone() is not None
                
                if exists:
                    logger.info(f"回测结果 {backtest_id} 已存在，跳过")
                    stats["skipped"] += 1
                    continue
                
                # 准备数据
                insert_data = {
                    'backtest_id': backtest_id,
                    'strategy_id': data.get('strategy_id', ''),
                    'status': data.get('status', 'completed'),
                    'start_date': data.get('start_date'),
                    'end_date': data.get('end_date'),
                    'initial_capital': data.get('initial_capital', 1000000),
                    'final_capital': data.get('final_capital'),
                    'total_return': data.get('total_return'),
                    'annualized_return': data.get('annualized_return'),
                    'sharpe_ratio': data.get('sharpe_ratio'),
                    'max_drawdown': data.get('max_drawdown'),
                    'win_rate': data.get('win_rate'),
                    'total_trades': data.get('total_trades'),
                    'equity_curve': json.dumps(data.get('equity_curve', [])),
                    'trades': json.dumps(data.get('trades', [])),
                    'metrics': json.dumps(data.get('metrics', {})),
                    'created_at': datetime.fromtimestamp(data.get('created_at', time.time())),
                    'updated_at': datetime.fromtimestamp(data.get('updated_at', time.time()))
                }
                
                if dry_run:
                    logger.info(f"[DRY RUN] 准备导入回测结果: {backtest_id}")
                    stats["success"] += 1
                else:
                    # 执行插入
                    self.cursor.execute("""
                        INSERT INTO backtest_results (
                            backtest_id, strategy_id, status, start_date, end_date,
                            initial_capital, final_capital, total_return, annualized_return,
                            sharpe_ratio, max_drawdown, win_rate, total_trades,
                            equity_curve, trades, metrics, created_at, updated_at
                        ) VALUES (
                            %(backtest_id)s, %(strategy_id)s, %(status)s, %(start_date)s, %(end_date)s,
                            %(initial_capital)s, %(final_capital)s, %(total_return)s, %(annualized_return)s,
                            %(sharpe_ratio)s, %(max_drawdown)s, %(win_rate)s, %(total_trades)s,
                            %(equity_curve)s, %(trades)s, %(metrics)s, %(created_at)s, %(updated_at)s
                        )
                    """, insert_data)
                    
                    self.conn.commit()
                    logger.info(f"回测结果 {backtest_id} 导入成功")
                    stats["success"] += 1
                    
            except Exception as e:
                logger.error(f"处理文件 {filename} 失败: {e}")
                stats["failed"] += 1
                stats["errors"].append({
                    "file": filename,
                    "error": str(e)
                })
                if self.conn:
                    self.conn.rollback()
        
        logger.info(f"回测结果数据迁移完成: 总计 {stats['total_files']}, 成功 {stats['success']}, 失败 {stats['failed']}, 跳过 {stats['skipped']}")
        return stats
    
    def validate_migration(self) -> Dict[str, Any]:
        """
        验证迁移结果
        
        Returns:
            验证结果
        """
        validation = {
            "strategy_conceptions": {
                "file_count": 0,
                "db_count": 0,
                "match": False
            },
            "backtest_results": {
                "file_count": 0,
                "db_count": 0,
                "match": False
            }
        }
        
        # 验证策略构思
        if os.path.exists(STRATEGY_CONCEPTIONS_DIR):
            file_count = len([f for f in os.listdir(STRATEGY_CONCEPTIONS_DIR) if f.endswith('.json')])
            validation["strategy_conceptions"]["file_count"] = file_count
        
        try:
            self.cursor.execute("SELECT COUNT(*) FROM strategy_conceptions")
            db_count = self.cursor.fetchone()[0]
            validation["strategy_conceptions"]["db_count"] = db_count
            validation["strategy_conceptions"]["match"] = (
                validation["strategy_conceptions"]["file_count"] == db_count
            )
        except Exception as e:
            logger.error(f"验证策略构思数据失败: {e}")
        
        # 验证回测结果
        if os.path.exists(BACKTEST_RESULTS_DIR):
            file_count = len([f for f in os.listdir(BACKTEST_RESULTS_DIR) if f.endswith('.json')])
            validation["backtest_results"]["file_count"] = file_count
        
        try:
            self.cursor.execute("SELECT COUNT(*) FROM backtest_results")
            db_count = self.cursor.fetchone()[0]
            validation["backtest_results"]["db_count"] = db_count
            validation["backtest_results"]["match"] = (
                validation["backtest_results"]["file_count"] == db_count
            )
        except Exception as e:
            logger.error(f"验证回测结果数据失败: {e}")
        
        logger.info("迁移验证结果:")
        logger.info(f"策略构思: 文件 {validation['strategy_conceptions']['file_count']}, 数据库 {validation['strategy_conceptions']['db_count']}, 匹配 {validation['strategy_conceptions']['match']}")
        logger.info(f"回测结果: 文件 {validation['backtest_results']['file_count']}, 数据库 {validation['backtest_results']['db_count']}, 匹配 {validation['backtest_results']['match']}")
        
        return validation


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="策略数据迁移脚本")
    parser.add_argument('--dry-run', action='store_true', help="仅模拟迁移，不实际执行")
    parser.add_argument('--skip-tables', action='store_true', help="跳过创建数据库表")
    parser.add_argument('--validate-only', action='store_true', help="仅验证迁移结果，不执行迁移")
    args = parser.parse_args()
    
    migrator = StrategyDataMigrator()
    
    try:
        # 连接数据库
        if not migrator.connect():
            logger.error("无法连接数据库，退出")
            return 1
        
        # 创建表（如果需要）
        if not args.skip_tables:
            if not migrator.create_tables():
                logger.error("创建数据库表失败，退出")
                return 1
        
        # 仅验证模式
        if args.validate_only:
            migrator.validate_migration()
            return 0
        
        # 执行迁移
        logger.info("开始数据迁移...")
        
        # 迁移策略构思
        strategy_stats = migrator.migrate_strategy_conceptions(args.dry_run)
        
        # 迁移回测结果
        backtest_stats = migrator.migrate_backtest_results(args.dry_run)
        
        # 验证迁移结果
        if not args.dry_run:
            migrator.validate_migration()
        
        # 打印迁移摘要
        logger.info("\n=== 迁移摘要 ===")
        logger.info(f"策略构思: 总计 {strategy_stats['total_files']}, 成功 {strategy_stats['success']}, 失败 {strategy_stats['failed']}, 跳过 {strategy_stats['skipped']}")
        logger.info(f"回测结果: 总计 {backtest_stats['total_files']}, 成功 {backtest_stats['success']}, 失败 {backtest_stats['failed']}, 跳过 {backtest_stats['skipped']}")
        
        if strategy_stats['errors']:
            logger.warning("策略构思迁移错误:")
            for error in strategy_stats['errors'][:5]:  # 只显示前5个错误
                logger.warning(f"  - {error['file']}: {error['error']}")
        
        if backtest_stats['errors']:
            logger.warning("回测结果迁移错误:")
            for error in backtest_stats['errors'][:5]:  # 只显示前5个错误
                logger.warning(f"  - {error['file']}: {error['error']}")
        
        logger.info("\n迁移完成！")
        return 0
        
    except Exception as e:
        logger.error(f"迁移过程中发生错误: {e}")
        return 1
    finally:
        migrator.disconnect()


if __name__ == "__main__":
    exit(main())
