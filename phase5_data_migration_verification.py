#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 5: 数据迁移验证

创建数据迁移验证脚本，确保生产数据安全迁移
包括数据完整性、一致性、质量评估、性能监控和回滚测试
"""

import hashlib
import json
import os
import shutil
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import logging
import concurrent.futures
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import random

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class MigrationConfig:
    """迁移配置"""
    source_db_path: str = "data/source_database.db"
    target_db_path: str = "data/target_database.db"
    backup_path: str = "data/backup_database.db"
    batch_size: int = 1000
    max_workers: int = 4
    enable_compression: bool = True
    enable_encryption: bool = False
    verify_integrity: bool = True


@dataclass
class MigrationStats:
    """迁移统计"""
    total_records: int = 0
    migrated_records: int = 0
    failed_records: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    data_size_bytes: int = 0
    checksum_source: str = ""
    checksum_target: str = ""


@dataclass
class DataQualityMetrics:
    """数据质量指标"""
    completeness_score: float = 0.0  # 完整性得分 0-1
    accuracy_score: float = 0.0     # 准确性得分 0-1
    consistency_score: float = 0.0  # 一致性得分 0-1
    timeliness_score: float = 0.0   # 时效性得分 0-1
    overall_score: float = 0.0      # 综合得分 0-1


@dataclass
class ValidationResult:
    """验证结果"""
    is_success: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class DataMigrationVerifier:
    """数据迁移验证器"""

    def __init__(self, config: MigrationConfig):
        self.config = config
        self.stats = MigrationStats()
        self.quality_metrics = DataQualityMetrics()
        self.validation_results: List[ValidationResult] = []
        self.error_log: List[Dict[str, Any]] = []

        # 确保数据目录存在
        os.makedirs("data", exist_ok=True)
        os.makedirs("test_logs", exist_ok=True)

        logger.info("数据迁移验证器初始化完成")

    def run_full_migration_test(self) -> bool:
        """运行完整迁移测试"""
        try:
            logger.info("🚀 开始完整数据迁移验证测试")

            # 1. 准备测试数据
            self._prepare_test_data()

            # 2. 执行数据迁移
            migration_success = self._execute_migration()

            # 3. 验证迁移结果
            validation_success = self._validate_migration()

            # 4. 数据质量评估
            quality_success = self._assess_data_quality()

            # 5. 性能评估
            performance_success = self._evaluate_performance()

            # 6. 回滚测试
            rollback_success = self._test_rollback_capability()

            # 7. 生成报告
            self._generate_migration_report()

            overall_success = all([
                migration_success, validation_success, quality_success,
                performance_success, rollback_success
            ])

            if overall_success:
                logger.info("✅ 数据迁移验证测试完成 - 所有验证通过")
            else:
                logger.error("❌ 数据迁移验证测试完成 - 发现问题需要修复")

            return overall_success

        except Exception as e:
            logger.error(f"数据迁移验证测试失败: {e}")
            self._log_error("migration_test_failed", str(e))
            return False

    def _prepare_test_data(self):
        """准备测试数据"""
        logger.info("📊 准备测试数据...")

        # 删除可能存在的旧数据库文件
        for db_path in [self.config.source_db_path, self.config.target_db_path, self.config.backup_path]:
            if os.path.exists(db_path):
                os.remove(db_path)

        # 创建源数据库和表结构
        self._create_source_database()

        # 生成测试数据
        self._generate_test_data()

        # 计算源数据校验和
        self.stats.checksum_source = self._calculate_database_checksum(self.config.source_db_path)

        logger.info(f"✅ 测试数据准备完成，共生成 {self.stats.total_records} 条记录")

    def _create_source_database(self):
        """创建源数据库"""
        conn = sqlite3.connect(self.config.source_db_path)
        cursor = conn.cursor()

        # 创建用户表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'active',
                balance REAL DEFAULT 0.0,
                risk_level TEXT DEFAULT 'low'
            )
        ''')

        # 创建订单表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                order_id INTEGER PRIMARY KEY,
                user_id INTEGER,
                symbol TEXT NOT NULL,
                order_type TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')

        # 创建交易记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                trade_id INTEGER PRIMARY KEY,
                order_id INTEGER,
                user_id INTEGER,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                fee REAL DEFAULT 0.0,
                FOREIGN KEY (order_id) REFERENCES orders (order_id),
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')

        # 创建持仓表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                position_id INTEGER PRIMARY KEY,
                user_id INTEGER,
                symbol TEXT NOT NULL,
                quantity REAL NOT NULL,
                avg_price REAL NOT NULL,
                current_price REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id),
                UNIQUE(user_id, symbol)
            )
        ''')

        conn.commit()
        conn.close()

    def _generate_test_data(self):
        """生成测试数据"""
        conn = sqlite3.connect(self.config.source_db_path)
        cursor = conn.cursor()

        # 生成用户数据
        users_data = []
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX']

        for i in range(1000):  # 1000个用户
            user = {
                'user_id': i + 1,
                'username': f'user_{i+1:04d}',
                'email': f'user_{i+1:04d}@example.com',
                'created_at': datetime.now() - timedelta(days=random.randint(1, 365)),
                'updated_at': datetime.now() - timedelta(hours=random.randint(1, 24)),
                'status': random.choice(['active', 'inactive', 'suspended']),
                'balance': round(random.uniform(1000, 100000), 2),
                'risk_level': random.choice(['low', 'medium', 'high'])
            }
            users_data.append(user)

        cursor.executemany('''
            INSERT OR REPLACE INTO users (user_id, username, email, created_at, updated_at, status, balance, risk_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', [(u['user_id'], u['username'], u['email'], u['created_at'], u['updated_at'],
               u['status'], u['balance'], u['risk_level']) for u in users_data])

        # 生成订单数据
        orders_data = []
        for i in range(5000):  # 5000个订单
            order = {
                'order_id': i + 1,
                'user_id': random.randint(1, 1000),
                'symbol': random.choice(symbols),
                'order_type': random.choice(['market', 'limit', 'stop']),
                'quantity': round(random.uniform(1, 100), 2),
                'price': round(random.uniform(100, 1000), 2) if random.random() > 0.3 else None,
                'status': random.choice(['pending', 'filled', 'cancelled', 'partially_filled']),
                'created_at': datetime.now() - timedelta(days=random.randint(1, 30)),
                'updated_at': datetime.now() - timedelta(hours=random.randint(1, 24))
            }
            orders_data.append(order)

        cursor.executemany('''
            INSERT OR REPLACE INTO orders (order_id, user_id, symbol, order_type, quantity, price, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', [(o['order_id'], o['user_id'], o['symbol'], o['order_type'], o['quantity'],
               o['price'], o['status'], o['created_at'], o['updated_at']) for o in orders_data])

        # 生成交易记录
        trades_data = []
        for i in range(3000):  # 3000条交易记录
            trade = {
                'trade_id': i + 1,
                'order_id': random.randint(1, 5000),
                'user_id': random.randint(1, 1000),
                'symbol': random.choice(symbols),
                'side': random.choice(['buy', 'sell']),
                'quantity': round(random.uniform(1, 50), 2),
                'price': round(random.uniform(100, 1000), 2),
                'executed_at': datetime.now() - timedelta(days=random.randint(1, 30)),
                'fee': round(random.uniform(0, 10), 2)
            }
            trades_data.append(trade)

        cursor.executemany('''
            INSERT OR REPLACE INTO trades (trade_id, order_id, user_id, symbol, side, quantity, price, executed_at, fee)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', [(t['trade_id'], t['order_id'], t['user_id'], t['symbol'], t['side'],
               t['quantity'], t['price'], t['executed_at'], t['fee']) for t in trades_data])

        # 生成持仓数据
        positions_data = []
        for i in range(2000):  # 2000个持仓记录
            position = {
                'position_id': i + 1,
                'user_id': random.randint(1, 1000),
                'symbol': random.choice(symbols),
                'quantity': round(random.uniform(-100, 100), 2),  # 可以是负数（空头）
                'avg_price': round(random.uniform(100, 1000), 2),
                'current_price': round(random.uniform(100, 1000), 2),
                'created_at': datetime.now() - timedelta(days=random.randint(1, 180)),
                'updated_at': datetime.now() - timedelta(hours=random.randint(1, 24))
            }
            positions_data.append(position)

        cursor.executemany('''
            INSERT OR REPLACE INTO positions (position_id, user_id, symbol, quantity, avg_price, current_price, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', [(p['position_id'], p['user_id'], p['symbol'], p['quantity'], p['avg_price'],
               p['current_price'], p['created_at'], p['updated_at']) for p in positions_data])

        conn.commit()
        conn.close()

        self.stats.total_records = 1000 + 5000 + 3000 + 2000  # 总记录数

    def _execute_migration(self) -> bool:
        """执行数据迁移"""
        logger.info("🔄 执行数据迁移...")

        try:
            self.stats.start_time = datetime.now()

            # 创建备份
            if os.path.exists(self.config.source_db_path):
                shutil.copy2(self.config.source_db_path, self.config.backup_path)
                logger.info(f"✅ 源数据库备份完成: {self.config.backup_path}")

            # 使用多线程执行迁移
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = []

                # 迁移各个表
                tables = ['users', 'orders', 'trades', 'positions']
                for table in tables:
                    future = executor.submit(self._migrate_table, table)
                    futures.append(future)

                # 等待所有迁移完成
                for future in concurrent.futures.as_completed(futures):
                    success, table_name, migrated_count = future.result()
                    if success:
                        self.stats.migrated_records += migrated_count
                        logger.info(f"✅ 表 {table_name} 迁移完成: {migrated_count} 条记录")
                    else:
                        logger.error(f"❌ 表 {table_name} 迁移失败")
                        return False

            self.stats.end_time = datetime.now()

            # 计算迁移数据大小
            if os.path.exists(self.config.target_db_path):
                self.stats.data_size_bytes = os.path.getsize(self.config.target_db_path)

            # 计算目标数据校验和
            self.stats.checksum_target = self._calculate_database_checksum(
                self.config.target_db_path)

            migration_time = (self.stats.end_time - self.stats.start_time).total_seconds()
            logger.info(f"✅ 数据迁移完成，耗时: {migration_time:.2f}秒")
            return True

        except Exception as e:
            logger.error(f"数据迁移执行失败: {e}")
            self._log_error("migration_execution_failed", str(e))
            return False

    def _migrate_table(self, table_name: str) -> Tuple[bool, str, int]:
        """迁移单个表"""
        try:
            source_conn = sqlite3.connect(self.config.source_db_path)
            target_conn = sqlite3.connect(self.config.target_db_path)

            source_cursor = source_conn.cursor()
            target_cursor = target_conn.cursor()

            # 创建目标表结构
            self._create_target_table(target_cursor, table_name)

            # 获取源数据
            source_cursor.execute(f"SELECT * FROM {table_name}")
            rows = source_cursor.fetchall()

            # 获取列名
            source_cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [col[1] for col in source_cursor.fetchall()]

            # 批量插入数据
            migrated_count = 0
            for i in range(0, len(rows), self.config.batch_size):
                batch = rows[i:i + self.config.batch_size]
                placeholders = ','.join(['?'] * len(columns))
                target_cursor.executemany(
                    f"INSERT OR REPLACE INTO {table_name} ({','.join(columns)}) VALUES ({placeholders})",
                    batch
                )
                migrated_count += len(batch)

            target_conn.commit()
            source_conn.close()
            target_conn.close()

            return True, table_name, migrated_count

        except Exception as e:
            logger.error(f"表 {table_name} 迁移失败: {e}")
            self._log_error(f"table_migration_failed_{table_name}", str(e))
            return False, table_name, 0

    def _create_target_table(self, cursor, table_name: str):
        """创建目标表结构"""
        if table_name == 'users':
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    email TEXT NOT NULL UNIQUE,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    status TEXT,
                    balance REAL,
                    risk_level TEXT
                )
            ''')
        elif table_name == 'orders':
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS orders (
                    order_id INTEGER PRIMARY KEY,
                    user_id INTEGER,
                    symbol TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL,
                    status TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            ''')
        elif table_name == 'trades':
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id INTEGER PRIMARY KEY,
                    order_id INTEGER,
                    user_id INTEGER,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    executed_at TIMESTAMP,
                    fee REAL
                )
            ''')
        elif table_name == 'positions':
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    position_id INTEGER PRIMARY KEY,
                    user_id INTEGER,
                    symbol TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    avg_price REAL NOT NULL,
                    current_price REAL,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            ''')

    def _validate_migration(self) -> bool:
        """验证迁移结果"""
        logger.info("🔍 验证迁移结果...")

        try:
            validation_result = ValidationResult(is_success=True)

            # 1. 记录数量验证
            source_counts = self._get_table_counts(self.config.source_db_path)
            target_counts = self._get_table_counts(self.config.target_db_path)

            for table in ['users', 'orders', 'trades', 'positions']:
                source_count = source_counts.get(table, 0)
                target_count = target_counts.get(table, 0)

                if source_count != target_count:
                    validation_result.is_success = False
                    validation_result.issues.append(
                        f"表 {table} 记录数量不匹配: 源表 {source_count}, 目标表 {target_count}"
                    )

            # 2. 数据完整性验证
            integrity_issues = self._validate_data_integrity()
            validation_result.issues.extend(integrity_issues)

            # 3. 参照完整性验证
            referential_issues = self._validate_referential_integrity()
            validation_result.issues.extend(referential_issues)

            # 4. 数据范围验证
            range_issues = self._validate_data_ranges()
            validation_result.issues.extend(range_issues)

            self.validation_results.append(validation_result)

            if validation_result.is_success:
                logger.info("✅ 迁移验证通过")
            else:
                logger.warning(f"⚠️  迁移验证发现 {len(validation_result.issues)} 个问题")

            return validation_result.is_success

        except Exception as e:
            logger.error(f"迁移验证失败: {e}")
            self._log_error("migration_validation_failed", str(e))
            return False

    def _get_table_counts(self, db_path: str) -> Dict[str, int]:
        """获取表记录数量"""
        counts = {}
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        for table in ['users', 'orders', 'trades', 'positions']:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            counts[table] = cursor.fetchone()[0]

        conn.close()
        return counts

    def _validate_data_integrity(self) -> List[str]:
        """验证数据完整性"""
        issues = []

        conn = sqlite3.connect(self.config.target_db_path)
        cursor = conn.cursor()

        # 检查必填字段
        cursor.execute("SELECT COUNT(*) FROM users WHERE username IS NULL OR email IS NULL")
        null_users = cursor.fetchone()[0]
        if null_users > 0:
            issues.append(f"用户表发现 {null_users} 条记录用户名或邮箱为空")

        cursor.execute("SELECT COUNT(*) FROM orders WHERE symbol IS NULL OR order_type IS NULL")
        null_orders = cursor.fetchone()[0]
        if null_orders > 0:
            issues.append(f"订单表发现 {null_orders} 条记录股票代码或订单类型为空")

        # 检查数据类型一致性
        cursor.execute("SELECT COUNT(*) FROM users WHERE typeof(balance) != 'real'")
        invalid_balance = cursor.fetchone()[0]
        if invalid_balance > 0:
            issues.append(f"用户表发现 {invalid_balance} 条记录余额字段类型不正确")

        conn.close()
        return issues

    def _validate_referential_integrity(self) -> List[str]:
        """验证参照完整性"""
        issues = []

        conn = sqlite3.connect(self.config.target_db_path)
        cursor = conn.cursor()

        # 检查订单的用户ID是否存在
        cursor.execute("""
            SELECT COUNT(*) FROM orders o
            LEFT JOIN users u ON o.user_id = u.user_id
            WHERE u.user_id IS NULL
        """)
        orphan_orders = cursor.fetchone()[0]
        if orphan_orders > 0:
            issues.append(f"发现 {orphan_orders} 个订单引用不存在的用户ID")

        # 检查交易记录的用户ID是否存在
        cursor.execute("""
            SELECT COUNT(*) FROM trades t
            LEFT JOIN users u ON t.user_id = u.user_id
            WHERE u.user_id IS NULL
        """)
        orphan_trades = cursor.fetchone()[0]
        if orphan_trades > 0:
            issues.append(f"发现 {orphan_trades} 个交易记录引用不存在的用户ID")

        # 检查持仓记录的用户ID是否存在
        cursor.execute("""
            SELECT COUNT(*) FROM positions p
            LEFT JOIN users u ON p.user_id = u.user_id
            WHERE u.user_id IS NULL
        """)
        orphan_positions = cursor.fetchone()[0]
        if orphan_positions > 0:
            issues.append(f"发现 {orphan_positions} 个持仓记录引用不存在的用户ID")

        conn.close()
        return issues

    def _validate_data_ranges(self) -> List[str]:
        """验证数据范围"""
        issues = []

        conn = sqlite3.connect(self.config.target_db_path)
        cursor = conn.cursor()

        # 检查余额范围
        cursor.execute("SELECT COUNT(*) FROM users WHERE balance < 0")
        negative_balance = cursor.fetchone()[0]
        if negative_balance > 0:
            issues.append(f"发现 {negative_balance} 个用户余额为负数")

        # 检查订单数量范围
        cursor.execute("SELECT COUNT(*) FROM orders WHERE quantity <= 0")
        invalid_quantity = cursor.fetchone()[0]
        if invalid_quantity > 0:
            issues.append(f"发现 {invalid_quantity} 个订单数量无效（<=0）")

        # 检查价格范围
        cursor.execute("SELECT COUNT(*) FROM orders WHERE price IS NOT NULL AND price <= 0")
        invalid_price = cursor.fetchone()[0]
        if invalid_price > 0:
            issues.append(f"发现 {invalid_price} 个订单价格无效（<=0）")

        conn.close()
        return issues

    def _assess_data_quality(self) -> bool:
        """评估数据质量"""
        logger.info("📊 评估数据质量...")

        try:
            conn = sqlite3.connect(self.config.target_db_path)
            cursor = conn.cursor()

            # 完整性评估
            total_users = cursor.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            complete_users = cursor.execute("""
                SELECT COUNT(*) FROM users
                WHERE username IS NOT NULL AND email IS NOT NULL AND balance IS NOT NULL
            """).fetchone()[0]
            self.quality_metrics.completeness_score = complete_users / total_users if total_users > 0 else 0

            # 准确性评估（检查数据格式）
            valid_emails = cursor.execute("""
                SELECT COUNT(*) FROM users
                WHERE email LIKE '%@%.%'
            """).fetchone()[0]
            self.quality_metrics.accuracy_score = valid_emails / total_users if total_users > 0 else 0

            # 一致性评估（检查枚举值）
            valid_statuses = cursor.execute("""
                SELECT COUNT(*) FROM users
                WHERE status IN ('active', 'inactive', 'suspended')
            """).fetchone()[0]
            self.quality_metrics.consistency_score = valid_statuses / total_users if total_users > 0 else 0

            # 时效性评估（检查更新时间）
            recent_updates = cursor.execute("""
                SELECT COUNT(*) FROM users
                WHERE updated_at > datetime('now', '-30 days')
            """).fetchone()[0]
            self.quality_metrics.timeliness_score = recent_updates / total_users if total_users > 0 else 0

            # 计算综合得分
            self.quality_metrics.overall_score = (
                self.quality_metrics.completeness_score * 0.3 +
                self.quality_metrics.accuracy_score * 0.3 +
                self.quality_metrics.consistency_score * 0.2 +
                self.quality_metrics.timeliness_score * 0.2
            )

            conn.close()

            logger.info(f"✅ 数据质量评估完成 - 综合得分: {self.quality_metrics.overall_score:.2f}")
            return self.quality_metrics.overall_score >= 0.8  # 80分以上算合格

        except Exception as e:
            logger.error(f"数据质量评估失败: {e}")
            self._log_error("data_quality_assessment_failed", str(e))
            return False

    def _evaluate_performance(self) -> bool:
        """评估迁移性能"""
        logger.info("⚡ 评估迁移性能...")

        try:
            if not self.stats.start_time or not self.stats.end_time:
                return False

            migration_time = (self.stats.end_time - self.stats.start_time).total_seconds()
            records_per_second = self.stats.migrated_records / migration_time if migration_time > 0 else 0

            logger.info(f"迁移性能指标:")
            logger.info(f"  总迁移时间: {migration_time:.2f}秒")
            logger.info(f"  迁移记录数: {self.stats.migrated_records}")
            logger.info(f"  每秒处理记录数: {records_per_second:.1f}")
            logger.info(f"  数据大小: {self.stats.data_size_bytes / 1024:.1f} KB")

            # 性能标准：至少每秒处理1000条记录
            return records_per_second >= 1000

        except Exception as e:
            logger.error(f"性能评估失败: {e}")
            self._log_error("performance_evaluation_failed", str(e))
            return False

    def _test_rollback_capability(self) -> bool:
        """测试回滚能力"""
        logger.info("🔄 测试回滚能力...")

        try:
            # 模拟回滚：从备份恢复
            if os.path.exists(self.config.backup_path):
                shutil.copy2(self.config.backup_path, self.config.target_db_path)
                logger.info("✅ 从备份恢复数据成功")

                # 验证恢复的数据完整性
                backup_checksum = self._calculate_database_checksum(self.config.backup_path)
                restored_checksum = self._calculate_database_checksum(self.config.target_db_path)

                if backup_checksum == restored_checksum:
                    logger.info("✅ 回滚后数据完整性验证通过")
                    return True
                else:
                    logger.error("❌ 回滚后数据完整性验证失败")
                    return False
            else:
                logger.error("❌ 备份文件不存在，无法测试回滚")
                return False

        except Exception as e:
            logger.error(f"回滚测试失败: {e}")
            self._log_error("rollback_test_failed", str(e))
            return False

    def _calculate_database_checksum(self, db_path: str) -> str:
        """计算数据库校验和"""
        try:
            if not os.path.exists(db_path):
                return ""

            hash_md5 = hashlib.md5()
            with open(db_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()

        except Exception as e:
            logger.error(f"计算校验和失败: {e}")
            return ""

    def _log_error(self, error_type: str, message: str):
        """记录错误"""
        error = {
            'timestamp': datetime.now(),
            'type': error_type,
            'message': message
        }
        self.error_log.append(error)

    def _generate_migration_report(self):
        """生成迁移报告"""
        report = {
            'migration_summary': {
                'start_time': self.stats.start_time.isoformat() if self.stats.start_time else None,
                'end_time': self.stats.end_time.isoformat() if self.stats.end_time else None,
                'duration_seconds': (self.stats.end_time - self.stats.start_time).total_seconds() if self.stats.start_time and self.stats.end_time else 0,
                'total_records': self.stats.total_records,
                'migrated_records': self.stats.migrated_records,
                'failed_records': self.stats.failed_records,
                'data_size_bytes': self.stats.data_size_bytes,
                'source_checksum': self.stats.checksum_source,
                'target_checksum': self.stats.checksum_target,
                'checksum_match': self.stats.checksum_source == self.stats.checksum_target
            },
            'data_quality_assessment': {
                'completeness_score': self.quality_metrics.completeness_score,
                'accuracy_score': self.quality_metrics.accuracy_score,
                'consistency_score': self.quality_metrics.consistency_score,
                'timeliness_score': self.quality_metrics.timeliness_score,
                'overall_score': self.quality_metrics.overall_score
            },
            'validation_results': [
                {
                    'is_success': vr.is_success,
                    'issues_count': len(vr.issues),
                    'warnings_count': len(vr.warnings),
                    'recommendations_count': len(vr.recommendations),
                    'issues': vr.issues,
                    'warnings': vr.warnings,
                    'recommendations': vr.recommendations
                }
                for vr in self.validation_results
            ],
            'error_summary': {
                'total_errors': len(self.error_log),
                'error_types': list(set(e['type'] for e in self.error_log)),
                'errors': self.error_log[-10:]  # 只显示最后10个错误
            },
            'recommendations': self._generate_recommendations()
        }

        # 保存报告
        report_file = f'test_logs/phase5_data_migration_verification_{int(time.time())}.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"📊 数据迁移验证报告已保存: {report_file}")

        # 打印总结报告
        self._print_summary_report(report)

    def _generate_recommendations(self) -> List[str]:
        """生成建议"""
        recommendations = []

        # 基于质量评估结果生成建议
        if self.quality_metrics.completeness_score < 0.9:
            recommendations.append("🔧 数据完整性不足，建议完善数据补全逻辑")

        if self.quality_metrics.accuracy_score < 0.95:
            recommendations.append("🎯 数据准确性需要提升，建议加强数据验证")

        if self.quality_metrics.consistency_score < 0.9:
            recommendations.append("📏 数据一致性不足，建议统一数据标准")

        if self.quality_metrics.timeliness_score < 0.8:
            recommendations.append("⏰ 数据时效性不足，建议优化数据更新频率")

        # 基于验证结果生成建议
        all_issues = []
        for vr in self.validation_results:
            all_issues.extend(vr.issues)

        if all_issues:
            recommendations.append(f"🚨 发现 {len(all_issues)} 个数据问题，建议优先修复")

        # 性能建议
        if self.stats.end_time and self.stats.start_time:
            duration = (self.stats.end_time - self.stats.start_time).total_seconds()
            if duration > 300:  # 超过5分钟
                recommendations.append("⚡ 迁移时间过长，建议优化批量处理和并发策略")

        if not recommendations:
            recommendations.append("✅ 数据迁移质量良好，无明显问题")

        return recommendations

    def _print_summary_report(self, report: Dict[str, Any]):
        """打印总结报告"""
        print("\n" + "="*80)
        print("📊 数据迁移验证总结报告")
        print("="*80)

        summary = report['migration_summary']
        print(f"🕐 迁移时间: {summary['start_time']} - {summary['end_time']}")
        print(f"⏱️  持续时间: {summary['duration_seconds']:.2f}秒")
        print(f"📊 总记录数: {summary['total_records']}")
        print(f"✅ 迁移记录数: {summary['migrated_records']}")
        print(f"❌ 失败记录数: {summary['failed_records']}")
        print(f"💾 数据大小: {summary['data_size_bytes'] / 1024:.1f} KB")
        print(f"🔒 校验和匹配: {'✅' if summary['checksum_match'] else '❌'}")

        quality = report['data_quality_assessment']
        print("\n📈 数据质量评估:")
        print(f"  完整性得分: {quality['completeness_score']:.2f}")
        print(f"  准确性得分: {quality['accuracy_score']:.2f}")
        print(f"  一致性得分: {quality['consistency_score']:.2f}")
        print(f"  时效性得分: {quality['timeliness_score']:.2f}")
        print(f"  综合得分: {quality['overall_score']:.2f}")
        validation = report['validation_results'][0] if report['validation_results'] else {}
        print("\n🔍 验证结果:")
        print(f"  验证通过: {'✅' if validation.get('is_success', False) else '❌'}")
        print(f"  问题数量: {validation.get('issues_count', 0)}")
        print(f"  警告数量: {validation.get('warnings_count', 0)}")

        errors = report['error_summary']
        print("\n🚨 错误统计:")
        print(f"  总错误数: {errors.get('total_errors', 0)}")

        print("\n💡 建议:")
        for rec in report['recommendations']:
            print(f"  • {rec}")

        print("\n" + "="*80)


def run_data_migration_verification():
    """运行数据迁移验证测试"""
    logger.info("开始Phase 5: 数据迁移验证测试")

    # 配置迁移参数
    config = MigrationConfig(
        source_db_path="data/source_database.db",
        target_db_path="data/target_database.db",
        backup_path="data/backup_database.db",
        batch_size=1000,
        max_workers=4,
        enable_compression=True,
        enable_encryption=False,
        verify_integrity=True
    )

    # 创建验证器并运行测试
    verifier = DataMigrationVerifier(config)

    try:
        success = verifier.run_full_migration_test()
        if success:
            logger.info("✅ 数据迁移验证测试完成")
        else:
            logger.error("❌ 数据迁移验证测试失败")
    except Exception as e:
        logger.error(f"测试过程中发生异常: {e}")


if __name__ == "__main__":
    run_data_migration_verification()
