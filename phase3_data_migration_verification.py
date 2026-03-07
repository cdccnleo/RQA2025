#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3 预投产验证 - 数据迁移验证脚本
验证数据迁移过程的完整性、一致性和性能
"""

import os
import sys
import json
import time
import hashlib
import sqlite3
import numpy as np
import psutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass, asdict
import shutil


@dataclass
class MigrationConfig:
    """迁移配置"""
    source_db_path: str = "data/source_database.db"
    target_db_path: str = "data/target_database.db"
    backup_path: str = "data/backup/"
    test_data_size: int = 100000  # 测试数据行数
    batch_size: int = 1000  # 批处理大小
    parallel_workers: int = 4  # 并行工作线程数
    migration_timeout_seconds: int = 3600  # 迁移超时时间


@dataclass
class MigrationMetrics:
    """迁移指标"""
    timestamp: datetime
    phase: str  # 'preparation', 'migration', 'validation', 'cleanup'
    records_processed: int
    records_per_second: float
    memory_usage_mb: float
    cpu_usage_percent: float
    elapsed_time_seconds: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class DataValidationResult:
    """数据验证结果"""
    table_name: str
    source_count: int
    target_count: int
    records_match: bool
    checksum_match: bool
    sample_validation_passed: bool
    data_integrity_score: float


class DataMigrationVerifier:
    """数据迁移验证器"""

    def __init__(self, config: MigrationConfig):
        self.config = config
        self.metrics_history: List[MigrationMetrics] = []
        self.validation_results: List[DataValidationResult] = []
        self.start_time = None
        self.end_time = None

        # 设置日志
        self.setup_logging()

        # 创建必要的目录
        self.setup_directories()

    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data_migration.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_directories(self):
        """创建必要的目录"""
        directories = [
            Path(self.config.backup_path),
            Path(self.config.source_db_path).parent,
            Path(self.config.target_db_path).parent
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def generate_test_data(self) -> None:
        """生成测试数据"""
        self.logger.info("开始生成测试数据...")

        # 清理现有的数据库文件
        source_path = Path(self.config.source_db_path)
        target_path = Path(self.config.target_db_path)

        if source_path.exists():
            source_path.unlink()
            self.logger.info(f"清理现有源数据库: {source_path}")

        if target_path.exists():
            target_path.unlink()
            self.logger.info(f"清理现有目标数据库: {target_path}")

        # 创建源数据库
        with sqlite3.connect(self.config.source_db_path) as conn:
            cursor = conn.cursor()

            # 创建用户表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    email TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    balance REAL DEFAULT 0.0,
                    status TEXT DEFAULT 'active'
                )
            ''')

            # 创建交易表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER,
                    symbol TEXT NOT NULL,
                    quantity INTEGER,
                    price REAL,
                    trade_type TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')

            # 创建市场数据表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    price REAL,
                    volume INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 生成测试用户数据
            self.logger.info(f"生成 {self.config.test_data_size} 条用户数据...")
            users_data = []
            for i in range(self.config.test_data_size):
                user = (
                    i + 1,
                    f'user_{i+1:06d}',
                    f'user_{i+1:06d}@example.com',
                    datetime.now() - timedelta(days=np.random.randint(0, 365)),
                    np.random.uniform(1000, 100000),
                    np.random.choice(['active', 'inactive', 'suspended'], p=[0.8, 0.15, 0.05])
                )
                users_data.append(user)

            cursor.executemany('INSERT INTO users VALUES (?, ?, ?, ?, ?, ?)', users_data)

            # 生成测试交易数据
            self.logger.info(f"生成 {self.config.test_data_size * 2} 条交易数据...")
            trades_data = []
            for i in range(self.config.test_data_size * 2):
                trade = (
                    i + 1,
                    np.random.randint(1, self.config.test_data_size + 1),
                    np.random.choice(['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']),
                    np.random.randint(1, 1000),
                    np.random.uniform(100, 1000),
                    np.random.choice(['buy', 'sell']),
                    datetime.now() - timedelta(hours=np.random.randint(0, 24*30))
                )
                trades_data.append(trade)

            cursor.executemany('INSERT INTO trades VALUES (?, ?, ?, ?, ?, ?, ?)', trades_data)

            # 生成市场数据
            self.logger.info(f"生成 {self.config.test_data_size} 条市场数据...")
            market_data = []
            for i in range(self.config.test_data_size):
                data = (
                    i + 1,
                    np.random.choice(['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']),
                    np.random.uniform(100, 1000),
                    np.random.randint(1000, 100000),
                    datetime.now() - timedelta(minutes=np.random.randint(0, 60*24))
                )
                market_data.append(data)

            cursor.executemany('INSERT INTO market_data VALUES (?, ?, ?, ?, ?)', market_data)

            conn.commit()

        self.logger.info("测试数据生成完成")

    def create_backup(self) -> bool:
        """创建源数据库备份"""
        try:
            self.logger.info("创建源数据库备份...")

            backup_file = Path(self.config.backup_path) / f"source_backup_{int(time.time())}.db"
            shutil.copy2(self.config.source_db_path, backup_file)

            self.logger.info(f"备份创建成功: {backup_file}")
            return True

        except Exception as e:
            self.logger.error(f"备份创建失败: {e}")
            return False

    def perform_migration(self) -> bool:
        """执行数据迁移"""
        self.logger.info("开始数据迁移...")
        self.start_time = datetime.now()

        try:
            # 记录迁移开始指标
            self.record_metrics("preparation", 0, True)

            # 创建目标数据库结构
            self.create_target_schema()

            # 执行数据迁移
            success = self.migrate_data()

            if success:
                self.logger.info("数据迁移成功完成")
                self.record_metrics("migration", self.get_total_records(), True)
            else:
                self.logger.error("数据迁移失败")
                self.record_metrics("migration", 0, False, "Migration failed")
                return False

            self.end_time = datetime.now()
            return True

        except Exception as e:
            self.logger.error(f"数据迁移过程中发生错误: {e}")
            self.record_metrics("migration", 0, False, str(e))
            return False

    def create_target_schema(self) -> None:
        """创建目标数据库结构"""
        self.logger.info("创建目标数据库结构...")

        with sqlite3.connect(self.config.target_db_path) as conn:
            cursor = conn.cursor()

            # 创建与源数据库相同的表结构
            cursor.execute('''
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    email TEXT NOT NULL,
                    created_at TIMESTAMP,
                    balance REAL DEFAULT 0.0,
                    status TEXT DEFAULT 'active'
                )
            ''')

            cursor.execute('''
                CREATE TABLE trades (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER,
                    symbol TEXT NOT NULL,
                    quantity INTEGER,
                    price REAL,
                    trade_type TEXT,
                    timestamp TIMESTAMP
                )
            ''')

            cursor.execute('''
                CREATE TABLE market_data (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    price REAL,
                    volume INTEGER,
                    timestamp TIMESTAMP
                )
            ''')

            conn.commit()

    def migrate_data(self) -> bool:
        """迁移数据"""
        tables = ['users', 'trades', 'market_data']

        for table in tables:
            if not self.migrate_table(table):
                return False

        return True

    def migrate_table(self, table_name: str) -> bool:
        """迁移单个表的数据"""
        self.logger.info(f"开始迁移表: {table_name}")

        try:
            with sqlite3.connect(self.config.source_db_path) as source_conn, \
                    sqlite3.connect(self.config.target_db_path) as target_conn:

                # 获取源表数据
                source_cursor = source_conn.cursor()
                source_cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                total_records = source_cursor.fetchone()[0]

                self.logger.info(f"表 {table_name} 共有 {total_records} 条记录")

                # 分批迁移数据
                offset = 0
                migrated_count = 0

                while offset < total_records:
                    batch_size = min(self.config.batch_size, total_records - offset)

                    # 获取一批数据
                    source_cursor.execute(f'''
                        SELECT * FROM {table_name}
                        LIMIT ? OFFSET ?
                    ''', (batch_size, offset))

                    batch_data = source_cursor.fetchall()

                    # 插入目标数据库
                    if batch_data:
                        placeholders = ','.join(['?' for _ in batch_data[0]])
                        target_conn.executemany(f'''
                            INSERT INTO {table_name} VALUES ({placeholders})
                        ''', batch_data)

                        target_conn.commit()

                    migrated_count += len(batch_data)
                    offset += batch_size

                    # 记录进度
                    progress = (migrated_count / total_records) * 100
                    self.logger.info(
                        f"表 {table_name} 迁移进度: {progress:.1f}% ({migrated_count}/{total_records})")

                    # 模拟迁移延迟
                    time.sleep(0.01)

                self.logger.info(f"表 {table_name} 迁移完成")
                return True

        except Exception as e:
            self.logger.error(f"迁移表 {table_name} 时发生错误: {e}")
            return False

    def validate_migration(self) -> bool:
        """验证数据迁移结果"""
        self.logger.info("开始数据迁移验证...")
        self.record_metrics("validation", 0, True)

        try:
            tables = ['users', 'trades', 'market_data']
            all_valid = True

            for table in tables:
                result = self.validate_table(table)
                self.validation_results.append(result)

                if not (result.records_match and result.checksum_match and result.sample_validation_passed):
                    all_valid = False
                    self.logger.error(f"表 {table} 验证失败")

            if all_valid:
                self.logger.info("数据迁移验证通过")
                self.record_metrics("validation", self.get_total_records(), True)
            else:
                self.logger.error("数据迁移验证失败")
                self.record_metrics("validation", 0, False, "Validation failed")

            return all_valid

        except Exception as e:
            self.logger.error(f"数据验证过程中发生错误: {e}")
            self.record_metrics("validation", 0, False, str(e))
            return False

    def validate_table(self, table_name: str) -> DataValidationResult:
        """验证单个表的数据"""
        self.logger.info(f"验证表: {table_name}")

        try:
            with sqlite3.connect(self.config.source_db_path) as source_conn, \
                    sqlite3.connect(self.config.target_db_path) as target_conn:

                # 比较记录数
                source_cursor = source_conn.cursor()
                target_cursor = target_conn.cursor()

                source_cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                source_count = source_cursor.fetchone()[0]

                target_cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                target_count = target_cursor.fetchone()[0]

                records_match = source_count == target_count

                # 计算校验和
                source_cursor.execute(f"SELECT * FROM {table_name} ORDER BY id")
                source_data = source_cursor.fetchall()

                target_cursor.execute(f"SELECT * FROM {table_name} ORDER BY id")
                target_data = target_cursor.fetchall()

                source_checksum = hashlib.md5(str(source_data).encode()).hexdigest()
                target_checksum = hashlib.md5(str(target_data).encode()).hexdigest()

                checksum_match = source_checksum == target_checksum

                # 采样验证
                sample_validation_passed = self.validate_sample_data(
                    table_name, source_data, target_data)

                # 计算数据完整性评分
                integrity_score = 0.0
                if records_match:
                    integrity_score += 40
                if checksum_match:
                    integrity_score += 40
                if sample_validation_passed:
                    integrity_score += 20

                result = DataValidationResult(
                    table_name=table_name,
                    source_count=source_count,
                    target_count=target_count,
                    records_match=records_match,
                    checksum_match=checksum_match,
                    sample_validation_passed=sample_validation_passed,
                    data_integrity_score=integrity_score
                )

                self.logger.info(
                    f"表 {table_name} 验证结果: 记录匹配={records_match}, 校验和匹配={checksum_match}, 采样验证={sample_validation_passed}")
                return result

        except Exception as e:
            self.logger.error(f"验证表 {table_name} 时发生错误: {e}")
            return DataValidationResult(
                table_name=table_name,
                source_count=0,
                target_count=0,
                records_match=False,
                checksum_match=False,
                sample_validation_passed=False,
                data_integrity_score=0.0
            )

    def validate_sample_data(self, table_name: str, source_data: List, target_data: List) -> bool:
        """验证采样数据"""
        if len(source_data) != len(target_data):
            return False

        # 随机选择10%的样本进行验证
        sample_size = max(1, len(source_data) // 10)
        indices = np.random.choice(len(source_data), sample_size, replace=False)

        for idx in indices:
            if source_data[idx] != target_data[idx]:
                self.logger.error(f"表 {table_name} 采样验证失败: 索引 {idx} 数据不匹配")
                return False

        return True

    def test_rollback(self) -> bool:
        """测试迁移回滚功能"""
        self.logger.info("测试迁移回滚功能...")

        try:
            # 模拟回滚：从备份恢复
            backup_files = list(Path(self.config.backup_path).glob("source_backup_*.db"))
            if not backup_files:
                self.logger.error("没有找到备份文件")
                return False

            latest_backup = max(backup_files, key=lambda x: x.stat().st_mtime)
            shutil.copy2(latest_backup, self.config.target_db_path)

            self.logger.info("迁移回滚测试成功")
            return True

        except Exception as e:
            self.logger.error(f"迁移回滚测试失败: {e}")
            return False

    def get_total_records(self) -> int:
        """获取总记录数"""
        try:
            with sqlite3.connect(self.config.target_db_path) as conn:
                cursor = conn.cursor()
                total = 0
                for table in ['users', 'trades', 'market_data']:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    total += cursor.fetchone()[0]
                return total
        except:
            return 0

    def record_metrics(self, phase: str, records_processed: int, success: bool,
                       error_message: Optional[str] = None) -> None:
        """记录迁移指标"""
        metrics = MigrationMetrics(
            timestamp=datetime.now(),
            phase=phase,
            records_processed=records_processed,
            records_per_second=records_processed /
            max(1, (datetime.now() - (self.start_time or datetime.now())).total_seconds()),
            memory_usage_mb=psutil.Process().memory_info().rss / (1024 * 1024),
            cpu_usage_percent=psutil.cpu_percent(),
            elapsed_time_seconds=(
                datetime.now() - (self.start_time or datetime.now())).total_seconds(),
            success=success,
            error_message=error_message
        )

        self.metrics_history.append(metrics)

    def generate_migration_report(self) -> Dict[str, Any]:
        """生成迁移报告"""
        # 计算总体指标
        total_records = sum(result.source_count for result in self.validation_results)
        successful_tables = sum(1 for result in self.validation_results
                                if result.records_match and result.checksum_match and result.sample_validation_passed)
        total_tables = len(self.validation_results)

        # 计算迁移性能
        migration_metrics = [m for m in self.metrics_history if m.phase == 'migration']
        avg_throughput = np.mean(
            [m.records_per_second for m in migration_metrics]) if migration_metrics else 0

        # 计算数据完整性评分
        avg_integrity_score = np.mean(
            [result.data_integrity_score for result in self.validation_results])

        # 评估迁移质量
        migration_quality_score = self.calculate_migration_quality_score()

        report = {
            "migration_info": {
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "duration_seconds": (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else 0,
                "config": asdict(self.config)
            },
            "migration_summary": {
                "total_records": total_records,
                "tables_processed": total_tables,
                "successful_tables": successful_tables,
                "failed_tables": total_tables - successful_tables,
                "success_rate": successful_tables / total_tables if total_tables > 0 else 0,
                "average_throughput": avg_throughput,
                "data_integrity_score": avg_integrity_score
            },
            "table_validation_results": [asdict(result) for result in self.validation_results],
            "performance_metrics": {
                "throughput_records_per_second": avg_throughput,
                "total_migration_time": (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else 0,
                "memory_peak_usage_mb": max([m.memory_usage_mb for m in self.metrics_history], default=0),
                "cpu_average_usage": np.mean([m.cpu_usage_percent for m in self.metrics_history])
            },
            "migration_quality": {
                "overall_score": migration_quality_score,
                "assessment": self.assess_migration_quality(migration_quality_score)
            },
            "recommendations": self.generate_migration_recommendations(migration_quality_score)
        }

        return report

    def calculate_migration_quality_score(self) -> float:
        """计算迁移质量评分"""
        if not self.validation_results:
            return 0.0

        # 数据完整性 (40%)
        integrity_score = np.mean(
            [result.data_integrity_score for result in self.validation_results])

        # 性能表现 (30%)
        migration_metrics = [m for m in self.metrics_history if m.phase == 'migration']
        performance_score = 0.0
        if migration_metrics:
            avg_throughput = np.mean([m.records_per_second for m in migration_metrics])
            # 假设目标吞吐量是1000 records/second
            performance_score = min(100, (avg_throughput / 1000) * 100)

        # 成功率 (30%)
        successful_tables = sum(1 for result in self.validation_results
                                if result.records_match and result.checksum_match and result.sample_validation_passed)
        success_rate = successful_tables / \
            len(self.validation_results) if self.validation_results else 0
        success_score = success_rate * 100

        return (integrity_score * 0.4) + (performance_score * 0.3) + (success_score * 0.3)

    def assess_migration_quality(self, score: float) -> Dict[str, str]:
        """评估迁移质量"""
        if score >= 95:
            status = "excellent"
            message = "数据迁移质量优秀，完全满足生产要求"
        elif score >= 90:
            status = "good"
            message = "数据迁移质量良好，基本满足生产要求"
        elif score >= 80:
            status = "acceptable"
            message = "数据迁移质量可接受，建议进行额外验证"
        elif score >= 70:
            status = "needs_attention"
            message = "数据迁移质量需要关注，建议重新检查"
        else:
            status = "poor"
            message = "数据迁移质量不佳，不建议投入生产"

        return {
            "status": status,
            "message": message,
            "score": score
        }

    def generate_migration_recommendations(self, score: float) -> List[str]:
        """生成迁移建议"""
        recommendations = []

        # 基于验证结果生成建议
        failed_tables = [result for result in self.validation_results
                         if not (result.records_match and result.checksum_match and result.sample_validation_passed)]

        if failed_tables:
            recommendations.append(
                f"🔴 以下表迁移失败或验证不通过: {', '.join([t.table_name for t in failed_tables])}")
            recommendations.append("建议重新执行这些表的迁移")

        if score < 80:
            recommendations.append("⚠️ 迁移质量需要提升，建议优化迁移脚本和验证逻辑")

        # 基于性能指标生成建议
        migration_metrics = [m for m in self.metrics_history if m.phase == 'migration']
        if migration_metrics:
            avg_throughput = np.mean([m.records_per_second for m in migration_metrics])
            if avg_throughput < 500:
                recommendations.append("🐌 迁移性能较低，建议优化批处理大小或增加并行度")

        if not recommendations:
            recommendations.append("✅ 数据迁移质量良好，可以准备投入生产")

        return recommendations


def main():
    """主函数"""
    print('🔄 Phase 3 数据迁移验证开始')
    print('=' * 60)

    # 配置迁移参数
    config = MigrationConfig(
        source_db_path="data/source_database.db",
        target_db_path="data/target_database.db",
        backup_path="data/backup/",
        test_data_size=50000,  # 为演示减少数据量
        batch_size=5000,
        parallel_workers=2,
        migration_timeout_seconds=1800
    )

    print('📊 迁移配置:')
    print(f'  源数据库: {config.source_db_path}')
    print(f'  目标数据库: {config.target_db_path}')
    print(f'  备份路径: {config.backup_path}')
    print(f'  测试数据量: {config.test_data_size} 条/表')
    print(f'  批处理大小: {config.batch_size}')
    print()

    # 创建验证器
    verifier = DataMigrationVerifier(config)

    try:
        # 1. 生成测试数据
        verifier.generate_test_data()

        # 2. 创建备份
        if not verifier.create_backup():
            print("❌ 备份创建失败")
            return "backup_failed", 0.0

        # 3. 执行数据迁移
        if not verifier.perform_migration():
            print("❌ 数据迁移失败")
            return "migration_failed", 0.0

        # 4. 验证迁移结果
        if not verifier.validate_migration():
            print("❌ 数据验证失败")
            return "validation_failed", 0.0

        # 5. 测试回滚功能
        if not verifier.test_rollback():
            print("⚠️ 回滚测试失败，但不影响迁移成功")

        # 6. 生成报告
        report = verifier.generate_migration_report()

        print('\n📊 数据迁移验证结果:')
        summary = report['migration_summary']
        print(f'总记录数: {summary["total_records"]:,}')
        print(f'表数量: {summary["tables_processed"]}')
        print(f'成功表数: {summary["successful_tables"]}')
        print(f'失败表数: {summary["failed_tables"]}')
        print(f'成功率: {summary["success_rate"]:.1%}')
        print(f'平均吞吐量: {summary["average_throughput"]:.0f} records/sec')
        print(f'数据完整性评分: {summary["data_integrity_score"]:.1f}/100')

        perf = report['performance_metrics']
        print(f'\n迁移总时间: {perf["total_migration_time"]:.1f}秒')
        print(f'内存峰值使用: {perf["memory_peak_usage_mb"]:.1f}MB')
        print(f'CPU平均使用率: {perf["cpu_average_usage"]:.1f}%')

        quality = report['migration_quality']
        print(f'\n迁移质量评分: {quality["overall_score"]:.1f}/100')
        print(f'质量评估: {quality["assessment"]["message"]}')

        print('\n💡 迁移建议:')
        for i, rec in enumerate(report['recommendations'], 1):
            print(f'{i}. {rec}')

        print('\n📋 表验证详情:')
        for result in report['table_validation_results']:
            status = "✅" if result['records_match'] and result['checksum_match'] and result['sample_validation_passed'] else "❌"
            print(
                f'{status} {result["table_name"]}: {result["source_count"]} -> {result["target_count"]} 条记录')

        # 保存详细报告
        os.makedirs('test_logs', exist_ok=True)
        report_file = f'phase3_data_migration_verification_{int(datetime.now().timestamp())}.json'
        with open(f'test_logs/{report_file}', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)

        print('=' * 60)
        print('✅ Phase 3 数据迁移验证完成')
        print(f'📄 详细报告已保存: test_logs/{report_file}')
        print('=' * 60)

        return quality['assessment']['status'], quality['overall_score']

    except Exception as e:
        print(f'\n❌ 数据迁移验证过程中发生错误: {e}')
        return "error", 0.0


if __name__ == "__main__":
    main()
