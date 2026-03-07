#!/usr/bin/env python3
"""
端到端数据迁移执行脚本 - Phase 6.2 Day 6
执行完整的生产环境数据迁移验证流程

迁移流程:
1. 环境准备 - 检查和启动生产环境
2. 数据导出 - 从源数据库导出数据
3. 数据迁移 - 将数据导入目标数据库
4. 进度监控 - 实时监控迁移进度
5. 结果验证 - 验证迁移结果的正确性
6. 性能测试 - 测试迁移后系统性能
7. 报告生成 - 生成完整的迁移报告

使用方法:
python scripts/end_to_end_migration.py --migration-id MIGRATION_001
python scripts/end_to_end_migration.py --dry-run  # 仅验证环境，不执行迁移
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import sys

# 导入迁移工具
from data_migration_tools import DataMigrationTools

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MigrationExecutionResult:
    """迁移执行结果"""
    migration_id: str
    phase: str
    status: str  # 'running', 'success', 'failed', 'error'
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    progress_percentage: float = 0.0
    current_step: str = ""
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = None

    def __post_init__(self):
        if not self.start_time:
            self.start_time = datetime.now()
        if not self.metrics:
            self.metrics = {}

    def complete(self, status: str, error_message: str = None):
        """完成迁移执行"""
        self.end_time = datetime.now()
        self.status = status
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        if error_message:
            self.error_message = error_message


class EndToEndMigrationExecutor:
    """端到端迁移执行器"""

    def __init__(self, migration_id: str, project_root: str = None):
        self.migration_id = migration_id
        self.project_root = Path(project_root or Path(__file__).parent.parent)
        self.production_env = self.project_root / "production_env"
        self.migration_dir = self.project_root / "data" / "migration"
        self.results: List[MigrationExecutionResult] = []

        # 迁移工具
        self.migration_tools = DataMigrationTools()

        # 迁移配置
        self.source_db = str(self.migration_dir / "test_source_database.db")
        self.target_db = str(self.migration_dir / "test_target_database.db")
        self.export_dir = str(self.migration_dir / "export")

        logger.info(f"🚀 初始化端到端迁移执行器: {migration_id}")

    def execute_full_migration(self) -> Dict[str, Any]:
        """执行完整迁移流程"""
        logger.info(f"🎯 开始执行完整数据迁移: {self.migration_id}")

        try:
            # Phase 1: 环境准备
            self._execute_phase("environment_preparation", self._prepare_environment)

            # Phase 2: 数据导出
            self._execute_phase("data_export", self._export_data)

            # Phase 3: 数据迁移
            self._execute_phase("data_import", self._import_data)

            # Phase 4: 数据验证
            self._execute_phase("data_validation", self._validate_migration)

            # Phase 5: 性能测试
            self._execute_phase("performance_test", self._performance_test)

            # Phase 6: 报告生成
            self._execute_phase("report_generation", self._generate_final_report)

            # 汇总结果
            return self._summarize_results()

        except Exception as e:
            logger.error(f"迁移执行失败: {e}")
            self._execute_phase("error_handling", lambda: None, error=str(e))
            return self._summarize_results()

    def _execute_phase(self, phase_name: str, phase_func, **kwargs):
        """执行单个迁移阶段"""
        logger.info(f"📍 开始执行阶段: {phase_name}")

        result = MigrationExecutionResult(
            migration_id=self.migration_id,
            phase=phase_name,
            status="running",
            start_time=datetime.now(),
            current_step=f"Starting {phase_name}"
        )

        try:
            # 执行阶段函数
            phase_result = phase_func(**kwargs)

            # 更新结果
            result.complete("success")
            result.progress_percentage = 100.0
            result.metrics = phase_result if isinstance(phase_result, dict) else {}

            logger.info(f"✅ 阶段 {phase_name} 执行成功")

        except Exception as e:
            error_msg = f"阶段 {phase_name} 执行失败: {str(e)}"
            logger.error(error_msg)
            result.complete("failed", error_msg)

        self.results.append(result)

    def _prepare_environment(self) -> Dict[str, Any]:
        """Phase 1: 环境准备"""
        logger.info("🏗️ 准备迁移环境...")

        # 检查生产环境配置
        if not self.production_env.exists():
            raise FileNotFoundError(f"生产环境目录不存在: {self.production_env}")

        # 检查必要的配置文件
        required_configs = [
            'docker-compose.yml',
            '.env.production',
            'configs/postgresql.conf',
            'configs/redis.conf'
        ]

        missing_configs = []
        for config in required_configs:
            if not (self.production_env / config).exists():
                missing_configs.append(config)

        if missing_configs:
            raise FileNotFoundError(f"缺少必要的配置文件: {missing_configs}")

        # 检查迁移工具依赖
        try:
            pass
        except ImportError as e:
            raise ImportError(f"缺少必要的依赖: {e}")

        # 验证源数据库存在
        if not Path(self.source_db).exists():
            raise FileNotFoundError(f"源数据库不存在: {self.source_db}")

        # 创建目标数据库目录
        Path(self.target_db).parent.mkdir(parents=True, exist_ok=True)

        # 创建导出目录
        Path(self.export_dir).mkdir(parents=True, exist_ok=True)

        logger.info("✅ 环境准备完成")

        return {
            'environment_status': 'ready',
            'configs_validated': len(required_configs) - len(missing_configs),
            'source_db_exists': Path(self.source_db).exists(),
            'export_dir_created': Path(self.export_dir).exists()
        }

    def _export_data(self) -> Dict[str, Any]:
        """Phase 2: 数据导出"""
        logger.info("📤 执行数据导出...")

        # 使用迁移工具执行导出
        export_result = self.migration_tools.export_data(
            source_db=self.source_db,
            export_dir=self.export_dir
        )

        # 验证导出结果
        export_path = Path(self.export_dir)
        manifest_file = export_path / "migration_manifest.json"

        if not manifest_file.exists():
            raise FileNotFoundError("迁移清单文件未生成")

        # 读取并验证清单
        with open(manifest_file, 'r', encoding='utf-8') as f:
            manifest = json.load(f)

        exported_tables = manifest.get('exported_tables', {})
        total_records = manifest.get('total_records', 0)

        logger.info(f"✅ 数据导出完成，共导出 {total_records} 条记录")

        return {
            'export_status': 'success',
            'total_records': total_records,
            'exported_tables': len(exported_tables),
            'manifest_file': str(manifest_file),
            'export_duration': export_result.get('duration_seconds', 0)
        }

    def _import_data(self) -> Dict[str, Any]:
        """Phase 3: 数据迁移"""
        logger.info("📥 执行数据导入...")

        # 使用迁移工具执行导入
        import_result = self.migration_tools.import_data(
            source_dir=self.export_dir,
            target_db=self.target_db
        )

        # 验证导入结果
        if not Path(self.target_db).exists():
            raise FileNotFoundError(f"目标数据库未生成: {self.target_db}")

        # 验证目标数据库中的表
        import sqlite3
        conn = sqlite3.connect(self.target_db)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()

        imported_tables = import_result.get('imported_tables', [])
        total_records = import_result.get('total_records', 0)

        logger.info(f"✅ 数据导入完成，共导入 {total_records} 条记录，{len(imported_tables)} 个表")

        return {
            'import_status': 'success',
            'total_records': total_records,
            'imported_tables': len(imported_tables),
            'target_db_size_mb': Path(self.target_db).stat().st_size / (1024 * 1024),
            'import_duration': import_result.get('duration_seconds', 0)
        }

    def _validate_migration(self) -> Dict[str, Any]:
        """Phase 4: 数据验证"""
        logger.info("🔍 执行数据验证...")

        # 使用迁移工具执行验证
        validation_result = self.migration_tools.validate_data(
            source_db=self.source_db,
            target_db=self.target_db
        )

        overall_result = validation_result.get('overall_result', {})
        validation_details = validation_result.get('validation_details', [])

        # 检查验证结果
        success_rate = overall_result.get('details', {}).get('success_rate', 0)
        if success_rate < 0.95:  # 95%成功率阈值
            failed_tests = [r for r in validation_details if not r.get('passed', False)]
            error_msg = f"验证失败，通过率仅 {success_rate:.1%}，失败项目: {len(failed_tests)}"
            logger.error(error_msg)
            raise Exception(error_msg)

        logger.info(f"✅ 数据验证完成，通过率: {success_rate:.1%}")

        return {
            'validation_status': 'success' if overall_result.get('passed') else 'failed',
            'success_rate': success_rate,
            'total_tests': len(validation_details),
            'passed_tests': sum(1 for r in validation_details if r.get('passed', False)),
            'validation_duration': 0  # 验证时间在validation_result中
        }

    def _performance_test(self) -> Dict[str, Any]:
        """Phase 5: 性能测试"""
        logger.info("⚡ 执行性能测试...")

        # 测试目标数据库的查询性能
        import sqlite3

        test_queries = [
            ("SELECT COUNT(*) FROM users", "用户计数查询"),
            ("SELECT COUNT(*) FROM trades", "交易计数查询"),
            ("SELECT * FROM positions WHERE user_id = 1 LIMIT 10", "持仓查询"),
            ("SELECT SUM(quantity * price) FROM trades GROUP BY symbol LIMIT 5", "交易汇总查询")
        ]

        performance_results = {}

        conn = sqlite3.connect(self.target_db)
        cursor = conn.cursor()

        for query, description in test_queries:
            start_time = time.time()
            cursor.execute(query)
            cursor.fetchall()  # 确保查询完全执行
            execution_time = time.time() - start_time

            performance_results[description] = {
                'query': query,
                'execution_time_seconds': execution_time,
                'acceptable': execution_time < 5.0
            }

        conn.close()

        # 计算平均性能
        avg_time = sum(r['execution_time_seconds']
                       for r in performance_results.values()) / len(performance_results)
        all_acceptable = all(r['acceptable'] for r in performance_results.values())

        if not all_acceptable:
            slow_queries = [desc for desc, result in performance_results.items()
                            if not result['acceptable']]
            logger.warning(f"⚠️ 发现慢查询: {slow_queries}")

        logger.info(f"✅ 性能测试完成，平均响应时间: {avg_time:.3f}秒")

        return {
            'performance_status': 'success' if all_acceptable else 'warning',
            'average_response_time': avg_time,
            'all_queries_acceptable': all_acceptable,
            'test_queries_count': len(test_queries),
            'slow_queries_count': len([r for r in performance_results.values() if not r['acceptable']])
        }

    def _generate_final_report(self) -> Dict[str, Any]:
        """Phase 6: 报告生成"""
        logger.info("📋 生成最终迁移报告...")

        # 使用迁移工具生成报告
        report_result = self.migration_tools.generate_report(self.migration_id)

        # 添加执行结果
        report_result['execution_results'] = [asdict(r) for r in self.results]

        # 保存最终报告
        final_report_file = self.migration_dir / f"final_migration_report_{self.migration_id}.json"
        with open(final_report_file, 'w', encoding='utf-8') as f:
            # 处理datetime序列化
            def datetime_handler(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

            json.dump(report_result, f, indent=2, ensure_ascii=False, default=datetime_handler)

        logger.info(f"✅ 最终迁移报告已生成: {final_report_file}")

        return {
            'report_generation_status': 'success',
            'report_file': str(final_report_file),
            'report_size_kb': final_report_file.stat().st_size / 1024
        }

    def _summarize_results(self) -> Dict[str, Any]:
        """汇总迁移结果"""
        total_phases = len(self.results)
        successful_phases = len([r for r in self.results if r.status == 'success'])
        failed_phases = len([r for r in self.results if r.status == 'failed'])
        error_phases = len([r for r in self.results if r.status == 'error'])

        success_rate = successful_phases / total_phases if total_phases > 0 else 0

        total_duration = sum(r.duration_seconds for r in self.results)

        # 收集关键指标
        key_metrics = {}
        for result in self.results:
            if result.phase == 'data_export' and result.metrics:
                key_metrics.update({
                    'exported_records': result.metrics.get('total_records', 0),
                    'export_duration': result.metrics.get('duration_seconds', 0)
                })
            elif result.phase == 'data_import' and result.metrics:
                key_metrics.update({
                    'imported_records': result.metrics.get('total_records', 0),
                    'import_duration': result.metrics.get('duration_seconds', 0)
                })
            elif result.phase == 'data_validation' and result.metrics:
                key_metrics.update({
                    'validation_success_rate': result.metrics.get('success_rate', 0),
                    'validation_passed_tests': result.metrics.get('passed_tests', 0)
                })

        overall_status = 'success' if success_rate >= 0.9 else 'failed'

        summary = {
            'migration_id': self.migration_id,
            'overall_status': overall_status,
            'success_rate': success_rate,
            'total_phases': total_phases,
            'successful_phases': successful_phases,
            'failed_phases': failed_phases,
            'error_phases': error_phases,
            'total_duration_seconds': total_duration,
            'completed_at': datetime.now().isoformat(),
            'key_metrics': key_metrics,
            'phase_results': [asdict(r) for r in self.results]
        }

        logger.info("="*60)
        logger.info("📊 端到端数据迁移执行总结")
        logger.info("="*60)
        logger.info(f"迁移ID: {self.migration_id}")
        logger.info(f"总体状态: {'✅ 成功' if overall_status == 'success' else '❌ 失败'}")
        logger.info(".1%")
        logger.info(f"阶段统计: {successful_phases}/{total_phases} 成功")
        logger.info(f"总耗时: {total_duration:.2f}秒")
        logger.info(f"导出记录: {key_metrics.get('exported_records', 0)}")
        logger.info(f"验证通过率: {key_metrics.get('validation_success_rate', 0):.1%}")
        logger.info("="*60)

        return summary

    def monitor_migration(self) -> Dict[str, Any]:
        """监控迁移进度"""
        return self.migration_tools.monitor_migration(self.migration_id)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='端到端数据迁移执行工具')
    parser.add_argument('--migration-id', required=True, help='迁移任务ID')
    parser.add_argument('--dry-run', action='store_true', help='仅验证环境，不执行迁移')
    parser.add_argument('--phase', choices=['all', 'prepare', 'export', 'import', 'validate', 'test', 'report'],
                        default='all', help='执行特定阶段')

    args = parser.parse_args()

    try:
        executor = EndToEndMigrationExecutor(args.migration_id)

        if args.dry_run:
            logger.info("🔍 执行环境验证 (dry-run模式)")
            # 只执行环境准备
            result = executor._prepare_environment()
            print("✅ 环境验证通过")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return

        if args.phase == 'all':
            # 执行完整迁移
            result = executor.execute_full_migration()
        else:
            # 执行特定阶段
            phase_map = {
                'prepare': executor._prepare_environment,
                'export': executor._export_data,
                'import': executor._import_data,
                'validate': executor._validate_migration,
                'test': executor._performance_test,
                'report': executor._generate_final_report
            }

            phase_func = phase_map.get(args.phase)
            if phase_func:
                logger.info(f"🎯 执行阶段: {args.phase}")
                result = phase_func()
                print(f"✅ 阶段 {args.phase} 执行完成")
                print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
            else:
                logger.error(f"未知阶段: {args.phase}")
                return

    except Exception as e:
        logger.error(f"迁移执行异常: {e}")
        print(f"❌ 迁移失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
