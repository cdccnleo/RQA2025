#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
策略框架迁移和合并实施脚本
Strategy Framework Migration and Consolidation Implementation Script

执行完整的策略框架迁移工作，将分散在src/backtest和src/trading中的策略相关代码
迁移合并到统一的src/strategy服务层。
"""

import sys
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Any, Set
from dataclasses import dataclass
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8',
    handlers=[
        logging.FileHandler(project_root / 'logs' / 'migration.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class MigrationTask:
    """迁移任务定义"""
    source_path: str
    target_path: str
    description: str
    priority: int  # 1=高, 2=中, 3=低
    dependencies: List[str] = None
    status: str = "pending"

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class StrategyFrameworkMigrator:
    """策略框架迁移器"""

    def __init__(self):
        """初始化迁移器"""
        self.project_root = project_root
        self.src_dir = project_root / "src"
        self.migration_tasks = self._define_migration_tasks()
        self.completed_tasks: Set[str] = set()
        self.backup_dir = project_root / "backups" / \
            f"migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def _define_migration_tasks(self) -> Dict[str, MigrationTask]:
        """定义迁移任务"""
        tasks = {}

        # 核心接口迁移
        tasks["migrate_strategy_interfaces"] = MigrationTask(
            source_path="src/backtest/interfaces.py",
            target_path="src/strategy/interfaces/strategy_interfaces.py",
            description="迁移回测层的策略接口定义",
            priority=1
        )

        tasks["migrate_trading_strategy_interfaces"] = MigrationTask(
            source_path="src/trading/strategies/base_strategy.py",
            target_path="src/strategy/interfaces/strategy_interfaces.py",
            description="迁移交易层的策略接口定义",
            priority=1
        )

        # 策略实现迁移
        tasks["migrate_basic_strategies"] = MigrationTask(
            source_path="src/trading/strategies/basic/",
            target_path="src/strategy/strategies/",
            description="迁移基础策略实现（均值回归、趋势跟随等）",
            priority=1,
            dependencies=["migrate_strategy_interfaces", "migrate_trading_strategy_interfaces"]
        )

        tasks["migrate_china_strategies"] = MigrationTask(
            source_path="src/trading/strategies/china/",
            target_path="src/strategy/strategies/",
            description="迁移中国市场专用策略",
            priority=1,
            dependencies=["migrate_strategy_interfaces", "migrate_trading_strategy_interfaces"]
        )

        tasks["migrate_advanced_strategies"] = MigrationTask(
            source_path="src/trading/strategies/",
            target_path="src/strategy/strategies/",
            description="迁移高级策略（强化学习、跨市场套利等）",
            priority=2,
            dependencies=["migrate_basic_strategies"]
        )

        # 回测功能迁移
        tasks["migrate_backtest_engine"] = MigrationTask(
            source_path="src/backtest/backtest_engine.py",
            target_path="src/strategy/backtest/backtest_engine.py",
            description="迁移回测引擎核心实现",
            priority=1,
            dependencies=["migrate_strategy_interfaces"]
        )

        tasks["migrate_backtest_service"] = MigrationTask(
            source_path="src/backtest/",
            target_path="src/strategy/backtest/",
            description="迁移完整的回测服务模块",
            priority=1,
            dependencies=["migrate_backtest_engine"]
        )

        # 优化功能迁移
        tasks["migrate_optimization"] = MigrationTask(
            source_path="src/backtest/optimization/",
            target_path="src/strategy/optimization/",
            description="迁移策略优化功能",
            priority=2,
            dependencies=["migrate_strategy_interfaces"]
        )

        tasks["migrate_trading_optimization"] = MigrationTask(
            source_path="src/trading/strategies/optimization/",
            target_path="src/strategy/optimization/",
            description="迁移交易层的优化功能",
            priority=2,
            dependencies=["migrate_strategy_interfaces"]
        )

        # 工作空间迁移
        tasks["migrate_workspace_core"] = MigrationTask(
            source_path="src/trading/strategy_workspace/",
            target_path="src/strategy/workspace/",
            description="迁移策略工作空间核心功能",
            priority=1,
            dependencies=["migrate_strategy_interfaces"]
        )

        tasks["migrate_workspace_web"] = MigrationTask(
            source_path="src/trading/strategy_workspace/web_interface.py",
            target_path="src/strategy/workspace/web_api.py",
            description="迁移Web界面功能",
            priority=2,
            dependencies=["migrate_workspace_core"]
        )

        # 分析功能迁移
        tasks["migrate_analysis"] = MigrationTask(
            source_path="src/backtest/analysis/",
            target_path="src/strategy/backtest/",
            description="迁移回测分析功能",
            priority=2,
            dependencies=["migrate_backtest_engine"]
        )

        # 监控功能迁移
        tasks["migrate_monitoring"] = MigrationTask(
            source_path="src/backtest/",
            target_path="src/strategy/monitoring/",
            description="迁移策略监控功能",
            priority=3,
            dependencies=["migrate_strategy_interfaces"]
        )

        # 评估功能迁移
        tasks["migrate_evaluation"] = MigrationTask(
            source_path="src/backtest/evaluation/",
            target_path="src/strategy/monitoring/",
            description="迁移策略评估功能",
            priority=3,
            dependencies=["migrate_strategy_interfaces"]
        )

        return tasks

    def execute_migration(self) -> Dict[str, Any]:
        """
        执行完整的迁移工作

        Returns:
            Dict[str, Any]: 迁移结果报告
        """
        logger.info("[START] 开始执行策略框架迁移工作...")

        results = {
            "total_tasks": len(self.migration_tasks),
            "completed_tasks": 0,
            "failed_tasks": 0,
            "skipped_tasks": 0,
            "task_results": {},
            "errors": []
        }

        # 按优先级和依赖关系排序任务
        sorted_tasks = self._sort_tasks_by_priority_and_dependencies()

        for task_id, task in sorted_tasks:
            try:
                logger.info(f"[TASK] 执行任务: {task.description}")

                # 检查依赖
                if not self._check_dependencies(task):
                    logger.warning(f"[SKIP] 跳过任务 {task_id}: 依赖未满足")
                    results["skipped_tasks"] += 1
                    results["task_results"][task_id] = {
                        "status": "skipped", "reason": "dependencies_not_met"}
                    continue

                # 执行迁移
                success = self._execute_task(task_id, task)
                if success:
                    results["completed_tasks"] += 1
                    results["task_results"][task_id] = {"status": "completed"}
                    self.completed_tasks.add(task_id)
                    logger.info(f"[SUCCESS] 任务完成: {task.description}")
                else:
                    results["failed_tasks"] += 1
                    results["task_results"][task_id] = {"status": "failed"}
                    logger.error(f"[FAILED] 任务失败: {task.description}")

            except Exception as e:
                results["failed_tasks"] += 1
                results["task_results"][task_id] = {"status": "error", "error": str(e)}
                results["errors"].append(f"{task_id}: {str(e)}")
                logger.error(f"[ERROR] 任务执行异常: {task_id} - {e}")

        # 生成迁移报告
        self._generate_migration_report(results)

        return results

    def _sort_tasks_by_priority_and_dependencies(self) -> List[tuple]:
        """按优先级和依赖关系排序任务"""
        # 简单实现：按优先级排序，相同优先级的按依赖关系
        return sorted(
            self.migration_tasks.items(),
            key=lambda x: (x[1].priority, len(x[1].dependencies))
        )

    def _check_dependencies(self, task: MigrationTask) -> bool:
        """检查任务依赖是否满足"""
        for dep in task.dependencies:
            if dep not in self.completed_tasks:
                return False
        return True

    def _execute_task(self, task_id: str, task: MigrationTask) -> bool:
        """执行单个迁移任务"""
        source_path = self.project_root / task.source_path
        target_path = self.project_root / task.target_path

        try:
            if source_path.is_file():
                # 文件迁移
                return self._migrate_file(task_id, source_path, target_path, task)
            elif source_path.is_dir():
                # 目录迁移
                return self._migrate_directory(task_id, source_path, target_path, task)
            else:
                logger.warning(f"源路径不存在: {source_path}")
                return False
        except Exception as e:
            logger.error(f"执行任务失败: {e}")
            return False

    def _migrate_file(self, task_id: str, source_path: Path, target_path: Path, task: MigrationTask) -> bool:
        """迁移单个文件"""
        try:
            # 创建目标目录
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # 备份现有文件（如果存在）
            if target_path.exists():
                backup_path = self.backup_dir / target_path.relative_to(self.project_root)
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target_path, backup_path)
                logger.info(f"[BACKUP] 已备份现有文件: {target_path}")

            # 复制文件
            shutil.copy2(source_path, target_path)
            logger.info(f"[COPY] 已复制文件: {source_path} -> {target_path}")

            # 如果需要，进行代码适配
            if self._needs_code_adaptation(task_id):
                self._adapt_code_for_unified_interface(target_path, task)

            return True
        except Exception as e:
            logger.error(f"文件迁移失败: {e}")
            return False

    def _migrate_directory(self, task_id: str, source_path: Path, target_path: Path, task: MigrationTask) -> bool:
        """迁移整个目录"""
        try:
            # 创建目标目录
            target_path.mkdir(parents=True, exist_ok=True)

            # 递归复制目录内容
            for item in source_path.rglob('*'):
                if item.is_file() and not item.name.startswith('__pycache__'):
                    relative_path = item.relative_to(source_path)
                    target_file = target_path / relative_path

                    # 创建子目录
                    target_file.parent.mkdir(parents=True, exist_ok=True)

                    # 复制文件
                    shutil.copy2(item, target_file)
                    logger.info(f"[COPY] 已复制: {item} -> {target_file}")

                    # 代码适配
                    if self._needs_code_adaptation(task_id):
                        self._adapt_code_for_unified_interface(target_file, task)

            return True
        except Exception as e:
            logger.error(f"目录迁移失败: {e}")
            return False

    def _needs_code_adaptation(self, task_id: str) -> bool:
        """判断是否需要代码适配"""
        adaptation_task_ids = {
            "migrate_strategy_interfaces",
            "migrate_trading_strategy_interfaces",
            "migrate_basic_strategies",
            "migrate_china_strategies",
            "migrate_advanced_strategies",
            "migrate_backtest_engine",
            "migrate_workspace_core"
        }
        return task_id in adaptation_task_ids

    def _adapt_code_for_unified_interface(self, file_path: Path, task: MigrationTask):
        """适配代码以使用统一的接口"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 基本适配规则
            adaptations = {
                # 导入语句适配
                'from src.backtest.strategy_framework import': 'from src.strategy.interfaces.strategy_interfaces import',
                'from src.trading.strategies.base_strategy import': 'from src.strategy.interfaces.strategy_interfaces import',
                'from .base_strategy import': 'from src.strategy.strategies.base_strategy import',

                # 类名适配
                'class BaseStrategy(': 'class BaseStrategy(BaseStrategy):',
                'class BacktestEngine(': 'class BacktestEngine(IBacktestEngine):',
            }

            modified = False
            for old, new in adaptations.items():
                if old in content:
                    content = content.replace(old, new)
                    modified = True
                    logger.info(f"[ADAPT] 适配代码: {old} -> {new}")

            if modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info(f"[ADAPT] 代码适配完成: {file_path}")

        except Exception as e:
            logger.error(f"代码适配失败: {file_path} - {e}")

    def _generate_migration_report(self, results: Dict[str, Any]):
        """生成迁移报告"""
        report_path = self.project_root / "docs" / "strategy" / "MIGRATION_REPORT.md"

        report_content = f"""# 策略框架迁移报告

## [STATS] 迁移概况

- **迁移任务总数**: {results['total_tasks']}
- **成功完成**: {results['completed_tasks']}
- **失败任务**: {results['failed_tasks']}
- **跳过任务**: {results['skipped_tasks']}
- **迁移时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **成功率**: {(results['completed_tasks'] / results['total_tasks'] * 100):.1f}%

## [RESULTS] 任务执行结果

"""

        for task_id, result in results['task_results'].items():
            status_emoji = {
                "completed": "[OK]",
                "failed": "[FAIL]",
                "skipped": "[SKIP]",
                "error": "[ERR]"
            }.get(result['status'], "[UNK]")

            report_content += f"### {status_emoji} {task_id}\n"
            report_content += f"- **状态**: {result['status']}\n"
            if 'error' in result:
                report_content += f"- **错误**: {result['error']}\n"
            if 'reason' in result:
                report_content += f"- **原因**: {result['reason']}\n"
            report_content += "\n"

        if results['errors']:
            report_content += "## [ERRORS] 错误详情\n\n"
            for error in results['errors']:
                report_content += f"- {error}\n"

        report_content += f"""
## [NEXT] 后续工作

1. **代码审查**: 检查所有迁移的代码是否正确
2. **接口统一**: 确保所有组件使用统一的接口
3. **依赖清理**: 移除旧的目录和文件
4. **测试验证**: 运行完整的测试套件验证功能
5. **文档更新**: 更新所有相关文档

## [BACKUP] 备份信息

所有修改的文件已备份到: `{self.backup_dir}`

---

**迁移报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        # 确保目录存在
        report_path.parent.mkdir(parents=True, exist_ok=True)

        # 写入报告
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        logger.info(f"[REPORT] 迁移报告已生成: {report_path}")


def main():
    """主函数"""
    try:
        migrator = StrategyFrameworkMigrator()
        results = migrator.execute_migration()

        if results['failed_tasks'] > 0:
            logger.error(f"[WARN] 迁移完成，但有 {results['failed_tasks']} 个任务失败")
            sys.exit(1)
        else:
            logger.info("[SUCCESS] 策略框架迁移全部完成！")
            sys.exit(0)

    except Exception as e:
        logger.error(f"[ERROR] 迁移过程异常: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
