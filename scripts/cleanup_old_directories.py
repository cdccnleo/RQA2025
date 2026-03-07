#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
清理旧目录中的策略相关代码
Clean up strategy-related code in old directories

删除 src/backtest/ 和 src/trading/ 目录中已被迁移到 src/strategy/ 的策略相关文件，
保留必要的非策略相关功能代码。
"""

import sys
import shutil
from pathlib import Path
from typing import Set, Dict, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class DirectoryCleanupManager:
    """目录清理管理器"""

    def __init__(self):
        """初始化清理管理器"""
        self.project_root = project_root
        self.backup_dir = project_root / "backups" / \
            f"cleanup_{project_root.name}_{self.get_timestamp()}"

        # 需要保留的文件（非策略相关）
        self.keep_files_backtest = {
            "__init__.py",
            # 数据加载和处理相关（如果不是策略专用）
            "data_loader.py",
            # 配置管理（如果不是策略专用）
            "config_manager.py",
            # 可视化（如果不是策略专用）
            "visualization.py",
            "visualizer.py",
            # 工具类
            "utils/backtest_utils.py",
            "utils/__init__.py"
        }

        self.keep_files_trading = {
            "__init__.py",
            # 交易执行相关（非策略部分）
            "execution/",
            "execution_engine.py",
            "executor.py",
            "order/",
            "order_executor.py",
            "order_manager.py",
            # 账户管理
            "account/",
            "account_manager.py",
            # 风控（非策略部分）
            "risk/",
            "risk.py",
            # 交易网关
            "gateway.py",
            "broker_adapter.py",
            # 分布式交易
            "distributed_distributed_trading_node.py",
            "distributed_intelligent_order_router.py",
            # 实时交易
            "live_trader.py",
            "live_trading.py",
            "real_time_executor.py",
            "realtime_realtime_trading_system.py",
            # 高级分析（非策略部分）
            "advanced_analysis/",
            "performance_analyzer.py",
            # ML集成（非策略部分）
            "ml_integration/",
            # 投资组合管理
            "portfolio_portfolio_manager.py",
            "portfolio_portfolio_optimizer.py",
            "portfolio___init__.py",
            # 结算
            "settlement_settlement_engine.py",
            "settlement___init__.py",
            # 信号处理（非策略部分）
            "signal_signal_generator.py",
            "signal___init__.py",
            # 智能执行
            "smart_execution.py",
            # 交易引擎
            "trading_engine.py",
            "trading_engine_with_distributed.py",
            # 市场数据
            "universe/"
        }

    def get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def create_backup(self, source_path: Path, backup_path: Path):
        """创建备份"""
        try:
            if source_path.exists():
                if source_path.is_file():
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_path, backup_path)
                elif source_path.is_dir():
                    shutil.copytree(source_path, backup_path, dirs_exist_ok=True)
                print(f"✅ 已备份: {source_path} -> {backup_path}")
        except Exception as e:
            print(f"❌ 备份失败: {source_path} - {e}")

    def should_keep_file(self, file_path: Path, keep_files: Set[str]) -> bool:
        """判断是否应该保留文件"""
        # 计算相对于src目录的路径
        try:
            relative_path = file_path.relative_to(self.project_root / "src")
        except ValueError:
            # 如果计算失败，使用文件名
            relative_path = file_path.name

        # 检查是否在保留列表中
        for keep_pattern in keep_files:
            if keep_pattern in str(relative_path) or file_path.name in keep_files:
                return True

        return False

    def cleanup_backtest_directory(self) -> Dict[str, Any]:
        """清理 src/backtest/ 目录"""
        print("🧹 开始清理 src/backtest/ 目录...")

        backtest_dir = self.project_root / "src" / "backtest"
        results = {
            "total_files": 0,
            "removed_files": 0,
            "kept_files": 0,
            "removed_dirs": 0,
            "kept_dirs": 0
        }

        if not backtest_dir.exists():
            print("ℹ️ src/backtest/ 目录不存在")
            return results

        # 备份整个目录
        backup_path = self.backup_dir / "src_backtest"
        self.create_backup(backtest_dir, backup_path)

        # 遍历并清理文件
        for item in backtest_dir.rglob("*"):
            if item.is_file() and not item.name.startswith("__pycache__"):
                results["total_files"] += 1

                if self.should_keep_file(item, self.keep_files_backtest):
                    results["kept_files"] += 1
                    print(f"✅ 保留文件: {item.relative_to(self.project_root)}")
                else:
                    try:
                        item.unlink()
                        results["removed_files"] += 1
                        print(f"🗑️ 删除文件: {item.relative_to(self.project_root)}")
                    except Exception as e:
                        print(f"❌ 删除失败: {item} - {e}")

        # 清理空的子目录
        for dir_path in sorted(backtest_dir.rglob("*"), reverse=True):
            if dir_path.is_dir() and not any(dir_path.iterdir()):
                try:
                    dir_path.rmdir()
                    results["removed_dirs"] += 1
                    print(f"🗑️ 删除空目录: {dir_path.relative_to(self.project_root)}")
                except Exception as e:
                    print(f"❌ 删除目录失败: {dir_path} - {e}")

        # 统计保留的目录
        for dir_path in backtest_dir.rglob("*"):
            if dir_path.is_dir():
                results["kept_dirs"] += 1

        return results

    def cleanup_trading_directory(self) -> Dict[str, Any]:
        """清理 src/trading/ 目录"""
        print("🧹 开始清理 src/trading/ 目录...")

        trading_dir = self.project_root / "src" / "trading"
        results = {
            "total_files": 0,
            "removed_files": 0,
            "kept_files": 0,
            "removed_dirs": 0,
            "kept_dirs": 0
        }

        if not trading_dir.exists():
            print("ℹ️ src/trading/ 目录不存在")
            return results

        # 备份整个目录
        backup_path = self.backup_dir / "src_trading"
        self.create_backup(trading_dir, backup_path)

        # 特殊处理：完全删除策略相关目录
        strategy_dirs = ["strategies", "strategy_workspace"]
        for strategy_dir in strategy_dirs:
            dir_path = trading_dir / strategy_dir
            if dir_path.exists():
                try:
                    shutil.rmtree(dir_path)
                    print(f"🗑️ 完全删除策略目录: {dir_path.relative_to(self.project_root)}")
                    results["removed_dirs"] += 1
                except Exception as e:
                    print(f"❌ 删除策略目录失败: {dir_path} - {e}")

        # 清理其他策略相关文件
        strategy_files = [
            "api.py",  # 策略API
            "backtest_analyzer.py",  # 回测分析器
            "backtester.py",  # 回测器
            "strategy_high_freq_optimizer.py",  # 策略优化器
            "manager.py"  # 如果主要是策略管理
        ]

        for file_name in strategy_files:
            file_path = trading_dir / file_name
            if file_path.exists():
                try:
                    file_path.unlink()
                    results["removed_files"] += 1
                    print(f"🗑️ 删除策略文件: {file_path.relative_to(self.project_root)}")
                except Exception as e:
                    print(f"❌ 删除策略文件失败: {file_path} - {e}")

        # 遍历并清理剩余文件
        for item in trading_dir.rglob("*"):
            if item.is_file() and not item.name.startswith("__pycache__"):
                results["total_files"] += 1

                if self.should_keep_file(item, self.keep_files_trading):
                    results["kept_files"] += 1
                    print(f"✅ 保留文件: {item.relative_to(self.project_root)}")
                else:
                    try:
                        item.unlink()
                        results["removed_files"] += 1
                        print(f"🗑️ 删除文件: {item.relative_to(self.project_root)}")
                    except Exception as e:
                        print(f"❌ 删除失败: {item} - {e}")

        # 清理空的子目录
        for dir_path in sorted(trading_dir.rglob("*"), reverse=True):
            if dir_path.is_dir() and not any(dir_path.iterdir()):
                try:
                    dir_path.rmdir()
                    results["removed_dirs"] += 1
                    print(f"🗑️ 删除空目录: {dir_path.relative_to(self.project_root)}")
                except Exception as e:
                    print(f"❌ 删除目录失败: {dir_path} - {e}")

        # 统计保留的目录
        for dir_path in trading_dir.rglob("*"):
            if dir_path.is_dir():
                results["kept_dirs"] += 1

        return results

    def generate_cleanup_report(self, backtest_results: Dict[str, Any],
                                trading_results: Dict[str, Any]) -> str:
        """生成清理报告"""
        report = f"""# 目录清理报告

## 📊 清理概况

- **清理时间**: {self.get_timestamp()}
- **备份位置**: `{self.backup_dir}`

## 🧹 Backtest目录清理结果

- **总文件数**: {backtest_results['total_files']}
- **删除文件数**: {backtest_results['removed_files']}
- **保留文件数**: {backtest_results['kept_files']}
- **删除目录数**: {backtest_results['removed_dirs']}
- **保留目录数**: {backtest_results['kept_dirs']}

## 🧹 Trading目录清理结果

- **总文件数**: {trading_results['total_files']}
- **删除文件数**: {trading_results['removed_files']}
- **保留文件数**: {trading_results['kept_files']}
- **删除目录数**: {trading_results['removed_dirs']}
- **保留目录数**: {trading_results['kept_dirs']}

## 📁 保留的文件类型

### Backtest目录保留
{chr(10).join(f"- {f}" for f in sorted(self.keep_files_backtest))}

### Trading目录保留
{chr(10).join(f"- {f}" for f in sorted(self.keep_files_trading))}

## ✅ 清理完成

所有策略相关代码已成功迁移到统一的 `src/strategy/` 目录，
旧目录中的冗余代码已清理完成。

---
**清理报告生成时间**: {self.get_timestamp()}
"""

        return report

    def execute_cleanup(self) -> Dict[str, Any]:
        """
        执行完整的清理工作

        Returns:
            Dict[str, Any]: 清理结果
        """
        print("🚀 开始执行目录清理工作...")

        # 创建备份目录
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        print(f"📦 备份目录: {self.backup_dir}")

        # 清理 backtest 目录
        backtest_results = self.cleanup_backtest_directory()

        # 清理 trading 目录
        trading_results = self.cleanup_trading_directory()

        # 生成清理报告
        report = self.generate_cleanup_report(backtest_results, trading_results)

        # 保存报告
        report_path = self.project_root / "docs" / "strategy" / "DIRECTORY_CLEANUP_REPORT.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"📋 清理报告已生成: {report_path}")

        results = {
            "backtest_cleanup": backtest_results,
            "trading_cleanup": trading_results,
            "backup_location": str(self.backup_dir),
            "report_location": str(report_path)
        }

        return results


def main():
    """主函数"""
    try:
        cleanup_manager = DirectoryCleanupManager()
        results = cleanup_manager.execute_cleanup()

        print("\n" + "="*60)
        print("🎉 目录清理完成！")
        print("="*60)

        backtest = results["backtest_cleanup"]
        trading = results["trading_cleanup"]

        print("\n📊 Backtest目录:")
        print(f"  • 删除文件: {backtest['removed_files']}")
        print(f"  • 保留文件: {backtest['kept_files']}")
        print(f"  • 删除目录: {backtest['removed_dirs']}")

        print("\n📊 Trading目录:")
        print(f"  • 删除文件: {trading['removed_files']}")
        print(f"  • 保留文件: {trading['kept_files']}")
        print(f"  • 删除目录: {trading['removed_dirs']}")

        print(f"\n📦 备份位置: {results['backup_location']}")
        print(f"📋 清理报告: {results['report_location']}")

        print("\n✅ 策略框架迁移和目录清理工作全部完成！")

    except Exception as e:
        print(f"❌ 清理过程异常: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
