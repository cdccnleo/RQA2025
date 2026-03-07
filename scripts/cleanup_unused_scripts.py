#!/usr/bin/env python3
"""
脚本清理工具
用于识别和清理scripts目录中的无用脚本
"""

import shutil
import json
import logging
from pathlib import Path
from typing import Dict, List, Set
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ScriptCleaner:
    """脚本清理器"""

    def __init__(self, scripts_dir: str = "scripts"):
        self.scripts_dir = Path(scripts_dir)
        self.core_scripts = self._load_core_scripts()
        self.usage_data = {}
        self.backup_dir = Path("scripts_backup")

    def _load_core_scripts(self) -> Set[str]:
        """加载核心脚本列表"""
        return {
            # 测试核心
            "testing/run_tests.py",
            "testing/run_focused_tests.py",
            "testing/run_e2e_tests.py",
            "testing/run_infrastructure_tests.py",
            "testing/verify_core_modules.py",
            "testing/enhance_test_coverage_plan.py",

            # 配置管理核心
            "testing/config_validation_test.py",
            "testing/config_sync_test.py",
            "testing/config_web_test.py",
            "testing/config_performance_test.py",

            # 环境管理核心
            "deployment/environment/health_check.py",
            "deployment/environment/quick_start.bat",
            "deployment/environment/run_conda_tests.bat",

            # 部署核心
            "deployment/auto_deployment.py",
            "deployment/production_deploy.py",
            "deployment/deployment_preparation.py",

            # 开发工具核心
            "development/optimize_imports.py",
            "development/smart_fix_engine.py",
            "development/fix_filename_issues.py",

            # 测试修复核心
            "testing/fixes/auto_fix_tests.py",
            "testing/fixes/update_test_imports.py",
            "testing/fixes/fix_infrastructure_tests.py",

            # 覆盖率分析核心
            "testing/optimization/test_coverage_analyzer.py",
            "testing/optimization/boost_infrastructure_coverage.py",
            "testing/optimization/analyze_infrastructure_coverage.py",

            # 监控报告核心
            "monitoring/progress_monitor.py",
            "monitoring/progress_tracker.py",
            "testing/tools/generate_test_reports.py",
            "testing/verify_core_modules.py",

            # 模型核心
            "models/model_deployment_controller.py",
            "models/auto_model_landing.py",
            "models/auto_model_landing_conda.py",
            "models/demos/pretrained_models_demo.py",
            "models/demos/optimized_pretrained_models_demo.py",

            # 工作流核心
            "workflows/minimal_e2e_main_flow.py",
            "workflows/minimal_infra_main_flow.py",
            "workflows/minimal_model_main_flow.py",

            # 压力测试核心
            "stress_testing/run_stress_test.py",
            "stress_testing/run_optimized_stress_test.py",
            "stress_testing/run_stable_infrastructure_tests.py",

            # 基础设施核心
            "infrastructure/optimization/technical_debt_manager.py",
            "infrastructure/optimization/optimize_system.py",
            "infrastructure/optimization/ops_optimizer.py",
            "infrastructure/validation/verify_fpga_modules.py",
            "infrastructure/validation/update_fpga_test_imports.py",

            # 交易回测核心
            "trading/minimal_trading_main_flow.py",
            "trading/risk/minimal_risk_main_flow.py",
            "backtest/backtest_optimizer.py",
            "backtest/portfolio_optimizer.py",

            # API集成核心
            "api/api_sdk_demo.py",
            "api/simple_api_server.py",
            "api/optimized_api_server.py",
            "integration/integration_test.py",
            "integration/run_complete_e2e_test.py",

            # 项目收尾核心
            "project/project_finalizer.py",
            "project/project_closure.py",

            # 文档和索引
            "SCRIPT_INDEX.md",
            "SCRIPT_USAGE_ANALYSIS.md",
            "QUICK_START_GUIDE.md",
            "ORGANIZATION_SUMMARY.md",
            "README.md"
        }

    def scan_scripts(self) -> Dict[str, Dict]:
        """扫描所有脚本文件"""
        logger.info("🔍 扫描scripts目录...")

        scripts_info = {}

        for file_path in self.scripts_dir.rglob("*.py"):
            if file_path.is_file():
                relative_path = str(file_path.relative_to(self.scripts_dir))

                # 获取文件信息
                stat = file_path.stat()
                scripts_info[relative_path] = {
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime),
                    "is_core": relative_path in self.core_scripts,
                    "path": str(file_path)
                }

        # 扫描其他脚本文件
        for file_path in self.scripts_dir.rglob("*.sh"):
            if file_path.is_file():
                relative_path = str(file_path.relative_to(self.scripts_dir))
                stat = file_path.stat()
                scripts_info[relative_path] = {
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime),
                    "is_core": relative_path in self.core_scripts,
                    "path": str(file_path)
                }

        logger.info(f"📊 发现 {len(scripts_info)} 个脚本文件")
        return scripts_info

    def identify_unused_scripts(self, scripts_info: Dict[str, Dict]) -> List[str]:
        """识别可能无用的脚本"""
        logger.info("🔍 识别可能无用的脚本...")

        unused_scripts = []
        cutoff_date = datetime.now() - timedelta(days=30)  # 30天未修改

        for script_path, info in scripts_info.items():
            # 跳过核心脚本
            if info["is_core"]:
                continue

            # 检查修改时间
            if info["modified"] < cutoff_date:
                unused_scripts.append(script_path)
                logger.info(
                    f"⚠️  可能无用: {script_path} (最后修改: {info['modified'].strftime('%Y-%m-%d')})")

        return unused_scripts

    def backup_scripts(self, scripts_to_backup: List[str]) -> bool:
        """备份脚本"""
        logger.info("💾 备份脚本...")

        self.backup_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_subdir = self.backup_dir / f"backup_{timestamp}"
        backup_subdir.mkdir(exist_ok=True)

        for script_path in scripts_to_backup:
            source_path = self.scripts_dir / script_path
            if source_path.exists():
                # 创建目标目录
                target_dir = backup_subdir / Path(script_path).parent
                target_dir.mkdir(parents=True, exist_ok=True)

                # 复制文件
                target_path = backup_subdir / script_path
                shutil.copy2(source_path, target_path)
                logger.info(f"📦 已备份: {script_path}")

        # 保存备份信息
        backup_info = {
            "timestamp": timestamp,
            "scripts": scripts_to_backup,
            "total_count": len(scripts_to_backup)
        }

        with open(backup_subdir / "backup_info.json", "w", encoding="utf-8") as f:
            json.dump(backup_info, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"✅ 备份完成，共备份 {len(scripts_to_backup)} 个脚本到 {backup_subdir}")
        return True

    def remove_scripts(self, scripts_to_remove: List[str]) -> bool:
        """删除脚本"""
        logger.info("🗑️  删除脚本...")

        removed_count = 0
        for script_path in scripts_to_remove:
            file_path = self.scripts_dir / script_path
            if file_path.exists():
                try:
                    file_path.unlink()
                    logger.info(f"🗑️  已删除: {script_path}")
                    removed_count += 1
                except Exception as e:
                    logger.error(f"❌ 删除失败 {script_path}: {e}")

        logger.info(f"✅ 删除完成，共删除 {removed_count} 个脚本")
        return True

    def generate_cleanup_report(self, scripts_info: Dict[str, Dict], unused_scripts: List[str]) -> str:
        """生成清理报告"""
        logger.info("📊 生成清理报告...")

        total_scripts = len(scripts_info)
        core_scripts = len([s for s in scripts_info.values() if s["is_core"]])
        unused_count = len(unused_scripts)

        report = f"""
# Scripts目录清理报告

## 📊 统计信息
- **总脚本数**: {total_scripts}
- **核心脚本**: {core_scripts}
- **可能无用脚本**: {unused_count}
- **清理比例**: {unused_count/total_scripts*100:.1f}%

## 🗑️ 建议删除的脚本
"""

        for script_path in unused_scripts:
            info = scripts_info[script_path]
            report += f"- `{script_path}` (最后修改: {info['modified'].strftime('%Y-%m-%d')})\n"

        report += f"""
## 📦 备份信息
- 备份位置: `{self.backup_dir}`
- 备份时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## ⚠️ 注意事项
1. 删除前已自动备份到 `{self.backup_dir}`
2. 如需恢复，请从备份目录复制文件
3. 建议在删除后运行测试，确保功能正常

## 🎯 后续建议
1. 定期运行此脚本进行清理
2. 新脚本创建前检查是否已有相似功能
3. 优先使用核心脚本，避免重复开发
"""

        # 保存报告
        report_file = Path("scripts_cleanup_report.md")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info(f"📄 清理报告已保存到: {report_file}")
        return report

    def cleanup(self, dry_run: bool = True) -> bool:
        """执行清理流程"""
        logger.info("🚀 开始脚本清理流程...")

        # 1. 扫描脚本
        scripts_info = self.scan_scripts()

        # 2. 识别无用脚本
        unused_scripts = self.identify_unused_scripts(scripts_info)

        if not unused_scripts:
            logger.info("✅ 没有发现需要清理的脚本")
            return True

        # 3. 生成报告
        report = self.generate_cleanup_report(scripts_info, unused_scripts)
        print(report)

        if dry_run:
            logger.info("🔍 这是预览模式，未实际删除文件")
            logger.info("💡 如需实际删除，请设置 dry_run=False")
            return True

        # 4. 备份脚本
        if not self.backup_scripts(unused_scripts):
            logger.error("❌ 备份失败，停止清理")
            return False

        # 5. 删除脚本
        if not self.remove_scripts(unused_scripts):
            logger.error("❌ 删除失败")
            return False

        logger.info("🎉 脚本清理完成!")
        return True


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="脚本清理工具")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="预览模式，不实际删除文件")
    parser.add_argument("--execute", action="store_true",
                        help="实际执行删除操作")

    args = parser.parse_args()

    cleaner = ScriptCleaner()

    if args.execute:
        # 实际执行删除
        success = cleaner.cleanup(dry_run=False)
        if success:
            print("✅ 清理完成")
        else:
            print("❌ 清理失败")
    else:
        # 预览模式
        success = cleaner.cleanup(dry_run=True)
        if success:
            print("✅ 预览完成，请检查报告")


if __name__ == "__main__":
    main()
