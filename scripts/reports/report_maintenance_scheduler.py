#!/usr/bin/env python3
"""
报告维护调度器

功能：
1. 定期清理过期报告
2. 检查命名规范
3. 自动归档历史报告
4. 更新索引文件
5. 发送维护通知
"""

import re
import json
import schedule
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/report_maintenance.log'),
        logging.StreamHandler()
    ]
)


class ReportMaintenanceScheduler:
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.config_file = Path("config/reports/report_generation_config.json")
        self.load_config()

        # 确保日志目录存在
        Path("logs").mkdir(exist_ok=True)

    def load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            logging.error(f"配置文件未找到: {self.config_file}")
            self.config = self.get_default_config()

    def get_default_config(self):
        """获取默认配置"""
        return {
            "report_generation": {
                "quantity_control": {
                    "max_reports_per_directory": 10,
                    "archive_days": 30
                },
                "naming_convention": {
                    "forbidden_patterns": [
                        r"_\d{8}_",
                        r"_\d{4}\d{2}\d{2}_",
                        r"_\d{4}-\d{2}-\d{2}_",
                        r"_\d{4}_\d{2}_\d{2}_",
                        r"_\d{10,}"
                    ]
                }
            }
        }

    def check_naming_convention(self) -> Dict[str, List[str]]:
        """检查命名规范"""
        violations = {}
        forbidden_patterns = self.config["report_generation"]["naming_convention"]["forbidden_patterns"]

        for file_path in self.reports_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix in ['.md', '.json', '.html']:
                # 排除归档目录和文档目录
                file_path_str = str(file_path)
                filename = file_path.name

                # 排除特殊文件类型
                if (not file_path_str.startswith(str(self.reports_dir / "archive")) and
                    not file_path_str.startswith(str(self.reports_dir / "docs")) and
                    not file_path_str.startswith("docs/") and
                    not filename.startswith("maintenance_report_") and  # 维护脚本生成的报告
                    not filename.startswith("memory_scan_") and        # 内存扫描测试文件
                    not filename.startswith("test_") and               # 测试文件
                    not filename.startswith("parallel_test_") and      # 并行测试文件
                        not filename.startswith("infrastructure_test_")):  # 基础设施测试文件

                    for pattern in forbidden_patterns:
                        if re.search(pattern, filename):
                            dir_name = str(file_path.parent.relative_to(self.reports_dir))
                            if dir_name not in violations:
                                violations[dir_name] = []
                            violations[dir_name].append(filename)
                            logging.debug(f"发现命名违规: {file_path_str}")
                            break

        return violations

    def check_quantity_limits(self) -> Dict[str, int]:
        """检查数量限制"""
        limits = {}
        max_files = self.config["report_generation"]["quantity_control"]["max_reports_per_directory"]

        for dir_path in self.reports_dir.rglob("*"):
            if dir_path.is_dir() and not str(dir_path).startswith(str(self.reports_dir / "archive")):
                files = [f for f in dir_path.iterdir() if f.is_file() and f.suffix in [
                    '.md', '.json', '.html']]
                if len(files) > max_files:
                    limits[str(dir_path.relative_to(self.reports_dir))] = len(files)

        return limits

    def check_old_files(self) -> List[Path]:
        """检查过期文件"""
        old_files = []
        archive_days = self.config["report_generation"]["quantity_control"]["archive_days"]
        cutoff_date = datetime.now() - timedelta(days=archive_days)

        for file_path in self.reports_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix in ['.md', '.json', '.html']:
                # 排除归档目录
                if not str(file_path).startswith(str(self.reports_dir / "archive")):
                    mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if mtime < cutoff_date:
                        old_files.append(file_path)

        return old_files

    def generate_maintenance_report(self) -> Dict:
        """生成维护报告"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "naming_violations": self.check_naming_convention(),
            "quantity_violations": self.check_quantity_limits(),
            "old_files_count": len(self.check_old_files()),
            "total_files": len(list(self.reports_dir.rglob("*.md")) +
                               list(self.reports_dir.rglob("*.json")) +
                               list(self.reports_dir.rglob("*.html")))
        }

        return report

    def send_notification(self, report: Dict):
        """发送通知"""
        violations_count = sum(len(files) for files in report["naming_violations"].values())
        quantity_violations = len(report["quantity_violations"])
        old_files = report["old_files_count"]

        message = f"""
报告维护检查结果 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 统计信息:
- 总文件数: {report['total_files']}
- 命名违规: {violations_count} 个文件
- 数量超限: {quantity_violations} 个目录
- 过期文件: {old_files} 个文件

🔧 建议行动:
"""

        if violations_count > 0:
            message += f"- 需要重命名 {violations_count} 个文件\n"

        if quantity_violations > 0:
            message += f"- 需要清理 {quantity_violations} 个目录\n"

        if old_files > 0:
            message += f"- 需要归档 {old_files} 个过期文件\n"

        if violations_count == 0 and quantity_violations == 0 and old_files == 0:
            message += "- ✅ 所有检查通过，无需维护\n"

        logging.info(message)

        # 保存报告到文件
        report_file = self.reports_dir / \
            f"maintenance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return message

    def run_maintenance_check(self):
        """运行维护检查"""
        logging.info("开始报告维护检查...")

        try:
            report = self.generate_maintenance_report()
            message = self.send_notification(report)

            # 如果有问题，建议运行清理脚本
            violations_count = sum(len(files) for files in report["naming_violations"].values())
            quantity_violations = len(report["quantity_violations"])
            old_files = report["old_files_count"]

            if violations_count > 0 or quantity_violations > 0 or old_files > 0:
                logging.info("发现问题，建议运行清理脚本: python scripts/reports/report_cleanup_and_rename.py")

            logging.info("维护检查完成")

        except Exception as e:
            logging.error(f"维护检查失败: {e}")

    def schedule_maintenance(self):
        """调度维护任务"""
        # 每天凌晨2点运行维护检查
        schedule.every().day.at("02:00").do(self.run_maintenance_check)

        # 每周一凌晨3点运行完整清理
        schedule.every().monday.at("03:00").do(self.run_full_cleanup)

        logging.info("报告维护调度器已启动")
        logging.info("每日检查: 02:00")
        logging.info("每周清理: 周一 03:00")

        while True:
            schedule.run_pending()
            time.sleep(60)  # 每分钟检查一次

    def run_full_cleanup(self):
        """运行完整清理"""
        logging.info("开始完整清理...")

        try:
            # 导入并运行清理脚本
            import sys
            sys.path.append(str(Path("scripts/reports")))

            from report_cleanup_and_rename import ReportCleanupManager
            manager = ReportCleanupManager()
            manager.run_cleanup()

            logging.info("完整清理完成")

        except Exception as e:
            logging.error(f"完整清理失败: {e}")

    def run_manual_check(self):
        """手动运行检查"""
        self.run_maintenance_check()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="报告维护调度器")
    parser.add_argument("--check", action="store_true", help="运行一次检查")
    parser.add_argument("--cleanup", action="store_true", help="运行完整清理")
    parser.add_argument("--schedule", action="store_true", help="启动调度器")

    args = parser.parse_args()

    scheduler = ReportMaintenanceScheduler()

    if args.check:
        scheduler.run_manual_check()
    elif args.cleanup:
        scheduler.run_full_cleanup()
    elif args.schedule:
        scheduler.schedule_maintenance()
    else:
        # 默认运行一次检查
        scheduler.run_manual_check()


if __name__ == "__main__":
    main()
