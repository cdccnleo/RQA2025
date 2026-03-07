#!/usr/bin/env python3
"""
测试覆盖率面板定时更新调度器

支持多种调度方式：
1. Cron作业 (Linux/Unix)
2. Windows任务计划程序
3. Python内置调度器
4. 系统服务
"""

import os
import sys
import logging
from pathlib import Path
import subprocess
import platform
import argparse

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/coverage_scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CoverageDashboardScheduler:
    """覆盖率面板调度器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.script_dir = self.project_root / 'scripts'
        self.logs_dir = self.project_root / 'logs'
        self.reports_dir = self.project_root / 'reports'

        # 确保目录存在
        self.logs_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)

    def setup_cron_job(self, interval_minutes: int = 60):
        """设置Linux/Unix cron作业"""
        if platform.system() != 'Linux':
            logger.error("Cron作业只支持Linux/Unix系统")
            return False

        script_path = self.script_dir / 'auto_update_coverage_dashboard.sh'
        cron_command = f"*/{interval_minutes} * * * * {script_path} --single"

        logger.info(f"设置cron作业: {cron_command}")

        # 备份现有的crontab
        try:
            result = subprocess.run(['crontab', '-l'],
                                    capture_output=True, text=True)
            existing_crontab = result.stdout if result.returncode == 0 else ""
        except subprocess.CalledProcessError:
            existing_crontab = ""

        # 检查是否已存在相同的作业
        if cron_command in existing_crontab:
            logger.info("Cron作业已存在")
            return True

        # 添加新作业
        new_crontab = existing_crontab + f"\n# RQA2025 Coverage Dashboard Update\n{cron_command}\n"

        try:
            process = subprocess.Popen(['crontab', '-'], stdin=subprocess.PIPE, text=True)
            process.communicate(new_crontab)
            if process.returncode == 0:
                logger.info("Cron作业设置成功")
                return True
            else:
                logger.error("Cron作业设置失败")
                return False
        except Exception as e:
            logger.error(f"设置cron作业时出错: {e}")
            return False

    def setup_windows_task(self, interval_minutes: int = 60):
        """设置Windows任务计划程序"""
        if platform.system() != 'Windows':
            logger.error("Windows任务计划程序只支持Windows系统")
            return False

        script_path = str(self.script_dir / 'auto_update_coverage_dashboard.bat')
        task_name = "RQA2025_Coverage_Dashboard_Update"

        # 使用schtasks创建任务
        cmd = [
            'schtasks', '/create', '/tn', task_name,
            '/tr', f'"{script_path} --single"',
            '/sc', 'minute', '/mo', str(interval_minutes),
            '/rl', 'highest', '/f'
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Windows任务计划程序设置成功")
                return True
            else:
                logger.error(f"Windows任务设置失败: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"设置Windows任务时出错: {e}")
            return False

    def remove_cron_job(self):
        """移除cron作业"""
        if platform.system() != 'Linux':
            logger.error("只支持Linux/Unix系统")
            return False

        try:
            # 获取现有crontab
            result = subprocess.run(['crontab', '-l'],
                                    capture_output=True, text=True)
            if result.returncode != 0:
                logger.info("没有找到现有的crontab")
                return True

            existing_crontab = result.stdout

            # 过滤掉相关的行
            lines = existing_crontab.split('\n')
            filtered_lines = [
                line for line in lines
                if 'RQA2025' not in line and 'coverage_dashboard' not in line
            ]

            # 更新crontab
            new_crontab = '\n'.join(filtered_lines) + '\n'
            process = subprocess.Popen(['crontab', '-'], stdin=subprocess.PIPE, text=True)
            process.communicate(new_crontab)

            if process.returncode == 0:
                logger.info("Cron作业移除成功")
                return True
            else:
                logger.error("Cron作业移除失败")
                return False

        except Exception as e:
            logger.error(f"移除cron作业时出错: {e}")
            return False

    def remove_windows_task(self):
        """移除Windows任务"""
        if platform.system() != 'Windows':
            logger.error("只支持Windows系统")
            return False

        task_name = "RQA2025_Coverage_Dashboard_Update"

        try:
            result = subprocess.run(['schtasks', '/delete', '/tn', task_name, '/f'],
                                    capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Windows任务移除成功")
                return True
            else:
                logger.error(f"Windows任务移除失败: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"移除Windows任务时出错: {e}")
            return False

    def run_manual_update(self):
        """手动运行一次更新"""
        logger.info("开始手动更新覆盖率面板")

        script_path = self.script_dir / 'generate_coverage_dashboard.py'

        try:
            cmd = [sys.executable, str(script_path)]
            result = subprocess.run(cmd, cwd=self.project_root,
                                    capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("覆盖率面板更新成功")
                logger.info(result.stdout)
                return True
            else:
                logger.error(f"覆盖率面板更新失败: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"运行更新脚本时出错: {e}")
            return False

    def generate_systemd_service(self):
        """生成systemd服务文件"""
        if platform.system() != 'Linux':
            logger.error("systemd只支持Linux系统")
            return None

        service_content = f"""[Unit]
Description=RQA2025 Coverage Dashboard Auto Update
After=network.target

[Service]
Type=simple
User={os.getenv('USER', 'www-data')}
WorkingDirectory={self.project_root}
ExecStart={sys.executable} {self.script_dir}/generate_coverage_dashboard.py --auto-update --interval 3600
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""

        service_path = Path('/etc/systemd/system/rqa2025-coverage-dashboard.service')

        try:
            with open(service_path, 'w') as f:
                f.write(service_content)

            logger.info(f"systemd服务文件已生成: {service_path}")
            logger.info("运行以下命令启用服务:")
            logger.info("  sudo systemctl daemon-reload")
            logger.info("  sudo systemctl enable rqa2025-coverage-dashboard")
            logger.info("  sudo systemctl start rqa2025-coverage-dashboard")

            return str(service_path)

        except Exception as e:
            logger.error(f"生成systemd服务文件失败: {e}")
            return None

    def show_status(self):
        """显示当前调度状态"""
        system = platform.system()

        print("=== 覆盖率面板调度状态 ===")
        print(f"操作系统: {system}")
        print(f"项目目录: {self.project_root}")

        if system == 'Linux':
            # 检查cron作业
            try:
                result = subprocess.run(['crontab', '-l'],
                                        capture_output=True, text=True)
                if result.returncode == 0:
                    crontab_content = result.stdout
                    coverage_jobs = [line for line in crontab_content.split('\n')
                                     if 'coverage' in line.lower() or 'rqa2025' in line.lower()]
                    if coverage_jobs:
                        print("✅ Cron作业状态: 已配置")
                        for job in coverage_jobs:
                            print(f"   {job}")
                    else:
                        print("❌ Cron作业状态: 未配置")
                else:
                    print("❌ Cron作业状态: 无法检查")
            except Exception as e:
                print(f"❌ Cron作业状态: 检查失败 ({e})")

            # 检查systemd服务
            service_path = Path('/etc/systemd/system/rqa2025-coverage-dashboard.service')
            if service_path.exists():
                print("✅ systemd服务状态: 已配置")
                try:
                    result = subprocess.run(['systemctl', 'is-active', 'rqa2025-coverage-dashboard'],
                                            capture_output=True, text=True)
                    status = result.stdout.strip()
                    print(f"   服务状态: {status}")
                except:
                    print("   服务状态: 未知")
            else:
                print("❌ systemd服务状态: 未配置")

        elif system == 'Windows':
            # 检查任务计划程序
            try:
                result = subprocess.run(['schtasks', '/query', '/tn', 'RQA2025_Coverage_Dashboard_Update'],
                                        capture_output=True, text=True)
                if result.returncode == 0:
                    print("✅ Windows任务状态: 已配置")
                    # 解析任务信息
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'RQA2025' in line:
                            print(f"   {line.strip()}")
                else:
                    print("❌ Windows任务状态: 未配置")
            except Exception as e:
                print(f"❌ Windows任务状态: 检查失败 ({e})")

        # 检查日志文件
        log_file = self.logs_dir / 'coverage_scheduler.log'
        if log_file.exists():
            print(f"✅ 日志文件: {log_file} (存在)")
            # 显示最后几行日志
            try:
                result = subprocess.run(['tail', '-5', str(log_file)],
                                        capture_output=True, text=True)
                if result.returncode == 0:
                    print("最后5行日志:")
                    for line in result.stdout.split('\n'):
                        if line.strip():
                            print(f"   {line}")
            except:
                pass
        else:
            print(f"❌ 日志文件: {log_file} (不存在)")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='测试覆盖率面板定时更新调度器')
    parser.add_argument('--project-root', default='.',
                        help='项目根目录路径')
    parser.add_argument('--setup-cron', type=int, metavar='MINUTES',
                        help='设置cron作业，指定间隔分钟数')
    parser.add_argument('--setup-windows-task', type=int, metavar='MINUTES',
                        help='设置Windows任务，指定间隔分钟数')
    parser.add_argument('--remove-cron', action='store_true',
                        help='移除cron作业')
    parser.add_argument('--remove-windows-task', action='store_true',
                        help='移除Windows任务')
    parser.add_argument('--manual-update', action='store_true',
                        help='手动运行一次更新')
    parser.add_argument('--generate-systemd', action='store_true',
                        help='生成systemd服务文件')
    parser.add_argument('--status', action='store_true',
                        help='显示当前调度状态')

    args = parser.parse_args()

    # 获取项目根目录
    project_root = Path(args.project_root).resolve()
    if not project_root.exists():
        logger.error(f"项目目录不存在: {project_root}")
        sys.exit(1)

    scheduler = CoverageDashboardScheduler(str(project_root))

    if args.setup_cron:
        success = scheduler.setup_cron_job(args.setup_cron)
        sys.exit(0 if success else 1)

    elif args.setup_windows_task:
        success = scheduler.setup_windows_task(args.setup_windows_task)
        sys.exit(0 if success else 1)

    elif args.remove_cron:
        success = scheduler.remove_cron_job()
        sys.exit(0 if success else 1)

    elif args.remove_windows_task:
        success = scheduler.remove_windows_task()
        sys.exit(0 if success else 1)

    elif args.manual_update:
        success = scheduler.run_manual_update()
        sys.exit(0 if success else 1)

    elif args.generate_systemd:
        service_path = scheduler.generate_systemd_service()
        if service_path:
            print(f"systemd服务文件已生成: {service_path}")
            sys.exit(0)
        else:
            sys.exit(1)

    elif args.status:
        scheduler.show_status()
        sys.exit(0)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
