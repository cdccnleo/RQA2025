#!/usr/bin/env python3
"""
脚本调度和终止控制器
管理多个测试脚本的运行，支持立即终止功能
"""

import os
import sys
import time
import json
import signal
import threading
import subprocess
import psutil
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
import queue
import atexit

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class ScriptInfo:
    """脚本信息数据类"""
    name: str
    path: str
    pid: Optional[int] = None
    start_time: Optional[datetime] = None
    status: str = 'stopped'  # 'running', 'stopped', 'error'
    exit_code: Optional[int] = None
    memory_usage: float = 0.0
    cpu_usage: float = 0.0


class ScriptScheduler:
    """脚本调度器"""

    def __init__(self, output_dir: str = "reports/script_scheduler"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()

        # 脚本管理
        self.running_scripts: Dict[str, ScriptInfo] = {}
        self.script_history: List[ScriptInfo] = []
        self.scheduler_active = False

        # 终止控制
        self.termination_queue = queue.Queue()
        self.force_kill_timeout = 10  # 强制终止超时时间（秒）

        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # 注册退出时的清理函数
        atexit.register(self.cleanup_on_exit)

    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            # 控制台处理器
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

            # 文件处理器
            log_file = self.output_dir / "script_scheduler.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        return logger

    def _signal_handler(self, signum, frame):
        """信号处理器"""
        self.logger.info(f"收到信号 {signum}，正在停止所有脚本...")
        self.terminate_all_scripts()
        sys.exit(0)

    def cleanup_on_exit(self):
        """退出时清理资源"""
        self.logger.info("正在清理资源...")
        self.terminate_all_scripts()
        self.save_scheduler_state()

    def register_script(self, name: str, script_path: str) -> ScriptInfo:
        """注册脚本"""
        script_info = ScriptInfo(
            name=name,
            path=script_path,
            status='stopped'
        )

        self.logger.info(f"注册脚本: {name} -> {script_path}")
        return script_info

    def start_script(self, script_info: ScriptInfo, args: List[str] = None) -> bool:
        """启动脚本"""
        try:
            self.logger.info(f"启动脚本: {script_info.name}")

            # 构建命令
            cmd = [sys.executable, script_info.path]
            if args:
                cmd.extend(args)

            # 启动进程
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # 更新脚本信息
            script_info.pid = process.pid
            script_info.start_time = datetime.now()
            script_info.status = 'running'
            script_info.exit_code = None

            # 添加到运行中列表
            self.running_scripts[script_info.name] = script_info

            self.logger.info(f"脚本 {script_info.name} 已启动 (PID: {process.pid})")
            return True

        except Exception as e:
            self.logger.error(f"启动脚本 {script_info.name} 失败: {e}")
            script_info.status = 'error'
            return False

    def stop_script(self, script_name: str, force: bool = False) -> bool:
        """停止脚本"""
        if script_name not in self.running_scripts:
            self.logger.warning(f"脚本 {script_name} 不在运行中")
            return False

        script_info = self.running_scripts[script_name]

        try:
            self.logger.info(f"停止脚本: {script_name} (PID: {script_info.pid})")

            if script_info.pid:
                # 尝试优雅终止
                try:
                    os.kill(script_info.pid, signal.SIGTERM)

                    # 等待进程结束
                    start_time = time.time()
                    while time.time() - start_time < self.force_kill_timeout:
                        try:
                            # 检查进程是否还存在
                            process = psutil.Process(script_info.pid)
                            if process.status() == psutil.STATUS_ZOMBIE:
                                break
                            time.sleep(0.1)
                        except psutil.NoSuchProcess:
                            break

                    # 如果进程仍然存在，强制终止
                    if force or time.time() - start_time >= self.force_kill_timeout:
                        try:
                            os.kill(script_info.pid, signal.SIGKILL)
                            self.logger.info(f"强制终止脚本: {script_name}")
                        except ProcessLookupError:
                            pass

                except ProcessLookupError:
                    self.logger.info(f"脚本 {script_name} 已经结束")
                except Exception as e:
                    self.logger.error(f"终止脚本 {script_name} 时发生错误: {e}")

            # 更新状态
            script_info.status = 'stopped'
            script_info.exit_code = 0

            # 从运行中列表移除
            del self.running_scripts[script_name]

            # 添加到历史记录
            self.script_history.append(script_info)

            self.logger.info(f"脚本 {script_name} 已停止")
            return True

        except Exception as e:
            self.logger.error(f"停止脚本 {script_name} 失败: {e}")
            return False

    def terminate_all_scripts(self):
        """终止所有运行中的脚本"""
        self.logger.info("终止所有运行中的脚本...")

        script_names = list(self.running_scripts.keys())
        for script_name in script_names:
            self.stop_script(script_name, force=True)

    def get_script_status(self, script_name: str) -> Optional[ScriptInfo]:
        """获取脚本状态"""
        if script_name in self.running_scripts:
            script_info = self.running_scripts[script_name]

            # 更新资源使用情况
            if script_info.pid:
                try:
                    process = psutil.Process(script_info.pid)
                    script_info.memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
                    script_info.cpu_usage = process.cpu_percent()
                except psutil.NoSuchProcess:
                    script_info.status = 'stopped'

            return script_info

        return None

    def list_running_scripts(self) -> List[ScriptInfo]:
        """列出所有运行中的脚本"""
        return list(self.running_scripts.values())

    def monitor_scripts(self, interval: float = 1.0):
        """监控脚本运行状态"""
        self.logger.info("开始监控脚本运行状态...")

        while self.scheduler_active:
            try:
                # 检查所有运行中的脚本
                script_names = list(self.running_scripts.keys())
                for script_name in script_names:
                    script_info = self.running_scripts[script_name]

                    if script_info.pid:
                        try:
                            process = psutil.Process(script_info.pid)

                            # 检查进程状态
                            if process.status() == psutil.STATUS_ZOMBIE:
                                self.logger.info(f"脚本 {script_name} 已结束")
                                script_info.status = 'stopped'
                                script_info.exit_code = process.returncode()

                                # 从运行中列表移除
                                del self.running_scripts[script_name]

                                # 添加到历史记录
                                self.script_history.append(script_info)

                            else:
                                # 更新资源使用情况
                                script_info.memory_usage = process.memory_info().rss / (1024 * 1024)
                                script_info.cpu_usage = process.cpu_percent()

                        except psutil.NoSuchProcess:
                            self.logger.info(f"脚本 {script_name} 进程不存在")
                            script_info.status = 'stopped'
                            del self.running_scripts[script_name]
                            self.script_history.append(script_info)

                # 检查终止队列
                try:
                    while not self.termination_queue.empty():
                        script_name = self.termination_queue.get_nowait()
                        self.stop_script(script_name, force=True)
                except queue.Empty:
                    pass

                time.sleep(interval)

            except Exception as e:
                self.logger.error(f"监控过程中发生错误: {e}")
                time.sleep(interval)

    def start_monitoring(self):
        """启动监控"""
        self.scheduler_active = True

        # 启动监控线程
        monitor_thread = threading.Thread(target=self.monitor_scripts)
        monitor_thread.daemon = True
        monitor_thread.start()

        self.logger.info("脚本监控已启动")
        return monitor_thread

    def stop_monitoring(self):
        """停止监控"""
        self.logger.info("停止脚本监控...")
        self.scheduler_active = False

        # 保存状态
        self.save_scheduler_state()

        # 生成监控报告
        self.generate_monitoring_report()

    def save_scheduler_state(self):
        """保存调度器状态"""
        state = {
            'timestamp': datetime.now().isoformat(),
            'running_scripts': {
                name: asdict(info) for name, info in self.running_scripts.items()
            },
            'script_history': [asdict(info) for info in self.script_history]
        }

        state_file = self.output_dir / "scheduler_state.json"
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False, default=str)

    def generate_monitoring_report(self) -> str:
        """生成监控报告"""
        report_lines = [
            "# 脚本调度器监控报告",
            "",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 📊 运行状态",
            "",
            f"**运行中脚本**: {len(self.running_scripts)}",
            f"**历史脚本**: {len(self.script_history)}",
            "",
            "## 🚀 运行中的脚本",
            "",
            "| 脚本名称 | PID | 状态 | 内存使用(MB) | CPU使用率(%) | 运行时间 |",
            "|----------|-----|------|---------------|--------------|----------|"
        ]

        for script_info in self.running_scripts.values():
            if script_info.start_time:
                runtime = datetime.now() - script_info.start_time
                runtime_str = str(runtime).split('.')[0]
            else:
                runtime_str = "N/A"

            report_lines.append(
                f"| {script_info.name} | {script_info.pid or 'N/A'} | {script_info.status} | "
                f"{script_info.memory_usage:.1f} | {script_info.cpu_usage:.1f} | {runtime_str} |"
            )

        report_lines.extend([
            "",
            "## 📋 历史脚本",
            "",
            "| 脚本名称 | 状态 | 退出码 | 运行时间 |",
            "|----------|------|--------|----------|"
        ])

        for script_info in self.script_history[-10:]:  # 只显示最近10个
            if script_info.start_time:
                runtime = script_info.start_time - script_info.start_time
                runtime_str = str(runtime).split('.')[0]
            else:
                runtime_str = "N/A"

            report_lines.append(
                f"| {script_info.name} | {script_info.status} | {script_info.exit_code or 'N/A'} | {runtime_str} |"
            )

        report_lines.extend([
            "",
            "## 🎯 控制命令",
            "",
            "- **启动脚本**: `scheduler.start_script(script_info)`",
            "- **停止脚本**: `scheduler.stop_script(script_name)`",
            "- **强制终止**: `scheduler.stop_script(script_name, force=True)`",
            "- **终止所有**: `scheduler.terminate_all_scripts()`",
            "- **查看状态**: `scheduler.get_script_status(script_name)`",
            "",
            "## ⚡ 立即终止功能",
            "",
            "系统支持以下立即终止方式：",
            "- **Ctrl+C**: 优雅终止所有脚本",
            "- **SIGTERM**: 发送终止信号",
            "- **强制终止**: 超时后自动强制终止",
            "- **队列终止**: 通过终止队列立即停止指定脚本",
            ""
        ])

        report_content = "\n".join(report_lines)
        report_file = self.output_dir / "script_monitoring_report.md"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        self.logger.info(f"监控报告已保存到: {report_file}")
        return str(report_file)


def create_test_scripts() -> Dict[str, ScriptInfo]:
    """创建测试脚本列表"""
    scripts = {}

    # 性能测试脚本
    scripts['performance_benchmark'] = ScriptInfo(
        name='performance_benchmark',
        path='scripts/testing/run_performance_benchmark.py'
    )

    scripts['simple_performance'] = ScriptInfo(
        name='simple_performance',
        path='scripts/testing/simple_performance_benchmark_system.py'
    )

    scripts['performance_monitor'] = ScriptInfo(
        name='performance_monitor',
        path='scripts/testing/performance_monitor.py'
    )

    # 回测测试脚本
    scripts['backtest_integration'] = ScriptInfo(
        name='backtest_integration',
        path='scripts/testing/run_backtest_integration_tests.py'
    )

    # 测试覆盖率脚本
    scripts['test_coverage'] = ScriptInfo(
        name='test_coverage',
        path='scripts/testing/enhance_test_coverage_plan.py'
    )

    return scripts


def main():
    """主函数"""
    print("🚀 启动脚本调度器")
    print("="*60)

    # 创建调度器
    scheduler = ScriptScheduler()

    try:
        # 创建测试脚本
        test_scripts = create_test_scripts()

        # 注册脚本
        for script_info in test_scripts.values():
            scheduler.register_script(script_info.name, script_info.path)

        # 启动监控
        monitor_thread = scheduler.start_monitoring()

        print("\n📋 可用脚本:")
        for name, script_info in test_scripts.items():
            print(f"  - {name}: {script_info.path}")

        print("\n🎮 控制命令:")
        print("  - 启动脚本: scheduler.start_script(script_info)")
        print("  - 停止脚本: scheduler.stop_script(script_name)")
        print("  - 强制终止: scheduler.stop_script(script_name, force=True)")
        print("  - 终止所有: scheduler.terminate_all_scripts()")
        print("  - 查看状态: scheduler.get_script_status(script_name)")

        print("\n⏹️ 按 Ctrl+C 停止所有脚本并退出")

        # 示例：启动一个性能测试脚本
        print("\n🧪 启动性能测试脚本示例...")
        performance_script = test_scripts['performance_benchmark']
        scheduler.start_script(performance_script)

        # 等待用户中断
        try:
            while True:
                time.sleep(1)

                # 显示运行状态
                running_scripts = scheduler.list_running_scripts()
                if running_scripts:
                    print(f"\r🔄 运行中脚本: {len(running_scripts)}", end='')

        except KeyboardInterrupt:
            print("\n\n⏹️ 用户中断，正在停止所有脚本...")
            scheduler.terminate_all_scripts()

        # 停止监控
        scheduler.stop_monitoring()

        print("\n✅ 脚本调度器已停止")

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        scheduler.terminate_all_scripts()
        scheduler.stop_monitoring()


if __name__ == "__main__":
    main()
