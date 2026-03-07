#!/usr/bin/env python3
"""
生产就绪的脚本调度和终止控制器
修复了所有代码审查中发现的问题，确保生产环境安全
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
import gc
from logging.handlers import RotatingFileHandler
import traceback

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
    error_message: Optional[str] = None


class ProductionScriptScheduler:
    """生产就绪的脚本调度器"""

    def __init__(self, output_dir: str = "reports/script_scheduler"):
        # 验证输出目录
        self.output_dir = Path(output_dir)
        self._validate_output_directory()

        self.logger = self._setup_logger()

        # 脚本管理
        self.running_scripts: Dict[str, ScriptInfo] = {}
        self.script_history: List[ScriptInfo] = []
        self.scheduler_active = False

        # 线程安全
        self._lock = threading.RLock()

        # 终止控制
        self.termination_queue = queue.Queue()
        self.force_kill_timeout = 10  # 强制终止超时时间（秒）

        # 内存监控
        self.memory_threshold = 1024  # MB
        self.initial_memory = psutil.virtual_memory().used / (1024 * 1024)

        # 设置信号处理
        self._setup_signal_handlers()

        # 注册退出时的清理函数
        atexit.register(self.cleanup_on_exit)

        self.logger.info("生产就绪脚本调度器已初始化")

    def _validate_output_directory(self):
        """验证输出目录"""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            # 测试写入权限
            test_file = self.output_dir / ".test_write"
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            raise RuntimeError(f"无法创建或写入输出目录 {self.output_dir}: {e}")

    def _setup_logger(self) -> logging.Logger:
        """设置日志（带轮转）"""
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

            # 文件处理器（带轮转）
            log_file = self.output_dir / "production_script_scheduler.log"
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        return logger

    def _setup_signal_handlers(self):
        """设置信号处理器"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """信号处理器（修复版本）"""
        self.logger.info(f"收到信号 {signum}，正在停止所有脚本...")
        try:
            self.terminate_all_scripts()
            self.stop_monitoring()
            self.save_scheduler_state()
        except Exception as e:
            self.logger.error(f"信号处理过程中发生错误: {e}")
            traceback.print_exc()
        finally:
            sys.exit(0)

    def cleanup_on_exit(self):
        """退出时清理资源"""
        self.logger.info("正在清理资源...")
        try:
            self.terminate_all_scripts()
            self.save_scheduler_state()
            # 强制垃圾回收
            gc.collect()
        except Exception as e:
            self.logger.error(f"清理资源时发生错误: {e}")

    def register_script(self, name: str, script_path: str) -> ScriptInfo:
        """注册脚本"""
        # 验证脚本文件存在
        if not Path(script_path).exists():
            raise FileNotFoundError(f"脚本文件不存在: {script_path}")

        script_info = ScriptInfo(
            name=name,
            path=script_path,
            status='stopped'
        )

        self.logger.info(f"注册脚本: {name} -> {script_path}")
        return script_info

    def start_script(self, script_info: ScriptInfo, args: List[str] = None) -> bool:
        """启动脚本（修复版本）"""
        try:
            self.logger.info(f"启动脚本: {script_info.name}")

            # 验证脚本文件存在
            if not Path(script_info.path).exists():
                raise FileNotFoundError(f"脚本文件不存在: {script_info.path}")

            # 检查内存使用
            self._check_memory_usage()

            # 构建命令
            cmd = [sys.executable, script_info.path]
            if args:
                cmd.extend(args)

            # 启动进程（带进程组）
            if sys.platform == "win32":
                # Windows下用CREATE_NEW_PROCESS_GROUP
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:
                # Linux/Unix下用os.setsid
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    preexec_fn=os.setsid,  # 创建新进程组
                    start_new_session=True  # 创建新会话
                )

            # 更新脚本信息
            with self._lock:
                script_info.pid = process.pid
                script_info.start_time = datetime.now()
                script_info.status = 'running'
                script_info.exit_code = None
                script_info.error_message = None

                # 添加到运行中列表
                self.running_scripts[script_info.name] = script_info

            self.logger.info(f"脚本 {script_info.name} 已启动 (PID: {process.pid})")
            return True

        except Exception as e:
            self.logger.error(f"启动脚本 {script_info.name} 失败: {e}")
            script_info.status = 'error'
            script_info.error_message = str(e)
            return False

    def _check_memory_usage(self):
        """检查内存使用"""
        current_memory = psutil.virtual_memory().used / (1024 * 1024)
        memory_increase = current_memory - self.initial_memory

        if memory_increase > self.memory_threshold:
            self.logger.warning(f"内存使用增加过多: {memory_increase:.1f}MB，执行垃圾回收")
            gc.collect()

    def _wait_for_process_termination(self, pid: int, timeout: float) -> bool:
        """等待进程终止（修复版本）"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                process = psutil.Process(pid)
                status = process.status()

                if status == psutil.STATUS_ZOMBIE:
                    # 等待父进程回收僵尸进程
                    try:
                        process.wait(timeout=1.0)
                    except psutil.TimeoutExpired:
                        pass
                    return True
                elif status == psutil.STATUS_DEAD:
                    return True

            except psutil.NoSuchProcess:
                return True

            time.sleep(0.1)

        return False

    def stop_script(self, script_name: str, force: bool = False) -> bool:
        """停止脚本（修复版本）"""
        with self._lock:
            if script_name not in self.running_scripts:
                self.logger.warning(f"脚本 {script_name} 不在运行中")
                return False

            script_info = self.running_scripts[script_name]

        try:
            self.logger.info(f"停止脚本: {script_name} (PID: {script_info.pid})")

            if script_info.pid:
                # 尝试优雅终止
                try:
                    # 发送SIGTERM到整个进程组
                    os.killpg(os.getpgid(script_info.pid), signal.SIGTERM)

                    # 等待进程结束
                    if not self._wait_for_process_termination(script_info.pid, self.force_kill_timeout):
                        # 如果进程仍然存在，强制终止
                        if force or True:  # 总是强制终止以确保清理
                            try:
                                os.killpg(os.getpgid(script_info.pid), signal.SIGKILL)
                                self.logger.info(f"强制终止脚本: {script_name}")
                                # 等待强制终止完成
                                time.sleep(0.5)
                            except ProcessLookupError:
                                pass

                except ProcessLookupError:
                    self.logger.info(f"脚本 {script_name} 已经结束")
                except Exception as e:
                    self.logger.error(f"终止脚本 {script_name} 时发生错误: {e}")

            # 更新状态
            with self._lock:
                script_info.status = 'stopped'
                script_info.exit_code = 0

                # 从运行中列表移除
                if script_name in self.running_scripts:
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

        with self._lock:
            script_names = list(self.running_scripts.keys())

        for script_name in script_names:
            self.stop_script(script_name, force=True)

    def get_script_status(self, script_name: str) -> Optional[ScriptInfo]:
        """获取脚本状态（线程安全）"""
        with self._lock:
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
        """列出所有运行中的脚本（线程安全）"""
        with self._lock:
            return list(self.running_scripts.values())

    def monitor_scripts(self, interval: float = 1.0):
        """监控脚本运行状态（修复版本）"""
        self.logger.info("开始监控脚本运行状态...")

        while self.scheduler_active:
            try:
                # 检查所有运行中的脚本
                with self._lock:
                    script_names = list(self.running_scripts.keys())

                for script_name in script_names:
                    script_info = None
                    with self._lock:
                        if script_name in self.running_scripts:
                            script_info = self.running_scripts[script_name]

                    if script_info and script_info.pid:
                        try:
                            process = psutil.Process(script_info.pid)

                            # 检查进程状态
                            if process.status() == psutil.STATUS_ZOMBIE:
                                self.logger.info(f"脚本 {script_name} 已结束")
                                with self._lock:
                                    script_info.status = 'stopped'
                                    script_info.exit_code = process.returncode()

                                    # 从运行中列表移除
                                    if script_name in self.running_scripts:
                                        del self.running_scripts[script_name]

                                    # 添加到历史记录
                                    self.script_history.append(script_info)

                            else:
                                # 更新资源使用情况
                                script_info.memory_usage = process.memory_info().rss / (1024 * 1024)
                                script_info.cpu_usage = process.cpu_percent()

                        except psutil.NoSuchProcess:
                            self.logger.info(f"脚本 {script_name} 进程不存在")
                            with self._lock:
                                script_info.status = 'stopped'
                                if script_name in self.running_scripts:
                                    del self.running_scripts[script_name]
                                self.script_history.append(script_info)

                # 检查终止队列
                try:
                    while not self.termination_queue.empty():
                        script_name = self.termination_queue.get_nowait()
                        self.stop_script(script_name, force=True)
                except queue.Empty:
                    pass

                # 检查内存使用
                self._check_memory_usage()

                time.sleep(interval)

            except Exception as e:
                self.logger.error(f"监控过程中发生错误: {e}")
                traceback.print_exc()
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
        try:
            with self._lock:
                state = {
                    'timestamp': datetime.now().isoformat(),
                    'running_scripts': {
                        name: asdict(info) for name, info in self.running_scripts.items()
                    },
                    # 只保存最近100个
                    'script_history': [asdict(info) for info in self.script_history[-100:]]
                }

            state_file = self.output_dir / "production_scheduler_state.json"
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False, default=str)

            self.logger.info("调度器状态已保存")
        except Exception as e:
            self.logger.error(f"保存调度器状态失败: {e}")

    def generate_monitoring_report(self) -> str:
        """生成监控报告"""
        try:
            report_lines = [
                "# 生产就绪脚本调度器监控报告",
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

            with self._lock:
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
                "| 脚本名称 | 状态 | 退出码 | 运行时间 | 错误信息 |",
                "|----------|------|--------|----------|----------|"
            ])

            with self._lock:
                for script_info in self.script_history[-10:]:  # 只显示最近10个
                    if script_info.start_time:
                        runtime = script_info.start_time - script_info.start_time
                        runtime_str = str(runtime).split('.')[0]
                    else:
                        runtime_str = "N/A"

                    error_msg = script_info.error_message or "N/A"
                    if len(error_msg) > 50:
                        error_msg = error_msg[:47] + "..."

                    report_lines.append(
                        f"| {script_info.name} | {script_info.status} | {script_info.exit_code or 'N/A'} | {runtime_str} | {error_msg} |"
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
                "",
                "## 🔧 生产就绪特性",
                "",
                "- ✅ 线程安全保护",
                "- ✅ 完善的异常处理",
                "- ✅ 内存使用监控",
                "- ✅ 进程组管理",
                "- ✅ 日志轮转",
                "- ✅ 资源清理",
                "- ✅ 状态持久化",
                ""
            ])

            report_content = "\n".join(report_lines)
            report_file = self.output_dir / "production_monitoring_report.md"

            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)

            self.logger.info(f"监控报告已保存到: {report_file}")
            return str(report_file)

        except Exception as e:
            self.logger.error(f"生成监控报告失败: {e}")
            return ""


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
    print("🚀 启动生产就绪脚本调度器")
    print("="*60)

    # 创建调度器
    scheduler = ProductionScriptScheduler()

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

        print("\n🎮 交互式控制命令:")
        print("  - 启动脚本: start <script_name> (例如: start test_coverage)")
        print("  - 停止脚本: stop <script_name> (例如: stop test_coverage)")
        print("  - 强制终止: kill <script_name> (例如: kill test_coverage)")
        print("  - 查看状态: status <script_name> (例如: status test_coverage)")
        print("  - 列出运行中: list")
        print("  - 终止所有: killall")
        print("  - 退出: quit 或 exit")

        print("\n⏹️ 按 Ctrl+C 停止所有脚本并退出")

        # 示例：启动一个性能测试脚本
        print("\n🧪 启动性能测试脚本示例...")
        performance_script = test_scripts['performance_benchmark']
        scheduler.start_script(performance_script)

        # 交互式控制循环
        print("\n💬 输入命令控制脚本 (例如: start test_coverage):")
        while True:
            try:
                # 显示运行状态
                running_scripts = scheduler.list_running_scripts()
                if running_scripts:
                    running_names = [s.name for s in running_scripts]
                    print(f"\r🔄 运行中: {', '.join(running_names)} | 输入命令: ", end='')
                else:
                    print(f"\r💤 无运行中脚本 | 输入命令: ", end='')

                # 获取用户输入
                user_input = input().strip().lower()

                if not user_input:
                    continue

                # 解析命令
                parts = user_input.split()
                command = parts[0]

                if command in ['quit', 'exit', 'q']:
                    print("\n⏹️ 用户退出，正在停止所有脚本...")
                    break

                elif command == 'start' and len(parts) > 1:
                    script_name = parts[1]
                    if script_name in test_scripts:
                        script_info = test_scripts[script_name]
                        if scheduler.start_script(script_info):
                            print(f"✅ 已启动脚本: {script_name}")
                        else:
                            print(f"❌ 启动脚本失败: {script_name}")
                    else:
                        print(f"❌ 未知脚本: {script_name}")

                elif command == 'stop' and len(parts) > 1:
                    script_name = parts[1]
                    if scheduler.stop_script(script_name):
                        print(f"✅ 已停止脚本: {script_name}")
                    else:
                        print(f"❌ 停止脚本失败: {script_name}")

                elif command == 'kill' and len(parts) > 1:
                    script_name = parts[1]
                    if scheduler.stop_script(script_name, force=True):
                        print(f"✅ 已强制终止脚本: {script_name}")
                    else:
                        print(f"❌ 强制终止脚本失败: {script_name}")

                elif command == 'status' and len(parts) > 1:
                    script_name = parts[1]
                    status = scheduler.get_script_status(script_name)
                    if status:
                        print(f"📊 {script_name} 状态:")
                        print(f"  - 状态: {status.status}")
                        print(f"  - PID: {status.pid}")
                        print(f"  - 启动时间: {status.start_time}")
                        print(f"  - 内存使用: {status.memory_usage:.1f}MB")
                        print(f"  - CPU使用: {status.cpu_usage:.1f}%")
                        if status.error_message:
                            print(f"  - 错误: {status.error_message}")
                    else:
                        print(f"❌ 未找到脚本: {script_name}")

                elif command == 'list':
                    running_scripts = scheduler.list_running_scripts()
                    if running_scripts:
                        print("📋 运行中的脚本:")
                        for script in running_scripts:
                            print(f"  - {script.name} (PID: {script.pid})")
                    else:
                        print("💤 当前无运行中的脚本")

                elif command == 'killall':
                    scheduler.terminate_all_scripts()
                    print("✅ 已终止所有脚本")

                elif command == 'help':
                    print("\n🎮 可用命令:")
                    print("  - start <script_name> - 启动脚本")
                    print("  - stop <script_name>  - 停止脚本")
                    print("  - kill <script_name>  - 强制终止脚本")
                    print("  - status <script_name> - 查看脚本状态")
                    print("  - list                - 列出运行中脚本")
                    print("  - killall             - 终止所有脚本")
                    print("  - help                - 显示帮助")
                    print("  - quit/exit/q         - 退出")

                else:
                    print(f"❌ 未知命令: {command}")
                    print("💡 输入 'help' 查看可用命令")

            except KeyboardInterrupt:
                print("\n\n⏹️ 用户中断，正在停止所有脚本...")
                break
            except Exception as e:
                print(f"\n❌ 命令执行错误: {e}")

        # 停止监控
        scheduler.stop_monitoring()

        print("\n✅ 生产就绪脚本调度器已停止")

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        traceback.print_exc()
        scheduler.terminate_all_scripts()
        scheduler.stop_monitoring()


if __name__ == "__main__":
    main()
