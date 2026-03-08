"""
System Processor Module
系统处理器模块

This module provides system - level processing capabilities for async operations
此模块为异步操作提供系统级别的处理能力

Author: RQA2025 Development Team
Date: 2025 - 01 - 28
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum
import threading
import time
import os
import sys
import platform
import psutil
import subprocess

logger = logging.getLogger(__name__)


class SystemCommand(Enum):

    """System command types"""
    SHELL = "shell"
    PYTHON = "python"
    SYSTEMCTL = "systemctl"
    SERVICE = "service"


class SystemProcessor:

    """
    System Processor Class
    系统处理器类

    Provides system - level operations and monitoring capabilities
    提供系统级别的操作和监控能力
    """

    def __init__(self, processor_name: str = "default_system_processor"):
        """
        Initialize the system processor
        初始化系统处理器

        Args:
            processor_name: Name of this processor
                          此处理器的名称
        """
        self.processor_name = processor_name
        self.is_running = False
        self.monitoring_thread = None
        self.system_metrics = {}
        self.command_history = []
        self.max_history_size = 100

        logger.info(f"System processor {processor_name} initialized")

    def start_system_monitoring(self) -> bool:
        """
        Start system monitoring
        开始系统监控

        Returns:
            bool: True if started successfully, False otherwise
                  启动成功返回True，否则返回False
        """
        if self.is_running:
            logger.warning(f"{self.processor_name} is already running")
            return False

        try:
            self.is_running = True
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info(f"System monitoring started for {self.processor_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to start system monitoring: {str(e)}")
            self.is_running = False
            return False

    def stop_system_monitoring(self) -> bool:
        """
        Stop system monitoring
        停止系统监控

        Returns:
            bool: True if stopped successfully, False otherwise
                  停止成功返回True，否则返回False
        """
        if not self.is_running:
            logger.warning(f"{self.processor_name} is not running")
            return False

        try:
            self.is_running = False
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5.0)
            logger.info(f"System monitoring stopped for {self.processor_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to stop system monitoring: {str(e)}")
            return False

    def execute_command(self,


                        command: str,
                        command_type: SystemCommand = SystemCommand.SHELL,
                        timeout: float = 30.0,
                        capture_output: bool = True) -> Dict[str, Any]:
        """
        Execute a system command
        执行系统命令

        Args:
            command: Command to execute
                    要执行的命令
            command_type: Type of command
                         命令类型
            timeout: Command timeout in seconds
                    命令超时时间（秒）
            capture_output: Whether to capture command output
                           是否捕获命令输出

        Returns:
            dict: Command execution result
                  命令执行结果
        """
        result = {
            'command': command,
            'command_type': command_type.value,
            'timestamp': datetime.now(),
            'success': False,
            'return_code': None,
            'stdout': None,
            'stderr': None,
            'execution_time': 0.0,
            'error': None
        }

        try:
            start_time = time.time()

            if command_type == SystemCommand.SHELL:
                # Execute shell command
                process = subprocess.run(
                    command,
                    shell=True,
                    capture_output=capture_output,
                    text=False,  # Return bytes to handle encoding manually
                    timeout=timeout,
                    encoding=None
                )

            elif command_type == SystemCommand.PYTHON:
                # Execute Python code
                process = subprocess.run(
                    [sys.executable, '-c', command],
                    capture_output=capture_output,
                    text=False,  # Return bytes to handle encoding manually
                    timeout=timeout,
                    encoding=None
                )

            else:
                # For other command types, treat as shell
                process = subprocess.run(
                    command,
                    shell=True,
                    capture_output=capture_output,
                    text=False,  # Return bytes to handle encoding manually
                    timeout=timeout,
                    encoding=None
                )

            result['success'] = process.returncode == 0
            result['return_code'] = process.returncode
            result['execution_time'] = time.time() - start_time

            if capture_output:
                # Handle encoding conversion for stdout and stderr
                def decode_output(output_bytes) -> Any:
                    """decode_output 函数的文档字符串"""

                    if output_bytes is None:
                        return None
                    if isinstance(output_bytes, str):
                        return output_bytes
                    # Try to decode with system default encoding, fallback to latin-1
                    try:
                        # Get system default encoding
                        system_encoding = sys.getdefaultencoding()
                        return output_bytes.decode(system_encoding)
                    except UnicodeDecodeError:
                        # Fallback to latin-1 which can handle any byte sequence
                        return output_bytes.decode('latin-1', errors='replace')

                result['stdout'] = decode_output(process.stdout)
                result['stderr'] = decode_output(process.stderr)

            # Log execution
            if result['success']:
                logger.info(f"Command executed successfully: {command[:50]}...")
            else:
                logger.warning(
                    f"Command failed: {command[:50]}... (code: {process.returncode})")

        except subprocess.TimeoutExpired:
            result['error'] = f"Command timed out after {timeout} seconds"
            logger.error(result['error'])
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Command execution error: {str(e)}")

        # Add to history
        self.command_history.append(result)
        if len(self.command_history) > self.max_history_size:
            self.command_history = self.command_history[-self.max_history_size:]

        return result

    def get_system_info(self) -> Dict[str, Any]:
        """
        Get comprehensive system information
        获取全面的系统信息

        Returns:
            dict: System information
                  系统信息
        """
        try:
            system_info = {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'hostname': platform.node(),
                'cpu_count': psutil.cpu_count(),
                'cpu_count_logical': psutil.cpu_count(logical=True),
                'memory_total_gb': psutil.virtual_memory().total / (1024 ** 3),
                'disk_total_gb': psutil.disk_usage('/').total / (1024 ** 3),
                'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat(),
                'current_time': datetime.now().isoformat()
            }

            return system_info

        except Exception as e:
            logger.error(f"Failed to get system info: {str(e)}")
            return {'error': str(e)}

    def get_process_info(self, process_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get information about running processes
        获取正在运行的进程信息

        Args:
            process_name: Filter by process name (None for all)
                         按进程名称过滤（None表示全部）

        Returns:
            list: List of process information
                  进程信息列表
        """
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
                try:
                    info = proc.info
                    if process_name is None or process_name.lower() in info['name'].lower():
                        processes.append({
                            'pid': info['pid'],
                            'name': info['name'],
                            'cpu_percent': info['cpu_percent'],
                            'memory_percent': info['memory_percent'],
                            'status': info['status']
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            return processes

        except Exception as e:
            logger.error(f"Failed to get process info: {str(e)}")
            return []

    def manage_service(self,


                       service_name: str,
                       action: str,
                       service_manager: str = "auto") -> Dict[str, Any]:
        """
        Manage system services
        管理系统服务

        Args:
            service_name: Name of the service
                         服务名称
            action: Action to perform ('start', 'stop', 'restart', 'status')
                   要执行的操作 ('start', 'stop', 'restart', 'status')
            service_manager: Service manager to use ('systemctl', 'service', 'auto')
                           要使用的服务管理器 ('systemctl', 'service', 'auto')

        Returns:
            dict: Service management result
                  服务管理结果
        """
        try:
            # Determine service manager
            if service_manager == "auto":
                if os.path.exists("/usr / bin / systemctl"):
                    service_manager = "systemctl"
                elif os.path.exists("/usr / sbin / service"):
                    service_manager = "service"
                else:
                    return {'success': False, 'error': 'No suitable service manager found'}

            # Construct command
            if service_manager == "systemctl":
                command = f"systemctl {action} {service_name}"
            elif service_manager == "service":
                command = f"service {service_name} {action}"
            else:
                return {'success': False, 'error': f'Unsupported service manager: {service_manager}'}

            # Execute command
            result = self.execute_command(command, SystemCommand.SHELL)

            return {
                'service_name': service_name,
                'action': action,
                'service_manager': service_manager,
                'success': result['success'],
                'return_code': result['return_code'],
                'output': result['stdout'],
                'error': result['stderr'] or result['error']
            }

        except Exception as e:
            logger.error(f"Failed to manage service {service_name}: {str(e)}")
            return {'success': False, 'error': str(e)}

    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Get current system performance metrics
        获取当前系统性能指标

        Returns:
            dict: System metrics
                  系统指标
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            metrics = {
                'timestamp': datetime.now(),
                'cpu': {
                    'usage_percent': cpu_percent,
                    'cores': psutil.cpu_count()
                },
                'memory': {
                    'total_gb': memory.total / (1024 ** 3),
                    'used_gb': memory.used / (1024 ** 3),
                    'free_gb': memory.free / (1024 ** 3),
                    'usage_percent': memory.percent
                },
                'disk': {
                    'total_gb': disk.total / (1024 ** 3),
                    'used_gb': disk.used / (1024 ** 3),
                    'free_gb': disk.free / (1024 ** 3),
                    'usage_percent': disk.percent
                },
                'network': self._get_network_metrics()
            }

            # Store metrics
            self.system_metrics = metrics

            return metrics

        except Exception as e:
            logger.error(f"Failed to get system metrics: {str(e)}")
            return {'error': str(e)}

    def _get_network_metrics(self) -> Dict[str, Any]:
        """Get network metrics"""
        try:
            net_io = psutil.net_io_counters()
            return {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv,
                'errin': net_io.errin,
                'errout': net_io.errout
            }
        except Exception as e:
            logger.error(f"Failed to get network metrics: {str(e)}")
            return {}

    def _monitoring_loop(self) -> None:
        """
        Main monitoring loop
        主要的监控循环
        """
        logger.info(f"System monitoring loop started for {self.processor_name}")

        while self.is_running:
            try:
                # Collect system metrics
                metrics = self.get_system_metrics()

                # Store metrics with timestamp
                if len(self.system_metrics) > 100:
                    # Keep only last 100 entries
                    keys = list(self.system_metrics.keys())
                    for key in keys[:-100]:
                        del self.system_metrics[key]

                # Sleep before next collection
                time.sleep(30)  # Collect every 30 seconds

            except Exception as e:
                logger.error(f"System monitoring loop error: {str(e)}")
                time.sleep(30)

        logger.info(f"System monitoring loop stopped for {self.processor_name}")

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get overall system status
        获取整体系统状态

        Returns:
            dict: System status information
                  系统状态信息
        """
        return {
            'processor_name': self.processor_name,
            'is_running': self.is_running,
            'system_info': self.get_system_info(),
            'current_metrics': self.system_metrics,
            'recent_commands': self.command_history[-10:],
            'total_commands_executed': len(self.command_history)
        }

    def cleanup_resources(self) -> Dict[str, Any]:
        """
        Cleanup system resources
        清理系统资源

        Returns:
            dict: Cleanup results
                  清理结果
        """
        try:
            # Force garbage collection
            import gc
            collected = gc.collect()

            # Clear old command history
            old_count = len(self.command_history)
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.command_history = [
                cmd for cmd in self.command_history
                if cmd['timestamp'] > cutoff_time
            ]
            cleaned_commands = old_count - len(self.command_history)

            result = {
                'garbage_collected': collected,
                'commands_cleaned': cleaned_commands,
                'timestamp': datetime.now()
            }

            logger.info(
                f"System cleanup completed: {collected} objects collected, {cleaned_commands} old commands removed")
            return result

        except Exception as e:
            logger.error(f"Failed to cleanup resources: {str(e)}")
            return {'error': str(e)}


# Global system processor instance
# 全局系统处理器实例
system_processor = SystemProcessor()

__all__ = [
    'SystemCommand',
    'SystemProcessor',
    'system_processor'
]
