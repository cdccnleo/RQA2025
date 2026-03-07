#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
统一Web管理界面快速启动脚本
使用配置文件启动，支持自动端口检测和进程管理
"""

from src.engine.logging.unified_logger import get_unified_logger
import sys
import json
import subprocess
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


logger = get_unified_logger(__name__)


def load_config(config_path: str = "config/web_dashboard_config.json") -> dict:
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"配置文件不存在: {config_path}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"配置文件格式错误: {e}")
        return {}


def check_port_availability(host: str, port: int) -> bool:
    """检查端口是否可用"""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex((host, port))
            return result != 0
    except Exception:
        return False


def find_available_port(host: str, start_port: int, max_attempts: int = 10) -> int:
    """查找可用端口"""
    for i in range(max_attempts):
        port = start_port + i
        if check_port_availability(host, port):
            return port
    raise RuntimeError(f"无法找到可用端口，尝试范围: {start_port}-{start_port + max_attempts - 1}")


def kill_process_on_port(port: int) -> bool:
    """终止占用指定端口的进程"""
    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'name', 'connections']):
            try:
                connections = proc.info['connections']
                for conn in connections:
                    if conn.laddr.port == port:
                        logger.info(
                            f"终止占用端口 {port} 的进程: {proc.info['name']} (PID: {proc.info['pid']})")
                        proc.terminate()
                        proc.wait(timeout=5)
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                continue
    except Exception as e:
        logger.warning(f"终止进程时出错: {e}")
    return False


def create_static_directories():
    """创建静态文件目录"""
    static_dirs = [
        "src/engine/web/static",
        "src/engine/web/templates",
        "logs/web"
    ]

    for dir_path in static_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    logger.info("静态文件目录创建完成")


def start_dashboard(config: dict):
    """启动统一Web管理界面"""
    dashboard_config = config.get("dashboard", {})

    # 获取配置参数
    host = dashboard_config.get("host", "127.0.0.1")
    port = dashboard_config.get("port", 8080)
    auto_port = dashboard_config.get("auto_port", True)
    force_kill = dashboard_config.get("force_kill", False)
    reload = dashboard_config.get("reload", False)
    workers = dashboard_config.get("workers", 1)
    log_level = dashboard_config.get("log_level", "info")
    env = dashboard_config.get("env", "development")

    # 处理端口问题
    if not check_port_availability(host, port):
        if force_kill:
            logger.info(f"端口 {port} 被占用，尝试终止占用进程...")
            if kill_process_on_port(port):
                logger.info(f"成功终止占用端口 {port} 的进程")
            else:
                logger.warning(f"无法终止占用端口 {port} 的进程")

        if auto_port or not check_port_availability(host, port):
            try:
                port = find_available_port(host, port)
                logger.info(f"自动切换到可用端口: {port}")
            except RuntimeError as e:
                logger.error(f"端口问题: {e}")
                return False

    # 创建静态文件目录
    create_static_directories()

    # 构建启动命令
    cmd = [
        sys.executable,
        "scripts/web/start_unified_dashboard.py",
        "--host", host,
        "--port", str(port),
        "--log-level", log_level,
        "--env", env,
        "--config", "config/web_dashboard_config.json"
    ]

    if reload:
        cmd.append("--reload")

    if workers > 1:
        cmd.extend(["--workers", str(workers)])

    # 启动进程
    try:
        logger.info("============================================================")
        logger.info("RQA2025 统一Web管理界面启动")
        logger.info("============================================================")
        logger.info(f"访问地址: http://{host}:{port}")
        logger.info(f"API文档: http://{host}:{port}/api/docs")
        logger.info(f"运行环境: {env}")
        logger.info(f"日志级别: {log_level}")
        logger.info(f"自动重载: {'启用' if reload else '禁用'}")
        logger.info("============================================================")

        # 启动子进程
        process = subprocess.Popen(
            cmd,
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        # 实时输出日志
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())

        return_code = process.poll()
        if return_code != 0:
            logger.error(f"进程异常退出，返回码: {return_code}")
            return False

        return True

    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭服务器...")
        if 'process' in locals():
            process.terminate()
            process.wait(timeout=5)
        return True
    except Exception as e:
        logger.error(f"启动失败: {e}")
        return False


def main():
    """主函数"""
    # 加载配置
    config = load_config()
    if not config:
        logger.error("无法加载配置文件，使用默认配置")
        config = {
            "dashboard": {
                "host": "127.0.0.1",
                "port": 8080,
                "auto_port": True,
                "force_kill": False,
                "reload": False,
                "workers": 1,
                "log_level": "info",
                "env": "development"
            }
        }

    # 启动仪表板
    success = start_dashboard(config)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
