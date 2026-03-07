#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RQA2025 引擎层统一Web管理界面启动脚本
启动引擎层的统一Web管理界面服务
"""

from src.engine.logging.unified_logger import get_unified_logger
from src.engine.web.unified_dashboard import create_dashboard, DashboardConfig
import os
import sys
import uvicorn
import socket
import psutil
from pathlib import Path

# 设置环境变量
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


logger = get_unified_logger(__name__)


def check_port_availability(host: str, port: int) -> bool:
    """检查端口是否可用"""
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
    """强制终止占用端口的进程"""
    try:
        # 查找占用端口的进程
        for proc in psutil.process_iter(['pid', 'name', 'connections']):
            try:
                connections = proc.info['connections']
                for conn in connections:
                    if conn.laddr.port == port:
                        logger.info(f"终止进程 {proc.info['name']} (PID: {proc.info['pid']})")
                        psutil.Process(proc.info['pid']).terminate()
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return False
    except Exception as e:
        logger.error(f"终止进程失败: {e}")
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


def check_dependencies():
    """检查依赖"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'websockets',
        'psutil',
        'jinja2'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        logger.warning(f"缺少依赖包: {', '.join(missing_packages)}")
        logger.info("请运行: pip install " + " ".join(missing_packages))
        return False

    return True


def setup_environment():
    """设置环境"""
    # 设置Python路径
    os.environ["PYTHONPATH"] = str(project_root)

    # 设置日志级别
    os.environ["LOG_LEVEL"] = "info"

    # 设置环境变量
    os.environ["RQA_ENV"] = "development"
    os.environ["RQA_DASHBOARD_PORT"] = "8080"


def main():
    """主函数"""
    try:
        # 设置环境
        setup_environment()

        # 检查依赖
        if not check_dependencies():
            logger.error("依赖检查失败，请安装缺失的包")
            sys.exit(1)

        # 创建静态文件目录
        create_static_directories()

        # 检查端口可用性
        host = "127.0.0.1"
        port = 8080

        if not check_port_availability(host, port):
            try:
                port = find_available_port(host, port)
                logger.info(f"端口8080被占用，自动切换到端口: {port}")
            except RuntimeError as e:
                logger.error(f"端口问题: {e}")
                sys.exit(1)

        # 创建配置
        config = DashboardConfig(
            title="RQA2025 统一管理平台",
            version="1.0.0",
            theme="modern",
            refresh_interval=30,
            max_connections=100,
            enable_websocket=True,
            enable_real_time=True
        )

        # 创建仪表板
        dashboard = create_dashboard(config)

        logger.info("============================================================")
        logger.info("RQA2025 引擎层统一Web管理界面启动")
        logger.info("============================================================")
        logger.info(f"访问地址: http://{host}:{port}")
        logger.info(f"API文档: http://{host}:{port}/api/docs")
        logger.info(f"运行环境: development")
        logger.info(f"日志级别: info")
        logger.info("============================================================")

        # 启动服务器
        uvicorn.run(
            dashboard.app,
            host=host,
            port=port,
            reload=False,
            log_level="info",
            access_log=True
        )

    except Exception as e:
        logger.error(f"启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
