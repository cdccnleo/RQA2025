#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
统一Web管理界面启动脚本
用于启动和管理统一Web管理界面
"""

from src.engine.logging.unified_logger import get_unified_logger
from src.engine.web.unified_dashboard import create_dashboard, DashboardConfig
import os
import sys
import argparse
import uvicorn
import socket
import psutil
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
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


def kill_process_on_port(port: int):
    """终止占用指定端口的进程"""
    try:
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


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="启动RQA2025统一Web管理界面")

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="监听主机地址 (默认: 127.0.0.1)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="监听端口 (默认: 8080)"
    )

    parser.add_argument(
        "--auto-port",
        action="store_true",
        help="自动查找可用端口"
    )

    parser.add_argument(
        "--force-kill",
        action="store_true",
        help="强制终止占用端口的进程"
    )

    parser.add_argument(
        "--reload",
        action="store_true",
        help="启用自动重载 (开发模式)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="工作进程数 (默认: 1)"
    )

    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="日志级别 (默认: info)"
    )

    parser.add_argument(
        "--config",
        help="配置文件路径"
    )

    parser.add_argument(
        "--env",
        default="development",
        choices=["development", "testing", "production"],
        help="运行环境 (默认: development)"
    )

    return parser.parse_args()


def load_config(config_path: str = None) -> DashboardConfig:
    """加载配置"""
    if config_path and os.path.exists(config_path):
        # 从配置文件加载
        import json
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        return DashboardConfig(**config_data)
    else:
        # 使用默认配置
        return DashboardConfig(
            title="RQA2025 统一管理平台",
            version="1.0.0",
            theme="modern",
            refresh_interval=30,
            max_connections=100,
            enable_websocket=True,
            enable_real_time=True
        )


def setup_environment(env: str):
    """设置运行环境"""
    # 设置环境变量
    os.environ["RQA_ENV"] = env
    os.environ["PYTHONPATH"] = str(project_root)

    # 设置日志级别
    if env == "development":
        os.environ["LOG_LEVEL"] = "DEBUG"
    elif env == "testing":
        os.environ["LOG_LEVEL"] = "INFO"
    else:
        os.environ["LOG_LEVEL"] = "WARNING"

    logger.info(f"运行环境设置为: {env}")


def check_dependencies():
    """检查依赖项"""
    required_packages = [
        "fastapi",
        "uvicorn",
        "jinja2",
        "websockets",
        "psutil"
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        logger.error(f"缺少依赖包: {missing_packages}")
        logger.info("请运行: pip install " + " ".join(missing_packages))
        return False

    logger.info("依赖检查通过")
    return True


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


def main():
    """主函数"""
    args = parse_arguments()

    # 设置环境
    setup_environment(args.env)

    # 检查依赖
    if not check_dependencies():
        sys.exit(1)

    # 创建静态文件目录
    create_static_directories()

    # 处理端口问题
    port = args.port
    if not check_port_availability(args.host, port):
        if args.force_kill:
            logger.info(f"端口 {port} 被占用，尝试终止占用进程...")
            if kill_process_on_port(port):
                logger.info(f"成功终止占用端口 {port} 的进程")
            else:
                logger.warning(f"无法终止占用端口 {port} 的进程")

        if args.auto_port or not check_port_availability(args.host, port):
            try:
                port = find_available_port(args.host, port)
                logger.info(f"自动切换到可用端口: {port}")
            except RuntimeError as e:
                logger.error(f"端口问题: {e}")
                sys.exit(1)

    # 加载配置
    config = load_config(args.config)

    # 创建仪表板应用
    try:
        dashboard = create_dashboard(config)
        logger.info("============================================================")
        logger.info("RQA2025 统一Web管理界面启动")
        logger.info("============================================================")
        logger.info(f"访问地址: http://{args.host}:{port}")
        logger.info(f"API文档: http://{args.host}:{port}/api/docs")
        logger.info(f"运行环境: {args.env}")
        logger.info(f"日志级别: {args.log_level}")
        logger.info(f"自动重载: {'启用' if args.reload else '禁用'}")
        logger.info("============================================================")

        # 启动服务器
        uvicorn.run(
            "src.engine.web.unified_dashboard:create_dashboard",
            host=args.host,
            port=port,
            reload=args.reload,
            workers=args.workers,
            log_level=args.log_level,
            access_log=True
        )

    except Exception as e:
        logger.error(f"启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
