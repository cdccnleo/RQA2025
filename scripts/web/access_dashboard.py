#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
统一Web管理界面快速访问脚本
自动检测服务状态并打开浏览器访问
"""

from src.engine.logging.unified_logger import get_unified_logger
import sys
import time
import webbrowser
import socket
import requests
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


logger = get_unified_logger(__name__)


def check_service_status(host: str, port: int) -> bool:
    """检查服务状态"""
    try:
        # 检查端口是否开放
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2)
            result = s.connect_ex((host, port))
            if result != 0:
                return False

        # 检查HTTP服务是否响应
        url = f"http://{host}:{port}/api/health"
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def find_running_port(host: str, start_port: int = 8080, max_attempts: int = 10) -> int:
    """查找运行中的服务端口"""
    for i in range(max_attempts):
        port = start_port + i
        if check_service_status(host, port):
            return port
    return None


def open_dashboard(host: str = "127.0.0.1"):
    """打开仪表板"""
    # 查找运行中的服务
    port = find_running_port(host)

    if port is None:
        logger.error("未找到运行中的统一Web管理界面服务")
        logger.info("请先启动服务:")
        logger.info("  python scripts/web/start_dashboard_fixed.py")
        return False

    # 构建访问URL
    dashboard_url = f"http://{host}:{port}"
    api_docs_url = f"http://{host}:{port}/api/docs"

    logger.info("============================================================")
    logger.info("RQA2025 统一Web管理界面")
    logger.info("============================================================")
    logger.info(f"主界面: {dashboard_url}")
    logger.info(f"API文档: {api_docs_url}")
    logger.info("============================================================")

    # 打开浏览器
    try:
        # 打开主界面
        webbrowser.open(dashboard_url)
        time.sleep(1)

        # 打开API文档
        webbrowser.open(api_docs_url)

        logger.info("已自动打开浏览器访问统一Web管理界面")
        return True

    except Exception as e:
        logger.error(f"打开浏览器失败: {e}")
        logger.info(f"请手动访问: {dashboard_url}")
        return False


def show_service_info():
    """显示服务信息"""
    logger.info("统一Web管理界面服务信息:")
    logger.info("- 服务地址: http://127.0.0.1:8081")
    logger.info("- API文档: http://127.0.0.1:8081/api/docs")
    logger.info("- WebSocket: ws://127.0.0.1:8081/ws")
    logger.info("- 已集成模块: config, fpga_monitoring, resource_monitoring, features_monitoring")


def main():
    """主函数"""
    try:
        # 显示服务信息
        show_service_info()
        print()

        # 打开仪表板
        success = open_dashboard()

        if success:
            logger.info("访问成功！")
        else:
            logger.error("访问失败，请检查服务状态")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("用户中断操作")
    except Exception as e:
        logger.error(f"访问失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
