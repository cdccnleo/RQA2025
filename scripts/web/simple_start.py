#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简化启动脚本 - 统一Web管理界面
直接启动，无需复杂配置
"""

from src.engine.logging.unified_logger import get_unified_logger
from src.engine.web.unified_dashboard import create_dashboard, DashboardConfig
import sys
import uvicorn
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


logger = get_unified_logger(__name__)


def main():
    """主函数"""
    try:
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
        logger.info("RQA2025 统一Web管理界面启动")
        logger.info("============================================================")
        logger.info("访问地址: http://127.0.0.1:8080")
        logger.info("API文档: http://127.0.0.1:8080/api/docs")
        logger.info("运行环境: development")
        logger.info("日志级别: info")
        logger.info("============================================================")

        # 启动服务器
        uvicorn.run(
            dashboard.app,
            host="127.0.0.1",
            port=8080,
            reload=False,
            log_level="info"
        )

    except Exception as e:
        logger.error(f"启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
