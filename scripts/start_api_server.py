#!/usr/bin/env python3
"""
RQA2025 数据采集API服务启动脚本
启动FastAPI数据采集服务
"""

import os
import sys
import logging
import asyncio
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/api_server.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

async def main():
    """主函数"""
    try:
        logger.info("正在启动RQA2025数据采集API服务...")

        # 导入完整的API应用
        from src.gateway.web.api import app

        if app is None:
            logger.error("API应用创建失败，请检查依赖安装")
            sys.exit(1)

        # 获取启动配置
        host = os.getenv('HOST', '0.0.0.0')
        port = int(os.getenv('PORT', '8000'))

        logger.info(f"API服务将在 http://{host}:{port} 启动")
        logger.info("API文档地址: http://localhost:8000/docs")

        # 调试：检查应用路由
        logger.info(f"应用路由总数: {len(app.routes)}")
        for route in app.routes[:5]:  # 只显示前5个路由
            if hasattr(route, 'path'):
                methods = list(route.methods) if hasattr(route, 'methods') else ['no methods']
                logger.info(f"路由: {route.path} - {methods}")

        # 启动服务器
        import uvicorn
        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )

        server = uvicorn.Server(config)
        logger.info("开始启动Uvicorn服务器...")
        await server.serve()

    except KeyboardInterrupt:
        logger.info("收到停止信号，正在关闭服务...")
    except Exception as e:
        logger.error(f"服务启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())