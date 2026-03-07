#!/usr/bin/env python3
"""
系统监控脚本
定期检查AI质量保障系统的运行状态
"""

import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def check_system_health():
    """检查系统健康状态"""
    try:
        from ai_quality.production_integration import ProductionIntegrationManager
        manager = ProductionIntegrationManager()
        status = manager.get_integration_status()
        return status
    except Exception as e:
        return {"status": "error", "error": str(e)}

def main():
    """主监控循环"""
    logging.basicConfig(
        filename='logs/system_monitor.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        encoding='utf-8'
    )

    logger = logging.getLogger(__name__)

    while True:
        try:
            health_status = check_system_health()
            logger.info(f"系统健康状态: {health_status}")

            # 检查是否有问题
            if health_status.get('status') != 'healthy':
                logger.warning(f"检测到系统问题: {health_status}")

        except Exception as e:
            logger.error(f"监控检查失败: {e}")

        time.sleep(300)  # 5分钟检查一次

if __name__ == "__main__":
    main()
