#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
策略分析器仪表板启动脚本
"""

from src.infrastructure.dashboard.strategy_analyzer_dashboard import StrategyAnalyzerDashboard
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    """启动策略分析器仪表板"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)
    logger.info("启动策略分析器仪表板...")

    try:
        dashboard = StrategyAnalyzerDashboard()
        dashboard.run(host="0.0.0.0", port=8051)
    except Exception as e:
        logger.error(f"启动仪表板失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
