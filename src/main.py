#!/usr/bin/env python3
"""
RQA2025量化交易系统主入口
"""
import logging
from data.china.adapters import ChinaStockDataAdapter
from trading import TradingEngine
from trading.strategies import QuantStrategy
from risk import RiskMonitor

def configure_logging():
    """配置系统日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('rqa2025.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """系统主入口"""
    configure_logging()
    logger = logging.getLogger(__name__)
    logger.info("启动RQA2025量化交易系统")

    try:
        # 初始化各模块
        data_adapter = ChinaStockDataAdapter()
        risk_monitor = RiskMonitor()
        strategy = QuantStrategy()
        trading_engine = TradingEngine()

        # 启动系统
        trading_engine.run(
            data_adapter=data_adapter,
            strategy=strategy,
            risk_monitor=risk_monitor
        )

    except Exception as e:
        logger.error(f"系统运行异常: {str(e)}", exc_info=True)
    finally:
        logger.info("系统正常退出")

if __name__ == "__main__":
    main()
