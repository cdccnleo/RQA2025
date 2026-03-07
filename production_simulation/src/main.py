#!/usr/bin/env python3
"""
RQA2025量化交易系统主入口

应用层统一入口，提供完整的量化交易系统功能。
采用分层架构设计，确保系统模块化和可扩展性。

分层架构：
- 应用入口：main.py, app.py
- 应用服务：ApplicationService, TradingApplication
- 应用配置：AppConfig, ApplicationManager
- 应用监控：AppMonitor, ApplicationMetrics
- 应用部署：AppDeployer, ApplicationServing
- 应用集成：AppIntegration, ApplicationAPI

典型用法：
    python src / main.py --config config / app_config.json
    python src / main.py --mode live --strategy momentum
    python src / main.py --backtest --start - date 2023 - 01 - 01 --end - date 2023 - 12 - 31
"""

from services import TradingService, DataValidationService, ModelServing
from utils.logging_utils import setup_logging
from infrastructure.config import UnifiedConfigManager
import logging
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入应用层组件

# 应用层组件（预留）
# from src.application import ApplicationService, TradingApplication
# from src.application import AppConfig, ApplicationManager
# from src.application import AppMonitor, ApplicationMetrics
# from src.application import AppDeployer, ApplicationServing
# from src.application import AppIntegration, ApplicationAPI


class ApplicationManager:

    """应用管理器，负责应用的生命周期管理"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """__init__ 函数的文档字符串"""

        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.services = {}
        self.is_running = False

    def initialize(self) -> Any:
        """初始化应用"""
        self.logger.info("初始化RQA2025应用...")

        # 初始化服务
        self.services['trading'] = TradingService()
        self.services['validation'] = DataValidationService()
        self.services['model'] = ModelServing()

        self.logger.info("应用初始化完成")

    def start(self) -> Any:
        """启动应用"""
        if self.is_running:
            self.logger.warning("应用已在运行中")
            return

        self.logger.info("启动RQA2025应用...")
        self.is_running = True

        try:
            # 启动各个服务
            for service_name, service in self.services.items():
                self.logger.info(f"启动服务: {service_name}")
                # 这里可以调用服务的启动方法
                # service.start()

            self.logger.info("应用启动完成")

        except Exception as e:
            self.logger.error(f"应用启动失败: {e}")
            self.stop()
            raise

    def stop(self) -> Any:
        """停止应用"""
        if not self.is_running:
            return

        self.logger.info("停止RQA2025应用...")
        self.is_running = False

        try:
            # 停止各个服务
            for service_name, service in self.services.items():
                self.logger.info(f"停止服务: {service_name}")
                # 这里可以调用服务的停止方法
                # service.stop()

            self.logger.info("应用停止完成")

        except Exception as e:
            self.logger.error(f"应用停止失败: {e}")

    def get_status(self) -> Dict[str, Any]:
        """获取应用状态"""
        return {
            'is_running': self.is_running,
            'services': list(self.services.keys()),
            'config': self.config
        }


class TradingApplication:

    """交易应用，提供完整的交易功能"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """__init__ 函数的文档字符串"""

        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.trading_service = TradingService()
        self.validation_service = DataValidationService()
        self.model_service = ModelServing()

    def run_live_trading(self, strategy_config: Dict[str, Any]):
        """运行实时交易"""
        self.logger.info("启动实时交易...")

        try:
            # 验证策略配置
            validation_result = self.validation_service.validate_data(strategy_config)
            if not validation_result.get('valid', False):
                raise ValueError(f"策略配置验证失败: {validation_result.get('issues', [])}")

            # 加载模型
            model = self.model_service.load_model(strategy_config.get('model_path'))

            # 启动交易服务
            # 这里实现具体的交易逻辑
            self.logger.info("实时交易启动成功")

        except Exception as e:
            self.logger.error(f"实时交易启动失败: {e}")
            raise

    def run_backtest(self, backtest_config: Dict[str, Any]):
        """运行回测"""
        self.logger.info("启动回测...")

        try:
            # 验证回测配置
            validation_result = self.validation_service.validate_data(backtest_config)
            if not validation_result.get('valid', False):
                raise ValueError(f"回测配置验证失败: {validation_result.get('issues', [])}")

            # 执行回测
            # 这里实现具体的回测逻辑
            self.logger.info("回测启动成功")

        except Exception as e:
            self.logger.error(f"回测启动失败: {e}")
            raise


def parse_arguments() -> Any:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='RQA2025量化交易系统')

    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--mode', choices=['live', 'backtest', 'paper'],

                        default='backtest', help='运行模式')
    parser.add_argument('--strategy', type=str, help='策略名称')
    parser.add_argument('--start - date', type=str, help='开始日期 (YYYY - MM - DD)')
    parser.add_argument('--end - date', type=str, help='结束日期 (YYYY - MM - DD)')
    parser.add_argument('--log - level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],

                        default='INFO', help='日志级别')

    return parser.parse_args()


def main() -> Any:
    """系统主入口"""
    # 解析命令行参数
    args = parse_arguments()

    # 设置日志
    setup_logging(level=getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)
    logger.info("启动RQA2025量化交易系统")

    try:
        # 加载配置
        config_manager = UnifiedConfigManager()
        if args.config:
            config = config_manager.load_config(args.config)
        else:
            config = config_manager.get_default_config()

        # 创建应用管理器
        app_manager = ApplicationManager(config)
        app_manager.initialize()

        # 创建交易应用
        trading_app = TradingApplication(config)

        # 根据模式运行应用
        if args.mode == 'live':
            # 实时交易模式
            strategy_config = {
                'name': args.strategy or 'default',
                'model_path': config.get('model_path'),
                'risk_limits': config.get('risk_limits', {})
            }
            trading_app.run_live_trading(strategy_config)

        elif args.mode == 'backtest':
            # 回测模式
            backtest_config = {
                'start_date': args.start_date or '2023 - 01 - 01',
                'end_date': args.end_date or '2023 - 12 - 31',
                'strategy': args.strategy or 'default',
                'initial_capital': config.get('initial_capital', 100000)
            }
            trading_app.run_backtest(backtest_config)

        elif args.mode == 'paper':
            # 模拟交易模式
            logger.info("启动模拟交易模式")
            # 实现模拟交易逻辑

        # 启动应用
        app_manager.start()

        # 保持应用运行
        try:
            while app_manager.is_running:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("收到停止信号")
        finally:
            app_manager.stop()

    except ImportError as e:
        logger.error(f"模块导入失败: {e}")
        print("请确保所有依赖模块已正确安装")
        sys.exit(1)
    except Exception as e:
        logger.error(f"系统运行异常: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("系统正常退出")


if __name__ == "__main__":
    main()
