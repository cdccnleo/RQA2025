#!/usr/bin/env python3
"""
RQA2025 分布式系统启动脚本
启动和管理整个量化交易系统的各个组件
"""

from src.adapters.market_adapters import MarketAdapterManager
from src.trading.execution.trade_execution_engine import TradeExecutionEngine
from src.trading.risk.risk_manager import RiskManager
from src.monitoring.trading_monitor import TradingMonitor
from src.realtime.data_stream_processor import DataStreamProcessor
from src.hft.hft_engine import HFTEngine
from src.ml.inference_service import InferenceService, InferenceAPIServer
from src.infrastructure.logging.unified_logger import get_logger
import sys
import os
import time
import threading
import signal
import argparse
from typing import Dict, Any, Optional

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


logger = get_logger(__name__)


class DistributedSystemManager:
    """分布式系统管理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.components = {}
        self.running = False

        # 系统配置
        self.node_type = os.environ.get('NODE_TYPE', 'core')
        self.node_id = os.environ.get('NODE_ID', f"{self.node_type}_001")

        logger.info(f"初始化分布式系统管理器: {self.node_type} - {self.node_id}")

    def initialize_components(self):
        """初始化系统组件"""
        logger.info("开始初始化系统组件...")

        try:
            # 1. 监控系统（所有节点都需要）
            self.components['monitor'] = TradingMonitor(self.config.get('monitoring', {}))
            logger.info("✅ 监控系统初始化完成")

            # 2. 风控管理器（所有节点都需要）
            self.components['risk_manager'] = RiskManager(self.config.get('risk', {}))
            logger.info("✅ 风控管理器初始化完成")

            # 根据节点类型初始化特定组件
            if self.node_type in ['core', 'master']:
                self._initialize_core_components()
            elif self.node_type == 'hft':
                self._initialize_hft_components()
            elif self.node_type == 'ml':
                self._initialize_ml_components()
            elif self.node_type == 'data_collector':
                self._initialize_data_components()
            else:
                logger.warning(f"未知节点类型: {self.node_type}")

            logger.info(f"所有组件初始化完成，共 {len(self.components)} 个组件")

        except Exception as e:
            logger.error(f"组件初始化失败: {e}")
            raise

    def _initialize_core_components(self):
        """初始化核心组件"""
        # 市场适配器管理器
        self.components['market_adapters'] = MarketAdapterManager()
        adapters = self.components['market_adapters'].create_default_adapters()
        logger.info(f"✅ 市场适配器管理器初始化完成，支持 {len(adapters)} 个市场")

        # 实时数据流处理器
        self.components['data_processor'] = DataStreamProcessor(
            self.config.get('data_processor', {})
        )
        logger.info("✅ 实时数据流处理器初始化完成")

        # 交易执行引擎
        self.components['execution_engine'] = TradeExecutionEngine(
            self.config.get('execution', {})
        )
        logger.info("✅ 交易执行引擎初始化完成")

    def _initialize_hft_components(self):
        """初始化高频交易组件"""
        # 高频交易引擎
        self.components['hft_engine'] = HFTEngine(self.config.get('hft', {}))
        logger.info("✅ 高频交易引擎初始化完成")

    def _initialize_ml_components(self):
        """初始化机器学习组件"""
        # ML推理服务
        self.components['inference_service'] = InferenceService(
            self.config.get('ml', {})
        )

        # API服务器
        self.components['api_server'] = InferenceAPIServer(
            self.components['inference_service'],
            self.config.get('api', {})
        )
        logger.info("✅ ML推理服务初始化完成")

    def _initialize_data_components(self):
        """初始化数据采集组件"""
        # 市场适配器管理器（数据采集专用）
        self.components['market_adapters'] = MarketAdapterManager()
        adapters = self.components['market_adapters'].create_default_adapters()
        logger.info("✅ 数据采集适配器初始化完成")

    def start_system(self):
        """启动系统"""
        if self.running:
            logger.warning("系统已在运行中")
            return

        logger.info("开始启动分布式系统...")

        try:
            # 1. 启动监控系统
            if 'monitor' in self.components:
                self.components['monitor'].start_monitoring()
                logger.info("✅ 监控系统已启动")

            # 2. 启动风控管理器（通常不需要特殊启动）

            # 3. 根据节点类型启动特定服务
            if self.node_type in ['core', 'master']:
                self._start_core_services()
            elif self.node_type == 'hft':
                self._start_hft_services()
            elif self.node_type == 'ml':
                self._start_ml_services()
            elif self.node_type == 'data_collector':
                self._start_data_services()

            self.running = True
            logger.info("🎉 分布式系统启动完成！")

            # 启动健康检查
            self._start_health_check()

        except Exception as e:
            logger.error(f"系统启动失败: {e}")
            self.stop_system()
            raise

    def _start_core_services(self):
        """启动核心服务"""
        # 连接市场适配器
        if 'market_adapters' in self.components:
            connection_results = self.components['market_adapters'].connect_all()
            successful_connections = sum(1 for result in connection_results.values() if result)
            logger.info(f"✅ 市场适配器连接完成: {successful_connections}/{len(connection_results)}")

        # 启动实时数据处理器
        if 'data_processor' in self.components:
            self.components['data_processor'].start()
            logger.info("✅ 实时数据处理器已启动")

        # 启动交易执行引擎
        if 'execution_engine' in self.components:
            logger.info("✅ 交易执行引擎已准备就绪")

    def _start_hft_services(self):
        """启动高频交易服务"""
        if 'hft_engine' in self.components:
            self.components['hft_engine'].start_engine()
            logger.info("✅ 高频交易引擎已启动")

    def _start_ml_services(self):
        """启动机器学习服务"""
        if 'inference_service' in self.components:
            self.components['inference_service'].start()
            logger.info("✅ ML推理服务已启动")

        if 'api_server' in self.components:
            # 在独立线程中启动API服务器
            api_thread = threading.Thread(target=self.components['api_server'].start)
            api_thread.daemon = True
            api_thread.start()
            logger.info("✅ ML API服务器已启动")

    def _start_data_services(self):
        """启动数据采集服务"""
        if 'market_adapters' in self.components:
            connection_results = self.components['market_adapters'].connect_all()
            successful_connections = sum(1 for result in connection_results.values() if result)
            logger.info(f"✅ 数据采集连接完成: {successful_connections}/{len(connection_results)}")

        logger.info("✅ 数据采集服务已启动")

    def stop_system(self):
        """停止系统"""
        if not self.running:
            return

        logger.info("正在停止分布式系统...")

        # 按相反顺序停止服务
        if self.node_type in ['core', 'master']:
            self._stop_core_services()
        elif self.node_type == 'hft':
            self._stop_hft_services()
        elif self.node_type == 'ml':
            self._stop_ml_services()
        elif self.node_type == 'data_collector':
            self._stop_data_services()

        # 停止监控系统
        if 'monitor' in self.components:
            self.components['monitor'].stop_monitoring()

        self.running = False
        logger.info("✅ 分布式系统已停止")

    def _stop_core_services(self):
        """停止核心服务"""
        if 'data_processor' in self.components:
            self.components['data_processor'].stop()

        if 'market_adapters' in self.components:
            self.components['market_adapters'].disconnect_all()

    def _stop_hft_services(self):
        """停止高频交易服务"""
        if 'hft_engine' in self.components:
            self.components['hft_engine'].stop_engine()

    def _stop_ml_services(self):
        """停止机器学习服务"""
        if 'inference_service' in self.components:
            self.components['inference_service'].stop()

    def _stop_data_services(self):
        """停止数据采集服务"""
        if 'market_adapters' in self.components:
            self.components['market_adapters'].disconnect_all()

    def _start_health_check(self):
        """启动健康检查"""
        def health_check_loop():
            while self.running:
                try:
                    self._perform_health_check()
                    time.sleep(30)  # 每30秒检查一次
                except Exception as e:
                    logger.error(f"健康检查异常: {e}")
                    time.sleep(10)

        health_thread = threading.Thread(target=health_check_loop)
        health_thread.daemon = True
        health_thread.start()
        logger.info("✅ 健康检查服务已启动")

    def _perform_health_check(self):
        """执行健康检查"""
        health_status = {
            'node_type': self.node_type,
            'node_id': self.node_id,
            'timestamp': time.time(),
            'components': {}
        }

        # 检查各个组件状态
        for name, component in self.components.items():
            try:
                if hasattr(component, 'get_service_status'):
                    status = component.get_service_status()
                elif hasattr(component, 'get_performance_stats'):
                    status = component.get_performance_stats()
                elif hasattr(component, 'health_check'):
                    status = component.health_check()
                else:
                    status = {'status': 'unknown'}

                health_status['components'][name] = status

            except Exception as e:
                health_status['components'][name] = {'status': 'error', 'error': str(e)}

        # 检查整体健康状态
        component_statuses = [comp.get('status', 'unknown')
                              for comp in health_status['components'].values()]

        if all(status in ['healthy', 'running', 'connected'] for status in component_statuses):
            health_status['overall_status'] = 'healthy'
        elif any(status in ['error', 'failed', 'disconnected'] for status in component_statuses):
            health_status['overall_status'] = 'unhealthy'
        else:
            health_status['overall_status'] = 'degraded'

        logger.info(f"健康检查完成: {health_status['overall_status']}")

        # 这里可以发送健康状态到监控系统
        return health_status

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'node_type': self.node_type,
            'node_id': self.node_id,
            'running': self.running,
            'components_count': len(self.components),
            'components': list(self.components.keys())
        }


def load_config() -> Dict[str, Any]:
    """加载配置文件"""
    config = {
        'system': {
            'name': 'RQA2025 Distributed Trading System',
            'version': '2.0.0'
        },
        'monitoring': {
            'monitoring_interval': 60,
            'alert_thresholds': {
                'cpu_threshold': 80,
                'memory_threshold': 80,
                'response_time_threshold': 1.0
            }
        },
        'risk': {
            'max_position_value': 100000,
            'max_single_position_ratio': 0.2,
            'max_portfolio_volatility': 0.25
        },
        'data_processor': {
            'buffer_size': 1000,
            'processing_interval': 1.0
        },
        'execution': {
            'max_latency_us': 1000,
            'risk_limits': {
                'max_position': 10000,
                'max_order_rate': 100
            }
        },
        'hft': {
            'max_position': 1000,
            'max_latency_us': 1000
        },
        'ml': {
            'max_workers': 4,
            'model_cache_size': 100
        },
        'api': {
            'host': '0.0.0.0',
            'port': 8080
        }
    }

    return config


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RQA2025 分布式系统管理器')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--node-type', type=str,
                        choices=['core', 'master', 'hft', 'ml', 'data_collector'],
                        default='core', help='节点类型')
    parser.add_argument('--node-id', type=str, help='节点ID')

    args = parser.parse_args()

    # 加载配置
    config = load_config()

    # 设置节点信息
    if args.node_id:
        config['node_id'] = args.node_id
    if args.node_type:
        config['node_type'] = args.node_type

    # 创建系统管理器
    system_manager = DistributedSystemManager(config)

    # 设置信号处理器
    def signal_handler(signum, frame):
        logger.info(f"接收到信号 {signum}, 正在停止系统...")
        system_manager.stop_system()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # 初始化组件
        system_manager.initialize_components()

        # 启动系统
        system_manager.start_system()

        # 保持运行状态
        logger.info("系统正在运行中... 按 Ctrl+C 停止")

        while system_manager.running:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("接收到键盘中断信号")
    except Exception as e:
        logger.error(f"系统运行异常: {e}")
        raise
    finally:
        system_manager.stop_system()


if __name__ == "__main__":
    main()
