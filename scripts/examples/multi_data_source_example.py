#!/usr/bin/env python
"""
多数据源使用示例
展示如何使用新的数据源适配器
"""

from src.data.adapters import (
    adapter_registry,
    AdapterInfo,
    AdapterConfig,
    InternationalStockAdapter,
    CryptoAdapter,
    MacroEconomicAdapter,
    NewsSentimentAdapter
)
import asyncio
from datetime import datetime, timedelta
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入数据源适配器


class MultiDataSourceExample:
    """多数据源使用示例类"""

    def __init__(self):
        self.registry = adapter_registry
        self._setup_adapters()

    def _setup_adapters(self):
        """设置数据源适配器"""
        try:
            # 注册国际股票数据适配器
            international_info = AdapterInfo(
                name='international_stock',
                adapter_class=InternationalStockAdapter,
                description='国际股票数据适配器，支持美股、港股等',
                supported_markets=['US', 'HK', 'JP', 'UK', 'DE'],
                supported_data_types=['stock', 'market_data'],
                config_schema={
                    'adapter_type': 'international_stock',
                    'connection_params': {},
                    'validation_rules': {}
                }
            )
            self.registry.register_adapter('international_stock', international_info)

            # 注册加密货币数据适配器
            crypto_info = AdapterInfo(
                name='crypto',
                adapter_class=CryptoAdapter,
                description='加密货币数据适配器，支持数字货币市场',
                supported_markets=['crypto'],
                supported_data_types=['crypto', 'market_data'],
                config_schema={
                    'adapter_type': 'crypto',
                    'connection_params': {
                        'exchange': 'binance',
                        'api_key': '',
                        'secret': '',
                        'sandbox': True
                    },
                    'validation_rules': {}
                }
            )
            self.registry.register_adapter('crypto', crypto_info)

            # 注册宏观经济数据适配器
            macro_info = AdapterInfo(
                name='macro_economic',
                adapter_class=MacroEconomicAdapter,
                description='宏观经济数据适配器，支持经济指标数据',
                supported_markets=['US', 'CN'],
                supported_data_types=['economic_indicators', 'macro_data'],
                config_schema={
                    'adapter_type': 'macro_economic',
                    'connection_params': {
                        'api_key': '',
                        'base_url': 'https://api.stlouisfed.org/fred'
                    },
                    'validation_rules': {}
                }
            )
            self.registry.register_adapter('macro_economic', macro_info)

            # 注册新闻情感数据适配器
            news_info = AdapterInfo(
                name='news_sentiment',
                adapter_class=NewsSentimentAdapter,
                description='新闻情感数据适配器，支持新闻和情感分析',
                supported_markets=['global'],
                supported_data_types=['news', 'sentiment', 'social_media'],
                config_schema={
                    'adapter_type': 'news_sentiment',
                    'connection_params': {
                        'api_key': '',
                        'base_url': 'https://newsapi.org/v2'
                    },
                    'validation_rules': {}
                }
            )
            self.registry.register_adapter('news_sentiment', news_info)

            logger.info("所有数据源适配器注册完成")

        except Exception as e:
            logger.error(f"设置数据源适配器失败: {e}")

    def configure_adapters(self):
        """配置适配器"""
        try:
            # 配置国际股票适配器
            international_config = AdapterConfig(
                adapter_type='international_stock',
                connection_params={},
                validation_rules={}
            )
            self.registry.configure_adapter('international_stock', international_config)

            # 配置加密货币适配器
            crypto_config = AdapterConfig(
                adapter_type='crypto',
                connection_params={
                    'exchange': 'binance',
                    'api_key': '',
                    'secret': '',
                    'sandbox': True
                },
                validation_rules={}
            )
            self.registry.configure_adapter('crypto', crypto_config)

            # 配置宏观经济适配器
            macro_config = AdapterConfig(
                adapter_type='macro_economic',
                connection_params={
                    'api_key': '',
                    'base_url': 'https://api.stlouisfed.org/fred'
                },
                validation_rules={}
            )
            self.registry.configure_adapter('macro_economic', macro_config)

            # 配置新闻情感适配器
            news_config = AdapterConfig(
                adapter_type='news_sentiment',
                connection_params={
                    'api_key': '',
                    'base_url': 'https://newsapi.org/v2'
                },
                validation_rules={}
            )
            self.registry.configure_adapter('news_sentiment', news_config)

            logger.info("所有适配器配置完成")

        except Exception as e:
            logger.error(f"配置适配器失败: {e}")

    async def demonstrate_international_stock_data(self):
        """演示国际股票数据"""
        logger.info("=== 国际股票数据演示 ===")

        try:
            # 获取适配器
            adapter = self.registry.get_adapter('international_stock')
            if not adapter:
                logger.error("国际股票适配器不可用")
                return

            # 获取支持的股票代码
            symbols = adapter.get_available_symbols()
            logger.info(f"支持的股票代码: {symbols[:5]}...")

            # 获取市场信息
            markets = adapter.get_supported_markets()
            logger.info(f"支持的市场: {markets}")

            # 加载数据示例
            start_date = datetime.now() - timedelta(days=30)
            end_date = datetime.now()

            # 尝试加载AAPL数据
            request = DataRequest(
                symbol='AAPL',
                start_date=start_date,
                end_date=end_date,
                interval='1d'
            )

            df = adapter.load(request)
            if not df.empty:
                logger.info(f"成功加载AAPL数据，共{len(df)}条记录")
                logger.info(f"数据列: {list(df.columns)}")
                logger.info(f"最新价格: {df['close'].iloc[-1]:.2f}")
            else:
                logger.warning("未获取到AAPL数据")

        except Exception as e:
            logger.error(f"国际股票数据演示失败: {e}")

    async def demonstrate_crypto_data(self):
        """演示加密货币数据"""
        logger.info("=== 加密货币数据演示 ===")

        try:
            # 获取适配器
            adapter = self.registry.get_adapter('crypto')
            if not adapter:
                logger.error("加密货币适配器不可用")
                return

            # 获取支持的交易所
            exchanges = adapter.get_supported_exchanges()
            logger.info(f"支持的交易所: {exchanges}")

            # 获取交易对
            symbols = adapter.get_available_symbols()
            logger.info(f"支持的交易对: {symbols[:5]}...")

            # 获取实时价格
            ticker = adapter.get_ticker('BTC/USDT')
            if ticker:
                logger.info(f"BTC/USDT实时价格: {ticker.get('last', 0):.2f}")

            # 获取订单簿
            order_book = adapter.get_order_book('BTC/USDT', limit=5)
            if order_book:
                logger.info(
                    f"BTC/USDT订单簿深度: {len(order_book.get('bids', []))} bids, {len(order_book.get('asks', []))} asks")

        except Exception as e:
            logger.error(f"加密货币数据演示失败: {e}")

    async def demonstrate_macro_economic_data(self):
        """演示宏观经济数据"""
        logger.info("=== 宏观经济数据演示 ===")

        try:
            # 获取适配器
            adapter = self.registry.get_adapter('macro_economic')
            if not adapter:
                logger.error("宏观经济适配器不可用")
                return

            # 获取支持的指标
            indicators = adapter.get_supported_indicators()
            logger.info(f"支持的美国指标: {list(indicators.keys())[:5]}...")

            china_indicators = adapter.get_china_indicators()
            logger.info(f"支持的中国指标: {list(china_indicators.keys())[:5]}...")

            # 获取经济日历
            calendar = adapter.get_economic_calendar()
            if calendar:
                logger.info(f"经济日历事件数量: {len(calendar)}")
                for event in calendar[:3]:
                    logger.info(
                        f"事件: {event['indicator']} - {event['date']} - {event['importance']}")

        except Exception as e:
            logger.error(f"宏观经济数据演示失败: {e}")

    async def demonstrate_news_sentiment_data(self):
        """演示新闻情感数据"""
        logger.info("=== 新闻情感数据演示 ===")

        try:
            # 获取适配器
            adapter = self.registry.get_adapter('news_sentiment')
            if not adapter:
                logger.error("新闻情感适配器不可用")
                return

            # 获取支持的数据类型
            data_types = adapter.get_supported_types()
            logger.info(f"支持的数据类型: {data_types}")

            # 获取情感分析摘要
            sentiment_summary = adapter.get_sentiment_summary('AAPL')
            if sentiment_summary:
                logger.info(f"AAPL情感分析摘要: {sentiment_summary}")

        except Exception as e:
            logger.error(f"新闻情感数据演示失败: {e}")

    async def demonstrate_adapter_registry(self):
        """演示适配器注册管理器功能"""
        logger.info("=== 适配器注册管理器演示 ===")

        try:
            # 列出所有适配器
            adapters = self.registry.list_adapters()
            logger.info(f"注册的适配器数量: {len(adapters)}")

            for adapter in adapters:
                logger.info(f"适配器: {adapter['name']} - {adapter['description']}")
                logger.info(f"  状态: {'已配置' if adapter['is_configured'] else '未配置'}")
                logger.info(f"  连接: {'已连接' if adapter['is_connected'] else '未连接'}")

            # 健康检查
            health_status = self.registry.health_check_all()
            logger.info(f"总体健康状态: {health_status['overall_status']}")
            logger.info(
                f"健康适配器: {health_status['healthy_adapters']}/{health_status['total_adapters']}")

            # 按市场查找适配器
            us_adapters = self.registry.find_adapters_by_market('US')
            logger.info(f"支持美国市场的适配器: {us_adapters}")

            # 按数据类型查找适配器
            market_data_adapters = self.registry.find_adapters_by_data_type('market_data')
            logger.info(f"支持市场数据的适配器: {market_data_adapters}")

        except Exception as e:
            logger.error(f"适配器注册管理器演示失败: {e}")

    async def run_all_demonstrations(self):
        """运行所有演示"""
        logger.info("开始多数据源演示...")

        # 配置适配器
        self.configure_adapters()

        # 运行各个演示
        await self.demonstrate_adapter_registry()
        await self.demonstrate_international_stock_data()
        await self.demonstrate_crypto_data()
        await self.demonstrate_macro_economic_data()
        await self.demonstrate_news_sentiment_data()

        logger.info("多数据源演示完成")

    def cleanup(self):
        """清理资源"""
        try:
            self.registry.cleanup()
            logger.info("资源清理完成")
        except Exception as e:
            logger.error(f"资源清理失败: {e}")


async def main():
    """主函数"""
    example = MultiDataSourceExample()

    try:
        await example.run_all_demonstrations()
    finally:
        example.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
