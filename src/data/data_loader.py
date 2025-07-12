from typing import Dict, Any, Optional
from adapters.base_adapter import BaseDataAdapter
from adapters.china import (
    ChinaStockAdapter,
    DragonBoardProcessor,
    MarginTradingAdapter
)
from cache.cache_manager import CacheManager
from parallel_loader import ParallelLoader
from quality.validator import DataValidator
from quality.monitor import DataQualityMonitor
import threading

class DataLoader:
    """主数据加载控制器"""

    def __init__(self):
        self.cache = CacheManager()
        self.parallel_loader = ParallelLoader()
        self.validator = DataValidator()
        self.monitor = DataQualityMonitor()
        self.adapters: Dict[str, BaseDataAdapter] = {
            'china_stock': ChinaStockAdapter(),
            'dragon_board': DragonBoardProcessor(),
            'margin_trading': MarginTradingAdapter()
        }
        self.lock = threading.Lock()

    def load_data(self, config: Dict) -> Optional[Dict]:
        """主数据加载方法"""
        # 从缓存获取
        cache_key = self._generate_cache_key(config)
        cached_data = self.cache.get(cache_key, config['data_type'])
        if cached_data is not None:
            return cached_data

        # 并行加载数据
        if config.get('parallel', False):
            return self._parallel_load(config)

        # 单线程加载
        return self._single_load(config)

    def _parallel_load(self, config: Dict) -> Optional[Dict]:
        """并行数据加载"""
        # 准备批量任务
        tasks = []
        if 'stock_list' in config:
            for stock in config['stock_list']:
                task_config = {
                    'market': config['market'],
                    'data_type': config['data_type'],
                    'symbol': stock
                }
                tasks.append((stock, task_config))

        # 批量加载
        results = self.parallel_loader.batch_load(tasks)

        # 处理结果
        final_result = {}
        for stock, result in results.items():
            if result.success:
                validated = self._validate_and_monitor(result.data)
                if validated:
                    final_result[stock] = result.data

        return final_result if final_result else None

    def _single_load(self, config: Dict) -> Optional[Dict]:
        """单线程数据加载"""
        adapter = self._get_adapter(config)
        if not adapter:
            return None

        # 加载数据
        try:
            data = adapter.load_data(config)
            validated = self._validate_and_monitor(data)
            if validated:
                # 缓存结果
                cache_key = self._generate_cache_key(config)
                self.cache.set(cache_key, data, config['data_type'])
                return data
        except Exception as e:
            print(f"数据加载失败: {str(e)}")

        return None

    def _get_adapter(self, config: Dict) -> Optional[BaseDataAdapter]:
        """获取适合的适配器"""
        market = config.get('market', '').lower()
        data_type = config.get('data_type', '').lower()

        # 中国市场特殊处理
        if market == 'china':
            if data_type == 'dragon_board':
                return self.adapters['dragon_board']
            elif data_type == 'margin':
                return self.adapters['margin_trading']
            else:
                return self.adapters['china_stock']

        # 其他市场适配器...
        return None

    def _validate_and_monitor(self, data: Dict) -> bool:
        """数据验证和监控"""
        with self.lock:
            # 数据验证
            result = self.validator.validate_stock_data(data)

            # 质量监控
            self.monitor.monitor(result)

            return result.is_valid

    def _generate_cache_key(self, config: Dict) -> str:
        """生成缓存键"""
        return f"{config['market']}_{config['data_type']}_{config.get('symbol', '')}"
