"""
A股市场数据模块 - v2.1

功能架构：
1. 基础数据适配器：
   - stock: 传统A股数据适配器
   - adapters: 新版基础适配器(推荐)
   - special: 特殊股票处理

2. 市场规则：
   - market: A股交易规则处理器

3. 行情处理：
   - level2: Level2行情解码

模块结构：
    china/
    ├── adapters.py      # 新版适配器
    ├── stock.py        # 传统适配器
    ├── market.py       # 市场规则
    ├── level2.py       # Level2处理
    └── special.py      # 特殊股票

使用示例：
    from src.data.china import ChinaStockAdapter
    from src.data.china.market import ChinaMarketRules

    # 使用新版适配器
    adapter = ChinaStockAdapter()
    data = adapter.get_stock_basic()

    # 检查交易规则
    rules = ChinaMarketRules()
    is_trading_day = rules.check_trading_date('2024-03-20')

注意事项：
1. 优先使用adapters.py中的新版适配器
2. 保持与通用数据层接口一致
3. 新增功能需添加集成测试

版本历史：
- v1.0 (2024-02-01): 初始版本
- v2.0 (2024-03-15): 添加新版适配器
- v2.1 (2024-03-20): 重构文档和架构
"""
from .stock import ChinaStockDataAdapter
from .market import ChinaMarketRules
from .level2 import ChinaLevel2Processor
from .special import SpecialStockHandler
from .adapters import ChinaStockAdapter, STARMarketAdapter

__all__ = [
    'ChinaStockDataAdapter',
    'ChinaMarketRules',
    'ChinaLevel2Processor',
    'SpecialStockHandler',
    'ChinaStockAdapter',
    'STARMarketAdapter'
]
