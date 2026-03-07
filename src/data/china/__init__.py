"""
A股市场数据模块 - v3.0

职责定位：
中国市场业务逻辑实现层，包含适配器实现、市场规则、行情处理等中国市场特定功能。

架构层次：
- adapters/china/  → 通用适配器接口层（定义接口）
- china/           → 中国市场业务逻辑实现层（本目录）

功能架构：
1. 适配器实现层：
   - adapters/          # 适配器实现（推荐使用）
     - AStockAdapter    # A股适配器
     - STARMarketAdapter # 科创板适配器
   - adapter.py         # 完整功能适配器（含Redis、T+1验证等）
   - stock.py           # 传统适配器（兼容旧系统）

2. 市场规则层：
   - market.py          # A股交易规则处理器

3. 行情处理层：
   - level2.py          # Level2行情解码

4. 特殊功能层：
   - special.py         # 特殊股票处理
   - dragon_board.py    # 龙虎榜
   - cache_policy.py    # 缓存策略

使用示例：
    # 使用新版适配器（推荐）
    from src.data.china.adapters import AStockAdapter, STARMarketAdapter
    
    adapter = AStockAdapter()
    data = adapter.get_stock_basic()
    
    # 使用完整功能适配器
    from src.data.china.adapter import ChinaDataAdapter
    
    adapter = ChinaDataAdapter(config={'redis': {...}})
    stock_info = adapter.get_stock_info('600519')
    
    # 检查交易规则
    from src.data.china.market import ChinaMarketRules
    
    rules = ChinaMarketRules()
    price_limit = rules.get_price_limit('600519')

注意事项：
1. 优先使用 adapters/ 目录中的新版适配器
2. adapter.py 提供完整功能（缓存、验证等）
3. 保持与通用数据层接口一致
4. 新增功能需添加集成测试

版本历史：
- v1.0 (2024-02-01): 初始版本
- v2.0 (2024-03-15): 添加新版适配器
- v2.1 (2024-03-20): 重构文档和架构
- v3.0 (2025-01-28): 重构目录结构，清晰分层
"""

# 适配器实现
# from .adapters import AStockAdapter, STARMarketAdapter, ChinaStockAdapter  # 暂时注释以修复循环导入
from .adapter import ChinaDataAdapter

# 传统适配器（向后兼容）
from .stock import ChinaStockDataAdapter

# 市场规则和处理器
from .market import ChinaMarketRules
from .level2 import ChinaLevel2Processor
from .special import SpecialStockHandler

__all__ = [
    # 新版适配器（推荐）
    'AStockAdapter',
    'STARMarketAdapter',
    'ChinaStockAdapter',
    # 完整功能适配器
    'ChinaDataAdapter',
    # 传统适配器（向后兼容）
    'ChinaStockDataAdapter',
    # 市场规则和处理器
    'ChinaMarketRules',
    'ChinaLevel2Processor',
    'SpecialStockHandler',
]
