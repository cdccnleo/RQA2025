"""
中国A股数据适配器模块（已迁移）

注意：此文件已重构，新的适配器实现已迁移到 china/adapters/ 目录。
此文件保留仅用于向后兼容。

新的导入方式：
    from src.data.china.adapters import AStockAdapter, STARMarketAdapter
    from src.data.china.adapters import ChinaStockAdapter  # 别名
"""

# 从新位置导入适配器
from .adapters import AStockAdapter, STARMarketAdapter, ChinaStockAdapter

# 向后兼容：导出到原位置
__all__ = [
    'AStockAdapter',
    'STARMarketAdapter',
    'ChinaStockAdapter',
]
