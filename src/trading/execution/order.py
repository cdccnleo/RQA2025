"""订单数据模型"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class Order:
    """交易订单"""
    symbol: str           # 交易标的代码
    price: float          # 价格
    quantity: float       # 数量
    id: Optional[str] = None  # 订单ID(可选)

    def __post_init__(self):
        """初始化后处理"""
        if self.id is None:
            # 生成简单ID (实际实现可能使用UUID等)
            self.id = f"{self.symbol}-{int(self.price*100)}-{int(self.quantity)}"
