from typing import Dict, Any
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PositionLimits:
    """A股持仓限制检查器
    
    Attributes:
        max_normal_position: 普通账户单只股票最大持仓(股)
        max_margin_position: 信用账户单只股票最大持仓(股)
    """
    max_normal_position: int = 1000000  # 默认普通账户限制
    max_margin_position: int = 500000   # 默认信用账户限制
    
    def validate(self, account: Dict[str, Any], symbol: str, quantity: int) -> bool:
        """验证持仓是否超过限制
        
        Args:
            account: 账户信息字典，需包含:
                - account_type: 账户类型('normal'/'margin')
                - position_limits: 当前持仓数据
            symbol: 股票代码(如'600000')
            quantity: 申报数量(股)
            
        Returns:
            bool: True表示未超限，False表示超过持仓限制
            
        Raises:
            ValueError: 如果输入参数无效
        """
        # 参数校验
        if not symbol or quantity <= 0:
            raise ValueError("无效的股票代码或数量")
            
        if not account or 'account_type' not in account:
            raise ValueError("账户信息不完整")
        
        account_type = account['account_type']
        current_pos = account.get('position_limits', {}).get(symbol, 0)
        
        # 获取对应账户类型的限制
        max_limit = self._get_position_limit(account_type, symbol)
        
        # 检查是否超限
        total = current_pos + quantity
        if total > max_limit:
            logger.warning(
                f"持仓超限: 账户{account_type} 股票{symbol} "
                f"当前{current_pos}+申报{quantity}>{max_limit}"
            )
            return False
            
        return True
    
    def _get_position_limit(self, account_type: str, symbol: str) -> int:
        """获取特定股票的持仓限制
        
        Args:
            account_type: 账户类型('normal'/'margin')
            symbol: 股票代码
            
        Returns:
            int: 该股票的最大允许持仓量
        """
        base_limit = {
            'normal': self.max_normal_position,
            'margin': self.max_margin_position
        }.get(account_type, 0)
        
        # ST股票特殊处理
        if symbol.startswith('ST'):
            return base_limit * 0.5  # ST股票减半
        return base_limit

# 保持向后兼容的全局函数
def validate_position(account: Dict[str, Any], symbol: str, quantity: int) -> bool:
    """兼容旧版本的持仓验证函数"""
    return PositionLimits().validate(account, symbol, quantity)
