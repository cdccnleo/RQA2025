"""
投资组合管理器模块
"""

class PortfolioManager:
    """投资组合管理器"""
    def __init__(self):
        self.positions = {}
        self.cash = 0
    
    def add_position(self, symbol, quantity, price):
        """添加持仓"""
        if symbol not in self.positions:
            self.positions[symbol] = {'quantity': 0, 'avg_price': 0}
        
        current = self.positions[symbol]
        total_quantity = current['quantity'] + quantity
        total_cost = current['quantity'] * current['avg_price'] + quantity * price
        
        self.positions[symbol] = {
            'quantity': total_quantity,
            'avg_price': total_cost / total_quantity if total_quantity > 0 else 0
        }
    
    def get_portfolio_value(self, current_prices):
        """获取投资组合价值"""
        total = self.cash
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                total += position['quantity'] * current_prices[symbol]
        return total

    def remove_position(self, symbol, quantity=None):
        """移除持仓"""
        if symbol not in self.positions:
            return False

        if quantity is None or quantity >= self.positions[symbol]['quantity']:
            del self.positions[symbol]
        else:
            self.positions[symbol]['quantity'] -= quantity

        return True

__all__ = ['PortfolioManager']
