class PriceLimitChecker:
    """A股个股涨跌停限制检查器"""
    
    def __init__(self):
        self.limits = {
            'normal': 0.10,  # 普通股票10%
            'st': 0.05,      # ST股票5%
            'new': 0.44      # 新股首日44%
        }
        self.st_prefixes = ['600***', '000***', '002***']  # ST股票前缀示例
        self.new_stock_days = 1  # 新股首日限制天数
        
    def check(self, symbol, current_price, prev_close):
        """
        检查个股价格是否超过涨跌停限制
        
        参数:
            symbol: 股票代码
            current_price: 当前价格
            prev_close: 前收盘价
            
        返回:
            bool: True表示价格在限制范围内，False表示超出限制
        """
        change = (current_price - prev_close) / prev_close
        
        if self._is_new_stock(symbol):
            limit = self.limits['new']
        elif self._is_st_stock(symbol):
            limit = self.limits['st']
        else:
            limit = self.limits['normal']
            
        return abs(change) <= limit
        
    def _is_st_stock(self, symbol):
        """判断是否为ST股票"""
        return any(symbol.startswith(prefix) for prefix in self.st_prefixes)
        
    def _is_new_stock(self, symbol):
        """判断是否为新股(首日)"""
        # 简化实现：假设以'688'开头的是科创板新股
        # 实际项目中应接入新股数据库或IPO日期信息
        return symbol.startswith('688') or symbol.startswith('300')
