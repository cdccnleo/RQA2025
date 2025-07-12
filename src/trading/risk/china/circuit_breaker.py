class CircuitBreaker:
    """A股熔断机制检查器"""
    
    def __init__(self, use_fpga=False):
        self.levels = [0.05, 0.07, 0.10]  # 熔断阈值: 5%,7%,10%
        self.status = {
            'triggered': False,
            'level': None,
            'recovery_time': None
        }
        self.logger = logging.getLogger('risk.circuit_breaker')
        self.use_fpga = use_fpga
        if use_fpga:
            from src.fpga.fpga_risk_engine import FpgaRiskEngine
            self.fpga_engine = FpgaRiskEngine()

    def check_market_status(self, index_data):
        """
        检查市场熔断状态
        
        参数:
            index_data: 市场指数数据，包含:
                - current: 当前指数
                - prev_close: 前收盘指数
                - timestamp: 时间戳

        返回:
            dict: 熔断状态 {
                'triggered': bool,  # 是否触发熔断
                'level': float,     # 触发的熔断级别
                'recovery_time': datetime  # 预计恢复时间
            }
        """
        try:
            drop_pct = (index_data['prev_close'] - index_data['current']) / index_data['prev_close']
            
            for level in sorted(self.levels, reverse=True):
                if drop_pct >= level:
                    if not self.status['triggered'] or level > self.status['level']:
                        self._trigger_breaker(level, index_data['timestamp'])
                    break
            
            if self.status['triggered'] and index_data['timestamp'] >= self.status['recovery_time']:
                self._recover_breaker()
                
            return self.status
        
        except Exception as e:
            self.logger.error(f"熔断检查异常: {str(e)}")
            raise

    def _trigger_breaker(self, level, trigger_time):
        """触发熔断"""
        self.status = {
            'triggered': True,
            'level': level,
            'recovery_time': trigger_time + timedelta(minutes=15 if level < 0.07 else 30)
        }
        self.logger.warning(f"熔断触发: 级别{level*100}% at {trigger_time}")
        
        # 通知交易系统暂停交易
        TradingSystem.pause_trading()

    def _recover_breaker(self):
        """熔断恢复"""
        self.logger.info(f"熔断恢复: 级别{self.status['level']*100}%")
        self.status = {
            'triggered': False,
            'level': None,
            'recovery_time': None
        }
        
        # 通知交易系统恢复交易
        TradingSystem.resume_trading()

def check_circuit_breaker(symbol, price_data):
    """
    检查个股熔断状态(兼容旧接口)
    
    参数:
        symbol: 股票代码
        price_data: 价格数据
        
    返回:
        bool: 是否处于熔断状态
    """
    return CircuitBreaker().check_market_status(price_data)['triggered']
