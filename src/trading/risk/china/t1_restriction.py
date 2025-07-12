class T1RestrictionChecker:
    """A股T+1交易限制检查器"""
    
    def __init__(self, use_fpga=False):
        self.position_service = PositionService()
        self.trading_calendar = TradingCalendar()
        self.logger = logging.getLogger('risk.t1_restriction')
        self.use_fpga = use_fpga
        self.star_market_checker = STARMarketRuleChecker()  # 科创板规则检查器
        self.metrics = RiskMetricsCollector('t1_restriction')
        if use_fpga:
            from src.fpga.fpga_risk_engine import FpgaRiskEngine
            self.fpga_engine = FpgaRiskEngine()
        
    def check_sell_restriction(self, account_id, symbol, sell_date):
        """
        检查卖出操作是否违反T+1规则
        
        参数:
            account_id: 账户ID
            symbol: 股票代码
            sell_date: 卖出日期(datetime.date)
            
        返回:
            bool: True表示违反T+1规则(不允许卖出)
        """
        start_time = time.time()
        try:
            # 批量获取持仓和交易数据
            position_data = self.position_service.get_positions_and_trades(
                account_id,
                symbols=[symbol],
                trade_date=sell_date
            )
            
            if not position_data:
                return False
                
            # 科创板特殊规则检查
            if symbol.startswith('688'):
                result = self.star_market_checker.check_star_market_rules({
                    'account_id': account_id,
                    'symbol': symbol,
                    'position_data': position_data
                })
                if not result[0]:
                    self.logger.warning(
                        f"科创板T+1违规阻止: 账户{account_id} 股票{symbol} - {result[1]}"
                    )
                return not result[0]
                
            # 普通A股T+1规则
            total_buy = sum(t['quantity'] for t in position_data['trades'])
            current_position = position_data['position']
            
            # 如果当日买入量大于等于持仓量，说明全部是当日买入
            violation = total_buy >= current_position
            if violation:
                self.logger.warning(
                    f"T+1违规阻止: 账户{account_id} 股票{symbol} "
                    f"试图卖出当日买入的{current_position}股"
                )
                
            # 记录性能指标
            elapsed = (time.time() - start_time) * 1000
            self.metrics.record_check(
                account_id=account_id,
                symbol=symbol,
                duration_ms=elapsed,
                is_violation=violation
            )
            
            return violation
            
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            self.metrics.record_error(
                account_id=account_id,
                symbol=symbol,
                duration_ms=elapsed,
                error=str(e)
            )
            self.logger.error(f"T+1检查异常: {str(e)}")
            # 出现异常时保守处理，阻止交易
            return True
            
    def _check_star_market_rule(self, buy_records, sell_date):
        """
        科创板特殊规则检查
        
        参数:
            buy_records: 买入记录列表
            sell_date: 卖出日期
            
        返回:
            bool: 是否违反规则
        """
        # 科创板目前也适用T+1规则
        # 未来如有特殊规则可在此实现
        return False

def check_t1_rule(account, symbol, trade_date):
    """
    检查T+1限制(兼容旧接口)
    
    参数:
        account: 账户信息
        symbol: 股票代码
        trade_date: 交易日期
        
    返回:
        bool: 是否违反T+1规则
    """
    return T1RestrictionChecker().check_sell_restriction(
        account['id'], 
        symbol,
        trade_date
    )
