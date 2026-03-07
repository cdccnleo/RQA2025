"""
多市场交易主流程脚本
集成多市场适配器、跨市场套利策略和策略自动优化器
"""

from src.trading.strategies.strategy_auto_optimizer import (
    StrategyAutoOptimizer, StrategyParameter, OptimizationMethod
)
from src.trading.strategies.cross_market_arbitrage import (
    CrossMarketArbitrageStrategy
)
from src.trading.execution.multi_market_adapter import (
    MultiMarketManager
)
import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiMarketTradingMainFlow:
    """多市场交易主流程"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 初始化组件
        self.multi_market_manager = MultiMarketManager(config)
        self.arbitrage_strategy = CrossMarketArbitrageStrategy(config)
        self.strategy_optimizer = StrategyAutoOptimizer(config)

        # 模拟市场数据
        self.market_data = self._generate_market_data()

        # 运行状态
        self.is_running = False
        self.start_time = None

    def _generate_market_data(self) -> Dict[str, pd.DataFrame]:
        """生成模拟市场数据"""
        market_data = {}

        # 生成A股数据
        a_share_symbols = ['600519.SH', '000858.SZ', '600036.SH']
        for symbol in a_share_symbols:
            dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
            prices = np.random.normal(100, 20, len(dates))
            prices = np.maximum(prices, 10)  # 确保价格为正

            df = pd.DataFrame({
                'date': dates,
                'open': prices * (1 + np.random.normal(0, 0.01, len(dates))),
                'high': prices * (1 + np.random.normal(0.02, 0.01, len(dates))),
                'low': prices * (1 + np.random.normal(-0.02, 0.01, len(dates))),
                'close': prices,
                'volume': np.random.randint(1000000, 10000000, len(dates))
            })
            market_data[symbol] = df

        # 生成港股数据
        h_share_symbols = ['02318.HK', '02319.HK', '0700.HK']
        for symbol in h_share_symbols:
            dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
            prices = np.random.normal(50, 10, len(dates))
            prices = np.maximum(prices, 5)

            df = pd.DataFrame({
                'date': dates,
                'open': prices * (1 + np.random.normal(0, 0.01, len(dates))),
                'high': prices * (1 + np.random.normal(0.02, 0.01, len(dates))),
                'low': prices * (1 + np.random.normal(-0.02, 0.01, len(dates))),
                'close': prices,
                'volume': np.random.randint(500000, 5000000, len(dates))
            })
            market_data[symbol] = df

        # 生成美股数据
        us_share_symbols = ['CMB', 'TCEHY', 'AAPL']
        for symbol in us_share_symbols:
            dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
            prices = np.random.normal(150, 30, len(dates))
            prices = np.maximum(prices, 10)

            df = pd.DataFrame({
                'date': dates,
                'open': prices * (1 + np.random.normal(0, 0.01, len(dates))),
                'high': prices * (1 + np.random.normal(0.02, 0.01, len(dates))),
                'low': prices * (1 + np.random.normal(-0.02, 0.01, len(dates))),
                'close': prices,
                'volume': np.random.randint(100000, 1000000, len(dates))
            })
            market_data[symbol] = df

        return market_data

    def run_multi_market_trading(self):
        """运行多市场交易主流程"""
        self.logger.info("启动多市场交易主流程")
        self.is_running = True
        self.start_time = datetime.now()

        try:
            # 1. 初始化市场状态检查
            self._check_market_status()

            # 2. 检测套利机会
            opportunities = self._detect_arbitrage_opportunities()

            # 3. 生成套利信号
            signals = self._generate_arbitrage_signals(opportunities)

            # 4. 执行套利交易
            execution_results = self._execute_arbitrage_trades(signals)

            # 5. 策略参数优化
            optimization_results = self._optimize_strategy_parameters()

            # 6. 监控持仓
            monitoring_results = self._monitor_positions()

            # 7. 生成报告
            self._generate_trading_report(
                execution_results, optimization_results, monitoring_results)

            self.logger.info("多市场交易主流程执行完成")

        except Exception as e:
            self.logger.error(f"多市场交易主流程执行失败: {e}")
            raise
        finally:
            self.is_running = False

    def _check_market_status(self):
        """检查市场状态"""
        self.logger.info("检查各市场交易状态...")

        market_status = self.multi_market_manager.get_market_status()

        for market_type, is_trading in market_status.items():
            status_str = "开市" if is_trading else "闭市"
            self.logger.info(f"{market_type.value}: {status_str}")

        # 获取账户信息
        accounts_info = self.multi_market_manager.get_all_accounts_info()
        self.logger.info(f"账户信息: {accounts_info}")

        # 获取持仓信息
        positions = self.multi_market_manager.get_all_positions()
        self.logger.info(f"持仓信息: {positions}")

    def _detect_arbitrage_opportunities(self) -> List[Any]:
        """检测套利机会"""
        self.logger.info("检测跨市场套利机会...")

        opportunities = self.arbitrage_strategy.detect_arbitrage_opportunities(self.market_data)

        self.logger.info(f"检测到 {len(opportunities)} 个套利机会")

        for i, opp in enumerate(opportunities):
            self.logger.info(f"机会 {i+1}: {opp.arbitrage_type.value}, "
                             f"价差: {opp.spread:.4f}, 置信度: {opp.confidence:.4f}")

        return opportunities

    def _generate_arbitrage_signals(self, opportunities: List[Any]) -> List[Any]:
        """生成套利信号"""
        self.logger.info("生成套利交易信号...")

        signals = self.arbitrage_strategy.generate_arbitrage_signals(opportunities)

        self.logger.info(f"生成了 {len(signals)} 个交易信号")

        for i, signal in enumerate(signals):
            self.logger.info(f"信号 {i+1}: {signal.action}, "
                             f"数量: {signal.quantity}, 预期收益: {signal.expected_profit:.4f}")

        return signals

    def _execute_arbitrage_trades(self, signals: List[Any]) -> List[Dict[str, Any]]:
        """执行套利交易"""
        self.logger.info("执行套利交易...")

        if not signals:
            self.logger.info("没有可执行的套利信号")
            return []

        execution_results = self.arbitrage_strategy.execute_arbitrage_signals(signals)

        success_count = sum(1 for result in execution_results if result['success'])
        self.logger.info(f"执行完成: {success_count}/{len(execution_results)} 成功")

        for result in execution_results:
            if result['success']:
                self.logger.info(f"交易成功: {result['signal_id']}, "
                                 f"预期收益: {result['expected_profit']:.4f}")
            else:
                self.logger.warning(f"交易失败: {result['signal_id']}, "
                                    f"原因: {result.get('error', '未知')}")

        return execution_results

    def _optimize_strategy_parameters(self) -> Dict[str, Any]:
        """优化策略参数"""
        self.logger.info("开始策略参数优化...")

        # 定义策略参数空间
        param_space = {
            'min_spread': StrategyParameter(
                name='min_spread',
                value=0.02,
                min_value=0.01,
                max_value=0.05,
                step=0.005,
                param_type='float'
            ),
            'max_position_size': StrategyParameter(
                name='max_position_size',
                value=10000,
                min_value=5000,
                max_value=20000,
                step=1000,
                param_type='int'
            ),
            'z_score_threshold': StrategyParameter(
                name='z_score_threshold',
                value=2.0,
                min_value=1.5,
                max_value=3.0,
                step=0.1,
                param_type='float'
            )
        }

        # 模拟策略类
        class MockStrategy:
            def __init__(self, min_spread=0.02, max_position_size=10000, z_score_threshold=2.0):
                self.min_spread = min_spread
                self.max_position_size = max_position_size
                self.z_score_threshold = z_score_threshold

        # 运行优化
        try:
            # 将market_data字典转换为单个DataFrame用于优化
            # 选择第一个市场的数据作为历史数据
            historical_df = list(self.market_data.values())[
                0] if self.market_data else pd.DataFrame()

            optimization_result = self.strategy_optimizer.optimize_strategy_parameters(
                strategy_name='CrossMarketArbitrage',
                strategy_class=MockStrategy,
                param_space=param_space,
                historical_data=historical_df,
                optimization_method=OptimizationMethod.BAYESIAN_OPTIMIZATION
            )

            self.logger.info(f"策略优化完成: 最佳分数 {optimization_result.best_score:.4f}")
            self.logger.info(f"最佳参数: {optimization_result.best_params}")

            return {
                'success': True,
                'optimization_result': optimization_result.to_dict(),
                'best_params': optimization_result.best_params,
                'best_score': optimization_result.best_score
            }

        except Exception as e:
            self.logger.error(f"策略优化失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _monitor_positions(self) -> List[Dict[str, Any]]:
        """监控持仓"""
        self.logger.info("监控当前持仓...")

        monitoring_results = self.arbitrage_strategy.monitor_positions()

        active_positions = len([r for r in monitoring_results if r.get('action') == 'monitor'])
        closed_positions = len([r for r in monitoring_results if r.get('action') == 'close'])

        self.logger.info(f"持仓监控完成: {active_positions} 个活跃持仓, {closed_positions} 个已平仓")

        for result in monitoring_results:
            if result.get('action') == 'close' and result.get('success'):
                self.logger.info(f"平仓成功: {result['signal_id']}, "
                                 f"最终盈亏: {result.get('final_pnl', 0):.4f}")

        return monitoring_results

    def _generate_trading_report(self,
                                 execution_results: List[Dict[str, Any]],
                                 optimization_results: Dict[str, Any],
                                 monitoring_results: List[Dict[str, Any]]):
        """生成交易报告"""
        self.logger.info("生成多市场交易报告...")

        # 计算统计信息
        total_executions = len(execution_results)
        successful_executions = sum(1 for r in execution_results if r.get('success'))
        success_rate = successful_executions / total_executions if total_executions > 0 else 0

        total_pnl = sum(r.get('expected_profit', 0) for r in execution_results if r.get('success'))

        # 生成报告
        report = {
            'report_time': datetime.now().isoformat(),
            'execution_summary': {
                'total_executions': total_executions,
                'successful_executions': successful_executions,
                'success_rate': success_rate,
                'total_pnl': total_pnl
            },
            'optimization_summary': optimization_results,
            'monitoring_summary': {
                'active_positions': len([r for r in monitoring_results if r.get('action') == 'monitor']),
                'closed_positions': len([r for r in monitoring_results if r.get('action') == 'close'])
            },
            'strategy_summary': self.arbitrage_strategy.get_strategy_summary(),
            'optimizer_summary': self.strategy_optimizer.get_optimization_summary()
        }

        # 保存报告
        report_file = f"reports/trading/multi_market_trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)

        import json
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.logger.info(f"交易报告已保存到: {report_file}")

        # 打印摘要
        self.logger.info("=== 多市场交易报告摘要 ===")
        self.logger.info(f"执行成功率: {success_rate:.2%}")
        self.logger.info(f"总预期收益: {total_pnl:.4f}")
        self.logger.info(f"活跃持仓数: {report['monitoring_summary']['active_positions']}")
        self.logger.info(f"已平仓数: {report['monitoring_summary']['closed_positions']}")

        if optimization_results.get('success'):
            self.logger.info(f"策略优化最佳分数: {optimization_results['best_score']:.4f}")

    def run_continuous_trading(self, interval_seconds: int = 60):
        """运行连续交易模式"""
        self.logger.info(f"启动连续交易模式，间隔: {interval_seconds}秒")

        try:
            while self.is_running:
                cycle_start = time.time()

                self.logger.info("=" * 50)
                self.logger.info(f"开始新的交易周期: {datetime.now()}")

                # 执行单次交易流程
                self.run_multi_market_trading()

                # 计算等待时间
                cycle_time = time.time() - cycle_start
                wait_time = max(0, interval_seconds - cycle_time)

                if wait_time > 0:
                    self.logger.info(f"等待 {wait_time:.1f} 秒后开始下一周期...")
                    time.sleep(wait_time)

        except KeyboardInterrupt:
            self.logger.info("收到中断信号，停止连续交易")
        except Exception as e:
            self.logger.error(f"连续交易异常: {e}")
            raise
        finally:
            self.is_running = False


def main():
    """主函数"""
    # 配置参数
    config = {
        'enable_a_share': True,
        'enable_h_share': True,
        'enable_us_share': True,
        'max_iterations': 50,
        'cv_folds': 3,
        'optimization_timeout': 1800,  # 30分钟
        'min_spread': 0.02,
        'max_position_size': 10000,
        'z_score_threshold': 2.0,
        'correlation_threshold': 0.8,
        'lookback_period': 60
    }

    # 创建主流程实例
    main_flow = MultiMarketTradingMainFlow(config)

    # 运行单次交易流程
    main_flow.run_multi_market_trading()

    # 或者运行连续交易模式（取消注释）
    # main_flow.run_continuous_trading(interval_seconds=300)  # 5分钟间隔


if __name__ == "__main__":
    main()
