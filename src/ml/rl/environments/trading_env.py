#!/usr/bin/env python3
"""
RQA2025交易强化学习环境
实现Gym风格的交易环境接口
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging

from .base_env import BaseTradingEnv, EnvConfig, TradeAction

logger = logging.getLogger(__name__)


class TradingEnvironment(BaseTradingEnv):
    """交易强化学习环境"""

    def __init__(self, config: Optional[EnvConfig] = None):
        """
        初始化交易环境

        Args:
            config: 环境配置，如果为None则使用默认配置
        """
        if config is None:
            config = EnvConfig()

        super().__init__(config)

        # 环境特定的状态
        self.observation_space_shape = self.get_observation_space_shape()
        self.action_space_size = self.get_action_space_size()

        # 渲染相关
        self.render_data = []

        logger.info("交易强化学习环境初始化完成")

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        重置环境到初始状态

        Args:
            seed: 随机种子

        Returns:
            初始观察值
        """
        if seed is not None:
            np.random.seed(seed)

        # 重置环境状态
        self.current_step = 0
        self.done = False

        # 重置投资组合
        self.portfolio = type(self.portfolio)(
            cash=self.config.initial_balance,
            position=0.0,
            entry_price=None,
            total_value=self.config.initial_balance,
            unrealized_pnl=0.0,
            realized_pnl=0.0
        )

        # 加载市场数据
        self.market_data = self._load_market_data()
        self.current_market_state = self._get_current_market_state()

        # 清空历史记录
        self.trade_history.clear()
        self.portfolio_history.clear()

        # 记录初始状态
        self.portfolio_history.append(self.portfolio)

        # 获取初始观察
        observation = self._get_observation()

        logger.info(f"环境重置完成 - 初始现金: ${self.portfolio.cash:,.2f}")

        return {
            'observation': observation,
            'info': {
                'portfolio_value': self.portfolio.cash,
                'position': self.portfolio.position,
                'current_price': self.current_market_state.price if self.current_market_state else 0
            }
        }

    def step(self, action: Union[int, TradeAction]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        执行一步动作

        Args:
            action: 动作 (整数索引或TradeAction对象)

        Returns:
            (observation, reward, done, info)
        """
        # 转换动作格式
        if isinstance(action, int):
            trade_action = self.action_to_trade(action)
        elif isinstance(action, TradeAction):
            trade_action = action
        else:
            raise ValueError(f"不支持的动作类型: {type(action)}")

        # 记录上一步的组合价值
        prev_portfolio_value = self._calculate_portfolio_value()

        # 执行交易
        trade_result = self._execute_trade(trade_action)

        # 前进一步
        self.current_step += 1

        # 更新市场状态
        if self.current_step < len(self.market_data):
            self.current_market_state = self._get_current_market_state()
        else:
            self.done = True

        # 计算当前组合价值
        current_portfolio_value = self._calculate_portfolio_value()

        # 计算奖励
        transaction_cost = trade_result.get(
            'commission', 0.0) if trade_result.get('success', False) else 0.0
        reward = self._calculate_reward(
            prev_portfolio_value, current_portfolio_value, transaction_cost)

        # 记录奖励
        self.episode_rewards.append(reward)

        # 更新投资组合历史
        self.portfolio.total_value = current_portfolio_value
        if self.current_market_state and self.portfolio.entry_price:
            position_value = self.portfolio.position * self.current_market_state.price
            self.portfolio.unrealized_pnl = position_value - \
                (self.portfolio.position * self.portfolio.entry_price)

        self.portfolio_history.append(self.portfolio)

        # 检查是否结束
        if self.current_step >= self.episode_length - 1:
            self.done = True
            # 计算总回报
            total_return = (current_portfolio_value - self.config.initial_balance) / \
                self.config.initial_balance
            self.episode_returns.append(total_return)

        # 获取观察
        observation = self._get_observation()

        # 构建info
        info = {
            'portfolio_value': current_portfolio_value,
            'cash': self.portfolio.cash,
            'position': self.portfolio.position,
            'current_price': self.current_market_state.price if self.current_market_state else 0,
            'trade_executed': trade_result.get('success', False),
            'transaction_cost': transaction_cost,
            'step': self.current_step,
            'total_trades': len(self.trade_history)
        }

        # 渲染数据收集
        if self.config.render_mode:
            self.render_data.append({
                'step': self.current_step,
                'portfolio_value': current_portfolio_value,
                'cash': self.portfolio.cash,
                'position': self.portfolio.position,
                'price': self.current_market_state.price if self.current_market_state else 0,
                'reward': reward,
                'action': trade_action.action_type,
                'quantity_pct': trade_action.quantity_pct
            })

        return observation, reward, self.done, info

    def render(self, mode: str = "human") -> Optional[Any]:
        """
        渲染环境状态

        Args:
            mode: 渲染模式

        Returns:
            渲染结果
        """
        if mode == "human":
            if not self.current_market_state:
                print("环境未初始化")
                return

            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${self._calculate_portfolio_value():,.2f}")
            print(f"Cash: ${self.portfolio.cash:,.2f}")
            print(f"Position: {self.portfolio.position:.2f} shares")
            print(f"Current Price: ${self.current_market_state.price:.2f}")
            print(f"Unrealized P&L: ${self.portfolio.unrealized_pnl:.2f}")
            print(f"Total Trades: {len(self.trade_history)}")
            print("-" * 50)

        elif mode == "rgb_array":
            # 返回渲染图像 (需要matplotlib)
            return self._render_chart()

        return None

    def _get_observation(self) -> np.ndarray:
        """获取当前观察状态"""
        if not self.current_market_state:
            # 返回零向量
            return np.zeros(self.observation_space_shape[0])

        # 使用AI特征作为观察状态
        features = self.current_market_state.features

        if not features:
            # 如果没有特征，返回零向量
            return np.zeros(self.observation_space_shape[0])

        # 转换为numpy数组
        observation = np.array(list(features.values()))

        # 确保维度正确
        if len(observation) < self.observation_space_shape[0]:
            # 填充零值
            padding = np.zeros(self.observation_space_shape[0] - len(observation))
            observation = np.concatenate([observation, padding])
        elif len(observation) > self.observation_space_shape[0]:
            # 截断
            observation = observation[:self.observation_space_shape[0]]

        return observation.astype(np.float32)

    def _render_chart(self) -> Optional[np.ndarray]:
        """渲染价格和投资组合表现图表"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.figure import Figure

            fig = Figure(figsize=(12, 8))

            # 子图1: 价格走势和持仓
            ax1 = fig.add_subplot(2, 1, 1)
            if self.market_data is not None and len(self.market_data) > 0:
                prices = self.market_data['close'].values[:self.current_step + 1]
                ax1.plot(prices, label='Price', color='blue')

                # 标记买入点
                buy_points = []
                sell_points = []
                for trade in self.trade_history:
                    if trade['type'] == 'buy':
                        buy_points.append((len(buy_points) + len(sell_points), trade['price']))
                    elif trade['type'] == 'sell':
                        sell_points.append((len(buy_points) + len(sell_points), trade['price']))

                if buy_points:
                    buy_x, buy_y = zip(*buy_points)
                    ax1.scatter(buy_x, buy_y, color='green', marker='^', s=100, label='Buy')

                if sell_points:
                    sell_x, sell_y = zip(*sell_points)
                    ax1.scatter(sell_x, sell_y, color='red', marker='v', s=100, label='Sell')

            ax1.set_title('Price Chart with Trading Signals')
            ax1.legend()
            ax1.grid(True)

            # 子图2: 投资组合价值
            ax2 = fig.add_subplot(2, 1, 2)
            if self.portfolio_history:
                values = [p.total_value for p in self.portfolio_history]
                ax2.plot(values, label='Portfolio Value', color='purple')

                # 添加基准线 (初始价值)
                ax2.axhline(y=self.config.initial_balance, color='gray',
                            linestyle='--', label='Initial Balance')

            ax2.set_title('Portfolio Value Over Time')
            ax2.legend()
            ax2.grid(True)

            # 转换为numpy数组
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            plt.close(fig)
            return image

        except ImportError:
            logger.warning("matplotlib未安装，无法渲染图表")
            return None
        except Exception as e:
            logger.error(f"渲染图表失败: {e}")
            return None

    def get_action_descriptions(self) -> List[str]:
        """获取动作描述"""
        return [
            "持有 (Hold)",
            "小仓位买入 (Buy 20%)",
            "中仓位买入 (Buy 50%)",
            "全仓位买入 (Buy 100%)",
            "小仓位卖出 (Sell 20%)",
            "中仓位卖出 (Sell 50%)",
            "全仓位卖出 (Sell 100%)"
        ]

    def get_state_descriptions(self) -> Dict[str, str]:
        """获取状态描述"""
        return {
            'portfolio_value': '投资组合总价值',
            'cash': '可用现金',
            'position': '持仓数量',
            'current_price': '当前价格',
            'unrealized_pnl': '未实现盈亏',
            'realized_pnl': '已实现盈亏',
            'total_trades': '总交易次数'
        }


class MultiAssetTradingEnvironment(TradingEnvironment):
    """多资产交易环境"""

    def __init__(self, config: EnvConfig, symbols: List[str]):
        super().__init__(config)
        self.symbols = symbols
        self.current_symbol_idx = 0

        # 为每个资产维护单独的组合状态
        self.portfolio_per_asset = {
            symbol: type(self.portfolio)(
                cash=config.initial_balance / len(symbols),
                position=0.0
            )
            for symbol in symbols
        }

    def step(self, action: Union[int, Dict[str, Union[int, TradeAction]]]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        多资产环境步进

        Args:
            action: 每个资产的动作字典或单个动作
        """
        if isinstance(action, dict):
            # 多资产动作
            total_reward = 0
            total_info = {}

            for symbol, symbol_action in action.items():
                if symbol in self.symbols:
                    # 暂时切换到该资产的环境
                    original_symbol = self.config.symbol
                    self.config.symbol = symbol

                    _, reward, _, info = super().step(symbol_action)
                    total_reward += reward

                    # 合并info
                    for key, value in info.items():
                        if key in total_info:
                            total_info[key] += value
                        else:
                            total_info[key] = value

                    # 恢复原始symbol
                    self.config.symbol = original_symbol

            total_reward /= len(action)  # 平均奖励

        else:
            # 单资产动作
            _, total_reward, _, total_info = super().step(action)

        # 检查episode结束条件
        if self.current_step >= self.episode_length - 1:
            self.done = True

        # 获取当前观察 (所有资产的组合观察)
        observation = self._get_multi_asset_observation()

        return observation, total_reward, self.done, total_info

    def _get_multi_asset_observation(self) -> np.ndarray:
        """获取多资产观察状态"""
        observations = []

        for symbol in self.symbols:
            # 暂时切换symbol获取观察
            original_symbol = self.config.symbol
            self.config.symbol = symbol

            obs = self._get_observation()
            observations.append(obs)

            # 恢复原始symbol
            self.config.symbol = original_symbol

        # 拼接所有资产的观察
        return np.concatenate(observations)


def create_trading_env(symbol: str = "AAPL",
                       initial_balance: float = 100000.0,
                       episode_length: int = 1000) -> TradingEnvironment:
    """
    创建交易环境便捷函数

    Args:
        symbol: 交易标的
        initial_balance: 初始资金
        episode_length: 每个episode的长度

    Returns:
        交易环境实例
    """
    config = EnvConfig(
        symbol=symbol,
        initial_balance=initial_balance,
        episode_length=episode_length
    )

    return TradingEnvironment(config)


if __name__ == "__main__":
    # 测试交易环境
    print("🧠 测试RQA2025交易强化学习环境")
    print("=" * 50)

    # 创建环境
    env = create_trading_env("AAPL", 100000.0, 100)
    print("✅ 交易环境创建成功")

    # 重置环境
    reset_result = env.reset(seed=42)
    print("✅ 环境重置完成")
    print(f"初始观察维度: {reset_result['observation'].shape}")
    print(f"初始组合价值: ${reset_result['info']['portfolio_value']:,.2f}")

    # 执行一些随机动作
    total_reward = 0
    for step in range(10):
        # 随机选择动作
        action = np.random.randint(env.action_space_size)

        observation, reward, done, info = env.step(action)
        total_reward += reward

        if step < 3:  # 只打印前几步
            action_desc = env.get_action_descriptions()[action]
            print(
                f"步骤 {step + 1}: 动作={action_desc}, 奖励={reward:.4f}, 组合价值=${info['portfolio_value']:,.2f}")

        if done:
            break

    print(f"\n总奖励: {total_reward:.4f}")
    print(f"最终组合价值: ${info['portfolio_value']:,.2f}")
    print(f"总交易次数: {info['total_trades']}")

    # 获取性能指标
    metrics = env.get_performance_metrics()
    if metrics:
        print("\n📊 性能指标:")
        print(",.2f")
        print(".2f")

    print("\n🎉 交易强化学习环境测试完成！")
