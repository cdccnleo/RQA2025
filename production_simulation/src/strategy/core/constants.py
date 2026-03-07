"""
策略层常量定义
Strategy Layer Constants

定义策略相关的常量，避免魔法数字
"""

# 回测参数
DEFAULT_LOOKBACK_PERIOD = 252  # 默认回望期(交易日)
DEFAULT_COMMISSION_RATE = 0.003  # 默认佣金率
DEFAULT_SLIPPAGE_RATE = 0.001  # 默认滑点率
DEFAULT_RISK_FREE_RATE = 0.03  # 默认无风险利率(年化)

# 策略参数
DEFAULT_INITIAL_CAPITAL = 1000000  # 默认初始资金
DEFAULT_POSITION_SIZE = 0.1  # 默认仓位大小
DEFAULT_STOP_LOSS_PCT = 0.05  # 默认止损百分比
DEFAULT_TAKE_PROFIT_PCT = 0.10  # 默认止盈百分比

# 风险控制
MAX_DRAWDOWN_LIMIT = 0.20  # 最大回撤限制
MAX_POSITION_SIZE = 1.0  # 最大仓位大小
MIN_POSITION_SIZE = 0.0  # 最小仓位大小

# 性能指标
MIN_SHARPE_RATIO = 1.0  # 最小夏普比率
MIN_WIN_RATE = 0.50  # 最小胜率
MIN_PROFIT_FACTOR = 1.2  # 最小盈利因子

# 信号强度
STRONG_SIGNAL_THRESHOLD = 0.8  # 强信号阈值
WEAK_SIGNAL_THRESHOLD = 0.2  # 弱信号阈值

# 时间窗口
DEFAULT_SHORT_WINDOW = 10  # 默认短期窗口
DEFAULT_LONG_WINDOW = 50  # 默认长期窗口
DEFAULT_SIGNAL_WINDOW = 20  # 默认信号窗口

# 技术指标参数
RSI_OVERBOUGHT = 70  # RSI超买线
RSI_OVERSOLD = 30  # RSI超卖线
BB_MULTIPLIER = 2.0  # 布林带倍数

# 机器学习参数
DEFAULT_TRAIN_SIZE = 0.7  # 默认训练集比例
DEFAULT_VAL_SIZE = 0.2  # 默认验证集比例
DEFAULT_TEST_SIZE = 0.1  # 默认测试集比例

# 分布式计算
DEFAULT_WORKER_COUNT = 4  # 默认工作进程数
MAX_WORKER_COUNT = 16  # 最大工作进程数

# 缓存设置
STRATEGY_CACHE_SIZE = 100  # 策略缓存大小
SIGNAL_CACHE_SIZE = 1000  # 信号缓存大小
CACHE_TTL_SECONDS = 3600  # 缓存过期时间

# 告警阈值
ALERT_THRESHOLD_HIGH = 0.8  # 高风险告警阈值
ALERT_THRESHOLD_MEDIUM = 0.6  # 中风险告警阈值

# 性能监控
MONITOR_UPDATE_INTERVAL = 60  # 监控更新间隔(秒)
PERFORMANCE_CHECK_INTERVAL = 300  # 性能检查间隔(秒)

# 批量处理
DEFAULT_BATCH_SIZE = 1000  # 默认批处理大小
MAX_BATCH_SIZE = 10000  # 最大批处理大小

# 重试和超时
MAX_RETRY_ATTEMPTS = 3  # 最大重试次数
OPERATION_TIMEOUT = 300  # 操作超时时间(秒)

# 评估指标
CONFIDENCE_THRESHOLD = 0.7  # 置信度阈值
STABILITY_THRESHOLD = 0.8  # 稳定性阈值
