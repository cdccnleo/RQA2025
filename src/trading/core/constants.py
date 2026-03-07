"""
交易层常量定义
Trading Layer Constants

定义交易相关的常量，避免魔法数字
"""

# 订单参数
DEFAULT_ORDER_TIMEOUT = 300  # 默认订单超时时间(秒)
DEFAULT_SLIPPAGE_TOLERANCE = 0.001  # 默认滑点容忍度
MAX_ORDER_RETRIES = 3  # 最大订单重试次数

# 交易限制
MAX_ORDERS_PER_SECOND = 100  # 每秒最大订单数
MAX_POSITION_SIZE = 1000000  # 最大持仓大小
MIN_ORDER_SIZE = 1  # 最小订单大小

# 风险控制
DEFAULT_STOP_LOSS_PCT = 0.05  # 默认止损百分比
DEFAULT_TAKE_PROFIT_PCT = 0.10  # 默认止盈百分比
MAX_DAILY_LOSS_PCT = 0.02  # 最大日损失百分比

# 手续费率
DEFAULT_COMMISSION_RATE = 0.003  # 默认佣金率
DEFAULT_MARKET_IMPACT_COST = 0.001  # 默认市场冲击成本

# 执行参数
DEFAULT_EXECUTION_TIMEOUT = 60  # 默认执行超时时间(秒)
EXECUTION_CHECK_INTERVAL = 1  # 执行检查间隔(秒)

# 连接参数
CONNECTION_TIMEOUT = 30  # 连接超时时间(秒)
RECONNECT_ATTEMPTS = 5  # 重连尝试次数
HEARTBEAT_INTERVAL = 30  # 心跳间隔(秒)

# 缓存设置
ORDER_CACHE_SIZE = 10000  # 订单缓存大小
POSITION_CACHE_SIZE = 1000  # 持仓缓存大小
CACHE_TTL_SECONDS = 3600  # 缓存过期时间

# 批量处理
DEFAULT_BATCH_SIZE = 100  # 默认批处理大小
MAX_BATCH_SIZE = 1000  # 最大批处理大小

# 监控阈值
ORDER_PROCESSING_TIME_THRESHOLD = 5  # 订单处理时间阈值(秒)
EXECUTION_LATENCY_THRESHOLD = 100  # 执行延迟阈值(毫秒)

# 资金参数
DEFAULT_LEVERAGE = 1.0  # 默认杠杆
MAX_LEVERAGE = 10.0  # 最大杠杆

# 市场数据
MARKET_DATA_TIMEOUT = 10  # 市场数据超时时间(秒)
PRICE_PRECISION = 4  # 价格精度
VOLUME_PRECISION = 0  # 成交量精度

# 报告参数
REPORT_UPDATE_INTERVAL = 60  # 报告更新间隔(秒)
PERFORMANCE_CHECK_INTERVAL = 300  # 性能检查间隔(秒)

# 告警阈值
ALERT_THRESHOLD_HIGH = 0.8  # 高风险告警阈值
ALERT_THRESHOLD_MEDIUM = 0.6  # 中风险告警阈值

# 系统限制
MAX_ACTIVE_ORDERS = 1000  # 最大活跃订单数
MAX_OPEN_POSITIONS = 100  # 最大未平仓位数
