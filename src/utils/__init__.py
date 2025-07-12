"""RQA2025通用工具库 - 跨模块共享功能

核心工具:
- date_utils: 日期时间处理
- math_utils: 数学计算
- data_utils: 数据转换
- logging_utils: 日志处理
- file_utils: 文件操作

使用示例:
    from src.utils import date_utils
    from src.utils.math_utils import calculate_returns

    # 日期处理
    biz_date = date_utils.get_business_date()

    # 计算收益率
    returns = calculate_returns(prices)

主要功能:
- 日期时间处理(支持A股交易日历)
- 数学和统计计算
- 数据格式转换
- 标准化日志记录
- 通用文件操作

注意事项:
1. 所有工具函数应保持无状态
2. 避免工具类之间的循环依赖
3. 保持函数功能单一明确

版本历史:
- v1.0 (2024-02-15): 初始版本
- v1.1 (2024-03-20): 添加A股交易日历支持
"""

from .date_utils import (
    get_business_date,
    is_trading_day,
    convert_timezone,
    next_trading_day,
    prev_trading_day,
    get_trading_days,
    is_trading_time
)
from .math_utils import (
    calculate_returns,
    annualized_volatility,
    sharpe_ratio
)
from .data_utils import (
    normalize_data,
    denormalize_data
)
from .logging_utils import setup_logging
from .logger import get_logger
from .file_utils import safe_file_write

__all__ = [
    # 日志工具
    'get_logger',
    # 日期工具
    'get_business_date',
    'is_trading_day',
    'convert_timezone',
    # 数学工具
    'calculate_returns',
    'annualized_volatility',
    'sharpe_ratio',
    # 数据工具
    'normalize_data',
    'denormalize_data',
    # 日志工具
    'setup_logging',
    # 文件工具
    'safe_file_write',
    # 子模块
    'date_utils',
    'math_utils',
    'data_utils',
    'logging_utils',
    'file_utils'
]
