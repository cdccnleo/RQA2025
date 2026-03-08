"""
工具模块（顶层别名）
提供向后兼容的导入路径
"""

# 从infrastructure.utils导入工具
try:
    # 只导入明确存在的函数，避免导入错误
    from src.infrastructure.utils.tools import (
        Convert, get_business_date, is_trading_day, convert_timezone,
        calculate_returns, annualized_volatility, sharpe_ratio,
        normalize_data, denormalize_data, safe_file_write, FileSystem,
        MarketAwareRetry
    )
    # 单独导入parse_datetime，如果存在的话
    try:
        from src.infrastructure.utils.tools import parse_datetime
    except ImportError:
        pass
except ImportError:
    # 从各个子模块单独导入
    try:
        from src.infrastructure.utils.tools.date_utils import *
        from src.infrastructure.utils.tools.convert import *
        from src.infrastructure.utils.tools.math_utils import *
        from src.infrastructure.utils.tools.data_utils import *
        from src.infrastructure.utils.tools.file_utils import *
        from src.infrastructure.utils.tools.market_aware_retry import *
    except ImportError:
        pass

__all__ = []

