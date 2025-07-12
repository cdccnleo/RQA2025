from datetime import datetime, time
from typing import Dict, List, Any

def validate_trading_hours(config: Dict[str, Any]) -> bool:
    """验证交易时段配置

    Args:
        config: 配置字典，包含trading_hours字段

    Returns:
        bool: True表示验证通过，False表示验证失败
    """
    if "trading_hours" not in config:
        return False

    trading_hours = config["trading_hours"]
    if not isinstance(trading_hours, dict):
        return False

    # 收集所有时间段
    time_slots = []
    for period_name, time_range in trading_hours.items():
        if not isinstance(time_range, list) or len(time_range) != 2:
            return False

        try:
            start = datetime.strptime(time_range[0], "%H:%M").time()
            end = datetime.strptime(time_range[1], "%H:%M").time()
            time_slots.append((start, end))
        except (ValueError, TypeError):
            return False

    # 检查时间段是否重叠
    time_slots.sort()  # 按开始时间排序
    for i in range(1, len(time_slots)):
        prev_end = time_slots[i-1][1]
        curr_start = time_slots[i][0]
        if curr_start < prev_end:
            return False

    return True
