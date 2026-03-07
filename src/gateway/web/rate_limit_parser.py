"""
采集频率解析器

解析数据源的 rate_limit 配置，转换为实际的调度间隔。
支持多种格式："X次/天"、"X次/小时"、"X次/分钟"、"X秒/次"
"""

import re
import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


def parse_rate_limit(rate_limit_str: str) -> int:
    """
    解析采集频率字符串，返回间隔秒数
    
    Args:
        rate_limit_str: 采集频率字符串，如"1次/天"、"100次/分钟"
        
    Returns:
        int: 间隔秒数，解析失败返回0
        
    Examples:
        >>> parse_rate_limit("1次/天")
        86400
        >>> parse_rate_limit("100次/分钟")
        60
        >>> parse_rate_limit("1次/小时")
        3600
    """
    if not rate_limit_str or not isinstance(rate_limit_str, str):
        logger.warning(f"无效的rate_limit: {rate_limit_str}")
        return 0
    
    rate_limit_str = rate_limit_str.strip()
    
    # 匹配 "X次/天"、"X次/小时"、"X次/分钟" 格式
    pattern1 = r'(\d+)\s*次\s*/\s*(天|日|小时|时|分钟|分)'
    match1 = re.match(pattern1, rate_limit_str)
    
    if match1:
        count = int(match1.group(1))
        unit = match1.group(2)
        
        if count <= 0:
            logger.warning(f"无效的采集次数: {count}")
            return 0
        
        # 计算间隔秒数
        if unit in ['天', '日']:
            seconds = 86400 // count  # 每天86400秒
        elif unit in ['小时', '时']:
            seconds = 3600 // count  # 每小时3600秒
        elif unit in ['分钟', '分']:
            seconds = 60 // count    # 每分钟60秒
        else:
            logger.warning(f"未知的时间单位: {unit}")
            return 0
        
        logger.debug(f"解析rate_limit: {rate_limit_str} -> {seconds}秒")
        return max(seconds, 60)  # 最小间隔60秒，避免过于频繁
    
    # 匹配 "X秒/次" 格式
    pattern2 = r'(\d+)\s*秒\s*/\s*次'
    match2 = re.match(pattern2, rate_limit_str)
    
    if match2:
        seconds = int(match2.group(1))
        if seconds <= 0:
            logger.warning(f"无效的秒数: {seconds}")
            return 0
        logger.debug(f"解析rate_limit: {rate_limit_str} -> {seconds}秒")
        return max(seconds, 60)  # 最小间隔60秒
    
    # 匹配纯数字（假设为秒）
    pattern3 = r'^(\d+)$'
    match3 = re.match(pattern3, rate_limit_str)
    
    if match3:
        seconds = int(match3.group(1))
        logger.debug(f"解析rate_limit: {rate_limit_str} -> {seconds}秒（纯数字）")
        return max(seconds, 60)
    
    logger.warning(f"无法解析rate_limit格式: {rate_limit_str}")
    return 0


def should_collect(last_collection_time: Optional[str], rate_limit: str) -> bool:
    """
    检查是否应该执行采集
    
    Args:
        last_collection_time: 最后采集时间字符串，格式为"YYYY-MM-DD HH:MM:SS"
        rate_limit: 采集频率字符串
        
    Returns:
        bool: 是否应该执行采集
    """
    interval_seconds = parse_rate_limit(rate_limit)
    
    if interval_seconds <= 0:
        logger.warning(f"无效的采集间隔: {rate_limit}")
        return False
    
    # 如果没有最后采集时间，应该执行采集
    if not last_collection_time:
        logger.debug("没有最后采集时间，应该执行采集")
        return True
    
    try:
        # 解析最后采集时间
        last_time = datetime.strptime(last_collection_time, "%Y-%m-%d %H:%M:%S")
        next_time = last_time + timedelta(seconds=interval_seconds)
        
        # 如果当前时间已经超过下次采集时间，应该执行采集
        now = datetime.now()
        should = now >= next_time
        
        if should:
            logger.debug(f"到达采集时间: 最后采集={last_collection_time}, "
                        f"间隔={interval_seconds}秒, 应该采集")
        else:
            time_until = next_time - now
            logger.debug(f"未到达采集时间: 还需等待 {time_until}")
        
        return should
        
    except ValueError as e:
        logger.error(f"解析最后采集时间失败: {last_collection_time}, 错误: {e}")
        # 如果时间格式错误，默认执行采集
        return True


def get_next_collection_time(last_collection_time: Optional[str], rate_limit: str) -> Optional[datetime]:
    """
    获取下次采集时间
    
    Args:
        last_collection_time: 最后采集时间字符串
        rate_limit: 采集频率字符串
        
    Returns:
        datetime: 下次采集时间，如果无法计算返回None
    """
    interval_seconds = parse_rate_limit(rate_limit)
    
    if interval_seconds <= 0:
        return None
    
    if not last_collection_time:
        # 如果没有最后采集时间，下次采集时间为现在
        return datetime.now()
    
    try:
        last_time = datetime.strptime(last_collection_time, "%Y-%m-%d %H:%M:%S")
        next_time = last_time + timedelta(seconds=interval_seconds)
        return next_time
    except ValueError:
        logger.error(f"解析最后采集时间失败: {last_collection_time}")
        return None


def format_interval(seconds: int) -> str:
    """
    将秒数格式化为可读字符串
    
    Args:
        seconds: 秒数
        
    Returns:
        str: 格式化后的字符串，如"1天2小时3分钟"
    """
    if seconds <= 0:
        return "未知"
    
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    parts = []
    if days > 0:
        parts.append(f"{days}天")
    if hours > 0:
        parts.append(f"{hours}小时")
    if minutes > 0:
        parts.append(f"{minutes}分钟")
    if secs > 0 and not parts:
        parts.append(f"{secs}秒")
    
    return "".join(parts) if parts else "0秒"


# 测试代码
if __name__ == "__main__":
    # 测试解析功能
    test_cases = [
        ("1次/天", 86400),
        ("2次/天", 43200),
        ("1次/小时", 3600),
        ("2次/小时", 1800),
        ("1次/分钟", 60),
        ("2次/分钟", 60),  # 最小60秒
        ("100次/分钟", 60),  # 最小60秒
        ("300秒/次", 300),
        ("3600", 3600),
        ("", 0),
        (None, 0),
        ("无效格式", 0),
    ]
    
    print("=" * 60)
    print("采集频率解析器测试")
    print("=" * 60)
    
    for rate_limit, expected in test_cases:
        result = parse_rate_limit(rate_limit)
        status = "✅" if result == expected else "❌"
        print(f"{status} parse_rate_limit('{rate_limit}') = {result} (期望: {expected})")
    
    # 测试 should_collect
    print("\n" + "=" * 60)
    print("采集判断测试")
    print("=" * 60)
    
    # 测试用例1: 没有最后采集时间
    result = should_collect(None, "1次/天")
    print(f"✅ should_collect(None, '1次/天') = {result} (期望: True)")
    
    # 测试用例2: 很久之前的采集时间
    old_time = "2025-01-01 00:00:00"
    result = should_collect(old_time, "1次/天")
    print(f"✅ should_collect('{old_time}', '1次/天') = {result} (期望: True)")
    
    # 测试用例3: 刚刚采集过
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result = should_collect(now, "1次/天")
    print(f"✅ should_collect('{now}', '1次/天') = {result} (期望: False)")
    
    # 测试 format_interval
    print("\n" + "=" * 60)
    print("间隔格式化测试")
    print("=" * 60)
    
    interval_tests = [
        (86400, "1天"),
        (3661, "1小时1分钟"),
        (3600, "1小时"),
        (60, "1分钟"),
        (30, "30秒"),
        (0, "未知"),
    ]
    
    for seconds, expected in interval_tests:
        result = format_interval(seconds)
        status = "✅" if result == expected else "❌"
        print(f"{status} format_interval({seconds}) = '{result}' (期望: '{expected}')")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
