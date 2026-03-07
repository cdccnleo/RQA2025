"""
特征质量评分工具
提供基于特征类型的差异化评分
"""

import logging

logger = logging.getLogger(__name__)

# 特征类型到质量评分的映射
FEATURE_QUALITY_MAP = {
    # 趋势类 - 高稳定性
    'SMA': 0.90,
    'EMA': 0.90,
    'WMA': 0.88,

    # 动量类 - 良好效果
    'RSI': 0.85,
    'MACD': 0.85,
    'CCI': 0.83,

    # 波动率类 - 中等稳定性
    'BOLL': 0.80,
    'ATR': 0.80,

    # 复杂指标 - 参数敏感
    'KDJ': 0.82,
    'STOCH': 0.81,

    # 成交量类 - 受异常影响
    'OBV': 0.78,
    'VWAP': 0.78,
    'ADL': 0.77,

    # 默认评分
    'DEFAULT': 0.80
}


def get_feature_quality_score(feature_name: str) -> float:
    """
    根据特征名称获取质量评分

    Args:
        feature_name: 特征名称，如 "SMA_5", "RSI", "boll_upper"

    Returns:
        质量评分 (0.0-1.0)
    """
    # 提取基础特征名（去掉参数后缀）
    base_name = feature_name.split('_')[0].upper()

    # 处理 BOLL 的特殊情况（boll_upper, boll_middle, boll_lower）
    if base_name in ['BOLL', 'BOL']:
        return FEATURE_QUALITY_MAP.get('BOLL', FEATURE_QUALITY_MAP['DEFAULT'])

    # 处理 KDJ 的特殊情况（kdj_k, kdj_d, kdj_j）
    if base_name in ['KDJ']:
        return FEATURE_QUALITY_MAP.get('KDJ', FEATURE_QUALITY_MAP['DEFAULT'])

    # 处理 MACD 的特殊情况（macd_macd, macd_signal, macd_histogram）
    if base_name in ['MACD']:
        return FEATURE_QUALITY_MAP.get('MACD', FEATURE_QUALITY_MAP['DEFAULT'])

    # 从映射表获取评分
    score = FEATURE_QUALITY_MAP.get(base_name, FEATURE_QUALITY_MAP['DEFAULT'])

    logger.debug(f"特征 {feature_name} (类型: {base_name}) 的质量评分: {score}")

    return score


def get_quality_category(score: float) -> str:
    """
    根据评分获取质量等级

    Args:
        score: 质量评分

    Returns:
        质量等级描述
    """
    if score >= 0.9:
        return "优秀"
    elif score >= 0.8:
        return "良好"
    elif score >= 0.7:
        return "一般"
    elif score >= 0.6:
        return "较差"
    else:
        return "差"


def calculate_quality_scores(features: list) -> dict:
    """
    为特征列表计算质量评分

    Args:
        features: 特征名称列表

    Returns:
        特征名到质量评分的字典
    """
    scores = {feature: get_feature_quality_score(feature) for feature in features}

    # 统计各质量等级的特征数量
    category_counts = {}
    for score in scores.values():
        category = get_quality_category(score)
        category_counts[category] = category_counts.get(category, 0) + 1

    logger.info(f"计算了 {len(features)} 个特征的质量评分")
    logger.info(f"质量分布: {category_counts}")

    return scores


def calculate_final_quality_score(
    feature_name: str,
    data_quality_factor: float = 1.0
) -> float:
    """
    计算最终质量评分（结合基础评分和数据质量因子）

    公式: final_score = base_score × data_quality_factor

    Args:
        feature_name: 特征名称
        data_quality_factor: 数据质量因子 (0-1)，默认为1.0（无影响）

    Returns:
        最终质量评分 (0-1)

    Example:
        >>> calculate_final_quality_score("SMA_5", 0.95)
        0.855  # 0.90 * 0.95
        >>> calculate_final_quality_score("RSI", 0.88)
        0.748  # 0.85 * 0.88
    """
    # 获取基础评分
    base_score = get_feature_quality_score(feature_name)

    # 确保数据质量因子在合理范围内
    data_quality_factor = max(0.0, min(1.0, data_quality_factor))

    # 计算最终评分
    final_score = base_score * data_quality_factor

    # 四舍五入到3位小数
    final_score = round(final_score, 3)

    logger.debug(f"特征 {feature_name} 的最终质量评分: {final_score} "
                f"(基础: {base_score}, 质量因子: {data_quality_factor})")

    return final_score


def calculate_quality_scores_with_data_quality(
    features: list,
    data_quality_metrics: dict = None,
    user_id: str = None
) -> dict:
    """
    为特征列表计算质量评分（结合数据质量因子和用户自定义评分）

    Args:
        features: 特征名称列表
        data_quality_metrics: 数据质量指标字典 {特征名: DataQualityMetrics}
        user_id: 用户ID，用于获取自定义评分

    Returns:
        特征名到最终质量评分的字典
    """
    scores = {}

    for feature in features:
        # 获取数据质量因子
        if data_quality_metrics and feature in data_quality_metrics:
            quality_factor = data_quality_metrics[feature].overall_factor
        else:
            quality_factor = 1.0  # 默认无影响

        # 计算最终评分（支持用户自定义）
        scores[feature] = calculate_final_quality_score_with_custom(
            feature, quality_factor, user_id
        )

    # 统计各质量等级的特征数量
    category_counts = {}
    for score in scores.values():
        category = get_quality_category(score)
        category_counts[category] = category_counts.get(category, 0) + 1

    logger.info(f"计算了 {len(features)} 个特征的最终质量评分（含数据质量因子和用户自定义）")
    logger.info(f"质量分布: {category_counts}")

    return scores


def calculate_final_quality_score_with_custom(
    feature_name: str,
    data_quality_factor: float = 1.0,
    user_id: str = None
) -> float:
    """
    计算最终质量评分（支持用户自定义评分）

    优先级:
    1. 用户自定义评分（如果存在且有效）
    2. 系统默认评分 × 数据质量因子

    Args:
        feature_name: 特征名称
        data_quality_factor: 数据质量因子 (0-1)，默认为1.0（无影响）
        user_id: 用户ID，用于获取自定义评分

    Returns:
        最终质量评分 (0-1)
    """
    # 1. 检查用户自定义评分
    if user_id:
        try:
            from src.gateway.web.feature_quality_config import get_user_custom_score
            custom_score = get_user_custom_score(user_id, feature_name)
            if custom_score is not None:
                logger.debug(f"使用用户自定义评分: {feature_name} = {custom_score}")
                return custom_score
        except Exception as e:
            logger.warning(f"获取用户自定义评分失败: {e}")

    # 2. 使用系统默认评分 × 数据质量因子
    return calculate_final_quality_score(feature_name, data_quality_factor)


def get_feature_quality_score_with_custom(
    feature_name: str,
    user_id: str = None
) -> float:
    """
    获取特征质量评分（支持用户自定义，便捷函数）

    优先级:
    1. 用户自定义评分（如果存在且有效）
    2. 系统默认评分

    Args:
        feature_name: 特征名称
        user_id: 用户ID

    Returns:
        质量评分 (0-1)
    """
    return calculate_final_quality_score_with_custom(feature_name, 1.0, user_id)
