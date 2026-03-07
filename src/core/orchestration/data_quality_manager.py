#!/usr/bin/env python3
"""
数据质量保证机制
提供多层次的数据质量检查、修复和验证
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable
import statistics
import numpy as np

logger = logging.getLogger(__name__)


class QualityCheckLevel(Enum):
    """质量检查级别"""
    BASIC = "basic"         # 基础检查：数据完整性、格式正确性
    STANDARD = "standard"   # 标准检查：加上合理性验证
    COMPREHENSIVE = "comprehensive"  # 全面检查：加上统计分析和异常检测


class DataQualityIssue(Enum):
    """数据质量问题类型"""
    MISSING_VALUES = "missing_values"           # 缺失值
    INVALID_FORMAT = "invalid_format"          # 格式错误
    OUTLIER_VALUES = "outlier_values"          # 异常值
    INCONSISTENT_DATA = "inconsistent_data"    # 数据不一致
    DUPLICATE_RECORDS = "duplicate_records"    # 重复记录
    INVALID_DATE_SEQUENCE = "invalid_date_sequence"  # 日期序列问题
    PRICE_ANOMALIES = "price_anomalies"        # 价格异常
    VOLUME_ANOMALIES = "volume_anomalies"      # 成交量异常
    DATA_GAPS = "data_gaps"                    # 数据缺口


@dataclass
class DataQualityResult:
    """数据质量检查结果"""
    overall_score: float = 0.0
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    issues: List[Dict[str, Any]] = field(default_factory=list)
    check_timestamp: datetime = field(default_factory=datetime.now)
    check_level: QualityCheckLevel = QualityCheckLevel.STANDARD
    recommendations: List[str] = field(default_factory=list)


@dataclass
class QualityThreshold:
    """质量阈值配置"""
    min_completeness: float = 0.90      # 最小完整度
    max_missing_rate: float = 0.10      # 最大缺失率
    outlier_threshold: float = 3.0      # 异常值阈值（标准差倍数）
    max_price_change: float = 0.20      # 最大价格变动率（20%）
    min_volume_threshold: int = 100     # 最小成交量阈值
    max_duplicate_rate: float = 0.05    # 最大重复率


class QualityChecker(ABC):
    """质量检查器基类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    async def check_quality(self, data: List[Dict[str, Any]], check_level: QualityCheckLevel) -> DataQualityResult:
        """检查数据质量"""
        pass

    def calculate_overall_score(self, issues: List[Dict[str, Any]], total_records: int) -> float:
        """计算总体质量分数"""
        if total_records == 0:
            return 0.0

        # 根据问题严重程度计算扣分
        total_penalty = 0.0
        severity_weights = {
            "critical": 0.5,    # 严重问题，每个扣0.5分
            "major": 0.2,       # 主要问题，每个扣0.2分
            "minor": 0.05,      # 次要问题，每个扣0.05分
            "warning": 0.01     # 警告，每个扣0.01分
        }

        for issue in issues:
            severity = issue.get('severity', 'warning')
            weight = severity_weights.get(severity, 0.01)
            total_penalty += weight

        # 基础分数100分，扣分制
        base_score = 1.0
        final_score = max(0.0, base_score - total_penalty)

        return final_score


class StockDataQualityChecker(QualityChecker):
    """股票数据质量检查器"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.thresholds = QualityThreshold(**config.get('thresholds', {}))

    async def check_quality(self, data: List[Dict[str, Any]], check_level: QualityCheckLevel) -> DataQualityResult:
        """检查股票数据质量"""
        if not data:
            return DataQualityResult(
                overall_score=0.0,
                total_records=0,
                valid_records=0,
                invalid_records=0,
                issues=[{
                    "type": DataQualityIssue.MISSING_VALUES.value,
                    "severity": "critical",
                    "description": "无数据记录",
                    "count": 0
                }]
            )

        result = DataQualityResult(
            total_records=len(data),
            check_level=check_level
        )

        issues = []

        # 基础检查
        issues.extend(await self._check_basic_integrity(data))

        # 标准检查
        if check_level in [QualityCheckLevel.STANDARD, QualityCheckLevel.COMPREHENSIVE]:
            issues.extend(await self._check_data_consistency(data))
            issues.extend(await self._check_price_anomalies(data))
            issues.extend(await self._check_volume_anomalies(data))

        # 全面检查
        if check_level == QualityCheckLevel.COMPREHENSIVE:
            issues.extend(await self._check_statistical_anomalies(data))
            issues.extend(await self._check_temporal_consistency(data))

        result.issues = issues
        result.overall_score = self.calculate_overall_score(issues, len(data))
        result.valid_records = len(data) - sum(issue.get('count', 0) for issue in issues if issue.get('severity') in ['critical', 'major'])
        result.invalid_records = result.total_records - result.valid_records

        # 生成修复建议
        result.recommendations = self._generate_recommendations(issues, result.overall_score)

        return result

    async def _check_basic_integrity(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """检查基础完整性"""
        issues = []

        # 检查必需字段
        required_fields = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
        missing_counts = {field: 0 for field in required_fields}

        for record in data:
            for field in required_fields:
                if record.get(field) is None or record.get(field) == '':
                    missing_counts[field] += 1

        # 报告缺失值问题
        for field, count in missing_counts.items():
            if count > 0:
                missing_rate = count / len(data)
                severity = "critical" if missing_rate > 0.1 else "major" if missing_rate > 0.05 else "minor"

                issues.append({
                    "type": DataQualityIssue.MISSING_VALUES.value,
                    "severity": severity,
                    "field": field,
                    "description": f"字段 '{field}' 缺失 {count} 条记录 ({missing_rate:.1%})",
                    "count": count,
                    "percentage": missing_rate
                })

        # 检查数据类型
        type_issues = 0
        for record in data:
            try:
                # 价格字段应该是数值
                for field in ['open', 'high', 'low', 'close']:
                    if record.get(field) is not None:
                        float(record[field])

                # 成交量应该是整数
                if record.get('volume') is not None:
                    int(float(record['volume']))

            except (ValueError, TypeError):
                type_issues += 1

        if type_issues > 0:
            issues.append({
                "type": DataQualityIssue.INVALID_FORMAT.value,
                "severity": "major" if type_issues > len(data) * 0.05 else "minor",
                "description": f"数据类型错误 {type_issues} 条记录",
                "count": type_issues
            })

        return issues

    async def _check_data_consistency(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """检查数据一致性"""
        issues = []

        # 检查OHLC关系：最高价 >= 最低价，收盘价在范围内
        consistency_issues = 0
        for record in data:
            try:
                high = float(record.get('high', 0))
                low = float(record.get('low', 0))
                open_price = float(record.get('open', 0))
                close = float(record.get('close', 0))

                if high < low or open_price < low or open_price > high or close < low or close > high:
                    consistency_issues += 1

            except (ValueError, TypeError):
                continue

        if consistency_issues > 0:
            issues.append({
                "type": DataQualityIssue.INCONSISTENT_DATA.value,
                "severity": "major",
                "description": f"OHLC价格关系不一致 {consistency_issues} 条记录",
                "count": consistency_issues
            })

        # 检查重复记录
        seen_records = set()
        duplicate_count = 0

        for record in data:
            # 使用(symbol, date)作为唯一键
            key = (record.get('symbol'), record.get('date'))
            if key in seen_records:
                duplicate_count += 1
            else:
                seen_records.add(key)

        if duplicate_count > 0:
            duplicate_rate = duplicate_count / len(data)
            severity = "major" if duplicate_rate > 0.1 else "minor"

            issues.append({
                "type": DataQualityIssue.DUPLICATE_RECORDS.value,
                "severity": severity,
                "description": f"发现 {duplicate_count} 条重复记录 ({duplicate_rate:.1%})",
                "count": duplicate_count
            })

        return issues

    async def _check_price_anomalies(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """检查价格异常"""
        issues = []

        # 计算价格变动率
        prices = []
        for record in data:
            try:
                close = float(record.get('close', 0))
                if close > 0:
                    prices.append(close)
            except (ValueError, TypeError):
                continue

        if len(prices) < 2:
            return issues

        # 计算日收益率
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(ret)

        if not returns:
            return issues

        # 检查极端价格变动
        extreme_changes = 0
        for ret in returns:
            if abs(ret) > self.thresholds.max_price_change:
                extreme_changes += 1

        if extreme_changes > 0:
            change_rate = extreme_changes / len(returns)
            severity = "major" if change_rate > 0.05 else "minor"

            issues.append({
                "type": DataQualityIssue.PRICE_ANOMALIES.value,
                "severity": severity,
                "description": f"极端价格变动 {extreme_changes} 次 ({change_rate:.1%})",
                "count": extreme_changes
            })

        # 使用IQR方法检测异常值
        if len(prices) > 10:
            try:
                q1 = np.percentile(prices, 25)
                q3 = np.percentile(prices, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                outlier_count = sum(1 for p in prices if p < lower_bound or p > upper_bound)

                if outlier_count > 0:
                    outlier_rate = outlier_count / len(prices)
                    if outlier_rate > 0.05:  # 5%以上为异常
                        issues.append({
                            "type": DataQualityIssue.OUTLIER_VALUES.value,
                            "severity": "minor",
                            "description": f"价格异常值 {outlier_count} 个 ({outlier_rate:.1%})",
                            "count": outlier_count
                        })

            except Exception as e:
                self.logger.warning(f"价格异常值检测失败: {e}")

        return issues

    async def _check_volume_anomalies(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """检查成交量异常"""
        issues = []

        volumes = []
        for record in data:
            try:
                volume = int(float(record.get('volume', 0)))
                if volume >= 0:
                    volumes.append(volume)
            except (ValueError, TypeError):
                continue

        if not volumes:
            return issues

        # 检查成交量为0的记录
        zero_volume_count = volumes.count(0)
        if zero_volume_count > 0:
            zero_rate = zero_volume_count / len(volumes)
            if zero_rate > 0.1:  # 超过10%为异常
                issues.append({
                    "type": DataQualityIssue.VOLUME_ANOMALIES.value,
                    "severity": "major",
                    "description": f"成交量为0的记录 {zero_volume_count} 条 ({zero_rate:.1%})",
                    "count": zero_volume_count
                })

        # 检查极小成交量
        small_volume_count = sum(1 for v in volumes if 0 < v < self.thresholds.min_volume_threshold)
        if small_volume_count > 0:
            small_rate = small_volume_count / len(volumes)
            if small_rate > 0.2:  # 超过20%为异常
                issues.append({
                    "type": DataQualityIssue.VOLUME_ANOMALIES.value,
                    "severity": "minor",
                    "description": f"成交量过小的记录 {small_volume_count} 条 ({small_rate:.1%})",
                    "count": small_volume_count
                })

        return issues

    async def _check_statistical_anomalies(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """检查统计异常"""
        issues = []

        # 分析价格分布
        prices = []
        for record in data:
            try:
                close = float(record.get('close', 0))
                if close > 0:
                    prices.append(close)
            except (ValueError, TypeError):
                continue

        if len(prices) < 30:  # 需要足够的样本
            return issues

        try:
            # 计算统计指标
            mean_price = statistics.mean(prices)
            stdev_price = statistics.stdev(prices)

            # 检查价格分布是否正常
            skewness = self._calculate_skewness(prices)
            if abs(skewness) > 2.0:  # 严重偏斜
                issues.append({
                    "type": DataQualityIssue.OUTLIER_VALUES.value,
                    "severity": "minor",
                    "description": f"价格分布严重偏斜 (skewness: {skewness:.2f})",
                    "count": len(prices),
                    "metric": "skewness"
                })

        except Exception as e:
            self.logger.warning(f"统计异常检查失败: {e}")

        return issues

    async def _check_temporal_consistency(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """检查时间一致性"""
        issues = []

        # 按日期排序
        sorted_data = sorted(data, key=lambda x: x.get('date', ''))

        # 检查日期连续性
        gaps = []
        for i in range(1, len(sorted_data)):
            try:
                current_date = datetime.strptime(sorted_data[i]['date'], '%Y-%m-%d').date()
                prev_date = datetime.strptime(sorted_data[i-1]['date'], '%Y-%m-%d').date()

                # 计算日期差（跳过周末）
                date_diff = (current_date - prev_date).days
                weekdays_diff = self._count_weekdays(prev_date, current_date)

                if weekdays_diff > 1:  # 存在缺口
                    gaps.append({
                        'start_date': prev_date,
                        'end_date': current_date,
                        'gap_days': date_diff,
                        'trading_gap_days': weekdays_diff - 1
                    })

            except (ValueError, KeyError, TypeError):
                continue

        if gaps:
            total_gap_days = sum(gap['trading_gap_days'] for gap in gaps)
            gap_rate = total_gap_days / max(len(sorted_data), 1)

            if gap_rate > 0.1:  # 缺口率超过10%
                issues.append({
                    "type": DataQualityIssue.DATA_GAPS.value,
                    "severity": "major",
                    "description": f"数据存在 {len(gaps)} 个缺口，总计 {total_gap_days} 个交易日 ({gap_rate:.1%})",
                    "count": len(gaps),
                    "total_gap_days": total_gap_days
                })

        return issues

    def _calculate_skewness(self, data: List[float]) -> float:
        """计算偏度"""
        if len(data) < 3:
            return 0.0

        mean = statistics.mean(data)
        stdev = statistics.stdev(data)

        if stdev == 0:
            return 0.0

        return sum(((x - mean) / stdev) ** 3 for x in data) / len(data)

    def _count_weekdays(self, start_date: datetime.date, end_date: datetime.date) -> int:
        """计算两个日期之间的工作日数量"""
        count = 0
        current = start_date
        while current <= end_date:
            if current.weekday() < 5:  # 周一到周五
                count += 1
            current += timedelta(days=1)
        return count

    def _generate_recommendations(self, issues: List[Dict[str, Any]], overall_score: float) -> List[str]:
        """生成修复建议"""
        recommendations = []

        if overall_score < 0.5:
            recommendations.append("数据质量严重不足，建议重新采集")
        elif overall_score < 0.7:
            recommendations.append("数据质量一般，建议进行数据清洗和修复")
        elif overall_score < 0.9:
            recommendations.append("数据质量良好，建议进行小幅修复")

        # 根据具体问题生成建议
        for issue in issues:
            issue_type = issue.get('type')
            severity = issue.get('severity')

            if issue_type == DataQualityIssue.MISSING_VALUES.value and severity in ['critical', 'major']:
                recommendations.append("建议补充缺失数据或使用数据插补方法")
            elif issue_type == DataQualityIssue.PRICE_ANOMALIES.value:
                recommendations.append("建议检查价格数据来源，移除异常价格点")
            elif issue_type == DataQualityIssue.DATA_GAPS.value:
                recommendations.append("建议检查数据源，补充缺失的交易日数据")

        return recommendations


class DataQualityManager:
    """数据质量管理器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 质量检查器映射
        self.checkers = {
            'stock': StockDataQualityChecker(config.get('stock_checker', {})),
            'index': StockDataQualityChecker(config.get('index_checker', {})),  # 复用股票检查器
            'fund': StockDataQualityChecker(config.get('fund_checker', {})),
        }

    async def check_data_quality(self, data: List[Dict[str, Any]], data_type: str,
                               check_level: QualityCheckLevel = QualityCheckLevel.STANDARD) -> DataQualityResult:
        """
        检查数据质量

        Args:
            data: 数据记录列表
            data_type: 数据类型
            check_level: 检查级别

        Returns:
            质量检查结果
        """
        if data_type not in self.checkers:
            # 使用默认检查器
            checker = StockDataQualityChecker({})
        else:
            checker = self.checkers[data_type]

        return await checker.check_quality(data, check_level)

    async def repair_data_quality(self, data: List[Dict[str, Any]], issues: List[Dict[str, Any]],
                                repair_level: str = "conservative") -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        修复数据质量问题

        Args:
            data: 原始数据
            issues: 质量问题列表
            repair_level: 修复级别 (conservative/aggressive)

        Returns:
            修复后的数据和修复日志
        """
        repaired_data = data.copy()
        repair_logs = []

        # 按严重程度排序修复
        sorted_issues = sorted(issues, key=lambda x: self._get_severity_priority(x.get('severity', 'warning')), reverse=True)

        for issue in sorted_issues:
            issue_type = issue.get('type')

            if issue_type == DataQualityIssue.MISSING_VALUES.value:
                repaired_data, logs = await self._repair_missing_values(repaired_data, issue, repair_level)
                repair_logs.extend(logs)

            elif issue_type == DataQualityIssue.OUTLIER_VALUES.value:
                repaired_data, logs = await self._repair_outliers(repaired_data, issue, repair_level)
                repair_logs.extend(logs)

            elif issue_type == DataQualityIssue.INCONSISTENT_DATA.value:
                repaired_data, logs = await self._repair_inconsistent_data(repaired_data, issue, repair_level)
                repair_logs.extend(logs)

        return repaired_data, repair_logs

    def _get_severity_priority(self, severity: str) -> int:
        """获取严重程度优先级"""
        priorities = {
            'critical': 4,
            'major': 3,
            'minor': 2,
            'warning': 1
        }
        return priorities.get(severity, 0)

    async def _repair_missing_values(self, data: List[Dict[str, Any]], issue: Dict[str, Any],
                                   repair_level: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        """修复缺失值"""
        logs = []
        field = issue.get('field', '')

        if repair_level == 'conservative':
            # 保守修复：只修复少量缺失值
            missing_rate = issue.get('percentage', 0)
            if missing_rate < 0.05:  # 少于5%的缺失
                # 使用前向填充
                repaired_count = 0
                last_value = None

                for record in data:
                    if record.get(field) is None and last_value is not None:
                        record[field] = last_value
                        repaired_count += 1

                    if record.get(field) is not None:
                        last_value = record[field]

                if repaired_count > 0:
                    logs.append(f"前向填充修复缺失值 {repaired_count} 条 ({field})")

        return data, logs

    async def _repair_outliers(self, data: List[Dict[str, Any]], issue: Dict[str, Any],
                             repair_level: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        """修复异常值"""
        logs = []

        if repair_level == 'conservative':
            # 保守修复：使用移动平均替换异常值
            field = 'close'  # 主要针对收盘价

            prices = []
            indices = []

            for i, record in enumerate(data):
                try:
                    price = float(record.get(field, 0))
                    if price > 0:
                        prices.append(price)
                        indices.append(i)
                except (ValueError, TypeError):
                    continue

            if len(prices) > 10:
                # 计算移动平均和标准差
                window_size = min(20, len(prices) // 5)

                repaired_count = 0
                for i in range(window_size, len(prices) - window_size):
                    window_prices = prices[i-window_size:i+window_size+1]
                    mean_price = statistics.mean(window_prices)
                    stdev_price = statistics.stdev(window_prices)

                    current_price = prices[i]
                    z_score = abs(current_price - mean_price) / stdev_price if stdev_price > 0 else 0

                    if z_score > 3.0:  # 3倍标准差外的异常值
                        # 替换为移动平均
                        original_index = indices[i]
                        data[original_index][field] = mean_price
                        repaired_count += 1

                if repaired_count > 0:
                    logs.append(f"移动平均修复异常值 {repaired_count} 条 ({field})")

        return data, logs

    async def _repair_inconsistent_data(self, data: List[Dict[str, Any]], issue: Dict[str, Any],
                                      repair_level: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        """修复不一致数据"""
        logs = []

        if repair_level == 'conservative':
            # 修复OHLC关系
            repaired_count = 0

            for record in data:
                try:
                    high = float(record.get('high', 0))
                    low = float(record.get('low', 0))
                    open_price = float(record.get('open', 0))
                    close = float(record.get('close', 0))

                    # 确保最高价 >= 最低价
                    if high < low:
                        # 交换高低价
                        record['high'], record['low'] = low, high
                        repaired_count += 1

                    # 确保开盘价在范围内
                    if open_price < low or open_price > high:
                        record['open'] = (high + low) / 2  # 使用平均价
                        repaired_count += 1

                    # 确保收盘价在范围内
                    if close < low or close > high:
                        record['close'] = (high + low) / 2  # 使用平均价
                        repaired_count += 1

                except (ValueError, TypeError):
                    continue

            if repaired_count > 0:
                logs.append(f"修复OHLC关系不一致 {repaired_count} 条记录")

        return data, logs

    async def validate_repair_effectiveness(self, original_data: List[Dict[str, Any]],
                                          repaired_data: List[Dict[str, Any]],
                                          data_type: str) -> Dict[str, Any]:
        """
        验证修复效果

        Args:
            original_data: 原始数据
            repaired_data: 修复后的数据
            data_type: 数据类型

        Returns:
            验证结果
        """
        # 检查原始数据质量
        original_quality = await self.check_data_quality(original_data, data_type)

        # 检查修复后数据质量
        repaired_quality = await self.check_data_quality(repaired_data, data_type)

        # 计算改善程度
        improvement = repaired_quality.overall_score - original_quality.overall_score

        return {
            'original_score': original_quality.overall_score,
            'repaired_score': repaired_quality.overall_score,
            'improvement': improvement,
            'improvement_percentage': (improvement / original_quality.overall_score) if original_quality.overall_score > 0 else 0,
            'issues_resolved': len(original_quality.issues) - len(repaired_quality.issues)
        }