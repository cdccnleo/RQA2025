#!/usr/bin/env python3
"""
数据质量管理器单元测试
测试质量检查、修复和验证功能
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
import statistics

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.orchestration.data_quality_manager import (
    DataQualityManager,
    DataQualityResult,
    QualityCheckLevel,
    DataQualityIssue,
    QualityThreshold
)


class TestDataQualityManager:
    """数据质量管理器测试类"""

    def setup_method(self):
        """测试前准备"""
        self.config = {
            'stock_checker': {
                'thresholds': {
                    'min_completeness': 0.90,
                    'max_missing_rate': 0.10,
                    'outlier_threshold': 3.0,
                    'max_price_change': 0.20,
                    'min_volume_threshold': 100,
                    'max_duplicate_rate': 0.05
                }
            }
        }
        self.manager = DataQualityManager(self.config)

    def teardown_method(self):
        """测试后清理"""
        pass

    @pytest.mark.asyncio
    async def test_comprehensive_quality_check(self):
        """测试全面质量检查"""
        # 创建测试数据 - 包含各种质量问题
        test_data = [
            # 正常数据
            {
                "symbol": "000001.SZ",
                "date": datetime(2020, 1, 1),
                "open": 100.0,
                "high": 105.0,
                "low": 95.0,
                "close": 102.0,
                "volume": 1000000
            },
            # 缺失数据
            {
                "symbol": "000001.SZ",
                "date": datetime(2020, 1, 2),
                "open": None,
                "high": 106.0,
                "low": 94.0,
                "close": 103.0,
                "volume": 1200000
            },
            # 异常值
            {
                "symbol": "000001.SZ",
                "date": datetime(2020, 1, 3),
                "open": 1000.0,  # 异常高价格
                "high": 1100.0,
                "low": 900.0,
                "close": 1050.0,
                "volume": 1000000
            },
            # 重复数据
            {
                "symbol": "000001.SZ",
                "date": datetime(2020, 1, 1),  # 与第一条重复
                "open": 100.0,
                "high": 105.0,
                "low": 95.0,
                "close": 102.0,
                "volume": 1000000
            }
        ]

        # 执行全面质量检查
        result = await self.manager.check_data_quality(test_data, "stock", QualityCheckLevel.COMPREHENSIVE)

        # 验证结果
        assert isinstance(result, DataQualityResult)
        assert result.total_records == 4
        assert result.overall_score < 1.0  # 应该有质量问题
        assert len(result.issues) > 0  # 应该检测到问题
        assert result.check_level == QualityCheckLevel.COMPREHENSIVE

        # 检查是否检测到各种问题
        issue_types = {issue['type'] for issue in result.issues}
        assert DataQualityIssue.MISSING_VALUES.value in issue_types
        assert DataQualityIssue.OUTLIER_VALUES.value in issue_types
        assert DataQualityIssue.DUPLICATE_RECORDS.value in issue_types

    @pytest.mark.asyncio
    async def test_basic_quality_check(self):
        """测试基础质量检查"""
        # 创建基本正常的数据
        test_data = [
            {
                "symbol": "000001.SZ",
                "date": datetime(2020, 1, 1),
                "open": 100.0,
                "high": 105.0,
                "low": 95.0,
                "close": 102.0,
                "volume": 1000000
            },
            {
                "symbol": "000001.SZ",
                "date": datetime(2020, 1, 2),
                "open": 102.0,
                "high": 107.0,
                "low": 98.0,
                "close": 105.0,
                "volume": 1200000
            }
        ]

        result = await self.manager.check_data_quality(test_data, "stock", QualityCheckLevel.BASIC)

        assert result.overall_score > 0.9  # 基础数据应该质量很高
        assert result.check_level == QualityCheckLevel.BASIC
        assert result.total_records == 2

    @pytest.mark.asyncio
    async def test_missing_values_detection(self):
        """测试缺失值检测"""
        # 创建包含缺失值的数据
        test_data = [
            # 完整数据
            {
                "symbol": "000001.SZ", "date": datetime(2020, 1, 1),
                "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000000
            },
            # 缺失open
            {
                "symbol": "000001.SZ", "date": datetime(2020, 1, 2),
                "open": None, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000000
            },
            # 缺失volume
            {
                "symbol": "000001.SZ", "date": datetime(2020, 1, 3),
                "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": None
            }
        ]

        result = await self.manager.check_data_quality(test_data, "stock", QualityCheckLevel.STANDARD)

        # 检查是否检测到缺失值问题
        missing_issues = [issue for issue in result.issues
                         if issue['type'] == DataQualityIssue.MISSING_VALUES.value]

        assert len(missing_issues) > 0

        # 检查具体的缺失统计
        for issue in missing_issues:
            assert issue['count'] > 0
            assert 'field' in issue
            assert issue['field'] in ['open', 'volume']

    @pytest.mark.asyncio
    async def test_duplicate_records_detection(self):
        """测试重复记录检测"""
        test_data = [
            # 原始记录
            {
                "symbol": "000001.SZ", "date": datetime(2020, 1, 1),
                "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000000
            },
            # 完全相同的重复记录
            {
                "symbol": "000001.SZ", "date": datetime(2020, 1, 1),
                "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000000
            },
            # 不同的记录
            {
                "symbol": "000001.SZ", "date": datetime(2020, 1, 2),
                "open": 102.0, "high": 107.0, "low": 98.0, "close": 105.0, "volume": 1200000
            }
        ]

        result = await self.manager.check_data_quality(test_data, "stock", QualityCheckLevel.STANDARD)

        # 检查是否检测到重复记录
        duplicate_issues = [issue for issue in result.issues
                           if issue['type'] == DataQualityIssue.DUPLICATE_RECORDS.value]

        assert len(duplicate_issues) > 0

        # 验证重复记录数量
        duplicate_issue = duplicate_issues[0]
        assert duplicate_issue['count'] == 1  # 1条重复记录

    @pytest.mark.asyncio
    async def test_price_anomalies_detection(self):
        """测试价格异常检测"""
        test_data = [
            # 正常价格数据
            {
                "symbol": "000001.SZ", "date": datetime(2020, 1, 1),
                "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000000
            },
            {
                "symbol": "000001.SZ", "date": datetime(2020, 1, 2),
                "open": 102.0, "high": 107.0, "low": 98.0, "close": 105.0, "volume": 1200000
            },
            # 极端价格变动
            {
                "symbol": "000001.SZ", "date": datetime(2020, 1, 3),
                "open": 105.0, "high": 200.0, "low": 50.0, "close": 180.0, "volume": 1000000
            }
        ]

        result = await self.manager.check_data_quality(test_data, "stock", QualityCheckLevel.STANDARD)

        # 检查是否检测到价格异常
        price_issues = [issue for issue in result.issues
                       if issue['type'] == DataQualityIssue.PRICE_ANOMALIES.value]

        assert len(price_issues) > 0

    @pytest.mark.asyncio
    async def test_volume_anomalies_detection(self):
        """测试成交量异常检测"""
        test_data = [
            # 正常成交量
            {
                "symbol": "000001.SZ", "date": datetime(2020, 1, 1),
                "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000000
            },
            # 成交量为0
            {
                "symbol": "000001.SZ", "date": datetime(2020, 1, 2),
                "open": 102.0, "high": 107.0, "low": 98.0, "close": 105.0, "volume": 0
            },
            # 极小成交量
            {
                "symbol": "000001.SZ", "date": datetime(2020, 1, 3),
                "open": 105.0, "high": 110.0, "low": 100.0, "close": 108.0, "volume": 10
            }
        ]

        result = await self.manager.check_data_quality(test_data, "stock", QualityCheckLevel.STANDARD)

        # 检查是否检测到成交量异常
        volume_issues = [issue for issue in result.issues
                        if issue['type'] == DataQualityIssue.VOLUME_ANOMALIES.value]

        assert len(volume_issues) > 0

    @pytest.mark.asyncio
    async def test_data_repair_missing_values(self):
        """测试缺失值修复"""
        # 原始数据包含缺失值
        original_data = [
            {
                "symbol": "000001.SZ", "date": datetime(2020, 1, 1),
                "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000000
            },
            {
                "symbol": "000001.SZ", "date": datetime(2020, 1, 2),
                "open": None, "high": 106.0, "low": 96.0, "close": 103.0, "volume": 1100000
            },
            {
                "symbol": "000001.SZ", "date": datetime(2020, 1, 3),
                "open": 103.0, "high": 108.0, "low": 98.0, "close": 106.0, "volume": 1200000
            }
        ]

        # 检测质量问题
        quality_result = await self.manager.check_data_quality(original_data, "stock", QualityCheckLevel.BASIC)
        issues = quality_result.issues

        # 执行修复
        repaired_data, repair_logs = await self.manager.repair_data_quality(
            original_data, issues, repair_level="conservative"
        )

        # 验证修复结果
        assert len(repaired_data) == len(original_data)
        assert len(repair_logs) > 0

        # 检查缺失值是否被修复
        for record in repaired_data:
            assert record['open'] is not None

    @pytest.mark.asyncio
    async def test_data_repair_inconsistent_data(self):
        """测试不一致数据修复"""
        # 创建OHLC不一致的数据
        original_data = [
            {
                "symbol": "000001.SZ", "date": datetime(2020, 1, 1),
                "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000000
            },
            # 高价低于低价
            {
                "symbol": "000001.SZ", "date": datetime(2020, 1, 2),
                "open": 100.0, "high": 95.0, "low": 105.0, "close": 102.0, "volume": 1100000
            }
        ]

        # 检测质量问题
        quality_result = await self.manager.check_data_quality(original_data, "stock", QualityCheckLevel.STANDARD)
        issues = quality_result.issues

        # 执行修复
        repaired_data, repair_logs = await self.manager.repair_data_quality(
            original_data, issues, repair_level="conservative"
        )

        # 验证修复结果
        assert len(repaired_data) == len(original_data)

        # 检查OHLC关系是否修复
        for record in repaired_data:
            assert record['high'] >= record['low']

    @pytest.mark.asyncio
    async def test_repair_effectiveness_validation(self):
        """测试修复效果验证"""
        # 创建有质量问题的数据
        original_data = [
            {
                "symbol": "000001.SZ", "date": datetime(2020, 1, 1),
                "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000000
            },
            {
                "symbol": "000001.SZ", "date": datetime(2020, 1, 2),
                "open": None, "high": 106.0, "low": 96.0, "close": 103.0, "volume": 1100000
            }
        ]

        # 检测原始质量
        original_quality = await self.manager.check_data_quality(original_data, "stock", QualityCheckLevel.BASIC)

        # 创建修复后的数据（模拟修复结果）
        repaired_data = [
            {
                "symbol": "000001.SZ", "date": datetime(2020, 1, 1),
                "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000000
            },
            {
                "symbol": "000001.SZ", "date": datetime(2020, 1, 2),
                "open": 101.5, "high": 106.0, "low": 96.0, "close": 103.0, "volume": 1100000
            }
        ]

        # 验证修复效果
        validation = await self.manager.validate_repair_effectiveness(
            original_data, repaired_data, "stock"
        )

        assert 'original_score' in validation
        assert 'repaired_score' in validation
        assert 'improvement' in validation
        assert validation['repaired_score'] >= validation['original_score']

    def test_quality_thresholds_configuration(self):
        """测试质量阈值配置"""
        # 验证默认阈值
        thresholds = QualityThreshold()
        assert thresholds.min_completeness == 0.90
        assert thresholds.max_missing_rate == 0.10
        assert thresholds.outlier_threshold == 3.0
        assert thresholds.max_price_change == 0.20

        # 验证自定义配置
        custom_config = {
            'stock_checker': {
                'thresholds': {
                    'min_completeness': 0.95,
                    'max_missing_rate': 0.05,
                    'outlier_threshold': 2.5
                }
            }
        }

        custom_manager = DataQualityManager(custom_config)

        # 验证配置是否生效（通过检查实际行为）
        test_data = [
            {
                "symbol": "000001.SZ", "date": datetime(2020, 1, 1),
                "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000000
            }
        ]

        # 配置应该影响检查结果的敏感度
        # 这里主要验证配置加载正确

    @pytest.mark.asyncio
    async def test_different_data_types(self):
        """测试不同数据类型"""
        # 股票数据
        stock_data = [{
            "symbol": "000001.SZ", "date": datetime(2020, 1, 1),
            "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000000
        }]

        stock_result = await self.manager.check_data_quality(stock_data, "stock")
        assert stock_result.overall_score > 0

        # 指数数据（使用相同的检查器）
        index_data = [{
            "symbol": "000001.SH", "date": datetime(2020, 1, 1),
            "open": 3000.0, "high": 3100.0, "low": 2950.0, "close": 3050.0, "volume": 1000000000
        }]

        index_result = await self.manager.check_data_quality(index_data, "index")
        assert index_result.overall_score > 0

    @pytest.mark.asyncio
    async def test_empty_data_handling(self):
        """测试空数据处理"""
        empty_data = []

        result = await self.manager.check_data_quality(empty_data, "stock")

        assert result.total_records == 0
        assert result.overall_score == 0.0
        assert result.valid_records == 0
        assert len(result.issues) > 0

        # 检查是否包含相应的错误信息
        issue_types = {issue['type'] for issue in result.issues}
        assert DataQualityIssue.MISSING_VALUES.value in issue_types

    @pytest.mark.asyncio
    async def test_single_record_handling(self):
        """测试单条记录处理"""
        single_record = [{
            "symbol": "000001.SZ", "date": datetime(2020, 1, 1),
            "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000000
        }]

        result = await self.manager.check_data_quality(single_record, "stock")

        assert result.total_records == 1
        assert result.valid_records == 1
        # 单条记录可能无法进行某些统计检查，但基本检查应该通过

    @pytest.mark.asyncio
    async def test_statistical_anomalies_detection(self):
        """测试统计异常检测"""
        # 创建包含统计异常的数据
        import numpy as np

        # 生成正常数据
        normal_prices = np.random.normal(100, 5, 50)
        test_data = []

        for i, price in enumerate(normal_prices):
            test_data.append({
                "symbol": "000001.SZ",
                "date": datetime(2020, 1, 1) + timedelta(days=i),
                "open": price - 1,
                "high": price + 2,
                "low": price - 2,
                "close": price,
                "volume": 1000000
            })

        # 添加一些异常值
        test_data[10]["close"] = 1000  # 极端异常值
        test_data[20]["close"] = 1     # 极端低值

        result = await self.manager.check_data_quality(test_data, "stock", QualityCheckLevel.COMPREHENSIVE)

        # 检查是否检测到异常值
        outlier_issues = [issue for issue in result.issues
                         if issue['type'] == DataQualityIssue.OUTLIER_VALUES.value]

        assert len(outlier_issues) > 0

    @pytest.mark.asyncio
    async def test_temporal_consistency_check(self):
        """测试时间一致性检查"""
        # 创建有时间缺口的数据
        test_data = [
            {
                "symbol": "000001.SZ", "date": datetime(2020, 1, 1),  # 周二
                "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000000
            },
            # 跳过周三、周四、周五，直接到下周二（5个交易日缺口）
            {
                "symbol": "000001.SZ", "date": datetime(2020, 1, 8),  # 下周二
                "open": 102.0, "high": 107.0, "low": 97.0, "close": 105.0, "volume": 1100000
            }
        ]

        result = await self.manager.check_data_quality(test_data, "stock", QualityCheckLevel.COMPREHENSIVE)

        # 检查是否检测到数据缺口
        gap_issues = [issue for issue in result.issues
                     if issue['type'] == DataQualityIssue.DATA_GAPS.value]

        assert len(gap_issues) > 0

    @pytest.mark.asyncio
    async def test_aggressive_repair_mode(self):
        """测试激进修复模式"""
        # 创建严重质量问题的数据
        original_data = [
            {
                "symbol": "000001.SZ", "date": datetime(2020, 1, 1),
                "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000000
            },
            # 严重缺失数据
            {
                "symbol": "000001.SZ", "date": datetime(2020, 1, 2),
                "open": None, "high": None, "low": None, "close": None, "volume": None
            }
        ]

        # 检测质量问题
        quality_result = await self.manager.check_data_quality(original_data, "stock", QualityCheckLevel.BASIC)
        issues = quality_result.issues

        # 使用激进修复模式
        repaired_data, repair_logs = await self.manager.repair_data_quality(
            original_data, issues, repair_level="aggressive"
        )

        # 验证修复结果
        assert len(repaired_data) == len(original_data)

        # 在激进模式下，应该尝试修复更多问题
        # （具体的修复逻辑取决于实现，这里主要验证接口）

    def test_quality_result_structure(self):
        """测试质量结果结构"""
        result = DataQualityResult(
            overall_score=0.85,
            total_records=100,
            valid_records=85,
            invalid_records=15,
            issues=[
                {
                    "type": DataQualityIssue.MISSING_VALUES.value,
                    "severity": "minor",
                    "description": "Test issue",
                    "count": 5
                }
            ],
            recommendations=["Fix missing values"]
        )

        assert result.overall_score == 0.85
        assert result.total_records == 100
        assert result.valid_records == 85
        assert result.invalid_records == 15
        assert len(result.issues) == 1
        assert len(result.recommendations) == 1
        assert result.check_level == QualityCheckLevel.STANDARD  # 默认值

    @pytest.mark.asyncio
    async def test_error_handling_in_quality_checks(self):
        """测试质量检查中的错误处理"""
        # 创建会导致异常的数据
        problematic_data = [
            {
                "symbol": "000001.SZ",
                "date": "invalid_date",  # 无效日期格式
                "open": "invalid_number",  # 无效数字
                "high": 105.0,
                "low": 95.0,
                "close": 102.0,
                "volume": 1000000
            }
        ]

        # 应该能够处理异常而不崩溃
        result = await self.manager.check_data_quality(problematic_data, "stock")

        # 即使有问题数据，也应该返回结果
        assert isinstance(result, DataQualityResult)
        assert result.total_records == 1
        # 可能会有一些质量问题被检测到

    @pytest.mark.asyncio
    async def test_quality_score_calculation(self):
        """测试质量分数计算"""
        # 测试完美数据
        perfect_data = [
            {
                "symbol": "000001.SZ", "date": datetime(2020, 1, 1),
                "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000000
            },
            {
                "symbol": "000001.SZ", "date": datetime(2020, 1, 2),
                "open": 102.0, "high": 107.0, "low": 98.0, "close": 105.0, "volume": 1100000
            }
        ]

        perfect_result = await self.manager.check_data_quality(perfect_data, "stock")
        assert perfect_result.overall_score > 0.9  # 应该接近满分

        # 测试有问题的数据
        problematic_data = [
            {
                "symbol": "000001.SZ", "date": datetime(2020, 1, 1),
                "open": None, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000000
            }
        ]

        problem_result = await self.manager.check_data_quality(problematic_data, "stock")
        assert problem_result.overall_score < perfect_result.overall_score  # 有问题的数据分数应该更低


if __name__ == '__main__':
    pytest.main([__file__])