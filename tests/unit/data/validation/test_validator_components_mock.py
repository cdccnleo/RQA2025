# -*- coding: utf-8 -*-
"""
数据验证器组件Mock测试
测试数据验证、质量检查、异常检测等功能
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class MockValidationResult:
    """模拟验证结果"""

    def __init__(self, is_valid: bool = True, metrics: Optional[Dict[str, float]] = None,
                 errors: Optional[List[str]] = None, timestamp: Optional[str] = None,
                 data_type: str = "unknown"):
        self.is_valid = is_valid
        self.metrics = metrics or {}
        self.errors = errors or []
        self.timestamp = timestamp or datetime.now().isoformat()
        self.data_type = data_type

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "is_valid": self.is_valid,
            "metrics": self.metrics,
            "errors": self.errors,
            "timestamp": self.timestamp,
            "data_type": self.data_type
        }


@dataclass
class MockQualityReport:
    """模拟质量报告"""

    def __init__(self, overall_score: float = 1.0, completeness: float = 1.0,
                 accuracy: float = 1.0, consistency: float = 1.0, timeliness: float = 1.0,
                 details: Optional[Dict[str, Any]] = None):
        self.overall_score = overall_score
        self.completeness = completeness
        self.accuracy = accuracy
        self.consistency = consistency
        self.timeliness = timeliness
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "overall_score": self.overall_score,
            "completeness": self.completeness,
            "accuracy": self.accuracy,
            "consistency": self.consistency,
            "timeliness": self.timeliness,
            "details": self.details
        }


@dataclass
class MockOutlierReport:
    """模拟离群值报告"""

    def __init__(self, outlier_count: int = 0, outlier_percentage: float = 0.0,
                 outlier_indices: Optional[List[int]] = None,
                 outlier_values: Optional[List[Any]] = None, threshold: float = 3.0):
        self.outlier_count = outlier_count
        self.outlier_percentage = outlier_percentage
        self.outlier_indices = outlier_indices or []
        self.outlier_values = outlier_values or []
        self.threshold = threshold

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "outlier_count": self.outlier_count,
            "outlier_percentage": self.outlier_percentage,
            "outlier_indices": self.outlier_indices,
            "outlier_values": self.outlier_values,
            "threshold": self.threshold
        }


class MockDataValidator:
    """模拟数据验证器"""

    def __init__(self, validation_rules: Optional[Dict[str, Any]] = None):
        self.validation_rules = validation_rules or {}
        self.is_initialized = False
        self.validation_count = 0
        self.error_count = 0
        self.logger = Mock()
        self.logger.info = Mock()
        self.logger.warning = Mock()
        self.logger.error = Mock()

    def initialize(self) -> bool:
        """初始化验证器"""
        try:
            self.is_initialized = True
            self.logger.info("DataValidator initialized")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize validator: {e}")
            return False

    def validate_data(self, data: Any, data_type: str = "unknown") -> MockValidationResult:
        """验证数据"""
        if not self.is_initialized:
            raise Exception("Validator not initialized")

        self.validation_count += 1
        errors = []
        metrics = {}

        try:
            # 基本验证
            if data is None:
                errors.append("Data is None")
            elif isinstance(data, dict):
                errors.extend(self._validate_dict_data(data))
                metrics.update(self._calculate_dict_metrics(data))
            elif isinstance(data, list):
                errors.extend(self._validate_list_data(data))
                metrics.update(self._calculate_list_metrics(data))
            elif isinstance(data, pd.DataFrame):
                errors.extend(self._validate_dataframe_data(data))
                metrics.update(self._calculate_dataframe_metrics(data))
            elif isinstance(data, np.ndarray):
                errors.extend(self._validate_array_data(data))
                metrics.update(self._calculate_array_metrics(data))
            else:
                errors.append(f"Unsupported data type: {type(data)}")

            if errors:
                self.error_count += 1

            return MockValidationResult(
                is_valid=len(errors) == 0,
                metrics=metrics,
                errors=errors,
                data_type=data_type
            )
        except Exception as e:
            self.error_count += 1
            return MockValidationResult(
                is_valid=False,
                metrics={},
                errors=[str(e)],
                data_type=data_type
            )

    def _validate_dict_data(self, data: Dict[str, Any]) -> List[str]:
        """验证字典数据"""
        errors = []
        if not data:
            errors.append("Empty dictionary")

        # 检查必需字段
        required_fields = self.validation_rules.get("required_fields", [])
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")

        return errors

    def _validate_list_data(self, data: List[Any]) -> List[str]:
        """验证列表数据"""
        errors = []
        if not data:
            errors.append("Empty list")

        min_length = self.validation_rules.get("min_length", 0)
        if len(data) < min_length:
            errors.append(f"List length {len(data)} below minimum {min_length}")

        return errors

    def _validate_dataframe_data(self, data: pd.DataFrame) -> List[str]:
        """验证DataFrame数据"""
        errors = []
        if data.empty:
            errors.append("Empty DataFrame")

        # 检查必需列
        required_columns = self.validation_rules.get("required_columns", [])
        for col in required_columns:
            if col not in data.columns:
                errors.append(f"Missing required column: {col}")

        # 检查数据类型
        for col, expected_type in self.validation_rules.get("column_types", {}).items():
            if col in data.columns:
                if not data[col].dtype.name.startswith(expected_type):
                    errors.append(f"Column {col} has wrong type: {data[col].dtype}, expected {expected_type}")

        return errors

    def _validate_array_data(self, data: np.ndarray) -> List[str]:
        """验证数组数据"""
        errors = []
        if data.size == 0:
            errors.append("Empty array")

        # 检查维度
        expected_shape = self.validation_rules.get("expected_shape")
        if expected_shape and data.shape != tuple(expected_shape):
            errors.append(f"Array shape {data.shape} doesn't match expected {expected_shape}")

        return errors

    def _calculate_dict_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """计算字典数据的指标"""
        return {
            "field_count": len(data),
            "null_value_count": sum(1 for v in data.values() if v is None),
            "completeness": 1.0 - (sum(1 for v in data.values() if v is None) / max(1, len(data)))
        }

    def _calculate_list_metrics(self, data: List[Any]) -> Dict[str, float]:
        """计算列表数据的指标"""
        return {
            "item_count": len(data),
            "null_item_count": sum(1 for item in data if item is None),
            "completeness": 1.0 - (sum(1 for item in data if item is None) / max(1, len(data)))
        }

    def _calculate_dataframe_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """计算DataFrame数据的指标"""
        total_cells = data.size
        null_cells = data.isnull().sum().sum()

        return {
            "row_count": len(data),
            "column_count": len(data.columns),
            "total_cells": total_cells,
            "null_cells": null_cells,
            "completeness": 1.0 - (null_cells / max(1, total_cells))
        }

    def _calculate_array_metrics(self, data: np.ndarray) -> Dict[str, float]:
        """计算数组数据的指标"""
        return {
            "total_elements": data.size,
            "null_elements": np.count_nonzero(pd.isna(data)),
            "completeness": 1.0 - (np.count_nonzero(pd.isna(data)) / max(1, data.size))
        }

    def get_statistics(self) -> Dict[str, Any]:
        """获取验证统计"""
        return {
            "validation_count": self.validation_count,
            "error_count": self.error_count,
            "success_rate": (self.validation_count - self.error_count) / max(1, self.validation_count),
            "error_rate": self.error_count / max(1, self.validation_count)
        }


class MockQualityAssessor:
    """模拟质量评估器"""

    def __init__(self, quality_thresholds: Optional[Dict[str, float]] = None):
        self.quality_thresholds = quality_thresholds or {
            "completeness": 0.95,
            "accuracy": 0.95,
            "consistency": 0.90,
            "timeliness": 0.95
        }
        self.assessment_count = 0
        self.logger = Mock()

    def assess_quality(self, data: Any, data_type: str = "unknown") -> MockQualityReport:
        """评估数据质量"""
        self.assessment_count += 1

        try:
            # 计算各项质量指标
            completeness = self._calculate_completeness(data)
            accuracy = self._calculate_accuracy(data)
            consistency = self._calculate_consistency(data)
            timeliness = self._calculate_timeliness(data)

            # 计算综合得分
            overall_score = (completeness + accuracy + consistency + timeliness) / 4.0

            details = {
                "completeness_score": completeness,
                "accuracy_score": accuracy,
                "consistency_score": consistency,
                "timeliness_score": timeliness,
                "thresholds": self.quality_thresholds,
                "data_type": data_type,
                "assessment_time": datetime.now().isoformat()
            }

            return MockQualityReport(
                overall_score=overall_score,
                completeness=completeness,
                accuracy=accuracy,
                consistency=consistency,
                timeliness=timeliness,
                details=details
            )
        except Exception as e:
            self.logger.error(f"Quality assessment failed: {e}")
            return MockQualityReport(
                overall_score=0.0,
                completeness=0.0,
                accuracy=0.0,
                consistency=0.0,
                timeliness=0.0,
                details={"error": str(e)}
            )

    def _calculate_completeness(self, data: Any) -> float:
        """计算完整性"""
        try:
            if isinstance(data, pd.DataFrame):
                total_cells = data.size
                null_cells = data.isnull().sum().sum()
                return 1.0 - (null_cells / max(1, total_cells))
            elif isinstance(data, dict):
                total_fields = len(data)
                null_fields = sum(1 for v in data.values() if v is None or v == "")
                return 1.0 - (null_fields / max(1, total_fields))
            elif isinstance(data, list):
                total_items = len(data)
                null_items = sum(1 for item in data if item is None)
                return 1.0 - (null_items / max(1, total_items))
            else:
                return 0.8  # 默认值
        except:
            return 0.0

    def _calculate_accuracy(self, data: Any) -> float:
        """计算准确性"""
        try:
            # 简单的准确性检查：数据类型一致性
            if isinstance(data, pd.DataFrame):
                # 检查数值列是否有异常值
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                accuracy_score = 0.9  # 基础分数

                for col in numeric_columns:
                    # 检查是否有无穷大或NaN
                    if data[col].isin([np.inf, -np.inf, np.nan]).any():
                        accuracy_score -= 0.1

                return max(0.0, accuracy_score)
            else:
                return 0.85  # 默认准确性分数
        except:
            return 0.0

    def _calculate_consistency(self, data: Any) -> float:
        """计算一致性"""
        try:
            if isinstance(data, pd.DataFrame):
                # 检查数据类型一致性
                consistency_score = 1.0
                for col in data.columns:
                    try:
                        # 尝试转换为统一类型
                        pd.to_numeric(data[col], errors='coerce')
                    except:
                        consistency_score -= 0.05
                return max(0.0, consistency_score)
            else:
                return 0.9  # 默认一致性分数
        except:
            return 0.0

    def _calculate_timeliness(self, data: Any) -> float:
        """计算及时性"""
        try:
            # 简单的及时性检查：数据时间范围
            if isinstance(data, pd.DataFrame) and 'timestamp' in data.columns:
                # 检查数据的新鲜度
                latest_time = pd.to_datetime(data['timestamp']).max()
                now = datetime.now()
                age_hours = (now - latest_time).total_seconds() / 3600

                # 如果数据在24小时内，给予高分
                if age_hours < 24:
                    return 0.95
                elif age_hours < 72:
                    return 0.85
                else:
                    return 0.7
            else:
                return 0.9  # 默认及时性分数
        except:
            return 0.0


class MockOutlierDetector:
    """模拟离群值检测器"""

    def __init__(self, method: str = "zscore", threshold: float = 3.0):
        self.method = method
        self.threshold = threshold
        self.detection_count = 0
        self.logger = Mock()

    def detect_outliers(self, data: Any, column: Optional[str] = None) -> MockOutlierReport:
        """检测离群值"""
        self.detection_count += 1

        try:
            if isinstance(data, pd.DataFrame):
                if column and column in data.columns:
                    return self._detect_series_outliers(data[column])
                else:
                    # 对所有数值列进行检测
                    all_outliers = []
                    for col in data.select_dtypes(include=[np.number]).columns:
                        col_report = self._detect_series_outliers(data[col])
                        all_outliers.extend(col_report.outlier_indices)

                    # 合并结果
                    unique_indices = list(set(all_outliers))
                    unique_indices.sort()

                    outlier_count = len(unique_indices)
                    total_count = len(data)

                    return MockOutlierReport(
                        outlier_count=outlier_count,
                        outlier_percentage=outlier_count / max(1, total_count),
                        outlier_indices=unique_indices,
                        outlier_values=[],  # 简化起见
                        threshold=self.threshold
                    )
            elif isinstance(data, (list, np.ndarray, pd.Series)):
                if isinstance(data, pd.Series):
                    series = data
                else:
                    series = pd.Series(data)
                return self._detect_series_outliers(series)
            else:
                return MockOutlierReport(threshold=self.threshold)
        except Exception as e:
            self.logger.error(f"Outlier detection failed: {e}")
            return MockOutlierReport(threshold=self.threshold)

    def _detect_series_outliers(self, series: pd.Series) -> MockOutlierReport:
        """检测序列中的离群值"""
        try:
            if self.method == "zscore":
                return self._zscore_detection(series)
            elif self.method == "iqr":
                return self._iqr_detection(series)
            else:
                return self._zscore_detection(series)
        except:
            return MockOutlierReport()

    def _zscore_detection(self, series: pd.Series) -> MockOutlierReport:
        """Z-Score离群值检测"""
        try:
            # 计算Z-Score
            mean_val = series.mean()
            std_val = series.std()

            if std_val == 0 or pd.isna(std_val):
                return MockOutlierReport(threshold=self.threshold)

            z_scores = np.abs((series - mean_val) / std_val)

            # 找出离群值
            outlier_mask = z_scores > self.threshold
            outlier_indices = series[outlier_mask].index.tolist()
            outlier_values = series[outlier_mask].values.tolist()

            return MockOutlierReport(
                outlier_count=len(outlier_indices),
                outlier_percentage=len(outlier_indices) / max(1, len(series)),
                outlier_indices=outlier_indices,
                outlier_values=outlier_values,
                threshold=self.threshold
            )
        except Exception:
            return MockOutlierReport(threshold=self.threshold)

    def _iqr_detection(self, series: pd.Series) -> MockOutlierReport:
        """IQR离群值检测"""
        try:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1

            if pd.isna(IQR) or IQR == 0:
                return MockOutlierReport(threshold=self.threshold)

            lower_bound = Q1 - (self.threshold * IQR)
            upper_bound = Q3 + (self.threshold * IQR)

            outlier_mask = (series < lower_bound) | (series > upper_bound)
            outlier_indices = series[outlier_mask].index.tolist()
            outlier_values = series[outlier_mask].values.tolist()

            return MockOutlierReport(
                outlier_count=len(outlier_indices),
                outlier_percentage=len(outlier_indices) / max(1, len(series)),
                outlier_indices=outlier_indices,
                outlier_values=outlier_values,
                threshold=self.threshold
            )
        except Exception:
            return MockOutlierReport(threshold=self.threshold)


class TestMockValidationResult:
    """模拟验证结果测试"""

    def test_validation_result_creation(self):
        """测试验证结果创建"""
        result = MockValidationResult(
            is_valid=True,
            metrics={"completeness": 0.95},
            errors=[],
            data_type="stock_data"
        )

        assert result.is_valid is True
        assert result.metrics["completeness"] == 0.95
        assert result.errors == []
        assert result.data_type == "stock_data"

    def test_validation_result_to_dict(self):
        """测试验证结果序列化"""
        result = MockValidationResult(is_valid=False, errors=["Missing data"])
        data = result.to_dict()

        assert data["is_valid"] is False
        assert "Missing data" in data["errors"]
        assert "timestamp" in data


class TestMockQualityReport:
    """模拟质量报告测试"""

    def test_quality_report_creation(self):
        """测试质量报告创建"""
        report = MockQualityReport(
            overall_score=0.85,
            completeness=0.90,
            accuracy=0.80,
            consistency=0.85,
            timeliness=0.90
        )

        assert report.overall_score == 0.85
        assert report.completeness == 0.90
        assert report.accuracy == 0.80
        assert report.consistency == 0.85
        assert report.timeliness == 0.90

    def test_quality_report_to_dict(self):
        """测试质量报告序列化"""
        report = MockQualityReport(overall_score=0.75)
        data = report.to_dict()

        assert data["overall_score"] == 0.75
        assert all(key in data for key in ["completeness", "accuracy", "consistency", "timeliness"])


class TestMockOutlierReport:
    """模拟离群值报告测试"""

    def test_outlier_report_creation(self):
        """测试离群值报告创建"""
        report = MockOutlierReport(
            outlier_count=5,
            outlier_percentage=0.05,
            outlier_indices=[1, 3, 5, 7, 9],
            threshold=2.5
        )

        assert report.outlier_count == 5
        assert report.outlier_percentage == 0.05
        assert report.outlier_indices == [1, 3, 5, 7, 9]
        assert report.threshold == 2.5

    def test_outlier_report_to_dict(self):
        """测试离群值报告序列化"""
        report = MockOutlierReport(outlier_count=3)
        data = report.to_dict()

        assert data["outlier_count"] == 3
        assert data["outlier_percentage"] == 0.0
        assert data["outlier_indices"] == []


class TestMockDataValidator:
    """模拟数据验证器测试"""

    def setup_method(self):
        """设置测试方法"""
        self.validator = MockDataValidator({
            "required_fields": ["symbol", "price"],
            "required_columns": ["date", "open", "close"],
            "min_length": 1
        })

    def test_validator_initialization(self):
        """测试验证器初始化"""
        assert not self.validator.is_initialized

        assert self.validator.initialize()
        assert self.validator.is_initialized

    def test_validate_dict_data(self):
        """测试验证字典数据"""
        self.validator.initialize()

        # 有效数据
        valid_data = {"symbol": "AAPL", "price": 150.0, "volume": 1000}
        result = self.validator.validate_data(valid_data, "stock_dict")

        assert result.is_valid is True
        assert result.data_type == "stock_dict"
        assert "field_count" in result.metrics

        # 无效数据 - 缺少必需字段
        invalid_data = {"price": 150.0}
        result = self.validator.validate_data(invalid_data, "stock_dict")

        assert result.is_valid is False
        assert len(result.errors) > 0
        assert "Missing required field" in result.errors[0]

    def test_validate_list_data(self):
        """测试验证列表数据"""
        self.validator.initialize()

        # 有效数据
        valid_data = [{"symbol": "AAPL", "price": 150.0}, {"symbol": "GOOGL", "price": 2500.0}]
        result = self.validator.validate_data(valid_data, "stock_list")

        assert result.is_valid is True
        assert result.metrics["item_count"] == 2

        # 无效数据 - 空列表
        invalid_data = []
        result = self.validator.validate_data(invalid_data, "stock_list")

        assert result.is_valid is False
        assert "Empty list" in result.errors

    def test_validate_dataframe_data(self):
        """测试验证DataFrame数据"""
        self.validator.initialize()

        # 创建测试DataFrame
        df = pd.DataFrame({
            "date": ["2023-01-01", "2023-01-02"],
            "open": [150.0, 152.0],
            "close": [152.0, 155.0],
            "volume": [1000, 1200]
        })

        result = self.validator.validate_data(df, "stock_dataframe")

        assert result.is_valid is True
        assert result.metrics["row_count"] == 2
        assert result.metrics["column_count"] == 4

        # 无效DataFrame - 缺少必需列
        invalid_df = pd.DataFrame({"price": [150.0]})
        result = self.validator.validate_data(invalid_df, "stock_dataframe")

        assert result.is_valid is False
        assert any("Missing required column" in error for error in result.errors)

    def test_validate_array_data(self):
        """测试验证数组数据"""
        self.validator.initialize()

        # 有效数组
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = self.validator.validate_data(arr, "numeric_array")

        assert result.is_valid is True
        assert result.metrics["total_elements"] == 4

    def test_validator_statistics(self):
        """测试验证器统计"""
        self.validator.initialize()

        # 执行几次验证
        self.validator.validate_data({"symbol": "AAPL"}, "test")
        self.validator.validate_data({}, "test")  # 这会失败
        self.validator.validate_data({"symbol": "GOOGL"}, "test")

        stats = self.validator.get_statistics()

        assert stats["validation_count"] == 3
        assert stats["error_count"] >= 1  # 至少有一个错误（空字典缺少必需字段）
        assert stats["success_rate"] < 1.0  # 成功率小于1
        assert stats["error_rate"] > 0.0  # 错误率大于0


class TestMockQualityAssessor:
    """模拟质量评估器测试"""

    def setup_method(self):
        """设置测试方法"""
        self.assessor = MockQualityAssessor()

    def test_quality_assessment_dataframe(self):
        """测试DataFrame质量评估"""
        # 创建测试DataFrame
        df = pd.DataFrame({
            "symbol": ["AAPL", "GOOGL", None],
            "price": [150.0, 2500.0, 100.0],
            "volume": [1000, None, 800]
        })

        report = self.assessor.assess_quality(df, "stock_data")

        assert isinstance(report, MockQualityReport)
        assert 0.0 <= report.overall_score <= 1.0
        assert 0.0 <= report.completeness <= 1.0
        assert report.details["data_type"] == "stock_data"

    def test_quality_assessment_dict(self):
        """测试字典质量评估"""
        data = {"symbol": "AAPL", "price": 150.0, "volume": None}
        report = self.assessor.assess_quality(data, "stock_dict")

        assert isinstance(report, MockQualityReport)
        assert report.completeness < 1.0  # 由于有None值

    def test_quality_assessment_list(self):
        """测试列表质量评估"""
        data = ["AAPL", "GOOGL", None, "MSFT"]
        report = self.assessor.assess_quality(data, "symbol_list")

        assert isinstance(report, MockQualityReport)
        assert report.completeness == 0.75  # 3/4 非空

    def test_quality_thresholds(self):
        """测试质量阈值"""
        custom_thresholds = {"completeness": 0.99, "accuracy": 0.99}
        assessor = MockQualityAssessor(custom_thresholds)

        assert assessor.quality_thresholds["completeness"] == 0.99
        assert assessor.quality_thresholds["accuracy"] == 0.99


class TestMockOutlierDetector:
    """模拟离群值检测器测试"""

    def setup_method(self):
        """设置测试方法"""
        self.detector = MockOutlierDetector(method="zscore", threshold=2.0)

    def test_outlier_detection_zscore(self):
        """测试Z-Score离群值检测"""
        # 创建带有离群值的数据
        data = [10, 12, 11, 13, 12, 100, 11, 12, 10]  # 100是离群值
        series = pd.Series(data)

        report = self.detector.detect_outliers(series)

        assert isinstance(report, MockOutlierReport)
        # 对于这个数据，100的Z-Score应该远大于2.0
        assert report.outlier_count > 0
        assert len(report.outlier_indices) > 0
        assert len(report.outlier_values) > 0
        assert 100 in report.outlier_values  # 确保检测到了100这个离群值

    def test_outlier_detection_iqr(self):
        """测试IQR离群值检测"""
        detector = MockOutlierDetector(method="iqr", threshold=1.5)

        data = [1, 2, 3, 4, 5, 100]  # 100是离群值
        series = pd.Series(data)

        report = detector.detect_outliers(series)

        assert report.outlier_count > 0

    def test_outlier_detection_dataframe(self):
        """测试DataFrame离群值检测"""
        df = pd.DataFrame({
            "price": [100, 105, 102, 200, 103],  # 200是离群值
            "volume": [1000, 1200, 1100, 1500, 1300]
        })

        report = self.detector.detect_outliers(df)

        assert isinstance(report, MockOutlierReport)
        assert report.outlier_count >= 0  # 可能检测到price列的离群值

    def test_outlier_detection_no_outliers(self):
        """测试无离群值的数据"""
        data = [10, 11, 12, 11, 10, 12, 11]
        series = pd.Series(data)

        report = self.detector.detect_outliers(series)

        assert report.outlier_count == 0
        assert report.outlier_percentage == 0.0

    def test_outlier_detection_empty_data(self):
        """测试空数据离群值检测"""
        empty_series = pd.Series([])
        report = self.detector.detect_outliers(empty_series)

        assert report.outlier_count == 0
        assert report.outlier_percentage == 0.0


class TestValidationIntegration:
    """验证器集成测试"""

    def test_complete_validation_workflow(self):
        """测试完整的验证工作流"""
        # 创建验证器
        validator = MockDataValidator({
            "required_columns": ["symbol", "price", "volume"],
            "column_types": {"price": "float", "volume": "int"}
        })

        # 创建质量评估器
        assessor = MockQualityAssessor()

        # 创建离群值检测器
        detector = MockOutlierDetector(threshold=1.5)  # 降低阈值以更容易检测离群值

        # 初始化组件
        assert validator.initialize()

        # 创建测试数据
        df = pd.DataFrame({
            "symbol": ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"],
            "price": [150.0, 2500.0, 300.0, 200.0, 400.0],
            "volume": [1000, 1500, 1200, 800, 2000],
            "outlier_price": [150.0, 160.0, 155.0, 158.0, 1000.0]  # 最后一个是明显的离群值
        })

        # 1. 数据验证
        validation_result = validator.validate_data(df, "stock_data")
        assert validation_result.is_valid is True

        # 2. 质量评估
        quality_report = assessor.assess_quality(df, "stock_data")
        assert quality_report.overall_score > 0.8

        # 3. 离群值检测
        outlier_report = detector.detect_outliers(df, "outlier_price")
        assert outlier_report.outlier_count > 0  # 应该检测到离群值

        # 4. 验证统计信息
        stats = validator.get_statistics()
        assert stats["validation_count"] == 1
        assert stats["success_rate"] == 1.0

    def test_validation_error_handling(self):
        """测试验证错误处理"""
        validator = MockDataValidator()

        # 未初始化验证器
        with pytest.raises(Exception, match="Validator not initialized"):
            validator.validate_data({"test": "data"})

        # 初始化后验证无效数据
        validator.initialize()

        # 验证None数据
        result = validator.validate_data(None)
        assert result.is_valid is False
        assert "Data is None" in result.errors

        # 验证不支持的数据类型
        result = validator.validate_data(set([1, 2, 3]))
        assert result.is_valid is False
        assert "Unsupported data type" in result.errors[0]

    def test_validation_performance_simulation(self):
        """测试验证性能模拟"""
        validator = MockDataValidator()
        validator.initialize()

        # 模拟大量数据验证
        test_data = [
            {"symbol": f"STOCK{i}", "price": 100.0 + i, "volume": 1000 + i}
            for i in range(100)
        ]

        # 批量验证
        results = []
        for data in test_data:
            result = validator.validate_data(data, "stock")
            results.append(result)

        # 验证结果
        valid_results = [r for r in results if r.is_valid]
        assert len(valid_results) == len(test_data)

        # 验证统计
        stats = validator.get_statistics()
        assert stats["validation_count"] == len(test_data)
        assert stats["success_rate"] == 1.0

    def test_cross_component_validation(self):
        """测试跨组件验证"""
        validator = MockDataValidator({"required_fields": ["price"]})
        assessor = MockQualityAssessor()
        detector = MockOutlierDetector()

        validator.initialize()

        # 测试不同类型的数据
        test_cases = [
            {"name": "dict_data", "data": {"price": 100.0, "volume": 1000}},
            {"name": "list_data", "data": [{"price": 100.0}, {"price": 200.0}]},
            {"name": "dataframe_data", "data": pd.DataFrame({"price": [100.0, 200.0]})}
        ]

        for test_case in test_cases:
            # 验证数据
            validation_result = validator.validate_data(test_case["data"], test_case["name"])
            assert validation_result.is_valid is True

            # 质量评估
            quality_report = assessor.assess_quality(test_case["data"], test_case["name"])
            assert quality_report.overall_score > 0.8

            # 离群值检测（如果适用）
            if isinstance(test_case["data"], (list, pd.DataFrame)):
                outlier_report = detector.detect_outliers(test_case["data"])
                assert isinstance(outlier_report, MockOutlierReport)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
