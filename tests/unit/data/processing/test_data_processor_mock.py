# -*- coding: utf-8 -*-
"""
数据处理器Mock测试
测试数据处理、清洗、转换和标准化的功能
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
from enum import Enum


class MockFillMethod(Enum):
    """模拟数据填充方法枚举"""
    FORWARD = "forward"
    BACKWARD = "backward"
    INTERPOLATE = "interpolate"
    MEAN = "mean"
    MEDIAN = "median"
    ZERO = "zero"
    DROP = "drop"


class MockDataProcessorConfig:
    """模拟数据处理器配置"""

    def __init__(self, fill_method: str = "forward", outlier_threshold: float = 3.0,
                 normalization_method: str = "zscore", remove_duplicates: bool = True,
                 handle_missing: bool = True):
        self.fill_method = fill_method
        self.outlier_threshold = outlier_threshold
        self.normalization_method = normalization_method
        self.remove_duplicates = remove_duplicates
        self.handle_missing = handle_missing

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "fill_method": self.fill_method,
            "outlier_threshold": self.outlier_threshold,
            "normalization_method": self.normalization_method,
            "remove_duplicates": self.remove_duplicates,
            "handle_missing": self.handle_missing
        }


class MockDataProcessor:
    """模拟数据处理器"""

    def __init__(self, config: MockDataProcessorConfig):
        self.config = config
        self.is_initialized = False
        self.processing_stats = {
            "processed_count": 0,
            "error_count": 0,
            "cleaned_rows": 0,
            "filled_values": 0,
            "removed_duplicates": 0,
            "normalized_columns": 0
        }
        self.logger = Mock()
        self.logger.info = Mock()
        self.logger.warning = Mock()
        self.logger.error = Mock()

    def initialize(self) -> bool:
        """初始化处理器"""
        try:
            self.is_initialized = True
            self.logger.info("DataProcessor initialized")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize processor: {e}")
            return False

    def process_data(self, data: Any, data_type: str = "dataframe") -> Any:
        """处理数据"""
        if not self.is_initialized:
            raise Exception("Processor not initialized")

        self.processing_stats["processed_count"] += 1

        try:
            if isinstance(data, pd.DataFrame):
                return self._process_dataframe(data)
            elif isinstance(data, dict):
                return self._process_dict(data)
            elif isinstance(data, list):
                return self._process_list(data)
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
        except Exception as e:
            self.processing_stats["error_count"] += 1
            self.logger.error(f"Data processing failed: {e}")
            raise

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理DataFrame数据"""
        processed_df = df.copy()

        # 1. 移除重复行
        if self.config.remove_duplicates:
            initial_rows = len(processed_df)
            processed_df = processed_df.drop_duplicates()
            removed = initial_rows - len(processed_df)
            self.processing_stats["removed_duplicates"] += removed
            self.processing_stats["cleaned_rows"] += removed

        # 2. 处理缺失值
        if self.config.handle_missing:
            filled_count = self._fill_missing_values(processed_df)
            self.processing_stats["filled_values"] += filled_count

        # 3. 数据标准化
        if self.config.normalization_method != "none":
            normalized_count = self._normalize_data(processed_df)
            self.processing_stats["normalized_columns"] += normalized_count

        return processed_df

    def _process_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理字典数据"""
        processed_data = data.copy()

        # 处理缺失值
        if self.config.handle_missing:
            for key, value in processed_data.items():
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    if self.config.fill_method == "zero":
                        processed_data[key] = 0
                    elif self.config.fill_method == "mean":
                        # 对于字典，计算其他数值的均值
                        numeric_values = [v for v in processed_data.values()
                                        if v is not None and isinstance(v, (int, float)) and not np.isnan(v)]
                        if numeric_values:
                            processed_data[key] = sum(numeric_values) / len(numeric_values)
                        else:
                            processed_data[key] = 0
                    else:
                        processed_data[key] = 0  # 默认填充0
                    self.processing_stats["filled_values"] += 1

        return processed_data

    def _process_list(self, data: List[Any]) -> List[Any]:
        """处理列表数据"""
        processed_data = data.copy()

        # 处理缺失值
        if self.config.handle_missing:
            for i, item in enumerate(processed_data):
                if item is None or (isinstance(item, float) and np.isnan(item)):
                    if self.config.fill_method == "zero":
                        processed_data[i] = 0
                    elif self.config.fill_method == "mean":
                        # 计算非空值的均值
                        numeric_values = [x for x in processed_data if x is not None and isinstance(x, (int, float)) and not np.isnan(x)]
                        if numeric_values:
                            processed_data[i] = sum(numeric_values) / len(numeric_values)
                        else:
                            processed_data[i] = 0
                    else:  # 默认填充为0
                        processed_data[i] = 0
                    self.processing_stats["filled_values"] += 1

        return processed_data

    def _fill_missing_values(self, df: pd.DataFrame) -> int:
        """填充缺失值"""
        filled_count = 0

        for column in df.columns:
            missing_count = df[column].isnull().sum()
            if missing_count > 0:
                if self.config.fill_method == "forward":
                    df[column] = df[column].ffill()
                elif self.config.fill_method == "backward":
                    df[column] = df[column].bfill()
                elif self.config.fill_method == "mean":
                    mean_val = df[column].mean()
                    df[column] = df[column].fillna(mean_val)
                elif self.config.fill_method == "median":
                    median_val = df[column].median()
                    df[column] = df[column].fillna(median_val)
                elif self.config.fill_method == "zero":
                    df[column] = df[column].fillna(0)
                elif self.config.fill_method == "drop":
                    df = df.dropna(subset=[column])

                filled_count += missing_count

        return filled_count

    def _normalize_data(self, df: pd.DataFrame) -> int:
        """数据标准化"""
        normalized_count = 0

        for column in df.select_dtypes(include=[np.number]).columns:
            if self.config.normalization_method == "zscore":
                mean_val = df[column].mean()
                std_val = df[column].std()
                if std_val != 0:
                    df[column] = (df[column] - mean_val) / std_val
                    normalized_count += 1
            elif self.config.normalization_method == "minmax":
                min_val = df[column].min()
                max_val = df[column].max()
                if max_val != min_val:
                    df[column] = (df[column] - min_val) / (max_val - min_val)
                    normalized_count += 1

        return normalized_count

    def get_processing_stats(self) -> Dict[str, Any]:
        """获取处理统计"""
        return self.processing_stats.copy()

    def reset_stats(self):
        """重置统计"""
        self.processing_stats = {
            "processed_count": 0,
            "error_count": 0,
            "cleaned_rows": 0,
            "filled_values": 0,
            "removed_duplicates": 0,
            "normalized_columns": 0
        }


class MockDataCleaner:
    """模拟数据清理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.is_initialized = False
        self.cleaned_count = 0
        self.removed_count = 0
        self.logger = Mock()

    def initialize(self) -> bool:
        """初始化清理器"""
        try:
            self.is_initialized = True
            return True
        except Exception:
            return False

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """清理数据"""
        if not self.is_initialized:
            raise Exception("Cleaner not initialized")

        self.cleaned_count += 1
        cleaned_df = data.copy()

        # 移除全为空的行
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.dropna(how='all')
        removed = initial_rows - len(cleaned_df)
        self.removed_count += removed

        # 移除全为空的列
        initial_cols = len(cleaned_df.columns)
        cleaned_df = cleaned_df.dropna(axis=1, how='all')
        removed_cols = initial_cols - len(cleaned_df.columns)

        return cleaned_df

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        return {
            "cleaned_count": self.cleaned_count,
            "removed_count": self.removed_count
        }


class MockDataTransformer:
    """模拟数据转换器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.is_initialized = False
        self.transform_count = 0
        self.logger = Mock()

    def initialize(self) -> bool:
        """初始化转换器"""
        try:
            self.is_initialized = True
            return True
        except Exception:
            return False

    def transform_data(self, data: pd.DataFrame, transformations: List[Dict[str, Any]]) -> pd.DataFrame:
        """转换数据"""
        if not self.is_initialized:
            raise Exception("Transformer not initialized")

        self.transform_count += 1
        transformed_df = data.copy()

        for transformation in transformations:
            transform_type = transformation.get("type")
            column = transformation.get("column")
            params = transformation.get("params", {})

            if transform_type == "log" and column in transformed_df.columns:
                transformed_df[column] = np.log(transformed_df[column] + 1)  # +1避免log(0)
            elif transform_type == "sqrt" and column in transformed_df.columns:
                transformed_df[column] = np.sqrt(transformed_df[column])
            elif transform_type == "scale" and column in transformed_df.columns:
                factor = params.get("factor", 1.0)
                transformed_df[column] = transformed_df[column] * factor

        return transformed_df

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        return {
            "transform_count": self.transform_count
        }


class TestMockDataProcessorConfig:
    """模拟数据处理器配置测试"""

    def test_config_creation(self):
        """测试配置创建"""
        config = MockDataProcessorConfig(
            fill_method="mean",
            outlier_threshold=2.5,
            normalization_method="minmax"
        )

        assert config.fill_method == "mean"
        assert config.outlier_threshold == 2.5
        assert config.normalization_method == "minmax"
        assert config.remove_duplicates is True
        assert config.handle_missing is True

    def test_config_to_dict(self):
        """测试配置序列化"""
        config = MockDataProcessorConfig(fill_method="zero", normalization_method="zscore")
        data = config.to_dict()

        assert data["fill_method"] == "zero"
        assert data["normalization_method"] == "zscore"


class TestMockDataProcessor:
    """模拟数据处理器测试"""

    def setup_method(self):
        """设置测试方法"""
        self.config = MockDataProcessorConfig(
            fill_method="mean",
            normalization_method="zscore",
            remove_duplicates=True,
            handle_missing=True
        )
        self.processor = MockDataProcessor(self.config)

    def test_processor_initialization(self):
        """测试处理器初始化"""
        assert not self.processor.is_initialized

        assert self.processor.initialize()
        assert self.processor.is_initialized

    def test_process_dataframe(self):
        """测试处理DataFrame"""
        self.processor.initialize()

        # 创建测试数据
        df = pd.DataFrame({
            "A": [1, 2, 2, 3, 2],  # 有重复行（第2和第5行都是2）
            "B": [10, 20, 30, 40, 20],  # 有对应的重复
            "C": ["a", "b", "c", "d", "b"]  # 有对应的重复
        })

        result = self.processor.process_data(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4  # 移除一个重复行（原始5行，移除1行重复）
        assert result["A"].isnull().sum() == 0  # 没有缺失值
        assert result["B"].isnull().sum() == 0  # 没有缺失值

        # 检查统计
        stats = self.processor.get_processing_stats()
        assert stats["processed_count"] == 1
        assert stats["removed_duplicates"] == 1
        assert stats["filled_values"] == 0  # 这个测试数据中没有缺失值

    def test_process_dict(self):
        """测试处理字典"""
        self.processor.initialize()

        data = {"x": 1, "y": None, "z": 3}
        result = self.processor.process_data(data, "dict")

        assert isinstance(result, dict)
        assert result["y"] == 2.0  # 缺失值被填充为均值 (1+3)/2 = 2

    def test_process_list(self):
        """测试处理列表"""
        self.processor.initialize()

        data = [1, None, 3, None, 5]
        result = self.processor.process_data(data, "list")

        assert isinstance(result, list)
        assert None not in result  # 所有缺失值被填充

    def test_fill_missing_values_forward(self):
        """测试前向填充"""
        config = MockDataProcessorConfig(fill_method="forward")
        processor = MockDataProcessor(config)
        processor.initialize()

        df = pd.DataFrame({"A": [1, None, 3, None, 5]})
        result = processor._fill_missing_values(df)

        assert result == 2  # 填充了2个缺失值
        assert df["A"].tolist() == [1, 1, 3, 3, 5]  # 前向填充结果

    def test_fill_missing_values_mean(self):
        """测试均值填充"""
        config = MockDataProcessorConfig(fill_method="mean")
        processor = MockDataProcessor(config)
        processor.initialize()

        df = pd.DataFrame({"A": [1, None, 3]})
        result = processor._fill_missing_values(df)

        assert result == 1
        assert df["A"].iloc[1] == 2.0  # 均值填充

    def test_normalize_data_zscore(self):
        """测试Z-Score标准化"""
        config = MockDataProcessorConfig(normalization_method="zscore")
        processor = MockDataProcessor(config)
        processor.initialize()

        df = pd.DataFrame({"A": [1, 2, 3, 4, 5]})
        result = processor._normalize_data(df)

        assert result == 1  # 标准化了一列
        # 检查标准化结果（均值接近0）
        assert abs(df["A"].mean()) < 0.001

    def test_normalize_data_minmax(self):
        """测试Min-Max标准化"""
        config = MockDataProcessorConfig(normalization_method="minmax")
        processor = MockDataProcessor(config)
        processor.initialize()

        df = pd.DataFrame({"A": [1, 2, 3, 4, 5]})
        result = processor._normalize_data(df)

        assert result == 1
        assert df["A"].min() == 0.0
        assert df["A"].max() == 1.0

    def test_processing_stats(self):
        """测试处理统计"""
        self.processor.initialize()

        # 处理多次数据
        df1 = pd.DataFrame({"A": [1, None, 3]})
        df2 = pd.DataFrame({"B": [None, 2, 3, 3]})  # 有重复

        self.processor.process_data(df1)
        self.processor.process_data(df2)

        stats = self.processor.get_processing_stats()
        assert stats["processed_count"] == 2
        assert stats["filled_values"] >= 1
        assert stats["removed_duplicates"] >= 1

    def test_reset_stats(self):
        """测试重置统计"""
        self.processor.initialize()
        self.processor.process_data(pd.DataFrame({"A": [1, None]}))

        # 确认有统计数据
        stats = self.processor.get_processing_stats()
        assert stats["processed_count"] == 1

        # 重置统计
        self.processor.reset_stats()
        stats = self.processor.get_processing_stats()
        assert stats["processed_count"] == 0


class TestMockDataCleaner:
    """模拟数据清理器测试"""

    def setup_method(self):
        """设置测试方法"""
        self.cleaner = MockDataCleaner()

    def test_cleaner_initialization(self):
        """测试清理器初始化"""
        assert not self.cleaner.is_initialized

        assert self.cleaner.initialize()
        assert self.cleaner.is_initialized

    def test_clean_empty_rows(self):
        """测试清理空行"""
        self.cleaner.initialize()

        df = pd.DataFrame({
            "A": [1, None, 3],
            "B": [2, None, 4],
            "C": [None, None, None]  # 全为空的行
        })

        result = self.cleaner.clean_data(df)

        assert len(result) == 2  # 移除了一行全空的行
        assert self.cleaner.removed_count == 1

    def test_clean_empty_columns(self):
        """测试清理空列"""
        self.cleaner.initialize()

        df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": [None, None, None],  # 全为空的列
            "C": [4, 5, 6]
        })

        result = self.cleaner.clean_data(df)

        assert len(result.columns) == 2  # 移除了全空的列
        assert "B" not in result.columns

    def test_cleaner_stats(self):
        """测试清理器统计"""
        self.cleaner.initialize()

        df = pd.DataFrame({
            "A": [1, None, None],  # 有两行全空（第1和第2行）
            "B": [None, None, None]  # 全空列
        })

        self.cleaner.clean_data(df)

        stats = self.cleaner.get_stats()
        assert stats["cleaned_count"] == 1
        assert stats["removed_count"] == 2  # 移除了两行全空行


class TestMockDataTransformer:
    """模拟数据转换器测试"""

    def setup_method(self):
        """设置测试方法"""
        self.transformer = MockDataTransformer()

    def test_transformer_initialization(self):
        """测试转换器初始化"""
        assert not self.transformer.is_initialized

        assert self.transformer.initialize()
        assert self.transformer.is_initialized

    def test_log_transformation(self):
        """测试对数转换"""
        self.transformer.initialize()

        df = pd.DataFrame({"A": [1, 2, 4, 8]})
        transformations = [{"type": "log", "column": "A"}]

        result = self.transformer.transform_data(df, transformations)

        # 检查对数转换结果
        expected = [np.log(2), np.log(3), np.log(5), np.log(9)]  # log(1+1), log(2+1), etc.
        assert abs(result["A"].iloc[0] - np.log(2)) < 0.001

    def test_sqrt_transformation(self):
        """测试平方根转换"""
        self.transformer.initialize()

        df = pd.DataFrame({"A": [1, 4, 9, 16]})
        transformations = [{"type": "sqrt", "column": "A"}]

        result = self.transformer.transform_data(df, transformations)

        assert result["A"].tolist() == [1, 2, 3, 4]

    def test_scale_transformation(self):
        """测试缩放转换"""
        self.transformer.initialize()

        df = pd.DataFrame({"A": [1, 2, 3, 4]})
        transformations = [{"type": "scale", "column": "A", "params": {"factor": 2.0}}]

        result = self.transformer.transform_data(df, transformations)

        assert result["A"].tolist() == [2, 4, 6, 8]

    def test_multiple_transformations(self):
        """测试多重转换"""
        self.transformer.initialize()

        df = pd.DataFrame({"A": [1, 2, 4], "B": [4, 9, 16]})
        transformations = [
            {"type": "log", "column": "A"},
            {"type": "sqrt", "column": "B"}
        ]

        result = self.transformer.transform_data(df, transformations)

        # A列进行了对数转换
        assert abs(result["A"].iloc[0] - np.log(2)) < 0.001
        # B列进行了平方根转换
        assert result["B"].tolist() == [2, 3, 4]

    def test_transformer_stats(self):
        """测试转换器统计"""
        self.transformer.initialize()

        df = pd.DataFrame({"A": [1, 2, 3]})
        transformations = [{"type": "log", "column": "A"}]

        self.transformer.transform_data(df, transformations)

        stats = self.transformer.get_stats()
        assert stats["transform_count"] == 1


class TestDataProcessingIntegration:
    """数据处理集成测试"""

    def test_complete_data_processing_pipeline(self):
        """测试完整的数据处理管道"""
        # 创建处理器
        processor_config = MockDataProcessorConfig(
            fill_method="mean",
            normalization_method="zscore"
        )
        processor = MockDataProcessor(processor_config)

        # 创建清理器
        cleaner = MockDataCleaner()

        # 创建转换器
        transformer = MockDataTransformer()

        # 初始化所有组件
        assert processor.initialize()
        assert cleaner.initialize()
        assert transformer.initialize()

        # 创建测试数据（包含各种问题）
        raw_data = pd.DataFrame({
            "symbol": ["AAPL", "AAPL", "GOOGL", "GOOGL", "MSFT"],  # 有重复
            "price": [150.0, None, 2500.0, 2600.0, 300.0],  # 有缺失值
            "volume": [1000, 1200, None, 1500, 800],  # 有缺失值
            "empty_col": [None, None, None, None, None],  # 全空列
            "empty_row": [None, None, None, None, None]  # 将创建全空行
        })

        # 添加全空行
        empty_row = pd.DataFrame({col: [None] for col in raw_data.columns})
        raw_data = pd.concat([raw_data, empty_row], ignore_index=True)

        # 1. 数据清理
        cleaned_data = cleaner.clean_data(raw_data)
        assert len(cleaned_data.columns) < len(raw_data.columns)  # 移除了空列
        assert len(cleaned_data) < len(raw_data)  # 移除了空行

        # 2. 数据处理
        processed_data = processor.process_data(cleaned_data)
        assert processed_data.isnull().sum().sum() == 0  # 所有缺失值被填充
        # 检查标准化（price列应该被标准化）
        assert abs(processed_data["price"].mean()) < 0.001  # Z-Score标准化后均值接近0

        # 3. 数据转换（确保volume都是正数以避免log(0)或log(负数)）
        # 先确保volume都是正数
        processed_data["volume"] = processed_data["volume"].abs() + 1  # 确保都是正数

        transformations = [
            {"type": "log", "column": "volume"},
            {"type": "scale", "column": "price", "params": {"factor": 0.01}}
        ]
        final_data = transformer.transform_data(processed_data, transformations)

        # 验证转换结果
        assert not final_data["volume"].isnull().any()  # 对数转换后不应该有NaN
        assert all(final_data["price"] <= 1.0)  # 缩放后值变小

        # 验证统计信息
        processor_stats = processor.get_processing_stats()
        cleaner_stats = cleaner.get_stats()
        transformer_stats = transformer.get_stats()

        assert processor_stats["processed_count"] == 1
        assert cleaner_stats["cleaned_count"] == 1
        assert transformer_stats["transform_count"] == 1

    def test_error_handling_in_processing(self):
        """测试处理中的错误处理"""
        processor = MockDataProcessor(MockDataProcessorConfig())

        # 未初始化处理
        with pytest.raises(Exception, match="Processor not initialized"):
            processor.process_data(pd.DataFrame())

        # 初始化后处理无效数据类型
        processor.initialize()
        with pytest.raises(ValueError, match="Unsupported data type"):
            processor.process_data("invalid_data")

        # 检查错误统计
        stats = processor.get_processing_stats()
        assert stats["error_count"] == 1

    def test_processing_different_data_formats(self):
        """测试处理不同数据格式"""
        processor = MockDataProcessor(MockDataProcessorConfig())
        processor.initialize()

        # 测试DataFrame
        df = pd.DataFrame({"A": [1, None, 3]})
        result_df = processor.process_data(df)
        assert isinstance(result_df, pd.DataFrame)

        # 测试字典
        dict_data = {"x": 1, "y": None, "z": 3}
        result_dict = processor.process_data(dict_data, "dict")
        assert isinstance(result_dict, dict)
        assert result_dict["y"] is not None  # 缺失值被填充
        assert result_dict["y"] == 0  # 默认配置使用forward填充，简化为0

        # 测试列表
        list_data = [1, None, 3]
        result_list = processor.process_data(list_data, "list")
        assert isinstance(result_list, list)
        assert None not in result_list  # 缺失值被填充

        # 验证处理统计
        stats = processor.get_processing_stats()
        assert stats["processed_count"] == 3

    def test_processing_with_different_configs(self):
        """测试不同配置的处理效果"""
        # 配置1: 前向填充，不标准化
        config1 = MockDataProcessorConfig(fill_method="forward", normalization_method="none")
        processor1 = MockDataProcessor(config1)
        processor1.initialize()

        # 配置2: 均值填充，Z-Score标准化
        config2 = MockDataProcessorConfig(fill_method="mean", normalization_method="zscore")
        processor2 = MockDataProcessor(config2)
        processor2.initialize()

        # 创建相同的测试数据
        df1 = pd.DataFrame({"A": [1, None, 3, None, 5]})
        df2 = df1.copy()

        result1 = processor1.process_data(df1)
        result2 = processor2.process_data(df2)

        # 结果应该不同
        assert not result1["A"].equals(result2["A"])

        # result1没有标准化，result2进行了标准化
        assert abs(result1["A"].mean() - 3.0) > 0.1  # 非标准化数据均值
        assert abs(result2["A"].mean()) < 0.001  # 标准化数据均值接近0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
