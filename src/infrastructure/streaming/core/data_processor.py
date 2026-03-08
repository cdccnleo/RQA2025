"""
Data Processor Module
数据处理器模块

This module provides data processing capabilities for streaming operations
此模块为流操作提供数据处理能力

Author: RQA2025 Development Team
Date: 2025 - 01 - 28
"""

import logging
from typing import Any, Dict, List, Optional, Callable, Union
import json


logger = logging.getLogger(__name__)


class DataProcessor:

    """
    Streaming Data Processor
    流数据处理器

    Processes various types of data in streaming pipelines
    处理流管道中的各种类型数据
    """

    def __init__(self, processor_name: str = "default_data_processor"):
        """
        Initialize the data processor
        初始化数据处理器

        Args:
            processor_name: Name of this processor
                          此处理器的名称
        """
        self.processor_name = processor_name
        self.processed_count = 0
        self.error_count = 0
        self.transformers: List[Callable] = []
        self.filters: List[Callable] = []
        self.validators: List[Callable] = []

        logger.info(f"Data processor {processor_name} initialized")

    def add_transformer(self, transformer: Callable) -> None:
        """
        Add a data transformer
        添加数据转换器

        Args:
            transformer: Function to transform data
                        数据转换函数
        """
        self.transformers.append(transformer)
        logger.info(f"Added transformer to {self.processor_name}")

    def add_filter(self, filter_func: Callable) -> None:
        """
        Add a data filter
        添加数据过滤器

        Args:
            filter_func: Function to filter data
                        数据过滤函数
        """
        self.filters.append(filter_func)
        logger.info(f"Added filter to {self.processor_name}")

    def add_validator(self, validator: Callable) -> None:
        """
        Add a data validator
        添加数据验证器

        Args:
            validator: Function to validate data
                      数据验证函数
        """
        self.validators.append(validator)
        logger.info(f"Added validator to {self.processor_name}")

    def process_data(self, data: Any) -> Optional[Any]:
        """
        Process input data through the pipeline
        通过管道处理输入数据

        Args:
            data: Input data to process
                 要处理的输入数据

        Returns:
            Processed data or None if filtered out
            处理后的数据，如果被过滤则返回None
        """
        try:
            # Apply filters first
            for filter_func in self.filters:
                if not filter_func(data):
                    logger.debug(f"Data filtered out by {filter_func.__name__}")
                    return None

            # Apply transformers
            processed_data = data
            for transformer in self.transformers:
                processed_data = transformer(processed_data)

            # Apply validators
            for validator in self.validators:
                if not validator(processed_data):
                    logger.warning(f"Data validation failed by {validator.__name__}")
                    self.error_count += 1
                    return None

            self.processed_count += 1
            return processed_data

        except Exception as e:
            logger.error(f"Data processing error in {self.processor_name}: {str(e)}")
            self.error_count += 1
            return None

    def process_batch(self, data_batch: List[Any]) -> List[Any]:
        """
        Process a batch of data
        处理一批数据

        Args:
            data_batch: Batch of data to process
                       要处理的数据批次

        Returns:
            List of processed data (filtered data removed)
            处理后的数据列表（已过滤掉不符合条件的数据）
        """
        results = []
        for item in data_batch:
            processed = self.process_data(item)
            if processed is not None:
                results.append(processed)

        logger.info(f"Processed batch of {len(data_batch)} items, resulted in {len(results)} items")
        return results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get processor statistics
        获取处理器统计信息

        Returns:
            dict: Processor statistics
                  处理器统计信息
        """
        return {
            'processor_name': self.processor_name,
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'success_rate': (self.processed_count / max(self.processed_count + self.error_count, 1)) * 100,
            'transformers_count': len(self.transformers),
            'filters_count': len(self.filters),
            'validators_count': len(self.validators)
        }

    def reset_stats(self) -> None:
        """
        Reset processor statistics
        重置处理器统计信息
        """
        self.processed_count = 0
        self.error_count = 0
        logger.info(f"Statistics reset for {self.processor_name}")


# Utility functions for common data processing


def json_parser(data: Union[str, bytes]) -> Dict[str, Any]:
    """
    Parse JSON data
    解析JSON数据

    Args:
        data: JSON string or bytes
             JSON字符串或字节

    Returns:
        Parsed JSON object
        解析后的JSON对象
    """
    if isinstance(data, bytes):
        data = data.decode('utf - 8')
    return json.loads(data)


def csv_parser(data: str, delimiter: str = ',') -> List[Dict[str, Any]]:
    """
    Parse CSV data
    解析CSV数据

    Args:
        data: CSV string data
             CSV字符串数据
        delimiter: CSV delimiter
                  CSV分隔符

    Returns:
        List of parsed records
        解析后的记录列表
    """
    lines = data.strip().split('\n')
    if not lines:
        return []

    headers = lines[0].split(delimiter)
    records = []

    for line in lines[1:]:
        if line.strip():
            values = line.split(delimiter)
            record = dict(zip(headers, values))
            records.append(record)

    return records


def data_filter_not_none(data: Any) -> bool:
    """
    Filter out None values
    过滤掉None值

    Args:
        data: Data to check
             要检查的数据

    Returns:
        True if data is not None, False otherwise
        如果数据不为None则返回True，否则返回False
    """
    return data is not None


def data_filter_has_field(data: Dict[str, Any], field: str) -> bool:
    """
    Filter data that has a specific field
    过滤包含特定字段的数据

    Args:
        data: Data dictionary
             数据字典
        field: Field name to check
              要检查的字段名

    Returns:
        True if field exists, False otherwise
        如果字段存在则返回True，否则返回False
    """
    return isinstance(data, dict) and field in data


def data_validator_schema(data: Dict[str, Any], required_fields: List[str]) -> bool:
    """
    Validate data against a schema
    根据模式验证数据

    Args:
        data: Data dictionary to validate
             要验证的数据字典
        required_fields: List of required field names
                        必需字段名列表

    Returns:
        True if validation passes, False otherwise
        如果验证通过则返回True，否则返回False
    """
    if not isinstance(data, dict):
        return False

    return all(field in data for field in required_fields)


# Global default processor instance
# 全局默认处理器实例

default_data_processor = DataProcessor("default_data_processor")

__all__ = [
    'DataProcessor',
    'default_data_processor',
    'json_parser',
    'csv_parser',
    'data_filter_not_none',
    'data_filter_has_field',
    'data_validator_schema'
]
