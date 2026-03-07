"""
中国股票数据验证器
"""
import pandas as pd
from typing import Dict, Any
from ..interfaces import IDataModel


class ChinaStockValidator:

    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """验证原始数据"""
        if data is None or not isinstance(data, pd.DataFrame) or data.empty:
            return {'is_valid': False, 'errors': ['数据为空或格式错误'], 'warnings': []}
        return {'is_valid': True, 'errors': [], 'warnings': []}

    def validate_data_model(self, model: IDataModel) -> Dict[str, Any]:
        """验证数据模型"""
        if model is None:
            return {'is_valid': False, 'errors': ['数据模型为空'], 'warnings': []}

        try:
            # 检查数据模型是否有data属性
            if not hasattr(model, 'data'):
                return {'is_valid': False, 'errors': ['数据模型缺少data属性'], 'warnings': []}

            # 验证数据
            data = model.data
            if data is None or not isinstance(data, pd.DataFrame) or data.empty:
                return {'is_valid': False, 'errors': ['数据为空或格式错误'], 'warnings': []}

            # 检查必要的列
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                return {
                    'is_valid': False,
                    'errors': [f'缺少必要的列: {missing_columns}'],
                    'warnings': []
                }

            # 检查数据类型
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
                    return {
                        'is_valid': False,
                        'errors': [f'列 {col} 不是数值类型'],
                        'warnings': []
                    }

            # 检查数据完整性
            null_counts = data.isnull().sum()
            if null_counts.sum() > 0:
                warnings = [f'列 {col} 有 {count} 个空值' for col,
                            count in null_counts.items() if count > 0]
                return {'is_valid': True, 'errors': [], 'warnings': warnings}

            return {'is_valid': True, 'errors': [], 'warnings': []}

        except Exception as e:
            return {'is_valid': False, 'errors': [f'验证过程中发生错误: {str(e)}'], 'warnings': []}
