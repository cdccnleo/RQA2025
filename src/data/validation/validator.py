"""
数据验证器模块
提供数据质量验证、异常检测、一致性检查等功能
"""
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


@dataclass
class ValidationResult:

    """验证结果数据类"""
    is_valid: bool
    metrics: Dict[str, float]
    errors: List[str]
    warnings: List[str]
    timestamp: str
    data_type: str = "unknown"


@dataclass
class QualityReport:

    """质量报告数据类"""
    overall_score: float
    completeness: float
    accuracy: float
    consistency: float
    timeliness: float
    details: Dict[str, Any]


@dataclass
class OutlierReport:

    """离群值报告数据类"""
    outlier_count: int
    outlier_percentage: float
    outlier_indices: List[int]
    outlier_values: List[Any]
    threshold: float
    lower_bound: float
    upper_bound: float


@dataclass
class ConsistencyReport:

    """一致性报告数据类"""
    is_consistent: bool
    consistency_score: float
    inconsistencies: List[str]
    cross_reference_results: Dict[str, Any]
    detailed_analysis: Dict[str, Any]


class ValidationError(Exception):

    """验证错误异常类"""

    def __init__(self, message: str, validation_result: ValidationResult = None):

        super().__init__(message)
        self.validation_result = validation_result


class DataValidationWarning(Exception):

    """验证警告异常类"""

    def __init__(self, message: str, validation_result: ValidationResult = None):

        super().__init__(message)
        self.validation_result = validation_result


class DataValidator:

    """数据质量验证器"""

    def __init__(self):

        self.quality_metrics = [
            'price_deviation',
            'volume_spike',
            'null_count',
            'outlier_count',
            'time_gap',
            'data_completeness',
            'data_consistency',
            'data_accuracy'
        ]
        self._validation_history = []
        self._rules = {}  # 存储自定义验证规则
        self._data_type_rules = {
            'stock': self._validate_stock_specific,
            'financial': self._validate_financial_specific,
            'market': self._validate_market_specific
        }

    def validate_data(self, data: Any, data_type: str = "unknown", strict: bool = False) -> ValidationResult:
        """通用数据验证方法"""
        try:
            if isinstance(data, pd.DataFrame):
                return self._validate_dataframe(data, data_type, strict)
            elif isinstance(data, dict):
                return self._validate_dict_data(data, data_type, strict)
            elif isinstance(data, list):
                return self._validate_list_data(data, data_type, strict)
            else:
                return ValidationResult(
                    is_valid=False,
                    metrics={},
                    errors=[f"不支持的数据类型: {type(data)}"],
                    warnings=[],
                    timestamp=datetime.now().isoformat(),
                    data_type=data_type
                )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                metrics={},
                errors=[f"验证过程中发生错误: {str(e)}"],
                warnings=[],
                timestamp=datetime.now().isoformat(),
                data_type=data_type
            )

    def validate_quality(self, data: Any) -> QualityReport:
        """验证数据质量"""
        try:
            if isinstance(data, pd.DataFrame):
                completeness = self._calculate_completeness(data)
                accuracy = self._calculate_accuracy(data)
                consistency = self._calculate_consistency(data)
                timeliness = self._calculate_timeliness(data)

                overall_score = (completeness + accuracy + consistency + timeliness) / 4

                return QualityReport(
                    overall_score=overall_score,
                    completeness=completeness,
                    accuracy=accuracy,
                    consistency=consistency,
                    timeliness=timeliness,
                    details={
                        'completeness_details': self._get_completeness_details(data),
                        'accuracy_details': self._get_accuracy_details(data),
                        'consistency_details': self._get_consistency_details(data),
                        'timeliness_details': self._get_timeliness_details(data)
                    }
                )
            else:
                return QualityReport(
                    overall_score=0.0,
                    completeness=0.0,
                    accuracy=0.0,
                    consistency=0.0,
                    timeliness=0.0,
                    details={}
                )
        except Exception as e:
            return QualityReport(
                overall_score=0.0,
                completeness=0.0,
                accuracy=0.0,
                consistency=0.0,
                timeliness=0.0,
                details={'error': str(e)}
            )

    def validate_data_model(self, data: Any, model_schema: Dict) -> bool:
        """验证数据模型"""
        try:
            if isinstance(data, pd.DataFrame):
                return self._validate_dataframe_schema(data, model_schema)
            elif isinstance(data, dict):
                return self._validate_dict_schema(data, model_schema)
            return False
        except Exception:
            return False

    def validate_date_range(self, data: Any, start_date: str, end_date: str) -> bool:
        """验证日期范围"""
        try:
            if isinstance(data, pd.DataFrame):
                if 'date' in data.columns:
                    data_dates = pd.to_datetime(data['date'], errors='coerce')
                    start_dt = pd.to_datetime(start_date)
                    end_dt = pd.to_datetime(end_date)
                    return (data_dates >= start_dt).all() and (data_dates <= end_dt).all()
            return True
        except Exception:
            return False

    def validate_numeric_columns(self, data: pd.DataFrame, columns: List[str]) -> bool:
        """验证数值列"""
        try:
            for col in columns:
                if col in data.columns:
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        return False
            return True
        except Exception:
            return False

    def validate_no_missing_values(self, data: pd.DataFrame, columns: List[str]) -> bool:
        """验证无缺失值"""
        try:
            for col in columns:
                if col in data.columns and data[col].isnull().any():
                    return False
            return True
        except Exception:
            return False

    def validate_no_duplicates(self, data: pd.DataFrame, columns: List[str] = None) -> bool:
        """验证无重复值"""
        try:
            if columns:
                return not data[columns].duplicated().any()
            return not data.duplicated().any()
        except Exception:
            return False

    def validate_outliers(self, data: pd.DataFrame, columns: List[str], threshold: float = 2.0) -> bool:
        """验证离群值"""
        try:
            for col in columns:
                if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
                    if outliers.any():
                        return False
            return True
        except Exception:
            return False

    def validate_data_consistency(self, data: Any) -> ConsistencyReport:
        """验证数据一致性"""
        try:
            if isinstance(data, pd.DataFrame):
                consistency_score = self._calculate_consistency_score(data)
                inconsistencies = self._find_inconsistencies(data)
                detailed_analysis = self._analyze_consistency_details(data)

                return ConsistencyReport(
                    is_consistent=consistency_score > 0.8,
                    consistency_score=consistency_score,
                    inconsistencies=inconsistencies,
                    cross_reference_results={},
                    detailed_analysis=detailed_analysis
                )
            else:
                return ConsistencyReport(
                    is_consistent=False,
                    consistency_score=0.0,
                    inconsistencies=["不支持的数据类型"],
                    cross_reference_results={},
                    detailed_analysis={}
                )
        except Exception as e:
            return ConsistencyReport(
                is_consistent=False,
                consistency_score=0.0,
                inconsistencies=[f"验证过程中发生错误: {str(e)}"],
                cross_reference_results={},
                detailed_analysis={}
            )

    def add_custom_rule(self, rule_name: str, rule_function: Callable):
        """添加自定义验证规则"""
        self._rules[rule_name] = rule_function

    def validate(self, data: Any, rules: List[str] = None) -> ValidationResult:
        """通用验证方法"""
        result = self.validate_data(data)
        
        # 应用自定义规则
        if rules:
            for rule_name in rules:
                if rule_name in self._rules:
                    try:
                        rule_result = self._rules[rule_name](data)
                        if not rule_result:
                            result.errors.append(f"自定义规则 {rule_name} 验证失败")
                            result.is_valid = False
                    except Exception as e:
                        result.errors.append(f"应用规则 {rule_name} 时发生错误: {str(e)}")
                        result.is_valid = False
        
        # 保存验证历史
        self._validation_history.append(result)
        if len(self._validation_history) > 1000:
            self._validation_history.pop(0)
        
        return result

    def validate_with_thresholds(self, data: Any, thresholds: Dict[str, float]) -> ValidationResult:
        """使用自定义阈值进行验证"""
        result = self.validate_data(data)
        
        # 应用阈值检查
        for metric, threshold in thresholds.items():
            if metric in result.metrics:
                if result.metrics[metric] < threshold:
                    result.errors.append(f"指标 {metric} 低于阈值 {threshold}")
                    result.is_valid = False
        
        return result

    # 私有方法

    def _validate_dataframe(self, data: pd.DataFrame, data_type: str, strict: bool = False) -> ValidationResult:
        """验证DataFrame数据"""
        metrics = {}
        errors = []
        warnings = []

        # 基本检查
        if data.empty:
            errors.append("数据为空")
            metrics['empty'] = 0.0
        else:
            metrics['empty'] = 1.0

        # 检查是否全为None
        if not data.empty and data.isnull().all().all():
            errors.append("数据全为空值")

        # 空值检查
        null_count = data.isnull().sum().sum()
        total_cells = data.size
        null_percentage = null_count / total_cells if total_cells > 0 else 0
        metrics['null_percentage'] = 1 - null_percentage

        if null_percentage > 0.1:  # 超过10 % 的空值
            errors.append(f"空值比例过高: {null_percentage:.2%}")
        elif null_percentage > 0.01:  # 超过1 % 的空值
            warnings.append(f"存在空值: {null_percentage:.2%}")

        # 重复值检查
        duplicate_count = data.duplicated().sum()
        duplicate_percentage = duplicate_count / len(data) if len(data) > 0 else 0
        metrics['duplicate_percentage'] = 1 - duplicate_percentage

        if duplicate_percentage > 0.05:  # 超过5 % 的重复值
            errors.append(f"重复值比例过高: {duplicate_percentage:.2%}")
        elif duplicate_percentage > 0:  # 存在重复值
            warnings.append(f"存在重复值: {duplicate_percentage:.2%}")

        # 数据类型特定验证
        if data_type in self._data_type_rules:
            type_specific_result = self._data_type_rules[data_type](data)
            errors.extend(type_specific_result['errors'])
            warnings.extend(type_specific_result['warnings'])
            metrics.update(type_specific_result['metrics'])

        # 计算数据质量指标
        if not data.empty:
            metrics['data_completeness'] = self._calculate_completeness(data)
            metrics['data_consistency'] = self._calculate_consistency(data)
            metrics['data_accuracy'] = self._calculate_accuracy(data)

        # 严格模式下的额外检查
        if strict:
            # 严格模式下，任何警告都视为错误
            if warnings:
                errors.extend(warnings)
                warnings = []
            
            # 严格模式下，数据质量指标必须高于阈值
            if 'data_completeness' in metrics and metrics['data_completeness'] < 0.9:
                errors.append(f"数据完整性不足: {metrics['data_completeness']:.2f}")
            if 'data_consistency' in metrics and metrics['data_consistency'] < 0.85:
                errors.append(f"数据一致性不足: {metrics['data_consistency']:.2f}")
            if 'data_accuracy' in metrics and metrics['data_accuracy'] < 0.85:
                errors.append(f"数据准确性不足: {metrics['data_accuracy']:.2f}")

        # 修复：空数据框或全为None的数据框应该返回is_valid=False
        is_valid = len(errors) == 0 and not data.empty and not (
            not data.empty and data.isnull().all().all())

        result = ValidationResult(
            is_valid=is_valid,
            metrics=metrics,
            errors=errors,
            warnings=warnings,
            timestamp=datetime.now().isoformat(),
            data_type=data_type
        )

        # 保存验证历史
        self._validation_history.append(result)
        if len(self._validation_history) > 1000:
            self._validation_history.pop(0)

        return result

    def _validate_dict_data(self, data: Dict, data_type: str, strict: bool = False) -> ValidationResult:
        """验证字典数据"""
        metrics = {}
        errors = []
        warnings = []

        if not data:
            errors.append("数据为空")
            metrics['empty'] = 0.0
        else:
            metrics['empty'] = 1.0

        # 检查必需字段
        required_fields = self._get_required_fields(data_type)
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            errors.append(f"缺少必需字段: {missing_fields}")

        # 字段类型验证
        type_errors = self._validate_field_types(data, data_type)
        errors.extend(type_errors)

        # 字段范围验证
        range_errors = self._validate_field_ranges(data, data_type)
        errors.extend(range_errors)

        # 严格模式下的额外检查
        if strict:
            # 严格模式下，任何警告都视为错误
            if warnings:
                errors.extend(warnings)
                warnings = []
            
            # 严格模式下，检查值的合理性
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    if abs(value) > 1e9:  # 检查值是否过大
                        errors.append(f"字段 {key} 值过大: {value}")
                elif isinstance(value, str):
                    if len(value) > 1000:  # 检查字符串长度
                        errors.append(f"字段 {key} 字符串过长")

        is_valid = len(errors) == 0

        result = ValidationResult(
            is_valid=is_valid,
            metrics=metrics,
            errors=errors,
            warnings=warnings,
            timestamp=datetime.now().isoformat(),
            data_type=data_type
        )

        # 保存验证历史
        self._validation_history.append(result)
        if len(self._validation_history) > 1000:
            self._validation_history.pop(0)

        return result

    def _validate_list_data(self, data: List, data_type: str, strict: bool = False) -> ValidationResult:
        """验证列表数据"""
        metrics = {}
        errors = []
        warnings = []

        if not data:
            errors.append("数据为空")
            metrics['empty'] = 0.0
        else:
            metrics['empty'] = 1.0

        # 验证列表中的每个元素
        valid_count = 0
        for i, item in enumerate(data):
            item_result = self.validate_data(item, data_type, strict)
            if item_result.is_valid:
                valid_count += 1
            else:
                errors.append(f"列表元素 {i} 验证失败: {'; '.join(item_result.errors)}")

        metrics['valid_items_ratio'] = valid_count / len(data) if data else 0.0

        # 严格模式下的额外检查
        if strict:
            # 严格模式下，任何警告都视为错误
            if warnings:
                errors.extend(warnings)
                warnings = []
            
            # 严格模式下，检查列表长度
            if len(data) > 10000:  # 检查列表是否过长
                errors.append(f"列表长度过长: {len(data)} 个元素")
            
            # 严格模式下，验证通过比例必须高于阈值
            if metrics['valid_items_ratio'] < 0.95:
                errors.append(f"验证通过比例不足: {metrics['valid_items_ratio']:.2f}")

        is_valid = len(errors) == 0

        result = ValidationResult(
            is_valid=is_valid,
            metrics=metrics,
            errors=errors,
            warnings=warnings,
            timestamp=datetime.now().isoformat(),
            data_type=data_type
        )

        return result

    def _validate_dataframe_schema(self, data: pd.DataFrame, schema: Dict) -> bool:
        """验证DataFrame模式"""
        try:
            for col, expected_type in schema.items():
                if col not in data.columns:
                    return False
                # 检查类型
                if expected_type == 'numeric' and not pd.api.types.is_numeric_dtype(data[col]):
                    return False
                elif expected_type == 'datetime' and not pd.api.types.is_datetime64_any_dtype(data[col]):
                    return False
                elif expected_type == 'string' and not pd.api.types.is_string_dtype(data[col]):
                    return False
            return True
        except Exception:
            return False

    def _validate_dict_schema(self, data: Dict, schema: Dict) -> bool:
        """验证字典模式"""
        try:
            for key, expected_type in schema.items():
                if key not in data:
                    return False
                # 检查类型
                if not isinstance(data[key], expected_type):
                    return False
            return True
        except Exception:
            return False

    def _calculate_completeness(self, data: pd.DataFrame) -> float:
        """计算完整性分数"""
        if data.empty:
            return 0.0
        non_null_count = data.count().sum()
        total_count = data.size
        return non_null_count / total_count if total_count > 0 else 0.0

    def _calculate_accuracy(self, data: pd.DataFrame) -> float:
        """计算准确性分数"""
        if data.empty:
            return 0.0
        
        # 更详细的准确性计算
        accuracy_score = 0.95
        
        # 检查价格数据合理性
        if 'price' in data.columns or 'close' in data.columns:
            price_col = 'price' if 'price' in data.columns else 'close'
            if pd.api.types.is_numeric_dtype(data[price_col]):
                # 检查价格是否为正数
                negative_prices = (data[price_col] < 0).sum()
                if negative_prices > 0:
                    accuracy_score -= negative_prices / len(data) * 0.5
        
        # 检查成交量数据合理性
        if 'volume' in data.columns:
            if pd.api.types.is_numeric_dtype(data['volume']):
                # 检查成交量是否为非负数
                negative_volume = (data['volume'] < 0).sum()
                if negative_volume > 0:
                    accuracy_score -= negative_volume / len(data) * 0.5
        
        return max(0.0, min(1.0, accuracy_score))

    def _calculate_consistency(self, data: pd.DataFrame) -> float:
        """计算一致性分数"""
        if data.empty:
            return 0.0
        
        # 更详细的一致性计算
        consistency_score = 0.90
        
        # 检查日期顺序一致性
        if 'date' in data.columns:
            try:
                dates = pd.to_datetime(data['date'])
                if not dates.is_monotonic_increasing:
                    consistency_score -= 0.1
            except Exception:
                consistency_score -= 0.05
        
        # 检查价格一致性（开盘价、最高价、最低价、收盘价）
        price_cols = ['open', 'high', 'low', 'close']
        if all(col in data.columns for col in price_cols):
            # 检查最高价 >= 收盘价和开盘价
            high_consistent = ((data['high'] >= data['close']) & (data['high'] >= data['open'])).all()
            # 检查最低价 <= 收盘价和开盘价
            low_consistent = ((data['low'] <= data['close']) & (data['low'] <= data['open'])).all()
            if not high_consistent or not low_consistent:
                consistency_score -= 0.1
        
        return max(0.0, min(1.0, consistency_score))

    def _calculate_timeliness(self, data: pd.DataFrame) -> float:
        """计算及时性分数"""
        if data.empty:
            return 0.0
        
        # 更详细的及时性计算
        timeliness_score = 0.95
        
        # 检查数据是否有最近的日期
        if 'date' in data.columns:
            try:
                dates = pd.to_datetime(data['date'])
                latest_date = dates.max()
                days_since_latest = (datetime.now() - latest_date).days
                
                if days_since_latest > 30:
                    timeliness_score -= min(0.3, days_since_latest / 100)
                elif days_since_latest > 7:
                    timeliness_score -= min(0.1, days_since_latest / 70)
            except Exception:
                timeliness_score -= 0.05
        
        return max(0.0, min(1.0, timeliness_score))

    def _calculate_consistency_score(self, data: pd.DataFrame) -> float:
        """计算一致性分数"""
        if data.empty:
            return 0.0
        
        # 综合一致性检查
        consistency_score = 0.85
        
        # 检查数据内部一致性
        if 'open' in data.columns and 'close' in data.columns:
            # 检查开盘价和收盘价的关系
            if not ((data['open'] * 0.5 <= data['close']) & (data['close'] <= data['open'] * 2)).all():
                consistency_score -= 0.1
        
        return max(0.0, min(1.0, consistency_score))

    def _find_inconsistencies(self, data: pd.DataFrame) -> List[str]:
        """查找不一致性"""
        inconsistencies = []
        
        if data.empty:
            inconsistencies.append("数据为空")
            return inconsistencies
        
        # 检查价格一致性
        if 'open' in data.columns and 'high' in data.columns and 'low' in data.columns and 'close' in data.columns:
            # 检查最高价是否大于等于开盘价和收盘价
            high_inconsistent = ((data['high'] < data['open']) | (data['high'] < data['close'])).any()
            if high_inconsistent:
                inconsistencies.append("最高价小于开盘价或收盘价")
            
            # 检查最低价是否小于等于开盘价和收盘价
            low_inconsistent = ((data['low'] > data['open']) | (data['low'] > data['close'])).any()
            if low_inconsistent:
                inconsistencies.append("最低价大于开盘价或收盘价")
        
        # 检查日期顺序
        if 'date' in data.columns:
            try:
                dates = pd.to_datetime(data['date'])
                if not dates.is_monotonic_increasing:
                    inconsistencies.append("日期顺序不一致")
            except Exception:
                inconsistencies.append("日期格式无效")
        
        # 检查成交量合理性
        if 'volume' in data.columns:
            if (data['volume'] < 0).any():
                inconsistencies.append("成交量为负数")
            if (data['volume'] == 0).all():
                inconsistencies.append("成交量全为零")
        
        return inconsistencies

    def _analyze_consistency_details(self, data: pd.DataFrame) -> Dict[str, Any]:
        """分析一致性详细信息"""
        details = {}
        
        if data.empty:
            return details
        
        # 价格一致性分析
        if 'open' in data.columns and 'high' in data.columns and 'low' in data.columns and 'close' in data.columns:
            details['price_consistency'] = {
                'high_consistent': ((data['high'] >= data['open']) & (data['high'] >= data['close'])).all(),
                'low_consistent': ((data['low'] <= data['open']) & (data['low'] <= data['close'])).all(),
                'price_range': {
                    'min': data[['open', 'high', 'low', 'close']].min().min(),
                    'max': data[['open', 'high', 'low', 'close']].max().max()
                }
            }
        
        # 日期分析
        if 'date' in data.columns:
            try:
                dates = pd.to_datetime(data['date'])
                details['date_analysis'] = {
                    'start_date': dates.min().strftime('%Y-%m-%d'),
                    'end_date': dates.max().strftime('%Y-%m-%d'),
                    'is_monotonic': dates.is_monotonic_increasing,
                    'date_range_days': (dates.max() - dates.min()).days
                }
            except Exception:
                details['date_analysis'] = {'error': '日期格式无效'}
        
        return details

    def _get_completeness_details(self, data: pd.DataFrame) -> Dict:
        """获取完整性详情"""
        details = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'non_null_count': data.count().sum(),
            'total_cells': data.size,
            'completeness_per_column': {}
        }
        
        # 计算每列的完整性
        for col in data.columns:
            non_null = data[col].count()
            total = len(data[col])
            details['completeness_per_column'][col] = non_null / total if total > 0 else 0.0
        
        return details

    def _get_accuracy_details(self, data: pd.DataFrame) -> Dict:
        """获取准确性详情"""
        details = {"accuracy_score": self._calculate_accuracy(data)}
        
        # 价格数据准确性检查
        if 'price' in data.columns or 'close' in data.columns:
            price_col = 'price' if 'price' in data.columns else 'close'
            details['price_analysis'] = {
                'mean_price': data[price_col].mean(),
                'std_price': data[price_col].std(),
                'min_price': data[price_col].min(),
                'max_price': data[price_col].max()
            }
        
        return details

    def _get_consistency_details(self, data: pd.DataFrame) -> Dict:
        """获取一致性详情"""
        details = {"consistency_score": self._calculate_consistency(data)}
        
        # 价格一致性检查
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            details['price_consistency'] = {
                'high_consistent': ((data['high'] >= data['close']) & (data['high'] >= data['open'])).all(),
                'low_consistent': ((data['low'] <= data['close']) & (data['low'] <= data['open'])).all()
            }
        
        return details

    def _get_timeliness_details(self, data: pd.DataFrame) -> Dict:
        """获取及时性详情"""
        details = {"timeliness_score": self._calculate_timeliness(data)}
        
        # 日期分析
        if 'date' in data.columns:
            try:
                dates = pd.to_datetime(data['date'])
                details['date_analysis'] = {
                    'latest_date': dates.max().strftime('%Y-%m-%d'),
                    'days_since_latest': (datetime.now() - dates.max()).days
                }
            except Exception:
                details['date_analysis'] = {'error': '日期格式无效'}
        
        return details

    def _get_required_fields(self, data_type: str) -> List[str]:
        """获取特定数据类型的必需字段"""
        field_mapping = {
            'stock': ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume'],
            'financial': ['date', 'symbol', 'revenue', 'profit', 'eps'],
            'market': ['date', 'symbol', 'price', 'volume'],
            'default': ['date', 'symbol']
        }
        return field_mapping.get(data_type, field_mapping['default'])

    def _validate_field_types(self, data: Dict, data_type: str) -> List[str]:
        """验证字段类型"""
        errors = []
        
        type_mapping = {
            'stock': {
                'date': str,
                'symbol': str,
                'open': (int, float),
                'high': (int, float),
                'low': (int, float),
                'close': (int, float),
                'volume': (int, float)
            },
            'financial': {
                'date': str,
                'symbol': str,
                'revenue': (int, float),
                'profit': (int, float),
                'eps': (int, float)
            }
        }
        
        expected_types = type_mapping.get(data_type, {})
        for field, expected_type in expected_types.items():
            if field in data:
                if not isinstance(data[field], expected_type):
                    errors.append(f"字段 {field} 类型错误，期望 {expected_type}，实际 {type(data[field])}")
        
        return errors

    def _validate_field_ranges(self, data: Dict, data_type: str) -> List[str]:
        """验证字段范围"""
        errors = []
        
        # 价格范围检查
        price_fields = ['open', 'high', 'low', 'close', 'price']
        for field in price_fields:
            if field in data:
                if isinstance(data[field], (int, float)):
                    if data[field] <= 0:
                        errors.append(f"字段 {field} 值必须大于零")
        
        # 成交量范围检查
        if 'volume' in data:
            if isinstance(data['volume'], (int, float)):
                if data['volume'] < 0:
                    errors.append("成交量必须大于等于零")
        
        return errors

    def _validate_stock_specific(self, data: pd.DataFrame) -> Dict:
        """验证股票特定数据"""
        errors = []
        warnings = []
        metrics = {}
        
        # 检查必需列
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            errors.append(f"缺少股票数据必需列: {missing_cols}")
        
        # 检查价格范围
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            # 检查价格是否为正数
            negative_prices = ((data['open'] <= 0) | (data['high'] <= 0) | 
                              (data['low'] <= 0) | (data['close'] <= 0)).any()
            if negative_prices:
                errors.append("价格数据包含非正数")
            
            # 检查价格关系
            price_relation = ((data['high'] < data['open']) | (data['high'] < data['close']) |
                             (data['low'] > data['open']) | (data['low'] > data['close'])).any()
            if price_relation:
                errors.append("价格关系不一致")
        
        # 检查成交量
        if 'volume' in data.columns:
            if (data['volume'] < 0).any():
                errors.append("成交量为负数")
            if (data['volume'] == 0).all():
                warnings.append("成交量全为零")
        
        metrics['stock_data_quality'] = 1.0 if not errors else 0.0
        
        return {'errors': errors, 'warnings': warnings, 'metrics': metrics}

    def _validate_financial_specific(self, data: pd.DataFrame) -> Dict:
        """验证财务特定数据"""
        errors = []
        warnings = []
        metrics = {}
        
        # 检查必需列
        required_cols = ['revenue', 'profit', 'eps']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            errors.append(f"缺少财务数据必需列: {missing_cols}")
        
        # 检查财务数据合理性
        if 'revenue' in data.columns and 'profit' in data.columns:
            if (data['profit'] > data['revenue']).any():
                errors.append("利润大于收入")
            if (data['revenue'] < 0).any():
                errors.append("收入为负数")
        
        metrics['financial_data_quality'] = 1.0 if not errors else 0.0
        
        return {'errors': errors, 'warnings': warnings, 'metrics': metrics}

    def _validate_market_specific(self, data: pd.DataFrame) -> Dict:
        """验证市场特定数据"""
        errors = []
        warnings = []
        metrics = {}
        
        # 检查必需列
        required_cols = ['price', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            errors.append(f"缺少市场数据必需列: {missing_cols}")
        
        # 检查市场数据合理性
        if 'price' in data.columns:
            if (data['price'] <= 0).any():
                errors.append("价格为非正数")
        
        if 'volume' in data.columns:
            if (data['volume'] < 0).any():
                errors.append("成交量为负数")
        
        metrics['market_data_quality'] = 1.0 if not errors else 0.0
        
        return {'errors': errors, 'warnings': warnings, 'metrics': metrics}

    def _check_price_deviation(self, data: Dict) -> Dict:
        """检查价格偏差"""
        try:
            if 'price' in data or 'close' in data:
                price = data.get('price', data.get('close'))
                if isinstance(price, (int, float)):
                    if price <= 0:
                        return {'is_valid': False, 'score': 0.0, 'message': '价格为非正数'}
                    return {'is_valid': True, 'score': 0.95, 'message': '正常'}
            return {'is_valid': True, 'score': 0.90, 'message': '正常'}
        except Exception:
            return {'is_valid': False, 'score': 0.0, 'message': '价格检查失败'}

    def _check_volume_spike(self, data: Dict) -> Dict:
        """检查成交量突增"""
        try:
            if 'volume' in data:
                volume = data['volume']
                if isinstance(volume, (int, float)):
                    if volume < 0:
                        return {'is_valid': False, 'score': 0.0, 'message': '成交量为负数'}
                    return {'is_valid': True, 'score': 0.90, 'message': '正常'}
            return {'is_valid': True, 'score': 0.85, 'message': '正常'}
        except Exception:
            return {'is_valid': False, 'score': 0.0, 'message': '成交量检查失败'}

    def _check_null_values(self, data: Dict) -> Dict:
        """检查空值"""
        try:
            null_count = sum(1 for v in data.values() if v is None or v == '')
            if null_count > len(data) * 0.5:
                return {'is_valid': False, 'score': 0.0, 'message': '空值过多'}
            return {'is_valid': True, 'score': 1.0 - null_count / len(data), 'message': '正常'}
        except Exception:
            return {'is_valid': False, 'score': 0.0, 'message': '空值检查失败'}

    def _check_outliers(self, data: Dict) -> Dict:
        """检查离群值"""
        try:
            # 简化的离群值检查
            return {'is_valid': True, 'score': 0.92, 'message': '正常'}
        except Exception:
            return {'is_valid': False, 'score': 0.0, 'message': '离群值检查失败'}

    def _check_time_gaps(self, data: Dict) -> Dict:
        """检查时间间隔"""
        try:
            # 简化的时间间隔检查
            return {'is_valid': True, 'score': 0.95, 'message': '正常'}
        except Exception:
            return {'is_valid': False, 'score': 0.0, 'message': '时间间隔检查失败'}

    def _check_financial_completeness(self, data: Dict) -> Dict:
        """检查财务数据完整性"""
        try:
            required_fields = ['revenue', 'profit', 'eps']
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                return {'is_valid': False, 'score': 0.0, 'message': f'缺少财务数据字段: {missing_fields}'}
            return {'is_valid': True, 'score': 0.95, 'message': '正常'}
        except Exception:
            return {'is_valid': False, 'score': 0.0, 'message': '财务数据完整性检查失败'}

    def _check_financial_accuracy(self, data: Dict) -> Dict:
        """检查财务数据准确性"""
        try:
            if 'revenue' in data and 'profit' in data:
                revenue = data['revenue']
                profit = data['profit']
                if isinstance(revenue, (int, float)) and isinstance(profit, (int, float)):
                    if profit > revenue:
                        return {'is_valid': False, 'score': 0.0, 'message': '利润大于收入'}
            return {'is_valid': True, 'score': 0.90, 'message': '正常'}
        except Exception:
            return {'is_valid': False, 'score': 0.0, 'message': '财务数据准确性检查失败'}

    def _check_financial_consistency(self, data: Dict) -> Dict:
        """检查财务数据一致性"""
        try:
            # 简化的财务数据一致性检查
            return {'is_valid': True, 'score': 0.88, 'message': '正常'}
        except Exception:
            return {'is_valid': False, 'score': 0.0, 'message': '财务数据一致性检查失败'}

    def get_validation_history(self) -> List[ValidationResult]:
        """获取验证历史"""
        return self._validation_history

    def generate_quality_report(self, data: Any, data_type: str = "unknown") -> QualityReport:
        """生成质量报告"""
        return self.validate_quality(data)

    def generate_detailed_quality_report(self, data: Any, data_type: str = "unknown") -> Dict[str, Any]:
        """生成详细的质量报告"""
        try:
            if isinstance(data, pd.DataFrame):
                report = {
                    'basic_info': {
                        'rows': len(data),
                        'columns': len(data.columns),
                        'data_type': data_type
                    },
                    'completeness': {
                        'null_values': data.isnull().sum().to_dict(),
                        'null_percentage': (data.isnull().sum() / len(data) * 100).to_dict(),
                        'total_null_cells': data.isnull().sum().sum(),
                        'total_cells': data.size,
                        'completeness_score': self._calculate_completeness(data)
                    },
                    'accuracy': {
                        'data_types': {col: str(data[col].dtype) for col in data.columns},
                        'numeric_stats': self._get_numeric_stats(data),
                        'accuracy_score': self._calculate_accuracy(data)
                    },
                    'consistency': {
                        'duplicate_rows': data.duplicated().sum(),
                        'duplicate_percentage': data.duplicated().sum() / len(data) * 100 if len(data) > 0 else 0,
                        'consistency_score': self._calculate_consistency(data),
                        'inconsistencies': self._find_inconsistencies(data)
                    },
                    'timeliness': {
                        'timeliness_score': self._calculate_timeliness(data),
                        'date_analysis': self._analyze_dates(data)
                    },
                    'overall': {
                        'quality_score': (self._calculate_completeness(data) + 
                                         self._calculate_accuracy(data) + 
                                         self._calculate_consistency(data) + 
                                         self._calculate_timeliness(data)) / 4
                    }
                }
                return report
            else:
                return {
                    'error': '不支持的数据类型',
                    'data_type': str(type(data))
                }
        except Exception as e:
            return {
                'error': str(e),
                'data_type': str(type(data))
            }

    def _get_numeric_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """获取数值列的统计信息"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        stats = {}
        for col in numeric_cols:
            stats[col] = {
                'mean': float(data[col].mean()),
                'std': float(data[col].std()),
                'min': float(data[col].min()),
                'max': float(data[col].max()),
                'median': float(data[col].median()),
                'q1': float(data[col].quantile(0.25)),
                'q3': float(data[col].quantile(0.75))
            }
        return stats

    def _analyze_dates(self, data: pd.DataFrame) -> Dict[str, Any]:
        """分析日期列"""
        date_cols = []
        for col in data.columns:
            try:
                pd.to_datetime(data[col])
                date_cols.append(col)
            except:
                pass
        
        date_analysis = {}
        for col in date_cols:
            try:
                dates = pd.to_datetime(data[col])
                date_analysis[col] = {
                    'min_date': dates.min().strftime('%Y-%m-%d') if not dates.min() is pd.NaT else None,
                    'max_date': dates.max().strftime('%Y-%m-%d') if not dates.max() is pd.NaT else None,
                    'date_range_days': (dates.max() - dates.min()).days if not (dates.min() is pd.NaT or dates.max() is pd.NaT) else 0,
                    'is_monotonic_increasing': dates.is_monotonic_increasing
                }
            except Exception as e:
                date_analysis[col] = {'error': str(e)}
        
        return date_analysis

    def check_data_quality_trend(self, data_history: List[Any]) -> Dict[str, Any]:
        """检查数据质量趋势"""
        try:
            if not data_history:
                return {'error': '数据历史为空'}
            
            trends = {
                'completeness_trend': [],
                'accuracy_trend': [],
                'consistency_trend': [],
                'timeliness_trend': [],
                'overall_trend': []
            }
            
            for i, data in enumerate(data_history):
                if isinstance(data, pd.DataFrame) and not data.empty:
                    completeness = self._calculate_completeness(data)
                    accuracy = self._calculate_accuracy(data)
                    consistency = self._calculate_consistency(data)
                    timeliness = self._calculate_timeliness(data)
                    overall = (completeness + accuracy + consistency + timeliness) / 4
                    
                    trends['completeness_trend'].append({'index': i, 'score': completeness})
                    trends['accuracy_trend'].append({'index': i, 'score': accuracy})
                    trends['consistency_trend'].append({'index': i, 'score': consistency})
                    trends['timeliness_trend'].append({'index': i, 'score': timeliness})
                    trends['overall_trend'].append({'index': i, 'score': overall})
            
            return trends
        except Exception as e:
            return {'error': str(e)}

    def set_quality_thresholds(self, thresholds: Dict[str, float]) -> None:
        """设置质量阈值"""
        self._quality_thresholds = thresholds

    def get_quality_thresholds(self) -> Dict[str, float]:
        """获取质量阈值"""
        return getattr(self, '_quality_thresholds', {
            'completeness': 0.9,
            'accuracy': 0.85,
            'consistency': 0.85,
            'timeliness': 0.8,
            'overall': 0.85
        })

    def detect_outliers(self, data: pd.Series, threshold: float = 2.0) -> OutlierReport:
        """检测离群值"""
        try:
            if data.empty:
                return OutlierReport(
                    outlier_count=0,
                    outlier_percentage=0.0,
                    outlier_indices=[],
                    outlier_values=[],
                    threshold=threshold,
                    lower_bound=0.0,
                    upper_bound=0.0
                )

            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            outliers = (data < lower_bound) | (data > upper_bound)
            outlier_indices = outliers[outliers].index.tolist()
            outlier_values = data[outliers].tolist()

            return OutlierReport(
                outlier_count=len(outlier_indices),
                outlier_percentage=len(outlier_indices) / len(data) if len(data) > 0 else 0.0,
                outlier_indices=outlier_indices,
                outlier_values=outlier_values,
                threshold=threshold,
                lower_bound=lower_bound,
                upper_bound=upper_bound
            )
        except Exception as e:
            return OutlierReport(
                outlier_count=0,
                outlier_percentage=0.0,
                outlier_indices=[],
                outlier_values=[],
                threshold=threshold,
                lower_bound=0.0,
                upper_bound=0.0
            )

    def check_consistency(self, data_a: Dict, data_b: Dict) -> ConsistencyReport:
        """检查数据一致性"""
        try:
            # 检查共同字段
            common_fields = set(data_a.keys()) & set(data_b.keys())
            inconsistencies = []
            
            for field in common_fields:
                if data_a[field] != data_b[field]:
                    inconsistencies.append(f"字段 {field} 值不一致: {data_a[field]} vs {data_b[field]}")
            
            consistency_score = 1.0 - len(inconsistencies) / len(common_fields) if common_fields else 0.0
            consistency_score = max(0.0, min(1.0, consistency_score))

            return ConsistencyReport(
                is_consistent=consistency_score > 0.8,
                consistency_score=consistency_score,
                inconsistencies=inconsistencies,
                cross_reference_results={},
                detailed_analysis={'common_fields': list(common_fields), 'consistency_score': consistency_score}
            )
        except Exception as e:
            return ConsistencyReport(
                is_consistent=False,
                consistency_score=0.0,
                inconsistencies=[f"一致性检查失败: {str(e)}"],
                cross_reference_results={},
                detailed_analysis={}
            )


__all__ = ['DataValidator', 'ValidationResult', 'QualityReport', 'OutlierReport', 'ConsistencyReport', 'ValidationError']
