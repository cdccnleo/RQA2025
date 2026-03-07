import logging
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 数据层统一质量监控系统

提供统一的数据质量监控、验证和修复功能，确保数据质量和完整性。
"""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

import pandas as pd

# 使用统一基础设施集成层（若不可用则降级）
try:
    from src.core.integration import (
        get_data_adapter,
        log_data_operation, record_data_metric
    )
except ImportError:  # pragma: no cover - 降级处理
    def get_data_adapter():
        return None

    def log_data_operation(*args, **kwargs):
        return None

    def record_data_metric(*args, **kwargs):
        return None

# 导入标准接口（本工程内提供完整 DataSourceType）
from ..interfaces.standard_interfaces import DataSourceType  # type: ignore
from .unified_quality_monitor_interface import IDataQualityMonitor
from typing import Protocol

class IDataValidator(Protocol):  # type: ignore
    def validate(self, data: Any, data_type: Any) -> Dict[str, Any]: ...
    def get_validation_rules(self, data_type: Any) -> Dict[str, Any]: ...


def _compat_type(name: str) -> Any:
    try:
        return getattr(DataSourceType, name)
    except Exception:
        class _Compat:
            def __init__(self, v: str):
                self.value = v.lower()
            def __repr__(self) -> str:
                return f"<CompatDataSourceType {self.value.upper()}>"
            def __eq__(self, other) -> bool:
                if hasattr(other, "value"):
                    return str(getattr(other, "value")).lower() == self.value
                return False
            def __hash__(self) -> int:
                return hash(self.value)
        return _Compat(name)

TYPE_STOCK = _compat_type("STOCK")
TYPE_CRYPTO = _compat_type("CRYPTO")
TYPE_NEWS = _compat_type("NEWS")
TYPE_MACRO = _compat_type("MACRO")
DEFAULT_DATA_TYPE = _compat_type("DATABASE")


@dataclass
class QualityConfig:

    """质量监控配置"""
    enable_realtime_monitoring: bool = True  # 启用实时监控
    enable_historical_analysis: bool = True  # 启用历史分析
    quality_threshold: float = 0.8          # 质量阈值
    anomaly_detection_sensitivity: float = 0.95  # 异常检测灵敏度
    max_quality_history: int = 1000         # 最大质量历史记录数
    alert_cooldown_minutes: int = 5         # 告警冷却时间(分钟)
    enable_auto_repair: bool = False        # 启用自动修复
    repair_confidence_threshold: float = 0.9  # 修复置信度阈值


@dataclass
class QualityMetrics:

    """质量指标"""
    completeness: float = 0.0      # 完整性
    accuracy: float = 0.0          # 准确性
    consistency: float = 0.0       # 一致性
    timeliness: float = 0.0        # 时效性
    validity: float = 0.0          # 有效性
    overall_score: float = 0.0     # 综合得分
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class QualityIssue:

    """质量问题"""
    issue_type: str
    severity: str  # critical, high, medium, low
    description: str
    affected_records: int = 0
    suggested_fix: Optional[str] = None
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class QualityReport:

    """质量报告"""
    data_type: DataSourceType
    period: str
    metrics: QualityMetrics
    issues: List[QualityIssue] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)


class UnifiedDataValidator(IDataValidator):

    """
    统一数据验证器

    提供标准化的数据验证功能，支持多种数据类型的验证规则。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        self.config = config or {}
        self.validation_rules = self._load_validation_rules()

    def _load_validation_rules(self) -> Dict[DataSourceType, Dict[str, Any]]:
        """加载验证规则"""
        templates = {
            TYPE_STOCK: {
                "price_range": {"min": 0, "max": 1000000},
                "volume_range": {"min": 0, "max": 1000000000},
                "price_change_limit": 0.11,
                "required_fields": ["symbol", "timestamp", "open", "high", "low", "close", "volume"],
                "timestamp_format": "%Y-%m-%d %H:%M:%S",
            },
            # 标准映射验证规则
            "standard_mapping": {
                "price_range": {"min": 0, "max": 1000000},
                "volume_range": {"min": 0, "max": 1000000000},
                "price_change_limit": 0.11,
                "required_fields": ["source_id", "symbol", "date", "data_type", "open_price", "high_price", "low_price", "close_price", "volume", "amount"],
                "required_keywords": ["pct_change", "change", "turnover_rate", "amplitude"],
                "valid_source_ids": ["akshare_stock_a", "akshare_stock_hk", "akshare_stock_us"],
                "valid_data_types": ["daily", "weekly", "monthly", "quarterly", "yearly"],
                "timestamp_format": "%Y-%m-%d",
            },
            TYPE_CRYPTO: {
                "price_range": {"min": 0, "max": 10000000},
                "volume_range": {"min": 0, "max": 10000000000},
                "required_fields": ["symbol", "timestamp", "price", "volume", "market_cap"],
                "timestamp_format": "%Y-%m-%d %H:%M:%S",
            },
            TYPE_NEWS: {
                "content_length": {"min": 10, "max": 10000},
                "required_fields": ["title", "content", "timestamp", "source"],
                "sentiment_range": {"min": -1.0, "max": 1.0},
                "timestamp_format": "%Y-%m-%d %H:%M:%S",
            },
            TYPE_MACRO: {
                "value_range": {"min": -1000000, "max": 1000000},
                "required_fields": ["indicator", "value", "timestamp", "unit"],
                "timestamp_format": "%Y-%m-%d",
            },
        }

        return {dtype: cfg for dtype, cfg in templates.items() if dtype is not None}

    def _normalize_data_type(self, data_type: DataSourceType) -> DataSourceType:
        """根据传入类型推断具体验证模板"""
        if data_type in (TYPE_STOCK, TYPE_CRYPTO, TYPE_NEWS, TYPE_MACRO):
            return data_type
        # 兼容：按名称字符串匹配，避免直接访问缺失的枚举成员
        name = str(getattr(data_type, "value", data_type)).upper()
        if name in ("DATABASE", "TABLE"):
            return TYPE_STOCK
        if name == "STREAM":
            return TYPE_CRYPTO
        if name == "API":
            return TYPE_NEWS
        # 默认使用通用配置
        return DEFAULT_DATA_TYPE

    def validate(self, data: Any, data_type: DataSourceType) -> Dict[str, Any]:
        """验证数据"""
        try:
            normalized_type = self._normalize_data_type(data_type)

            if data is None:
                return {
                    "valid": False,
                    "issues": [QualityIssue(
                        issue_type="null_data",
                        severity="critical",
                        description="数据为空",
                        confidence=1.0
                    )]
                }

            rules = self.validation_rules.get(normalized_type, {})
            issues = []

            # 检查是否为标准映射数据
            if isinstance(data_type, str) and data_type == "standard_mapping":
                # 使用标准映射验证规则
                standard_rules = self.validation_rules.get("standard_mapping", {})
                issues.extend(self._validate_standard_mapping(data, standard_rules))
            else:
                # 基础验证
                issues.extend(self._validate_basic_structure(data, rules))

                # 数据类型特定验证
                if normalized_type == TYPE_STOCK:
                    issues.extend(self._validate_structured_data(data, rules))
                elif normalized_type == TYPE_CRYPTO:
                    issues.extend(self._validate_stream_data(data, rules))
                elif normalized_type == TYPE_NEWS:
                    issues.extend(self._validate_api_data(data, rules))

            # 计算整体有效性
            valid = len([i for i in issues if i.severity == "critical"]) == 0

            return {
                "valid": valid,
                "issues": issues,
                "issue_count": len(issues),
                "critical_issues": len([i for i in issues if i.severity == "critical"])
            }

        except Exception as e:
            logger.error(f"数据验证失败: {e}")
            return {
                "valid": False,
                "issues": [QualityIssue(
                    issue_type="validation_error",
                    severity="critical",
                    description=f"验证过程出错: {str(e)}",
                    confidence=1.0
                )]
            }

    def _validate_basic_structure(self, data: Any, rules: Dict[str, Any]) -> List[QualityIssue]:
        """验证基本结构"""
        issues = []

        # 检查必需字段
        required_fields = rules.get("required_fields", [])
        if required_fields:
            if hasattr(data, 'columns'):
                # DataFrame
                missing_fields = [field for field in required_fields if field not in data.columns]
                if missing_fields:
                    issues.append(QualityIssue(
                        issue_type="missing_fields",
                        severity="critical",
                        description=f"缺少必需字段: {missing_fields}",
                        affected_records=len(data) if hasattr(data, '__len__') else 0,
                        confidence=1.0
                    ))
            elif isinstance(data, dict):
                # 字典
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    issues.append(QualityIssue(
                        issue_type="missing_fields",
                        severity="critical",
                        description=f"缺少必需字段: {missing_fields}",
                        confidence=1.0
                    ))

        return issues

    def _validate_structured_data(self, data: Any, rules: Dict[str, Any]) -> List[QualityIssue]:
        """验证结构化表格数据"""
        issues = []

        if hasattr(data, 'columns'):
            # 检查价格合理性
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in data.columns:
                    price_range = rules.get("price_range", {})
                    min_price = price_range.get("min", 0)
                    max_price = price_range.get("max", 1000000)

                    invalid_prices = ((data[col] < min_price) | (data[col] > max_price)).sum()
                    if invalid_prices > 0:
                        issues.append(QualityIssue(
                            issue_type="invalid_price",
                            severity="high",
                            description=f"{col}列存在无效价格值",
                            affected_records=int(invalid_prices),
                            confidence=0.9
                        ))

            # 检查涨跌停限制
            if 'close' in data.columns and 'open' in data.columns:
                price_change_limit = rules.get("price_change_limit", 0.11)
                price_changes = abs((data['close'] - data['open']) / data['open'])
                limit_breaches = (price_changes > price_change_limit).sum()
                if limit_breaches > 0:
                    issues.append(QualityIssue(
                        issue_type="price_limit_breach",
                        severity="medium",
                        description="价格变动超过涨跌停限制",
                        affected_records=int(limit_breaches),
                        confidence=0.8
                    ))

        return issues

    def _validate_stream_data(self, data: Any, rules: Dict[str, Any]) -> List[QualityIssue]:
        """验证流式价格数据"""
        issues = []

        # 类似股票数据的验证，但有不同的价格范围
        if hasattr(data, 'columns'):
            if 'price' in data.columns:
                price_range = rules.get("price_range", {})
                min_price = price_range.get("min", 0)
                max_price = price_range.get("max", 10000000)

                invalid_prices = ((data['price'] < min_price) | (data['price'] > max_price)).sum()
                if invalid_prices > 0:
                    issues.append(QualityIssue(
                        issue_type="invalid_price",
                        severity="high",
                        description="价格值超出合理范围",
                        affected_records=int(invalid_prices),
                        confidence=0.9
                    ))

        return issues

    def _validate_api_data(self, data: Any, rules: Dict[str, Any]) -> List[QualityIssue]:
        """验证 API 数据"""
        issues = []

        if hasattr(data, 'columns'):
            # 检查内容长度
            if 'content' in data.columns:
                content_length = rules.get("content_length", {})
                min_length = content_length.get("min", 10)
                max_length = content_length.get("max", 10000)

                data['content_length'] = data['content'].astype(str).str.len()
                invalid_length = ((data['content_length'] < min_length) |
                                  (data['content_length'] > max_length)).sum()
                if invalid_length > 0:
                    issues.append(QualityIssue(
                        issue_type="invalid_content_length",
                        severity="medium",
                        description="内容长度不符合要求",
                        affected_records=int(invalid_length),
                        confidence=0.7
                    ))

            # 检查情感值范围
            if 'sentiment' in data.columns:
                sentiment_range = rules.get("sentiment_range", {})
                min_sentiment = sentiment_range.get("min", -1.0)
                max_sentiment = sentiment_range.get("max", 1.0)

                invalid_sentiment = ((data['sentiment'] < min_sentiment)
                                     | (data['sentiment'] > max_sentiment)).sum()
                if invalid_sentiment > 0:
                    issues.append(QualityIssue(
                        issue_type="invalid_sentiment",
                        severity="medium",
                        description="情感值超出合理范围",
                        affected_records=int(invalid_sentiment),
                        confidence=0.8
                    ))

        return issues
    
    def _validate_standard_mapping(self, data: Any, rules: Dict[str, Any]) -> List[QualityIssue]:
        """验证标准映射数据"""
        issues = []

        if isinstance(data, list):
            # 验证标准格式的字典列表
            for i, record in enumerate(data):
                record_issues = self._validate_standard_record(record, rules)
                for issue in record_issues:
                    issue.affected_records = 1
                    issues.append(issue)
        elif hasattr(data, 'columns'):
            # 验证DataFrame格式
            issues.extend(self._validate_structured_data(data, rules))

        return issues
    
    def _validate_standard_record(self, record: Dict[str, Any], rules: Dict[str, Any]) -> List[QualityIssue]:
        """验证单个标准格式记录"""
        issues = []

        # 检查必需字段
        required_fields = rules.get("required_fields", [])
        missing_fields = [field for field in required_fields if field not in record or record[field] is None]
        if missing_fields:
            issues.append(QualityIssue(
                issue_type="missing_fields",
                severity="critical",
                description=f"缺少必需字段: {missing_fields}",
                confidence=1.0
            ))

        # 检查关键字段
        required_keywords = rules.get("required_keywords", [])
        missing_keywords = [field for field in required_keywords if field not in record]
        if missing_keywords:
            issues.append(QualityIssue(
                issue_type="missing_keywords",
                severity="high",
                description=f"缺少关键字段: {missing_keywords}",
                confidence=0.9
            ))

        # 检查数据源标识
        valid_source_ids = rules.get("valid_source_ids", [])
        if "source_id" in record:
            source_id = record["source_id"]
            if valid_source_ids and source_id not in valid_source_ids:
                issues.append(QualityIssue(
                    issue_type="invalid_source_id",
                    severity="high",
                    description=f"无效的数据源标识: {source_id}",
                    confidence=1.0
                ))

        # 检查数据类型
        valid_data_types = rules.get("valid_data_types", [])
        if "data_type" in record:
            data_type = record["data_type"]
            if valid_data_types and data_type not in valid_data_types:
                issues.append(QualityIssue(
                    issue_type="invalid_data_type",
                    severity="high",
                    description=f"无效的数据类型: {data_type}",
                    confidence=1.0
                ))

        # 检查价格范围
        price_fields = ["open_price", "high_price", "low_price", "close_price"]
        price_range = rules.get("price_range", {"min": 0, "max": 1000000})
        for field in price_fields:
            if field in record and record[field] is not None:
                price = record[field]
                if price < price_range["min"] or price > price_range["max"]:
                    issues.append(QualityIssue(
                        issue_type="invalid_price",
                        severity="medium",
                        description=f"{field} 超出合理范围: {price}",
                        confidence=0.8
                    ))

        # 检查成交量范围
        if "volume" in record and record["volume"] is not None:
            volume = record["volume"]
            volume_range = rules.get("volume_range", {"min": 0, "max": 1000000000})
            if volume < volume_range["min"] or volume > volume_range["max"]:
                issues.append(QualityIssue(
                    issue_type="invalid_volume",
                    severity="medium",
                    description=f"成交量超出合理范围: {volume}",
                    confidence=0.8
                ))

        return issues

    def get_validation_rules(self, data_type: DataSourceType) -> Dict[str, Any]:
        """获取验证规则"""
        return self.validation_rules.get(data_type, {})

    def repair_data(self, data: Any, issues: List[QualityIssue]) -> Any:
        """修复数据问题"""
        try:
            if not self.config.get("enable_auto_repair", False):
                logger.info("自动修复已禁用")
                return data

            repaired_data = data.copy() if hasattr(data, 'copy') else data

            for issue in issues:
                if issue.confidence < self.config.get("repair_confidence_threshold", 0.9):
                    continue

                if issue.issue_type == "missing_fields":
                    # 尝试填充缺失字段
                    repaired_data = self._repair_missing_fields(repaired_data, issue)
                elif issue.issue_type == "invalid_price":
                    # 尝试修复无效价格
                    repaired_data = self._repair_invalid_prices(repaired_data, issue)

            return repaired_data

        except Exception as e:
            logger.error(f"数据修复失败: {e}")
            return data

    def _repair_missing_fields(self, data: Any, issue: QualityIssue) -> Any:
        """修复缺失字段"""
        # 这里实现具体的修复逻辑
        return data

    def _repair_invalid_prices(self, data: Any, issue: QualityIssue) -> Any:
        """修复无效价格"""
        # 这里实现具体的修复逻辑
        return data


class UnifiedQualityMonitor(IDataQualityMonitor):

    """
    统一质量监控器

    提供全面的数据质量监控、分析和报告功能。
    """

    @staticmethod
    def _coerce_config(config: Optional[Any]) -> QualityConfig:
        if config is None:
            return QualityConfig()
        if isinstance(config, QualityConfig):
            return config
        if isinstance(config, dict):
            allowed_keys = QualityConfig.__annotations__.keys()
            filtered = {k: v for k, v in config.items() if k in allowed_keys}
            return QualityConfig(**filtered)
        raise TypeError("Unsupported config type for UnifiedQualityMonitor")

    def __init__(self, *args, **kwargs):

        monitor_id = kwargs.pop("monitor_id", "default")
        config_input = kwargs.pop("config", None)

        if args:
            first = args[0]
            if isinstance(first, (QualityConfig, dict)):
                config_input = first
            else:
                monitor_id = str(first)
                if len(args) > 1:
                    config_input = args[1]

        # 使用基础设施集成管理器获取配置
        self.monitor_id = monitor_id
        self.config_obj = self._coerce_config(config_input)
        merged_config = self._load_config_from_integration_manager()

        # 初始化统一基础设施集成层适配器
        try:
            self.data_adapter = get_data_adapter()
        except Exception as e:
            print(f"获取统一基础设施集成层适配器失败: {e}")
            self.data_adapter = None

        self.validator = UnifiedDataValidator()
        self.quality_history: Dict[DataSourceType, List[QualityMetrics]] = defaultdict(list)
        self.alerts_sent: Dict[str, datetime] = {}
        self._alerts_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._monitored_sources: Dict[str, Dict[str, Any]] = {}
        self._thresholds: Dict[str, Dict[str, Any]] = {}
        self._alert_handlers: List[Callable[[Dict[str, Any]], None]] = []
        self.alert_threshold: float = float(self.config_obj.quality_threshold)
        self.auto_repair: bool = bool(self.config_obj.enable_auto_repair)

        # 注册健康检查
        self._register_health_checks()

        # 初始化质量监控
        self._initialize_quality_monitoring()

        log_data_operation("quality_monitor_init", TYPE_STOCK,
                           {"config": self.config_obj.__dict__, "monitor_id": self.monitor_id}, "info")

    def _load_config_from_integration_manager(self) -> Dict[str, Any]:
        """从统一基础设施集成层加载配置"""
        try:
            # 使用统一基础设施集成层获取配置
            merged_config = self.config_obj.__dict__.copy()

            # 尝试从统一基础设施集成层获取配置
            if hasattr(self, 'data_adapter') and self.data_adapter:
                config_manager = self.data_adapter.get_config_manager()
                if config_manager:
                    # 这里可以添加从配置中心获取配置的逻辑
                    pass

            return merged_config

        except Exception as e:
            # 如果集成管理器不可用，使用默认配置
            return self.config_obj.__dict__.copy()

    def _register_health_checks(self) -> None:
        """注册健康检查"""
        try:
            if hasattr(self, 'data_adapter') and self.data_adapter:
                health_checker = self.data_adapter.get_health_checker()
                if health_checker:
                    # 注册质量监控器健康检查
                    health_checker.register_check(
                        "quality_monitor",
                        self._quality_monitor_health_check
                    )

        except Exception as e:
            log_data_operation("quality_monitor_health_check_registration_error", TYPE_STOCK,
                               {"error": str(e)}, "warning")

    def _initialize_quality_monitoring(self) -> None:
        """初始化质量监控"""
        try:
            # 设置质量监控的初始状态
            self.monitoring_active = True
            self.last_quality_check = datetime.now()

            # 初始化质量基准线（如果有历史数据）
            self._initialize_quality_baselines()

        except Exception as e:
            log_data_operation("quality_monitoring_init_error", TYPE_STOCK,
                               {"error": str(e)}, "error")

    def _initialize_quality_baselines(self) -> None:
        """初始化质量基准线"""
        try:
            # 为每个数据类型设置质量基准
            self.quality_baselines = {}

            for data_type in DataSourceType:
                self.quality_baselines[data_type] = {
                    'completeness': 0.95,
                    'accuracy': 0.98,
                    'consistency': 0.90,
                    'timeliness': 0.85,
                    'validity': 0.99,
                    'overall_score': 0.93
                }

        except Exception as e:
            log_data_operation("quality_baselines_init_error", TYPE_STOCK,
                               {"error": str(e)}, "error")

    def _quality_monitor_health_check(self) -> Dict[str, Any]:
        """质量监控器健康检查"""
        try:
            health_status = {
                'component': 'UnifiedQualityMonitor',
                'status': 'healthy',
                'monitoring_active': self.monitoring_active,
                'quality_history_records': sum(len(history) for history in self.quality_history.values()),
                'alerts_sent_count': len(self.alerts_sent),
                'last_quality_check': self.last_quality_check.isoformat() if self.last_quality_check else None,
                'timestamp': datetime.now().isoformat()
            }

            # 检查关键指标
            if not self.monitoring_active:
                health_status['status'] = 'warning'
                health_status['message'] = '质量监控未激活'

            # 检查是否有过期的质量检查
            if self.last_quality_check and (datetime.now() - self.last_quality_check).seconds > 3600:
                health_status['status'] = 'warning'
                health_status['message'] = '质量检查已过期'

            return health_status

        except Exception as e:
            return {
                'component': 'UnifiedQualityMonitor',
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _normalize_data_type(self, data_type: Optional[DataSourceType]) -> DataSourceType:
        # 支持真实枚举或字符串，默认回退 STOCK
        if data_type is None:
            return TYPE_STOCK
        if isinstance(data_type, DataSourceType):
            return data_type
        if hasattr(data_type, "value"):
            val = getattr(data_type, "value")
            if isinstance(val, str):
                return getattr(DataSourceType, val.upper(), TYPE_STOCK)
        if isinstance(data_type, str):
            return getattr(DataSourceType, data_type.upper(), TYPE_STOCK)
        return TYPE_STOCK

    def _check_completeness(self, data: Any) -> float:
        if not hasattr(data, "isnull") or not hasattr(data, "shape"):
            return 1.0
        total_cells = int(data.shape[0] * data.shape[1]) or 1
        missing = int(data.isnull().sum().sum())
        completeness = 1.0 - (missing / total_cells)
        return float(max(0.0, min(1.0, completeness)))

    def _check_accuracy(self, data: Any) -> float:
        if not hasattr(data, "select_dtypes"):
            return 1.0
        numeric = data.select_dtypes(include=["number"])
        if numeric.empty:
            return 1.0
        q1 = numeric.quantile(0.25)
        q3 = numeric.quantile(0.75)
        iqr = (q3 - q1).replace({0: 1e-9})
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((numeric < lower) | (numeric > upper)).sum().sum()
        total = numeric.size or 1
        accuracy = 1.0 - (outliers / total)
        return float(max(0.0, min(1.0, accuracy)))

    def _check_timeliness(self, data: Any) -> float:
        if hasattr(data, "columns") and "timestamp" in data.columns:
            timestamps = pd.to_datetime(data["timestamp"])
            if timestamps.empty:
                return 1.0
            age_days = (pd.Timestamp.now() - timestamps.max()).days
            recency_score = max(0.0, 1.0 - (age_days / 365) * 0.3)
            recency_score = max(0.0, min(1.0, recency_score))
            return float(recency_score)
        return float(self._calculate_timeliness(data, TYPE_STOCK))

    def _check_consistency(self, data: Any) -> float:
        return float(self._calculate_consistency(data, TYPE_STOCK))

    def _send_alert(self, message: str) -> None:
        logger.warning(message)
        record = {"message": message, "timestamp": datetime.now(), "resolved": False}
        self._alerts_history["manual"].append(record)

    def check_quality(self, data: Any, data_type: Optional[DataSourceType] = None) -> Dict[str, Any]:
        """检查数据质量 - 基础设施层深度集成版本"""
        start_time = datetime.now()
        normalized_type = self._normalize_data_type(data_type)

        try:
            # 记录质量检查开始
            log_data_operation(
                "quality_check_start",
                normalized_type,
                {"data_size": len(str(data)) if data is not None else 0},
                "info",
            )

            # 数据验证
            validation_result = self.validator.validate(data, normalized_type)

            # 自动修复处理
            repair_actions: List[str] = []
            auto_repair_enabled = bool(self.auto_repair or self.config_obj.enable_auto_repair)
            if auto_repair_enabled and validation_result.get("issues"):
                issues = validation_result.get("issues", [])
                repaired_data = self.validator.repair_data(data, issues)
                if issues:
                    repair_actions = [
                        f"auto_repair:{getattr(issue, 'issue_type', 'unknown')}"
                        for issue in issues
                    ]
                if repaired_data is not None:
                    data = repaired_data

            # 计算质量指标
            metrics = self._calculate_quality_metrics(data, normalized_type, validation_result)

            # 更新最后检查时间
            self.last_quality_check = datetime.now()

            # 存储质量历史
            self.quality_history[normalized_type].append(metrics)
            if len(self.quality_history[normalized_type]) > self.config_obj.max_quality_history:
                self.quality_history[normalized_type].pop(0)
            
            # 持久化质量指标到数据湖（可选功能）
            try:
                self._persist_quality_metrics_to_data_lake(normalized_type, metrics)
            except Exception as e:
                logger.debug(f"持久化质量指标到数据湖失败（可选功能）: {e}")

            # 检测异常
            anomalies = self._detect_anomalies(normalized_type, metrics)

            # 生成告警
            if anomalies:
                self._generate_alerts(normalized_type, anomalies)

            # 阈值触发告警
            if metrics.overall_score < self.alert_threshold:
                alert_message = f"质量分数低于阈值({self.alert_threshold:.2f}): {metrics.overall_score:.2f}"
                self._send_alert(alert_message)
                alert_payload = {
                    "quality_score": metrics.overall_score,
                    "threshold": self.alert_threshold,
                    "data_type": normalized_type.value,
                    "anomalies": anomalies,
                    "message": alert_message,
                }
                for handler in list(self._alert_handlers):
                    try:
                        handler(alert_payload)
                    except Exception as exc:
                        logger.error("质量告警处理器执行失败: %s", exc)

            # 生成建议
            recommendations = self._generate_recommendations(normalized_type, metrics, validation_result)

            # 记录性能指标
            duration = (datetime.now() - start_time).total_seconds()
            record_data_metric("quality_check_duration", duration, normalized_type,
                               {"success": True})

            # 记录质量指标
            record_data_metric("quality_score", metrics.overall_score, normalized_type,
                               {"completeness": metrics.completeness, "accuracy": metrics.accuracy})

            # 发布质量检查事件
            if metrics.overall_score < self.config_obj.quality_threshold:
                logger.warning("质量阈值违规: 数据类型=%s, 质量分数=%.2f, 阈值=%.2f, 异常数量=%d",
                               normalized_type.value, metrics.overall_score,
                               self.config_obj.quality_threshold, len(anomalies))

            # 记录质量检查完成
            log_data_operation("quality_check_complete", normalized_type,
                               {
                                   "quality_score": metrics.overall_score,
                                   "anomalies_count": len(anomalies),
                                   "duration": duration
                               }, "info")

            return {
                "metrics": metrics,
                "overall_score": metrics.overall_score,
                "completeness": metrics.completeness,
                "accuracy": metrics.accuracy,
                "consistency": metrics.consistency,
                "timeliness": metrics.timeliness,
                "validity": metrics.validity,
                "validation": validation_result,
                "anomalies": anomalies,
                "recommendations": recommendations,
                "processing_time": duration,
                "data_type": normalized_type.value,
                "repair_actions": repair_actions,
            }

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()

            log_data_operation("quality_check_error", normalized_type,
                               {"error": str(e), "duration": duration}, "error")

            record_data_metric("quality_check_error", 1, normalized_type,
                               {"error_type": type(e).__name__})

            return {
                "metrics": QualityMetrics(),
                "overall_score": 0.0,
                "completeness": 0.0,
                "accuracy": 0.0,
                "consistency": 0.0,
                "timeliness": 0.0,
                "validity": 0.0,
                "validation": {"valid": False, "issues": []},
                "anomalies": [],
                "recommendations": ["检查系统日志以获取详细错误信息"],
                "error": str(e),
                "processing_time": duration,
                "data_type": normalized_type.value,
                "repair_actions": [],
            }

    def _calculate_quality_metrics(self, data: Any, data_type: DataSourceType,


                                   validation_result: Dict[str, Any]) -> QualityMetrics:
        """计算质量指标"""
        metrics = QualityMetrics()

        try:
            # 完整性：基于非空值比例
            if hasattr(data, 'columns'):
                non_null_ratio = 1 - data.isnull().sum().sum() / (len(data) * len(data.columns))
                metrics.completeness = max(0, min(1, non_null_ratio))

            # 有效性：基于验证结果
            validation_valid = validation_result.get("valid", False)
            critical_issues = validation_result.get("critical_issues", 0)
            metrics.validity = 1.0 if validation_valid and critical_issues == 0 else 0.5

            # 一致性：检查数据内部一致性
            metrics.consistency = self._calculate_consistency(data, data_type)

            # 时效性：基于数据时间戳
            metrics.timeliness = self._calculate_timeliness(data, data_type)

            # 准确性：基于业务规则验证
            metrics.accuracy = 1.0 - (validation_result.get("issue_count", 0) * 0.1)

            # 综合得分
            metrics.overall_score = statistics.mean([
                metrics.completeness,
                metrics.accuracy,
                metrics.consistency,
                metrics.timeliness,
                metrics.validity
            ])

            return metrics

        except Exception as e:
            logger.error(f"计算质量指标失败: {e}")
            return QualityMetrics()

    def _calculate_consistency(self, data: Any, data_type: DataSourceType) -> float:
        """计算一致性"""
        try:
            if data_type == TYPE_STOCK and hasattr(data, 'columns'):
                # 检查OHLC关系：High >= max(Open, Close), Low <= min(Open, Close)
                if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                    valid_ohlc = (
                        (data['high'] >= data[['open', 'close']].max(axis=1))
                        & (data['low'] <= data[['open', 'close']].min(axis=1))
                    ).mean()
                    return float(valid_ohlc)
            return 0.8  # 默认值
        except BaseException:
            return 0.5

    def _calculate_timeliness(self, data: Any, data_type: DataSourceType) -> float:
        """计算时效性"""
        try:
            if hasattr(data, 'columns') and 'timestamp' in data.columns:
                # 检查数据的新鲜度
                latest_timestamp = pd.to_datetime(data['timestamp']).max()
                now = pd.Timestamp.now()
                hours_old = (now - latest_timestamp).total_seconds() / 3600

                # 根据数据类型设置不同的新鲜度要求
                freshness_thresholds = {
                    TYPE_STOCK: 24,    # 24小时
                    TYPE_CRYPTO: 1,    # 1小时
                    TYPE_NEWS: 12,     # 12小时
                }

                threshold = freshness_thresholds.get(data_type, 24)
                timeliness = max(0, 1 - (hours_old / threshold))
                return timeliness

            return 0.5
        except BaseException:
            return 0.5

    def _detect_anomalies(self, data_type: DataSourceType, current_metrics: QualityMetrics) -> List[str]:
        """检测异常"""
        anomalies = []

        try:
            history = self.quality_history[data_type]
            if len(history) < 5:  # 需要足够的历史数据
                return anomalies

            # 计算历史平均值和标准差
            scores = [m.overall_score for m in history[-10:]]  # 最近10个数据点
            avg_score = statistics.mean(scores)
            std_score = statistics.stdev(scores) if len(scores) > 1 else 0

            # 检测显著下降
            if current_metrics.overall_score < avg_score - 2 * std_score:
                anomalies.append(
                    f"质量得分显著下降: {current_metrics.overall_score:.2f} vs 平均 {avg_score:.2f}")

            # 检测完整性问题
            if current_metrics.completeness < self.config_obj.quality_threshold:
                anomalies.append(f"数据完整性不足: {current_metrics.completeness:.2f}")

            # 检测有效性问题
            if current_metrics.validity < 0.8:
                anomalies.append(f"数据有效性不足: {current_metrics.validity:.2f}")

        except Exception as e:
            logger.error(f"异常检测失败: {e}")

        return anomalies

    def _generate_alerts(self, data_type: DataSourceType, anomalies: List[str]):
        """生成告警"""
        if not anomalies:
            return

        alert_key = f"{data_type.value}_quality"
        now = datetime.now()

        # 检查告警冷却时间
        if alert_key in self.alerts_sent:
            last_alert = self.alerts_sent[alert_key]
            cooldown = timedelta(minutes=self.config_obj.alert_cooldown_minutes)
            if now - last_alert < cooldown:
                return  # 冷却时间内不重复告警

        # 发送告警
        alert_message = f"数据质量告警 - {data_type.value}: {'; '.join(anomalies)}"
        logger.warning(alert_message)
        self._send_alert(alert_message)

        # 更新告警时间
        self.alerts_sent[alert_key] = now
        alert_record = {
            "message": alert_message,
            "timestamp": now,
            "resolved": False,
            "data_type": data_type.value if hasattr(data_type, 'value') else str(data_type),
            "anomalies": anomalies
        }
        self._alerts_history[alert_key].append(alert_record)
        
        # 持久化告警历史到数据湖（符合架构设计：双重持久化）
        try:
            self._persist_quality_alert_to_data_lake(alert_record, data_type)
        except Exception as e:
            logger.debug(f"持久化告警历史到数据湖失败（可选功能）: {e}")

    def _generate_recommendations(self, data_type: DataSourceType,


                                  metrics: QualityMetrics,
                                  validation_result: Dict[str, Any]) -> List[str]:
        """生成建议"""
        recommendations = []

        if metrics.completeness < 0.9:
            recommendations.append("建议检查数据源完整性，补充缺失字段")

        if metrics.validity < 0.8:
            recommendations.append("建议加强数据验证，修复无效值")

        if metrics.consistency < 0.8:
            recommendations.append("建议检查数据一致性，修复逻辑错误")

        if metrics.timeliness < 0.8:
            recommendations.append("建议优化数据更新频率，确保数据新鲜度")

        issues = validation_result.get("issues", [])
        critical_issues = [i for i in issues if i.severity == "critical"]
        if critical_issues:
            recommendations.append(f"紧急修复 {len(critical_issues)} 个严重质量问题")

        return recommendations

    def register_alert_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        if not callable(handler):
            raise TypeError("alert handler must be callable")
        self._alert_handlers.append(handler)

    def get_quality_history(self, data_type: Optional[DataSourceType] = None) -> List[QualityMetrics]:
        # 当指定类型时，返回该类型的历史；未指定时，聚合所有类型历史
        if data_type is not None:
            normalized = self._normalize_data_type(data_type)
            return list(self.quality_history.get(normalized, []))
        aggregated: List[QualityMetrics] = []
        for history in self.quality_history.values():
            aggregated.extend(history)
        return aggregated

    def _get_quality_metrics_for_type(self, data_type: DataSourceType) -> Dict[str, Any]:
        """获取质量指标"""
        history = self.quality_history.get(data_type, [])
        
        # 尝试从数据湖加载历史（如果内存中没有足够的历史）
        if len(history) < 5:
            try:
                lake_history = self._load_quality_history_from_data_lake(data_type)
                if lake_history:
                    history.extend(lake_history)
            except Exception as e:
                logger.debug(f"从数据湖加载质量历史失败（可选功能）: {e}")

        result: Dict[str, Any] = {
            "history_length": len(history),
            "latest": history[-1].__dict__ if history else None,
            "trend": self._calculate_trend(history) if history else "insufficient_data",
            "alerts_today": len(
                [
                    k
                    for k, v in self.alerts_sent.items()
                    if k.startswith(data_type.value) and v.date() == datetime.now().date()
                ]
            ),
        }

        if history:
            latest = history[-1]
            result["current"] = {
                "completeness": latest.completeness,
                "accuracy": latest.accuracy,
                "consistency": latest.consistency,
                "timeliness": latest.timeliness,
                "validity": latest.validity,
                "overall_score": latest.overall_score,
            }

        return result

    def _calculate_trend(self, history: List[QualityMetrics]) -> str:
        """计算趋势"""
        if len(history) < 2:
            return "insufficient_data"

        recent_scores = [m.overall_score for m in history[-5:]]
        older_scores = [m.overall_score for m in history[-10:-5]
                        ] if len(history) >= 10 else recent_scores

        recent_avg = statistics.mean(recent_scores)
        older_avg = statistics.mean(older_scores)

        if recent_avg > older_avg + 0.05:
            return "improving"
        elif recent_avg < older_avg - 0.05:
            return "declining"
        else:
            return "stable"

    def generate_report(self, data_type: DataSourceType, period: str) -> Dict[str, Any]:
        """生成质量报告"""
        try:
            history = self.quality_history.get(data_type, [])
            if not history:
                return {"error": "无质量历史数据"}

            # 根据周期过滤数据
            filtered_history = self._filter_by_period(history, period)

            if not filtered_history:
                return {"error": f"指定周期内无数据: {period}"}

            # 计算统计信息
            scores = [m.overall_score for m in filtered_history]
            completeness_scores = [m.completeness for m in filtered_history]
            validity_scores = [m.validity for m in filtered_history]

            report = QualityReport(
                data_type=data_type,
                period=period,
                metrics=filtered_history[-1],  # 最新指标
                issues=[],  # 这里可以扩展为收集历史问题
                recommendations=self._generate_report_recommendations(
                    scores, completeness_scores, validity_scores)
            )

            return {
                "data_type": data_type.value,
                "period": period,
                "metrics": {
                    "avg_score": round(statistics.mean(scores), 3),
                    "min_score": round(min(scores), 3),
                    "max_score": round(max(scores), 3),
                    "score_std": round(statistics.stdev(scores), 3) if len(scores) > 1 else 0,
                    "avg_completeness": round(statistics.mean(completeness_scores), 3),
                    "avg_validity": round(statistics.mean(validity_scores), 3)
                },
                "data_points": len(filtered_history),
                "recommendations": report.recommendations,
                "generated_at": report.generated_at.isoformat()
            }

        except Exception as e:
            logger.error(f"生成质量报告失败: {e}")
            return {"error": str(e)}

    def _filter_by_period(self, history: List[QualityMetrics], period: str) -> List[QualityMetrics]:
        """按周期过滤历史数据"""
        if period == "all":
            return history

        now = datetime.now()
        if period == "1h":
            cutoff = now - timedelta(hours=1)
        elif period == "24h":
            cutoff = now - timedelta(days=1)
        elif period == "7d":
            cutoff = now - timedelta(days=7)
        elif period == "30d":
            cutoff = now - timedelta(days=30)
        else:
            return history[-10:]  # 默认返回最近10个

        return [m for m in history if m.timestamp >= cutoff]

    def _generate_report_recommendations(self, scores: List[float],


                                         completeness_scores: List[float],
                                         validity_scores: List[float]) -> List[str]:
        """生成报告建议"""
        recommendations = []

        avg_score = statistics.mean(scores)
        if avg_score < 0.8:
            recommendations.append("整体质量需要提升，建议加强数据验证和清理")

        if statistics.mean(completeness_scores) < 0.9:
            recommendations.append("数据完整性不足，建议补充缺失字段")

        if statistics.mean(validity_scores) < 0.8:
            recommendations.append("数据有效性需要改进，建议加强业务规则验证")

        score_std = statistics.stdev(scores) if len(scores) > 1 else 0
        if score_std > 0.1:
            recommendations.append("质量波动较大，建议稳定数据源和处理流程")

        return recommendations

    # --- IDataQualityMonitor interface implementations ---

    def monitor_data_source(self, source_id: str, config: Dict[str, Any]) -> bool:
        self._monitored_sources[source_id] = config or {}
        return True

    def stop_monitoring(self, source_id: str) -> bool:
        return self._monitored_sources.pop(source_id, None) is not None

    def check_data_quality(self, data: Any, source_id: str) -> List[Dict[str, Any]]:
        source_config = self._monitored_sources.get(source_id, {})
        data_type = source_config.get("data_type", TYPE_STOCK)
        result = self.check_quality(data, data_type)
        return [result]

    def get_quality_metrics(self, source_id: str) -> Dict[str, Any]:
        source_config = self._monitored_sources.get(source_id, {})
        data_type = source_config.get("data_type", TYPE_STOCK)
        metrics = self._get_quality_metrics_for_type(data_type)
        metrics["source_id"] = source_id
        return metrics

    def get_alerts(self, source_id: str, level: Optional[Any] = None, resolved: bool = False) -> List[Dict[str, Any]]:
        key = f"{source_id}_quality" if not source_id.endswith("_quality") else source_id
        alerts = self._alerts_history.get(key, [])
        if resolved:
            return [alert for alert in alerts if alert.get("resolved")]
        if level is not None:
            return []
        return [alert for alert in alerts if not alert.get("resolved")]

    def resolve_alert(self, source_id: str, alert_id: Optional[str] = None) -> bool:
        key = f"{source_id}_quality" if not source_id.endswith("_quality") else source_id
        alerts = self._alerts_history.get(key, [])
        if not alerts:
            return False
        alerts[-1]["resolved"] = True
        return True

    def set_thresholds(self, source_id: str, thresholds: Dict[str, Any]) -> bool:
        self._thresholds[source_id] = thresholds
        return True

    def get_thresholds(self, source_id: str) -> Dict[str, Any]:
        return self._thresholds.get(source_id, {})
    
    def _persist_quality_metrics_to_data_lake(self, data_type: DataSourceType, metrics: QualityMetrics) -> bool:
        """持久化质量指标到数据湖（可选功能）"""
        try:
            from ...lake.data_lake_manager import DataLakeManager, LakeConfig
            import pandas as pd
            
            # 创建数据湖管理器
            lake_config = LakeConfig(
                base_path="data_lake/quality_metrics",
                approach="date",
                compression="parquet",
                metadata_enabled=True
            )
            lake_manager = DataLakeManager(lake_config)
            
            # 将质量指标转换为DataFrame
            data_type_value = data_type.value if hasattr(data_type, 'value') else str(data_type)
            metrics_dict = {
                'data_type': [data_type_value],
                'completeness': [metrics.completeness],
                'accuracy': [metrics.accuracy],
                'consistency': [metrics.consistency],
                'timeliness': [metrics.timeliness],
                'validity': [metrics.validity],
                'overall_score': [metrics.overall_score],
                'timestamp': [metrics.timestamp]
            }
            df = pd.DataFrame(metrics_dict)
            
            # 存储到数据湖
            dataset_name = f"quality_metrics_{data_type_value}"
            partition_key = metrics.timestamp.strftime("%Y-%m-%d")
            metadata = {
                'data_type': data_type_value,
                'timestamp': metrics.timestamp.isoformat(),
                'overall_score': metrics.overall_score
            }
            
            lake_manager.store_data(df, dataset_name, partition_key=partition_key, metadata=metadata)
            logger.debug(f"质量指标已持久化到数据湖: {dataset_name}")
            return True
            
        except Exception as e:
            logger.debug(f"持久化质量指标到数据湖失败: {e}")
            return False
    
    def _persist_quality_alert_to_data_lake(self, alert_record: Dict[str, Any], data_type: DataSourceType) -> bool:
        """持久化质量告警到数据湖（可选功能）"""
        try:
            from ...lake.data_lake_manager import DataLakeManager, LakeConfig
            import pandas as pd
            
            # 创建数据湖管理器
            lake_config = LakeConfig(
                base_path="data_lake/quality_alerts",
                approach="date",
                compression="parquet",
                metadata_enabled=True
            )
            lake_manager = DataLakeManager(lake_config)
            
            # 将告警记录转换为DataFrame
            data_type_value = data_type.value if hasattr(data_type, 'value') else str(data_type)
            
            # 处理timestamp字段
            timestamp = alert_record.get('timestamp')
            if isinstance(timestamp, datetime):
                timestamp_str = timestamp.isoformat()
                timestamp_date = timestamp
            else:
                timestamp_str = str(timestamp)
                try:
                    timestamp_date = datetime.fromisoformat(timestamp_str) if isinstance(timestamp_str, str) else datetime.now()
                except:
                    timestamp_date = datetime.now()
            
            # 处理anomalies字段（可能是列表）
            anomalies = alert_record.get('anomalies', [])
            anomalies_str = ', '.join(anomalies) if isinstance(anomalies, list) else str(anomalies)
            
            alert_dict = {
                'data_type': [data_type_value],
                'message': [alert_record.get('message', '')],
                'timestamp': [timestamp_date],
                'resolved': [alert_record.get('resolved', False)],
                'anomalies': [anomalies_str]
            }
            df = pd.DataFrame(alert_dict)
            
            # 存储到数据湖
            dataset_name = f"quality_alerts_{data_type_value}"
            partition_key = timestamp_date.strftime("%Y-%m-%d")
            metadata = {
                'data_type': data_type_value,
                'timestamp': timestamp_str,
                'resolved': alert_record.get('resolved', False),
                'anomalies_count': len(anomalies) if isinstance(anomalies, list) else 0
            }
            
            lake_manager.store_data(df, dataset_name, partition_key=partition_key, metadata=metadata)
            logger.debug(f"质量告警已持久化到数据湖: {dataset_name}")
            return True
            
        except Exception as e:
            logger.debug(f"持久化质量告警到数据湖失败: {e}")
            return False
    
    def _load_quality_history_from_data_lake(self, data_type: DataSourceType, limit: int = 100) -> List[QualityMetrics]:
        """从数据湖加载质量历史（可选功能）"""
        try:
            from ...lake.data_lake_manager import DataLakeManager, LakeConfig
            import pandas as pd
            
            # 创建数据湖管理器
            lake_config = LakeConfig(
                base_path="data_lake/quality_metrics",
                approach="date",
                compression="parquet"
            )
            lake_manager = DataLakeManager(lake_config)
            
            # 构建数据集名称
            data_type_value = data_type.value if hasattr(data_type, 'value') else str(data_type)
            dataset_name = f"quality_metrics_{data_type_value}"
            
            # 尝试加载数据
            try:
                # 获取数据集信息
                dataset_info = lake_manager.get_dataset_info(dataset_name)
                if not dataset_info:
                    return []
                
                # 加载最近的数据
                files = dataset_info.get('files', [])
                if not files:
                    return []
                
                # 加载最近的文件
                latest_file = sorted(files, key=lambda x: x.get('timestamp', 0), reverse=True)[0]
                df = lake_manager.load_data(dataset_name, partition_key=latest_file.get('partition'))
                
                if df is None or df.empty:
                    return []
                
                # 转换为QualityMetrics列表
                metrics_list = []
                for _, row in df.tail(limit).iterrows():
                    metrics = QualityMetrics(
                        completeness=row.get('completeness', 0.0),
                        accuracy=row.get('accuracy', 0.0),
                        consistency=row.get('consistency', 0.0),
                        timeliness=row.get('timeliness', 0.0),
                        validity=row.get('validity', 0.0),
                        overall_score=row.get('overall_score', 0.0),
                        timestamp=pd.to_datetime(row.get('timestamp', datetime.now()))
                    )
                    metrics_list.append(metrics)
                
                logger.debug(f"从数据湖加载质量历史: {len(metrics_list)} 条")
                return metrics_list
                
            except Exception as e:
                logger.debug(f"从数据湖加载数据失败: {e}")
                return []
                
        except Exception as e:
            logger.debug(f"从数据湖加载质量历史失败: {e}")
            return []


# 工厂函数

def create_unified_quality_monitor(config: Optional[QualityConfig] = None) -> UnifiedQualityMonitor:
    """创建统一质量监控器"""
    return UnifiedQualityMonitor(config)


def create_unified_data_validator(config: Optional[Dict[str, Any]] = None) -> UnifiedDataValidator:
    """创建统一数据验证器"""
    return UnifiedDataValidator(config)


# 导出主要类和函数
__all__ = [
    'QualityConfig',
    'QualityMetrics',
    'QualityIssue',
    'QualityReport',
    'UnifiedDataValidator',
    'UnifiedQualityMonitor',
    'create_unified_quality_monitor',
    'create_unified_data_validator'
]

# Logger setup
logger = logging.getLogger(__name__)
