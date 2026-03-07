"""
specialized_validators 模块

提供 specialized_validators 相关功能和接口。
"""

import re
import os
import logging
import re

import datetime
import ipaddress

from typing import Dict, Any, List, Optional, Tuple, Union
from .validator_base import (
    ValidationSeverity, ValidationType, ValidationResult, ValidationRule,
    BaseConfigValidator
)

"""
专用配置验证器

包含各种特定领域的配置验证器实现
拆分自validators.py，提高代码组织性和可维护性
"""

logger = logging.getLogger(__name__)

# 常量定义
MAX_PORT_NUMBER = 65535  # 最大端口号
MIN_PRIVILEGED_PORT = 1024  # 特权端口最小值
MAX_HOSTNAME_LENGTH = 253  # 最大主机名长度
DEFAULT_CONNECTION_POOL_SIZE = 10  # 默认连接池大小
WELL_KNOWN_PORTS = {80, 443, 22, 21, 25, 53, 110, 143, 993, 995}  # 知名端口

# ==================== 交易时间验证器 ====================


class TradingHoursValidator(BaseConfigValidator):
    """交易时间验证器"""

    def __init__(self):
        """初始化交易时间验证器"""
        super().__init__(
            name="TradingHoursValidator",
            description="验证交易时段配置"
        )

        # 添加验证规则 - start和end不是必需的，因为支持分段格式
        self.add_rule(ValidationRule(
            ValidationType.REQUIRED,
            "trading_hours.start",
            required=False
        ))
        self.add_rule(ValidationRule(
            ValidationType.REQUIRED,
            "trading_hours.end",
            required=False
        ))
        self.add_rule(ValidationRule(
            ValidationType.TYPE,
            "trading_hours.timezone",
            required=False,
            type=str
        ))

    def _validate_custom(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """自定义交易时间验证"""
        results = []

        # 验证基础字段结构
        field_validation = self._validate_trading_hours_basic(config)
        results.extend(field_validation)

        # 如果基础验证失败，直接返回
        if any(r.severity == ValidationSeverity.ERROR for r in results):
            return results

        # 继续验证交易时间格式和逻辑
        trading_hours = config.get('trading_hours')
        format_validation = self._validate_trading_hours_advanced(trading_hours)
        results.extend(format_validation)

        return results

    def _validate_trading_hours_basic(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """验证trading_hours字段的基础结构"""
        results = []
        trading_hours = config.get('trading_hours')

        # 如果trading_hours字段完全缺失，返回一个错误
        if trading_hours is None:
            results.append(ValidationResult(
                is_valid=False,
                errors=["缺少trading_hours字段"],
                severity=ValidationSeverity.ERROR,
                field="trading_hours"
            ))
            return results

        # 如果trading_hours不是字典，返回类型错误
        if not isinstance(trading_hours, dict):
            results.append(ValidationResult(
                is_valid=False,
                errors=["trading_hours必须是字典类型"],
                severity=ValidationSeverity.ERROR,
                field="trading_hours",
                value=trading_hours
            ))

        return results

    def _validate_trading_hours_advanced(self, trading_hours: Dict[str, Any]) -> List[ValidationResult]:
        """验证交易时间的格式和逻辑"""
        results = []

        # 提取不同格式的交易时间
        start_time = trading_hours.get('start')
        end_time = trading_hours.get('end')
        segments = self._extract_segments(trading_hours)

        # 验证分段格式
        if segments:
            segment_results = self._validate_segments(segments)
            results.extend(segment_results)

        # 验证传统格式
        if start_time or end_time:
            traditional_results = self._validate_traditional_format(start_time, end_time)
            results.extend(traditional_results)

        # 验证时区
        timezone = trading_hours.get('timezone')
        if timezone and not self._is_valid_timezone(timezone):
            results.append(ValidationResult(
                is_valid=False,
                errors=[f"无效的时区: {timezone}"],
                severity=ValidationSeverity.WARNING,
                field="trading_hours.timezone",
                value=timezone,
                suggestions=["使用标准的时区名称，如 'UTC', 'America/New_York'"]
            ))

        return results

    def _extract_segments(self, trading_hours: Dict[str, Any]) -> Dict[str, List[str]]:
        """提取分段格式的交易时间"""
        segments = {}
        for key, value in trading_hours.items():
            if key not in ['start', 'end', 'timezone'] and isinstance(value, list) and len(value) == 2:
                segments[key] = value
        return segments

    def _validate_segments(self, segments: Dict[str, List[str]]) -> List[ValidationResult]:
        """验证分段格式"""
        results = []
        valid_segments = []

        for segment_name, times in segments.items():
            if len(times) != 2:
                results.append(ValidationResult(
                    is_valid=False,
                    errors=[f"交易时段 '{segment_name}' 必须包含开始和结束时间"],
                    severity=ValidationSeverity.ERROR,
                    field=f"trading_hours.{segment_name}",
                    value=times
                ))
                continue

            start, end = times
            if not self._is_valid_time_format(start) or not self._is_valid_time_format(end):
                results.append(ValidationResult(
                    is_valid=False,
                    errors=["时间格式不正确"],
                    severity=ValidationSeverity.ERROR,
                    field=f"trading_hours.{segment_name}",
                    value=f"{start}-{end}"
                ))
            elif not self._is_valid_time_range(start, end):
                results.append(ValidationResult(
                    is_valid=False,
                    errors=[f"交易时段 '{segment_name}' 时间范围无效: {start} - {end}"],
                    severity=ValidationSeverity.WARNING,
                    field=f"trading_hours.{segment_name}",
                    value=f"{start}-{end}",
                    suggestions=["确保结束时间晚于开始时间"]
                ))
            else:
                valid_segments.append((segment_name, start, end))

        # 检测时段重叠
        if len(valid_segments) > 1:
            overlap_results = self._detect_segment_overlaps(valid_segments)
            results.extend(overlap_results)

        return results

    def _detect_segment_overlaps(self, segments: List[tuple]) -> List[ValidationResult]:
        """检测时段重叠"""
        results = []

        # 简单的重叠检测：检查是否有任何两个时段重叠
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                segment1 = segments[i]
                segment2 = segments[j]

                name1, start1, end1 = segment1
                name2, start2, end2 = segment2

                # 简单的时间比较（假设格式为HH:MM）
                if self._segments_overlap(start1, end1, start2, end2):
                    results.append(ValidationResult(
                        is_valid=True,  # 重叠只是警告，不影响验证结果
                        warnings=[f"交易时段 '{name1}' 和 '{name2}' 存在重叠"],
                        severity=ValidationSeverity.WARNING,
                        field=f"trading_hours.{name1}",
                        value=f"{start1}-{end1}",
                        suggestions=["考虑调整交易时段以避免重叠"]
                    ))

        return results

    def _segments_overlap(self, start1: str, end1: str, start2: str, end2: str) -> bool:
        """检查两个时间段是否重叠"""
        # 简单的字符串比较（可以根据需要改进为更精确的时间比较）
        return not (end1 <= start2 or end2 <= start1)

    def _validate_traditional_format(self, start_time: Optional[str], end_time: Optional[str]) -> List[ValidationResult]:
        """验证传统格式"""
        results = []

        if start_time and not self._is_valid_time_format(start_time):
            results.append(ValidationResult(
                is_valid=False,
                errors=[f"交易开始时间格式无效: {start_time}，期望格式 HH:MM"],
                severity=ValidationSeverity.ERROR,
                field="trading_hours.start",
                value=start_time
            ))

        if end_time and not self._is_valid_time_format(end_time):
            results.append(ValidationResult(
                is_valid=False,
                errors=[f"交易结束时间格式无效: {end_time}，期望格式 HH:MM"],
                severity=ValidationSeverity.ERROR,
                field="trading_hours.end",
                value=end_time
            ))

        # 验证时间逻辑
        if (start_time and end_time and
            self._is_valid_time_format(start_time) and
                self._is_valid_time_format(end_time)):
            if not self._is_valid_time_range(start_time, end_time):
                results.append(ValidationResult(
                    is_valid=False,
                    errors=[f"交易时间范围无效: {start_time} - {end_time}"],
                    severity=ValidationSeverity.WARNING,
                    field="trading_hours",
                    value=f"{start_time}-{end_time}",
                    suggestions=["确保结束时间晚于开始时间"]
                ))

        return results

    def _is_valid_time_format(self, time_str: str) -> bool:
        """验证时间格式 (HH:MM)"""
        if not isinstance(time_str, str):
            return False
        return bool(re.match(r'^([01]\d|2[0-3]):([0-5]\d)$', time_str))

    def _is_valid_time_range(self, start: str, end: str) -> bool:
        """验证时间范围"""
        try:
            start_dt = datetime.datetime.strptime(start, '%H:%M')
            end_dt = datetime.datetime.strptime(end, '%H:%M')
            return end_dt > start_dt
        except ValueError:
            return False

    def _check_segment_overlaps(self, valid_segments: List[Tuple[str, str, str]], results: List[ValidationResult]):
        """检测时段重叠"""
        if len(valid_segments) < 2:
            return

        # 将时段转换为分钟进行比较
        def time_to_minutes(time_str: str) -> int:
            """将HH:MM格式转换为分钟"""
            hours, minutes = map(int, time_str.split(':'))
            return hours * 60 + minutes

        # 检查每对时段是否重叠
        for i, (name1, start1, end1) in enumerate(valid_segments):
            start1_min = time_to_minutes(start1)
            end1_min = time_to_minutes(end1)

            for j, (name2, start2, end2) in enumerate(valid_segments[i+1:], i+1):
                start2_min = time_to_minutes(start2)
                end2_min = time_to_minutes(end2)

                # 检查是否重叠：两个时段的交集不为空
                if max(start1_min, start2_min) < min(end1_min, end2_min):
                    results.append(ValidationResult(
                        is_valid=True,  # 重叠只是警告，不影响整体有效性
                        warnings=[
                            f"交易时段 '{name1}' ({start1}-{end1}) 与 '{name2}' ({start2}-{end2}) 重叠"],
                        severity=ValidationSeverity.WARNING,
                        field="trading_hours",
                        value=f"{name1}({start1}-{end1}) vs {name2}({start2}-{end2})",
                        suggestions=["调整时段以避免重叠"]
                    ))

    def _is_valid_timezone(self, timezone: str) -> bool:
        """验证时区"""
        # 简单的时区验证，实际应该使用pytz等库进行完整验证
        common_timezones = [
            'UTC', 'GMT', 'EST', 'CST', 'PST', 'CET', 'JST',
            'America/New_York', 'Europe/London', 'Asia/Tokyo'
        ]
        return timezone in common_timezones or '/' in timezone

# ==================== 数据库配置验证器 ====================


class DatabaseConfigValidator(BaseConfigValidator):
    """数据库配置验证器"""

    def __init__(self):
        """初始化数据库配置验证器"""
        super().__init__(
            name="DatabaseConfigValidator",
            description="验证数据库配置"
        )

        # 添加验证规则
        self.add_rule(ValidationRule(
            ValidationType.REQUIRED,
            "database.host",
            required=True
        ))
        self.add_rule(ValidationRule(
            ValidationType.TYPE,
            "database.port",
            required=False,
            type=int
        ))
        self.add_rule(ValidationRule(
            ValidationType.RANGE,
            "database.port",
            required=False,
            min=1,
            max=MAX_PORT_NUMBER
        ))
        self.add_rule(ValidationRule(
            ValidationType.REQUIRED,
            "database.name",
            required=True
        ))
        self.add_rule(ValidationRule(
            ValidationType.TYPE,
            "database.username",
            required=True,
            type=str
        ))

    def _validate_custom(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """自定义数据库配置验证"""
        results = []

        database = config.get('database', {})

        # 验证主机地址
        host = database.get('host')
        if host and not self._is_valid_host(host):
            results.append(ValidationResult(is_valid=False, errors=[f"无效的数据库主机地址: {host}"],
                                            severity=ValidationSeverity.ERROR,
                                            field="database.host",
                                            value=host,
                                            suggestions=["使用有效的IP地址或主机名"]
                                            ))

        # 验证端口范围（产生期望的错误消息）
        port = database.get('port')
        if port is not None:
            if not isinstance(port, int) or port < MIN_PRIVILEGED_PORT or port > MAX_PORT_NUMBER:
                results.append(ValidationResult(is_valid=False, errors=["端口必须是1024-65535之间的整数"],
                                                severity=ValidationSeverity.ERROR,
                                                field="database.port",
                                                value=port
                                                ))

        # 验证连接池配置
        pool = database.get('pool', {})
        if pool:
            min_size = pool.get('min_size')
            max_size = pool.get('max_size')
            if min_size is not None and max_size is not None:
                if min_size > max_size:
                    results.append(ValidationResult(is_valid=False, errors=["最小大小不能大于最大大小"],
                                                    severity=ValidationSeverity.ERROR,
                                                    field="database.pool",
                                                    value=f"min_size={min_size}, max_size={max_size}"
                                                    ))

        # 验证数据库类型
        db_type = database.get('type', '').lower()
        if db_type and db_type not in ['mysql', 'postgresql', 'oracle', 'sqlite', 'mongodb']:
            results.append(ValidationResult(is_valid=False, errors=[f"不支持的数据库类型: {db_type}"],
                                            severity=ValidationSeverity.WARNING,
                                            field="database.type",
                                            value=db_type,
                                            suggestions=[
                                                "支持的类型: mysql, postgresql, oracle, sqlite, mongodb"]
                                            ))

        # 验证连接池配置
        pool_config = database.get('pool', {})
        if pool_config:
            min_size = pool_config.get('min_size', 1)
            max_size = pool_config.get('max_size', DEFAULT_CONNECTION_POOL_SIZE)

            if min_size > max_size:
                results.append(ValidationResult(is_valid=False, errors=[f"连接池最小大小({min_size})不能大于最大大小({max_size})"],
                                                severity=ValidationSeverity.ERROR,
                                                field="database.pool",
                                                value=pool_config
                                                ))

        return results

    def _is_valid_host(self, host: str) -> bool:
        """验证主机地址"""
        if not isinstance(host, str):
            return False

        # 检查是否为IP地址
        try:
            ipaddress.ip_address(host)
            return True
        except ValueError:
            pass

        # 检查是否为有效的主机名
        # 简单的正则检查，实际应该更严格
        if re.match(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', host):
            return True

        # localhost和简单的单字主机名
        return host in ['localhost', '127.0.0.1'] or len(host) <= MAX_HOSTNAME_LENGTH

    def validate_database_config(self, config: Dict[str, Any]) -> ValidationResult:
        """验证数据库配置（独立方法）

        Args:
            config: 数据库配置

        Returns:
            ValidationResult: 验证结果
        """
        return self.validate(config)

# ==================== 日志配置验证器 ====================


class LoggingConfigValidator(BaseConfigValidator):
    """日志配置验证器"""

    def __init__(self):
        """初始化日志配置验证器"""
        super().__init__(
            name="LoggingConfigValidator",
            description="验证日志配置"
        )

        # 添加验证规则
        self.add_rule(ValidationRule(
            ValidationType.REQUIRED,
            "logging.level",
            required=False  # 修改为非必需，缺少时只是警告
        ))
        self.add_rule(ValidationRule(
            ValidationType.TYPE,
            "logging.file",
            required=False,
            type=(str, dict)  # 可以是字符串路径或文件配置对象
        ))
        self.add_rule(ValidationRule(
            ValidationType.TYPE,
            "logging.max_size",
            required=False,
            type=(str, int)  # 允许字符串（如"10MB"）或整数
        ))

    def _validate_custom(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """自定义日志配置验证"""
        results = []

        logging_config = config.get('logging', {})

        # 验证日志级别
        level = logging_config.get('level', '').upper()
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

        if level and level not in valid_levels:
            results.append(ValidationResult(is_valid=False, errors=[f"无效的日志级别: {level}"],
                                            severity=ValidationSeverity.ERROR,
                                            field="logging.level",
                                            value=level,
                                            suggestions=[f"有效的级别: {', '.join(valid_levels)}"]
                                            ))

        # 验证日志文件路径
        log_file = logging_config.get('file')
        if log_file:
            # 如果是字符串，直接验证路径
            if isinstance(log_file, str) and not self._is_valid_log_path(log_file):
                results.append(ValidationResult(is_valid=False, errors=[f"无效的日志文件路径: {log_file}"],
                                                severity=ValidationSeverity.WARNING,
                                                field="logging.file",
                                                value=log_file,
                                                suggestions=["确保路径存在且可写"]
                                                ))
            # 如果是dict，验证其中的path字段
            elif isinstance(log_file, dict):
                path = log_file.get('path')
                if not path:
                    results.append(ValidationResult(is_valid=False, errors=["缺少path字段"],
                                                    severity=ValidationSeverity.ERROR,
                                                    field="logging.file.path",
                                                    value=log_file,
                                                    suggestions=["日志文件配置必须包含path字段"]
                                                    ))
                elif not self._is_valid_log_path(path):
                    results.append(ValidationResult(is_valid=False, errors=[f"无效的日志文件路径: {path}"],
                                                    severity=ValidationSeverity.WARNING,
                                                    field="logging.file.path",
                                                    value=path,
                                                    suggestions=["确保路径存在且可写"]
                                                    ))

        # 验证日志格式
        format_str = logging_config.get('format')
        if format_str and not self._is_valid_log_format(format_str):
            results.append(ValidationResult(is_valid=False, errors=[f"日志格式可能无效: {format_str}"],
                                            severity=ValidationSeverity.INFO,
                                            field="logging.format",
                                            value=format_str,
                                            suggestions=["使用标准的Python日志格式化字符串"]
                                            ))

        # 验证日志文件大小
        max_size = logging_config.get('file', {}).get('max_size') if isinstance(
            logging_config.get('file'), dict) else logging_config.get('max_size')
        if max_size and not self._is_valid_max_size(max_size):
            results.append(ValidationResult(is_valid=False, errors=[f"无效的日志文件大小格式: {max_size}"],
                                            severity=ValidationSeverity.ERROR,
                                            field="logging.max_size",
                                            value=max_size
                                            ))

        # 验证日志轮转配置
        rotation = logging_config.get('rotation')
        if rotation and not self._is_valid_rotation_config(rotation):
            results.append(ValidationResult(is_valid=False, errors=[f"无效的日志轮转配置: {rotation}"],
                                            severity=ValidationSeverity.WARNING,
                                            field="logging.rotation",
                                            value=rotation
                                            ))

        return results

    def _is_valid_log_path(self, path: str) -> bool:
        """验证日志文件路径"""
        if not isinstance(path, str):
            return False

        # 检查路径格式（不检查实际存在性，只检查格式）
        try:
            # 尝试规范化路径
            normalized = os.path.normpath(path)
            return len(normalized) > 0 and not normalized.startswith('..')
        except (ValueError, OSError):
            return False

    def _is_valid_log_format(self, format_str: str) -> bool:
        """验证日志格式字符串"""
        if not isinstance(format_str, str):
            return False

        # 检查是否包含基本的日志格式化字段
        required_fields = ['%(levelname)s', '%(message)s']
        return all(field in format_str for field in required_fields)

    def _is_valid_max_size(self, max_size: Any) -> bool:
        """验证日志文件最大大小格式"""
        if isinstance(max_size, int):
            return max_size > 0
        elif isinstance(max_size, str):
            # 支持如 "10MB", "1GB" 等格式
            pattern = r'^\d+(B|KB|MB|GB|TB)$'
            return bool(re.match(pattern, max_size.upper()))
        return False

    def _is_valid_rotation_config(self, rotation: Any) -> bool:
        """验证日志轮转配置"""
        if isinstance(rotation, str):
            return rotation.lower() in ['daily', 'weekly', 'monthly', 'hourly']
        elif isinstance(rotation, int):
            return rotation > 0
        elif isinstance(rotation, dict):
            # 检查轮转字典配置
            return 'when' in rotation or 'interval' in rotation
        return False

# ==================== 网络配置验证器 ====================


class NetworkConfigValidator(BaseConfigValidator):
    """网络配置验证器"""

    def __init__(self):
        """初始化网络配置验证器"""
        super().__init__(
            name="NetworkConfigValidator",
            description="验证网络配置"
        )

        # 添加验证规则
        self.add_rule(ValidationRule(
            ValidationType.REQUIRED,
            "network.host",
            required=False  # 修改为非必需，缺少时只是警告
        ))
        self.add_rule(ValidationRule(
            ValidationType.TYPE,
            "network.port",
            required=False,  # 修改为非必需，缺少时只是警告
            type=int
        ))
        self.add_rule(ValidationRule(
            ValidationType.RANGE,
            "network.port",
            required=False,  # 修改为非必需，缺少时只是警告
            min=1,
            max=MAX_PORT_NUMBER
        ))

    def _validate_custom(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """自定义网络配置验证"""
        results = []

        network = config.get('network', {})

        # 验证主机地址
        host = network.get('host')
        if host and not self._is_valid_network_host(host):
            results.append(ValidationResult(is_valid=False, errors=[f"无效的网络主机地址: {host}"],
                                            severity=ValidationSeverity.ERROR,
                                            field="network.host",
                                            value=host,
                                            suggestions=["使用有效的IP地址、主机名或 '0.0.0.0'"]
                                            ))

        # 验证端口范围
        if 'port' in network:
            port = network.get('port')
        else:
            port = None

        if 'port' in network and port in (None, ""):
            results.append(ValidationResult(
                is_valid=False,
                errors=["网络端口不能为空"],
                severity=ValidationSeverity.ERROR,
                field="network.port",
                value=port,
                suggestions=["提供1-65535范围内的有效端口号"]
            ))
        else:
            normalized_port = port
            if isinstance(normalized_port, str):
                stripped = normalized_port.strip()
                if not stripped:
                    results.append(ValidationResult(
                        is_valid=False,
                        errors=["网络端口不能为空"],
                        severity=ValidationSeverity.ERROR,
                        field="network.port",
                        value=port,
                        suggestions=["提供1-65535范围内的有效端口号"]
                    ))
                    normalized_port = None
                elif stripped.isdigit():
                    normalized_port = int(stripped)
                else:
                    normalized_port = None

            well_known_ports = WELL_KNOWN_PORTS

            if isinstance(normalized_port, int) and normalized_port in well_known_ports:
                results.append(ValidationResult(is_valid=False, errors=[f"端口 {normalized_port} 是知名端口，建议使用其他端口"],
                                                severity=ValidationSeverity.WARNING,
                                                field="network.port",
                                                value=port,
                                                suggestions=["使用1024以上的端口"]
                                                ))

        # 验证SSL配置
        ssl_config = network.get('ssl', {})
        if ssl_config:
            results.extend(self._validate_ssl_config(ssl_config))

        # 验证代理配置
        proxy_config = network.get('proxy')
        if proxy_config:
            results.extend(self._validate_proxy_config(proxy_config))

        return results

    def _is_valid_network_host(self, host: str) -> bool:
        """验证网络主机地址"""
        if not isinstance(host, str):
            return False

        # 允许的特殊值
        if host in ['0.0.0.0', '127.0.0.1', 'localhost']:
            return True

        # 检查IP地址
        try:
            ipaddress.ip_address(host)
            return True
        except ValueError:
            pass

        # 检查主机名格式
        if re.match(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', host):
            return True

        return False

    def _is_valid_ip(self, ip: str) -> bool:
        """验证IP地址格式"""
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False

    def _validate_ssl_config(self, ssl_config: Dict[str, Any]) -> List[ValidationResult]:
        """验证SSL配置"""
        results = []

        # 检查SSL是否启用
        ssl_enabled = ssl_config.get('enabled', False)
        if ssl_enabled:
            # 当SSL启用时，cert_file和key_file是必需的
            cert_file = ssl_config.get('cert_file')
            key_file = ssl_config.get('key_file')

            if not cert_file:
                results.append(ValidationResult(
                    is_valid=False,
                    errors=["SSL启用时必须配置cert_file"],
                    severity=ValidationSeverity.ERROR,
                    field="network.ssl.cert_file",
                    value=None
                ))

            if not key_file:
                results.append(ValidationResult(
                    is_valid=False,
                    errors=["SSL启用时必须配置key_file"],
                    severity=ValidationSeverity.ERROR,
                    field="network.ssl.key_file",
                    value=None
                ))

            # 验证证书文件路径（如果存在）
            if cert_file and not self._is_valid_file_path(cert_file):
                results.append(ValidationResult(
                    is_valid=False,
                    errors=[f"无效的SSL证书文件路径: {cert_file}"],
                    severity=ValidationSeverity.ERROR,
                    field="network.ssl.cert_file",
                    value=cert_file
                ))

            # 验证密钥文件路径（如果存在）
            if key_file and not self._is_valid_file_path(key_file):
                results.append(ValidationResult(
                    is_valid=False,
                    errors=[f"无效的SSL密钥文件路径: {key_file}"],
                    severity=ValidationSeverity.ERROR,
                    field="network.ssl.key_file",
                    value=key_file
                ))
        else:
            # SSL未启用时，验证现有字段（如果存在）
            cert_file = ssl_config.get('cert_file')
            key_file = ssl_config.get('key_file')

            # 验证证书文件路径（如果存在）
            if cert_file and not self._is_valid_file_path(cert_file):
                results.append(ValidationResult(
                    is_valid=False,
                    errors=[f"无效的SSL证书文件路径: {cert_file}"],
                    severity=ValidationSeverity.ERROR,
                    field="network.ssl.cert_file",
                    value=cert_file
                ))

            # 验证密钥文件路径（如果存在）
            if key_file and not self._is_valid_file_path(key_file):
                results.append(ValidationResult(
                    is_valid=False,
                    errors=[f"无效的SSL密钥文件路径: {key_file}"],
                    severity=ValidationSeverity.ERROR,
                    field="network.ssl.key_file",
                    value=key_file
                ))

        # 验证SSL版本
        ssl_version = ssl_config.get('version')
        if ssl_version and ssl_version not in ['TLSv1', 'TLSv1.1', 'TLSv1.2', 'TLSv1.3']:
            results.append(ValidationResult(
                is_valid=False,
                errors=[f"不支持的SSL版本: {ssl_version}"],
                severity=ValidationSeverity.WARNING,
                field="network.ssl.version",
                value=ssl_version,
                suggestions=["推荐使用 TLSv1.2 或 TLSv1.3"]
            ))

        return results

    def _validate_proxy_config(self, proxy_config: Union[str, Dict[str, Any]]) -> List[ValidationResult]:
        """验证代理配置"""
        results = []

        if isinstance(proxy_config, str):
            # 简单的URL验证
            if not re.match(r'^https?://.*', proxy_config):
                results.append(ValidationResult(is_valid=False, errors=[f"无效的代理URL格式: {proxy_config}"],
                                                severity=ValidationSeverity.ERROR,
                                                field="network.proxy",
                                                value=proxy_config,
                                                suggestions=["使用 http:// 或 https:// 开头的URL"]
                                                ))
        elif isinstance(proxy_config, dict):
            # 验证代理字典配置
            for protocol, url in proxy_config.items():
                if not isinstance(url, str) or not re.match(r'^https?://.*', url):
                    results.append(ValidationResult(is_valid=False, errors=[f"协议 {protocol} 的代理URL格式无效: {url}"],
                                                    severity=ValidationSeverity.ERROR,
                                                    field=f"network.proxy.{protocol}",
                                                    value=url
                                                    ))

        return results

    def _is_valid_file_path(self, path: str) -> bool:
        """验证文件路径"""
        if not isinstance(path, str):
            return False

        try:
            # 检查路径是否可解析
            normalized = os.path.normpath(path)
            return len(normalized) > 0
        except (ValueError, OSError):
            return False




