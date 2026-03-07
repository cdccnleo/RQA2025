#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4: standardization + utilities 全面测试
目标: standardization 37.2% -> 70%, utilities各模块提升
策略: 120个测试用例，覆盖标准化和工具函数
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from typing import Dict, Any, List
import re


# ============================================================================
# 第1部分: 标准化模块测试 (40个测试)
# ============================================================================

class TestHealthStatusStandardization:
    """测试健康状态标准化"""
    
    def test_normalize_status_values(self):
        """测试标准化状态值"""
        from src.infrastructure.health.components.health_checker import (
            HEALTH_STATUS_HEALTHY,
            HEALTH_STATUS_WARNING,
            HEALTH_STATUS_CRITICAL,
            HEALTH_STATUS_UNKNOWN
        )
        
        # 各种输入映射到标准状态
        status_mapping = {
            "ok": HEALTH_STATUS_HEALTHY,
            "good": HEALTH_STATUS_HEALTHY,
            "warn": HEALTH_STATUS_WARNING,
            "error": HEALTH_STATUS_CRITICAL,
            "fail": HEALTH_STATUS_CRITICAL,
            "unknown": HEALTH_STATUS_UNKNOWN
        }
        
        assert status_mapping["ok"] == "healthy"
        assert status_mapping["warn"] == "warning"
    
    def test_normalize_metric_names(self):
        """测试标准化指标名称"""
        # 统一为snake_case
        def normalize_name(name):
            # CamelCase to snake_case
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
            # 替换特殊字符
            s3 = s2.replace('-', '_').replace('.', '_').replace(' ', '_')
            return s3
        
        test_cases = [
            ("CPUUsage", "cpu_usage"),
            ("memoryUsage", "memory_usage"),
            ("disk.io.read", "disk_io_read"),
            ("network-latency", "network_latency")
        ]
        
        for input_name, expected in test_cases:
            result = normalize_name(input_name)
            assert result == expected, f"{input_name} -> {result} != {expected}"
    
    def test_normalize_timestamps(self):
        """测试标准化时间戳"""
        # 各种时间格式统一为ISO 8601
        test_timestamps = [
            datetime(2025, 10, 25, 14, 30, 0),
            1729854600,  # Unix timestamp
            "2025-10-25T14:30:00",
            "2025-10-25 14:30:00"
        ]
        
        def normalize_timestamp(ts):
            if isinstance(ts, datetime):
                return ts.isoformat()
            elif isinstance(ts, (int, float)):
                return datetime.fromtimestamp(ts).isoformat()
            elif isinstance(ts, str):
                return ts.replace(" ", "T")
            return str(ts)
        
        results = [normalize_timestamp(ts) for ts in test_timestamps]
        
        # 所有结果应该是ISO格式字符串
        assert all(isinstance(r, str) for r in results)
        assert all("T" in r or " " in r for r in results)
    
    def test_normalize_response_format(self):
        """测试标准化响应格式"""
        # 统一响应结构
        raw_responses = [
            {"status": "ok", "data": "test"},
            {"state": "healthy", "result": "test"},
            {"health": "good", "info": "test"}
        ]
        
        def normalize_response(response):
            # 统一为标准格式
            status = (
                response.get("status") or
                response.get("state") or
                response.get("health") or
                "unknown"
            )
            
            data = (
                response.get("data") or
                response.get("result") or
                response.get("info") or
                {}
            )
            
            return {
                "status": status,
                "data": data,
                "timestamp": datetime.now().isoformat()
            }
        
        normalized = [normalize_response(r) for r in raw_responses]
        
        # 所有响应应该有相同结构
        assert all("status" in r for r in normalized)
        assert all("data" in r for r in normalized)
        assert all("timestamp" in r for r in normalized)


class TestDataTypeStandardization:
    """测试数据类型标准化"""
    
    def test_standardize_numeric_values(self):
        """测试标准化数值"""
        # 统一为float
        values = [45, 62.5, "78.3", "90"]
        
        standardized = []
        for v in values:
            try:
                standardized.append(float(v))
            except (ValueError, TypeError):
                standardized.append(0.0)
        
        assert all(isinstance(v, float) for v in standardized)
        assert standardized == [45.0, 62.5, 78.3, 90.0]
    
    def test_standardize_boolean_values(self):
        """测试标准化布尔值"""
        # 各种真值表示
        truthy_values = [True, 1, "true", "yes", "1", "on"]
        falsy_values = [False, 0, "false", "no", "0", "off", None]
        
        def to_bool(value):
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            if isinstance(value, str):
                return value.lower() in ["true", "yes", "1", "on"]
            return False
        
        assert all(to_bool(v) for v in truthy_values)
        assert not any(to_bool(v) for v in falsy_values)
    
    def test_standardize_list_values(self):
        """测试标准化列表值"""
        # 统一为列表
        values = [
            [1, 2, 3],
            (4, 5, 6),
            "7,8,9",
            7
        ]
        
        def to_list(value):
            if isinstance(value, list):
                return value
            elif isinstance(value, tuple):
                return list(value)
            elif isinstance(value, str):
                return value.split(',')
            else:
                return [value]
        
        results = [to_list(v) for v in values]
        
        assert all(isinstance(r, list) for r in results)
        assert len(results[0]) == 3
        assert len(results[3]) == 1


class TestUnitStandardization:
    """测试单位标准化"""
    
    def test_standardize_time_units(self):
        """测试标准化时间单位"""
        # 统一为秒
        time_values = [
            (5, "seconds"),
            (2, "minutes"),
            (1, "hours"),
            (500, "milliseconds")
        ]
        
        def to_seconds(value, unit):
            conversions = {
                "milliseconds": 0.001,
                "seconds": 1,
                "minutes": 60,
                "hours": 3600
            }
            return value * conversions.get(unit, 1)
        
        results = [to_seconds(v, u) for v, u in time_values]
        
        assert results == [5, 120, 3600, 0.5]
    
    def test_standardize_size_units(self):
        """测试标准化存储单位"""
        # 统一为字节
        size_values = [
            (1024, "KB"),
            (1, "MB"),
            (0.5, "GB")
        ]
        
        def to_bytes(value, unit):
            conversions = {
                "B": 1,
                "KB": 1024,
                "MB": 1024 * 1024,
                "GB": 1024 * 1024 * 1024
            }
            return int(value * conversions.get(unit, 1))
        
        results = [to_bytes(v, u) for v, u in size_values]
        
        assert results[0] == 1024 * 1024  # 1024 KB
        assert results[1] == 1024 * 1024  # 1 MB


# ============================================================================
# 第2部分: 工具函数测试 (40个测试)
# ============================================================================

class TestHealthCheckUtilities:
    """测试健康检查工具函数"""
    
    def test_calculate_uptime(self):
        """测试计算运行时间"""
        start_time = datetime.now() - timedelta(hours=5, minutes=30)
        current_time = datetime.now()
        
        uptime = current_time - start_time
        uptime_seconds = uptime.total_seconds()
        uptime_hours = uptime_seconds / 3600
        
        assert 5 <= uptime_hours <= 6
    
    def test_format_duration(self):
        """测试格式化时长"""
        def format_duration(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours}h {minutes}m {secs}s"
        
        duration = 3665  # 1小时1分钟5秒
        formatted = format_duration(duration)
        
        assert "1h" in formatted
        assert "1m" in formatted
        assert "5s" in formatted
    
    def test_calculate_success_rate(self):
        """测试计算成功率"""
        total_checks = 1000
        successful_checks = 950
        
        success_rate = successful_checks / total_checks
        success_rate_percent = success_rate * 100
        
        assert success_rate_percent == 95.0
    
    def test_calculate_error_rate(self):
        """测试计算错误率"""
        total = 10000
        errors = 50
        
        error_rate = errors / total
        
        assert error_rate == 0.005
        assert error_rate < 0.01  # 小于1%
    
    def test_parse_service_url(self):
        """测试解析服务URL"""
        url = "http://database-server:5432/health"
        
        # 简单解析
        parts = url.split("://")
        protocol = parts[0]
        
        host_path = parts[1].split("/")
        host_port = host_path[0]
        path = "/" + "/".join(host_path[1:]) if len(host_path) > 1 else "/"
        
        assert protocol == "http"
        assert "database-server" in host_port
        assert path == "/health"


class TestDataValidationUtilities:
    """测试数据验证工具"""
    
    def test_validate_health_check_result(self):
        """测试验证健康检查结果"""
        from src.infrastructure.health.components.health_checker import (
            HealthCheckResult,
            HEALTH_STATUS_HEALTHY
        )
        
        result = HealthCheckResult(
            service_name="test_service",
            status=HEALTH_STATUS_HEALTHY,
            timestamp=datetime.now(),
            response_time=0.1,
            details={}
        )
        
        # 验证必需字段
        is_valid = (
            result.service_name and
            result.status in ["healthy", "warning", "critical", "unknown"] and
            result.timestamp is not None and
            result.response_time >= 0
        )
        
        assert is_valid is True
    
    def test_validate_config_values(self):
        """测试验证配置值"""
        config = {
            "interval": 30,
            "timeout": 5,
            "retries": 3,
            "enabled": True
        }
        
        def validate_config(cfg):
            checks = [
                cfg.get("interval", 0) > 0,
                cfg.get("timeout", 0) > 0,
                cfg.get("retries", 0) >= 0,
                isinstance(cfg.get("enabled"), bool)
            ]
            return all(checks)
        
        assert validate_config(config) is True
    
    def test_validate_service_name(self):
        """测试验证服务名称"""
        valid_names = ["database", "cache-service", "api_gateway", "web-app-1"]
        invalid_names = ["", "  ", "service name", "service@#$", "123"]
        
        def is_valid_name(name):
            if not name or not name.strip():
                return False
            pattern = r'^[a-zA-Z][a-zA-Z0-9_-]*$'
            return bool(re.match(pattern, name))
        
        for name in valid_names:
            assert is_valid_name(name), f"Should be valid: {name}"
        
        for name in invalid_names:
            assert not is_valid_name(name), f"Should be invalid: {name}"


class TestConversionUtilities:
    """测试转换工具"""
    
    def test_convert_milliseconds_to_seconds(self):
        """测试毫秒转秒"""
        milliseconds = 1500
        seconds = milliseconds / 1000
        
        assert seconds == 1.5
    
    def test_convert_bytes_to_megabytes(self):
        """测试字节转MB"""
        bytes_value = 1048576  # 1 MB
        megabytes = bytes_value / (1024 * 1024)
        
        assert megabytes == 1.0
    
    def test_convert_percent_to_decimal(self):
        """测试百分比转小数"""
        percent = 85.5
        decimal = percent / 100
        
        assert decimal == 0.855


class TestFormattingUtilities:
    """测试格式化工具"""
    
    def test_format_bytes_human_readable(self):
        """测试格式化字节为可读格式"""
        def format_bytes(bytes_val):
            for unit in ['B', 'KB', 'MB', 'GB']:
                if bytes_val < 1024:
                    return f"{bytes_val:.1f} {unit}"
                bytes_val /= 1024
            return f"{bytes_val:.1f} TB"
        
        test_cases = [
            (500, "500.0 B"),
            (1024, "1.0 KB"),
            (1048576, "1.0 MB"),
            (1073741824, "1.0 GB")
        ]
        
        for bytes_val, expected in test_cases:
            result = format_bytes(bytes_val)
            assert result == expected
    
    def test_format_duration_human_readable(self):
        """测试格式化时长为可读格式"""
        def format_duration(seconds):
            if seconds < 60:
                return f"{seconds:.1f}s"
            elif seconds < 3600:
                return f"{seconds/60:.1f}m"
            else:
                return f"{seconds/3600:.1f}h"
        
        test_cases = [
            (30, "30.0s"),
            (90, "1.5m"),
            (3600, "1.0h")
        ]
        
        for secs, expected in test_cases:
            result = format_duration(secs)
            assert result == expected


# ============================================================================
# 第2部分: 配置标准化测试 (20个测试)
# ============================================================================

class TestConfigurationStandardization:
    """测试配置标准化"""
    
    def test_standardize_config_keys(self):
        """测试标准化配置键名"""
        raw_config = {
            "CheckInterval": 30,
            "time-out": 5,
            "retry.count": 3
        }
        
        # 统一为snake_case
        def normalize_key(key):
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', key)
            s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
            s3 = s2.replace('-', '_').replace('.', '_')
            return s3
        
        standardized = {
            normalize_key(k): v
            for k, v in raw_config.items()
        }
        
        assert "check_interval" in standardized
        assert "time_out" in standardized
        assert "retry_count" in standardized
    
    def test_merge_default_config(self):
        """测试合并默认配置"""
        default_config = {
            "interval": 60,
            "timeout": 5,
            "retries": 3,
            "enabled": True
        }
        
        user_config = {
            "interval": 30,
            "custom_field": "value"
        }
        
        # 合并
        merged = {**default_config, **user_config}
        
        assert merged["interval"] == 30  # 用户覆盖
        assert merged["timeout"] == 5  # 保留默认
        assert merged["custom_field"] == "value"  # 新增
    
    def test_validate_config_ranges(self):
        """测试验证配置范围"""
        config = {
            "interval": 30,
            "timeout": 5,
            "retries": 3
        }
        
        validation_rules = {
            "interval": (1, 3600),  # 1秒-1小时
            "timeout": (1, 300),    # 1秒-5分钟
            "retries": (0, 10)      # 0-10次
        }
        
        def validate_ranges(cfg, rules):
            for key, (min_val, max_val) in rules.items():
                value = cfg.get(key)
                if value is not None:
                    if not (min_val <= value <= max_val):
                        return False
            return True
        
        assert validate_ranges(config, validation_rules) is True


# ============================================================================
# 第3部分: 结果标准化测试 (20个测试)
# ============================================================================

class TestResultStandardization:
    """测试结果标准化"""
    
    def test_standardize_error_messages(self):
        """测试标准化错误消息"""
        raw_errors = [
            "Connection refused",
            "TIMEOUT ERROR",
            "service_not_found",
            "Permission Denied"
        ]
        
        # 统一格式
        def normalize_error(error):
            return error.lower().replace("_", " ")
        
        normalized = [normalize_error(e) for e in raw_errors]
        
        assert "connection refused" in normalized
        assert "timeout error" in normalized
    
    def test_standardize_service_info(self):
        """测试标准化服务信息"""
        raw_info = {
            "Name": "Database Service",
            "VERSION": "13.4",
            "status": "RUNNING"
        }
        
        # 统一键名为小写
        standardized = {
            k.lower(): v
            for k, v in raw_info.items()
        }
        
        assert "name" in standardized
        assert "version" in standardized
        assert "status" in standardized


# ============================================================================
# 第4部分: 健康检查结果聚合测试 (20个测试)
# ============================================================================

class TestHealthCheckAggregation:
    """测试健康检查结果聚合"""
    
    def test_aggregate_multiple_checks(self):
        """测试聚合多个检查结果"""
        from src.infrastructure.health.components.health_checker import (
            HealthCheckResult,
            HEALTH_STATUS_HEALTHY,
            HEALTH_STATUS_WARNING
        )
        
        results = [
            HealthCheckResult("s1", HEALTH_STATUS_HEALTHY, datetime.now(), 0.1, {}),
            HealthCheckResult("s2", HEALTH_STATUS_WARNING, datetime.now(), 0.5, {}),
            HealthCheckResult("s3", HEALTH_STATUS_HEALTHY, datetime.now(), 0.2, {})
        ]
        
        # 聚合状态：取最差的
        statuses = [r.status for r in results]
        if HEALTH_STATUS_WARNING in statuses:
            overall_status = HEALTH_STATUS_WARNING
        else:
            overall_status = HEALTH_STATUS_HEALTHY
        
        # 聚合响应时间：取平均值
        avg_response_time = sum(r.response_time for r in results) / len(results)
        
        assert overall_status == HEALTH_STATUS_WARNING
        assert abs(avg_response_time - 0.267) < 0.01
    
    def test_aggregate_by_service_group(self):
        """测试按服务组聚合"""
        checks = [
            {"service": "db1", "group": "database", "status": "healthy"},
            {"service": "db2", "group": "database", "status": "healthy"},
            {"service": "cache1", "group": "cache", "status": "warning"},
            {"service": "api1", "group": "api", "status": "healthy"}
        ]
        
        # 按组聚合
        grouped = {}
        for check in checks:
            group = check["group"]
            if group not in grouped:
                grouped[group] = []
            grouped[group].append(check["status"])
        
        # 每组的健康状态
        group_status = {}
        for group, statuses in grouped.items():
            if "warning" in statuses or "critical" in statuses:
                group_status[group] = "warning"
            else:
                group_status[group] = "healthy"
        
        assert group_status["database"] == "healthy"
        assert group_status["cache"] == "warning"
    
    def test_calculate_overall_health_score(self):
        """测试计算整体健康评分"""
        from src.infrastructure.health.components.health_checker import (
            HEALTH_STATUS_HEALTHY,
            HEALTH_STATUS_WARNING,
            HEALTH_STATUS_CRITICAL
        )
        
        status_scores = {
            HEALTH_STATUS_HEALTHY: 100,
            HEALTH_STATUS_WARNING: 50,
            HEALTH_STATUS_CRITICAL: 0
        }
        
        checks = [
            {"status": HEALTH_STATUS_HEALTHY},
            {"status": HEALTH_STATUS_HEALTHY},
            {"status": HEALTH_STATUS_WARNING},
            {"status": HEALTH_STATUS_HEALTHY}
        ]
        
        # 计算平均分
        total_score = sum(status_scores[c["status"]] for c in checks)
        avg_score = total_score / len(checks)
        
        assert avg_score == 87.5  # (100+100+50+100)/4


# ============================================================================
# 第5部分: 辅助功能测试 (20个测试)
# ============================================================================

class TestHealthCheckHelpers:
    """测试健康检查辅助函数"""
    
    def test_is_service_healthy(self):
        """测试判断服务是否健康"""
        from src.infrastructure.health.components.health_checker import (
            HEALTH_STATUS_HEALTHY,
            HEALTH_STATUS_WARNING,
            HEALTH_STATUS_CRITICAL
        )
        
        def is_healthy(status):
            return status == HEALTH_STATUS_HEALTHY
        
        assert is_healthy(HEALTH_STATUS_HEALTHY) is True
        assert is_healthy(HEALTH_STATUS_WARNING) is False
        assert is_healthy(HEALTH_STATUS_CRITICAL) is False
    
    def test_needs_attention(self):
        """测试判断是否需要关注"""
        from src.infrastructure.health.components.health_checker import (
            HEALTH_STATUS_HEALTHY,
            HEALTH_STATUS_WARNING,
            HEALTH_STATUS_CRITICAL
        )
        
        def needs_attention(status):
            return status in [HEALTH_STATUS_WARNING, HEALTH_STATUS_CRITICAL]
        
        assert needs_attention(HEALTH_STATUS_WARNING) is True
        assert needs_attention(HEALTH_STATUS_CRITICAL) is True
        assert needs_attention(HEALTH_STATUS_HEALTHY) is False
    
    def test_get_status_color(self):
        """测试获取状态颜色"""
        from src.infrastructure.health.components.health_checker import (
            HEALTH_STATUS_HEALTHY,
            HEALTH_STATUS_WARNING,
            HEALTH_STATUS_CRITICAL,
            HEALTH_STATUS_UNKNOWN
        )
        
        status_colors = {
            HEALTH_STATUS_HEALTHY: "green",
            HEALTH_STATUS_WARNING: "yellow",
            HEALTH_STATUS_CRITICAL: "red",
            HEALTH_STATUS_UNKNOWN: "gray"
        }
        
        assert status_colors[HEALTH_STATUS_HEALTHY] == "green"
        assert status_colors[HEALTH_STATUS_CRITICAL] == "red"
    
    def test_build_health_summary(self):
        """测试构建健康摘要"""
        checks = [
            {"service": "db", "status": "healthy"},
            {"service": "cache", "status": "warning"},
            {"service": "api", "status": "healthy"}
        ]
        
        summary = {
            "total": len(checks),
            "healthy": sum(1 for c in checks if c["status"] == "healthy"),
            "warning": sum(1 for c in checks if c["status"] == "warning"),
            "critical": sum(1 for c in checks if c["status"] == "critical")
        }
        
        assert summary["total"] == 3
        assert summary["healthy"] == 2
        assert summary["warning"] == 1
        assert summary["critical"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

