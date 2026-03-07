"""
数据隐私保护组件模拟测试
测试数据脱敏、隐私保护、合规加密、访问审计功能
"""
import asyncio
import pandas as pd
from unittest.mock import Mock

# Mock数据管理器模块以绕过复杂的导入问题
mock_data_manager = Mock()
mock_data_manager.DataManager = Mock()
mock_data_manager.DataLoaderError = Exception

# 配置DataManager实例方法
mock_instance = Mock()
mock_instance.validate_all_configs.return_value = True
mock_instance.health_check.return_value = {"status": "healthy"}
mock_instance.store_data.return_value = True
mock_instance.has_data.return_value = True
mock_instance.get_metadata.return_value = {"data_type": "test", "symbol": "X"}
mock_instance.retrieve_data.return_value = pd.DataFrame({"col": [1, 2, 3]})
mock_instance.get_stats.return_value = {"total_items": 1}
mock_instance.validate_data.return_value = {"valid": True}
mock_instance.shutdown.return_value = None

mock_data_manager.DataManager.return_value = mock_instance

# Mock整个模块
import sys
sys.modules["src.data.data_manager"] = mock_data_manager


import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional, Set
import json
import hashlib
import re


# Mock 依赖
class MockLogger:
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass
    def debug(self, msg): pass


class MockAuditEventType:
    SECURITY = "security"
    ACCESS = "access"
    DATA_OPERATION = "data_operation"
    CONFIG_CHANGE = "config_change"
    USER_MANAGEMENT = "user_management"
    SYSTEM_EVENT = "system_event"
    COMPLIANCE = "compliance"


class MockAuditSeverity:
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MockAuditEvent:
    def __init__(self, event_id, event_type, severity, timestamp, user_id=None, session_id=None,
                 resource=None, action="", result="", details=None, ip_address=None,
                 user_agent=None, location=None, risk_score=0.0, tags=None):
        self.event_id = event_id
        self.event_type = event_type
        self.severity = severity
        self.timestamp = timestamp
        self.user_id = user_id
        self.session_id = session_id
        self.resource = resource
        self.action = action
        self.result = result
        self.details = details or {}
        self.ip_address = ip_address
        self.user_agent = user_agent
        self.location = location
        self.risk_score = risk_score
        self.tags = tags or set()

    def to_dict(self):
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'severity': self.severity,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'session_id': self.session_id,
            'resource': self.resource,
            'action': self.action,
            'result': self.result,
            'details': self.details,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'location': self.location,
            'risk_score': self.risk_score,
            'tags': list(self.tags)
        }


class MockPrivacyProtector:
    def __init__(self):
        self.logger = MockLogger()

    def protect(self, data, level="standard"):
        if not isinstance(data, str):
            return data

        if not level or level not in ["standard", "encrypted", "none"]:
            level = "standard"

        if level == "none":
            return data
        elif level == "encrypted":
            return hashlib.sha256(data.encode()).hexdigest()
        elif level == "standard":
            return self._mask_data(data)

        return data

    def _mask_data(self, data):
        if not data:
            return data

        # 手机号模式
        if re.match(r'^1[3-9]\d{9}$', data):
            return data[:2] + "*******" + data[-2:]

        # 邮箱模式
        if '@' in data:
            parts = data.split('@')
            username = parts[0]
            domain = parts[1]
            if len(username) > 1:
                masked_username = username[0] + "***"
            else:
                masked_username = "***"
            return masked_username + "@***." + domain.split('.')[-1]

        # 身份证号模式
        if re.match(r'^\d{17}[\dXx]$', data):
            return data[:6] + "****" + data[-4:]

        # 信用卡号模式
        if re.match(r'^\d{16}$', data):
            return data[:4] + "****" + data[-4:]

        # 银行账号模式
        if re.match(r'^\d{16,19}$', data):
            return data[:4] + "****" + data[-4:]

        # 地址模式
        if len(data) > 8:
            return data[:2] + "****" + data[-2:]

        # 姓名模式
        if len(data) == 2:
            return data[0] + "*"

        # 默认模式：短字符串全部脱敏
        if len(data) <= 4:
            return "*" * len(data)

        # 长字符串：显示前2后2位
        return data[:2] + "****" + data[-2:]


class MockDataPolicyManager:
    def __init__(self):
        self.policies = {}

    def _validate_policy(self, policy):
        if not policy or not isinstance(policy, dict):
            return False
        if "name" not in policy:
            return False
        if "required_fields" not in policy:
            return False
        return True

    def _add_timestamps(self, policy):
        now = datetime.now().isoformat()
        if "created_at" not in policy:
            policy["created_at"] = now
        policy["updated_at"] = now

    def register_policy(self, policy):
        if not self._validate_policy(policy):
            return False

        policy_id = policy.get("id") or f"policy_{len(self.policies) + 1}"
        if policy_id in self.policies:
            return False

        policy["id"] = policy_id
        self._add_timestamps(policy)
        self.policies[policy_id] = policy
        return True

    def get_policy(self, policy_id):
        return self.policies.get(policy_id)

    def update_policy(self, policy_id, updates):
        if policy_id not in self.policies:
            return False
        if not isinstance(updates, dict):
            return False
        self.policies[policy_id].update(updates)
        self._add_timestamps(self.policies[policy_id])
        return True

    def delete_policy(self, policy_id):
        if policy_id not in self.policies:
            return False
        del self.policies[policy_id]
        return True

    def list_policies(self):
        return self.policies.copy()


class MockComplianceChecker:
    def __init__(self, policy_manager):
        self.policy_manager = policy_manager
        self.logger = MockLogger()

    def check(self, data, policy_id=None):
        start_time = datetime.now()
        policy = self.policy_manager.get_policy(policy_id) if policy_id else None
        issues = []
        compliance = True

        try:
            field_issues = self._check_required_fields(data, policy)
            issues.extend(field_issues)
            type_issues = self._check_data_types(data, policy)
            issues.extend(type_issues)
            business_issues = self._check_business_rules(data, policy)
            issues.extend(business_issues)
            security_issues = self._check_security_requirements(data, policy)
            issues.extend(security_issues)

            if issues:
                compliance = False
        except Exception as e:
            issues.append(f"检查过程错误: {str(e)}")
            compliance = False

        return {
            "compliance": compliance,
            "issues": issues,
            "checked_at": datetime.now().isoformat(),
            "check_duration_seconds": (datetime.now() - start_time).total_seconds(),
            "policy_applied": policy_id,
            "check_type": "comprehensive"
        }

    def _check_required_fields(self, data, policy):
        issues = []
        if not policy or "required_fields" not in policy:
            return issues
        if not isinstance(data, dict):
            issues.append("数据格式错误：期望字典格式")
            return issues

        for field in policy["required_fields"]:
            if field not in data or data[field] is None or str(data[field]).strip() == "":
                issues.append(f"缺失必需字段: {field}")
        return issues

    def _check_data_types(self, data, policy):
        issues = []
        if not policy or "field_types" not in policy:
            return issues
        if not isinstance(data, dict):
            return issues

        type_mapping = {
            "string": str,
            "integer": int,
            "float": float,
            "boolean": bool,
            "list": list,
            "dict": dict
        }

        for field, expected_type in policy["field_types"].items():
            if field in data:
                expected_class = type_mapping.get(expected_type.lower())
                if expected_class and not isinstance(data[field], expected_class):
                    issues.append(f"字段 {field} 类型错误：期望 {expected_type}，实际 {type(data[field]).__name__}")
        return issues

    def _check_business_rules(self, data, policy):
        issues = []
        if not policy or "business_rules" not in policy:
            return issues
        if not isinstance(data, dict):
            return issues

        rules = policy["business_rules"]
        if "value_ranges" in rules:
            for field, range_spec in rules["value_ranges"].items():
                if field in data and isinstance(data[field], (int, float)):
                    min_val = range_spec.get("min")
                    max_val = range_spec.get("max")
                    if min_val is not None and data[field] < min_val:
                        issues.append(f"字段 {field} 值 {data[field]} 小于最小值 {min_val}")
                    if max_val is not None and data[field] > max_val:
                        issues.append(f"字段 {field} 值 {data[field]} 大于最大值 {max_val}")

        if "enum_values" in rules:
            for field, allowed_values in rules["enum_values"].items():
                if field in data and data[field] not in allowed_values:
                    issues.append(f"字段 {field} 值 {data[field]} 不在允许值列表中: {allowed_values}")
        return issues

    def _check_security_requirements(self, data, policy):
        issues = []
        if not isinstance(data, dict):
            return issues

        sensitive_patterns = ["password", "token", "key", "secret", "ssn", "credit_card"]
        for field, value in data.items():
            field_lower = field.lower()
            value_str = str(value).lower()
            for pattern in sensitive_patterns:
                if pattern in field_lower or pattern in value_str:
                    issues.append(f"检测到潜在敏感信息字段: {field}")
                    break

        if policy and "max_field_lengths" in policy:
            for field, max_length in policy["max_field_lengths"].items():
                if field in data and isinstance(data[field], str) and len(data[field]) > max_length:
                    issues.append(f"字段 {field} 长度 {len(data[field])} 超过最大限制 {max_length}")
        return issues

    def check_trading_compliance(self, trade_data):
        issues = []

        if "amount" in trade_data:
            amount = trade_data["amount"]
            if amount <= 0:
                issues.append("交易金额必须大于0")
            elif amount > 10000000:
                issues.append("交易金额超过单笔交易上限")

        if "timestamp" in trade_data:
            try:
                trade_time = datetime.fromisoformat(trade_data["timestamp"].replace('Z', '+00:00'))
                if trade_time.weekday() >= 5:
                    issues.append("交易时间不在工作日")
                elif not (9 <= trade_time.hour <= 15):
                    issues.append("交易时间不在交易时段（9:00-15:00）")
            except:
                issues.append("交易时间戳格式错误")

        valid_trade_types = ["buy", "sell", "short", "cover"]
        if "trade_type" in trade_data and trade_data["trade_type"] not in valid_trade_types:
            issues.append(f"无效的交易类型: {trade_data['trade_type']}")

        return {
            "compliance": len(issues) == 0,
            "issues": issues,
            "checked_at": datetime.now().isoformat(),
            "check_type": "trading_specific"
        }

    def check_bulk_data(self, data_list, policy_id=None):
        total_records = len(data_list)
        compliant_records = 0
        all_issues = []

        for i, data in enumerate(data_list):
            result = self.check(data, policy_id)
            if result["compliance"]:
                compliant_records += 1
            else:
                all_issues.extend([f"记录 {i+1}: {issue}" for issue in result["issues"]])

        return {
            "total_records": total_records,
            "compliant_records": compliant_records,
            "non_compliant_records": total_records - compliant_records,
            "compliance_rate": compliant_records / total_records if total_records > 0 else 0,
            "all_issues": all_issues[:100],
            "checked_at": datetime.now().isoformat(),
            "check_type": "bulk_check"
        }


class MockDataComplianceManager:
    def __init__(self):
        self.policy_manager = MockDataPolicyManager()
        self.compliance_checker = MockComplianceChecker(self.policy_manager)
        self.privacy_protector = MockPrivacyProtector()
        self.logger = MockLogger()

    def register_policy(self, policy):
        return self.policy_manager.register_policy(policy)

    def check_compliance(self, data, policy_id=None):
        return self.compliance_checker.check(data, policy_id)

    def check_bulk_compliance(self, data_list, policy_id=None):
        return self.compliance_checker.check_bulk_data(data_list, policy_id)

    def check_trading_compliance(self, trade_data):
        return self.compliance_checker.check_trading_compliance(trade_data)

    def protect_privacy(self, data, level="standard"):
        return self.privacy_protector.protect(data, level)

    def generate_compliance_report(self, data, policy_id=None):
        result = self.check_compliance(data, policy_id)
        report = {
            "policy_id": policy_id,
            "compliance": result.get("compliance", False),
            "issues": result.get("issues", []),
            "checked_at": result.get("checked_at"),
            "check_duration": result.get("check_duration_seconds", 0),
            "check_type": result.get("check_type", "standard"),
            "recommendations": self._generate_recommendations(result)
        }
        return report

    def generate_bulk_compliance_report(self, data_list, policy_id=None):
        bulk_result = self.check_bulk_compliance(data_list, policy_id)
        report = {
            "policy_id": policy_id,
            "total_records": bulk_result["total_records"],
            "compliant_records": bulk_result["compliant_records"],
            "non_compliant_records": bulk_result["non_compliant_records"],
            "compliance_rate": bulk_result["compliance_rate"],
            "sample_issues": bulk_result["all_issues"][:10],
            "checked_at": bulk_result["checked_at"],
            "severity_assessment": self._assess_bulk_severity(bulk_result),
            "recommendations": self._generate_bulk_recommendations(bulk_result)
        }
        return report

    def setup_default_policies(self):
        default_policies = [
            {
                "id": "user_data_policy",
                "name": "用户数据合规策略",
                "required_fields": ["user_id", "username", "email"],
                "field_types": {
                    "user_id": "integer",
                    "username": "string",
                    "email": "string",
                    "balance": "float",
                    "status": "string"
                },
                "business_rules": {
                    "value_ranges": {
                        "balance": {"min": 0, "max": 10000000}
                    },
                    "enum_values": {
                        "status": ["active", "inactive", "suspended"]
                    }
                },
                "max_field_lengths": {
                    "username": 50,
                    "email": 100
                }
            },
            {
                "id": "trade_data_policy",
                "name": "交易数据合规策略",
                "required_fields": ["user_id", "symbol", "quantity", "price", "trade_type"],
                "field_types": {
                    "user_id": "integer",
                    "symbol": "string",
                    "quantity": "integer",
                    "price": "float",
                    "trade_type": "string"
                },
                "business_rules": {
                    "value_ranges": {
                        "quantity": {"min": 1, "max": 100000},
                        "price": {"min": 0.01, "max": 10000}
                    },
                    "enum_values": {
                        "trade_type": ["buy", "sell", "short", "cover"]
                    }
                }
            }
        ]

        for policy in default_policies:
            if not self.policy_manager.get_policy(policy["id"]):
                success = self.register_policy(policy)
                if success:
                    self.logger.info(f"已注册默认策略: {policy['name']}")
                else:
                    self.logger.error(f"注册默认策略失败: {policy['name']}")

    def audit_compliance_status(self):
        policies = self.policy_manager.list_policies()
        audit_result = {
            "total_policies": len(policies),
            "active_policies": len([p for p in policies.values() if p.get("active", True)]),
            "policies_by_category": {},
            "last_updated": datetime.now().isoformat(),
            "recommendations": []
        }

        for policy in policies.values():
            category = policy.get("category", "uncategorized")
            if category not in audit_result["policies_by_category"]:
                audit_result["policies_by_category"][category] = 0
            audit_result["policies_by_category"][category] += 1

        if audit_result["total_policies"] == 0:
            audit_result["recommendations"].append("建议建立基础的合规策略")
        elif audit_result["total_policies"] < 3:
            audit_result["recommendations"].append("建议完善合规策略覆盖范围")

        return audit_result

    def _generate_recommendations(self, check_result):
        recommendations = []
        if not check_result.get("compliance", False):
            issues = check_result.get("issues", [])
            if any("缺失字段" in issue for issue in issues):
                recommendations.append("完善数据字段，确保所有必需字段都已提供")
            if any("类型错误" in issue for issue in issues):
                recommendations.append("修正数据类型，确保字段类型符合策略要求")
            if any("敏感信息" in issue for issue in issues):
                recommendations.append("加强敏感数据保护，使用数据脱敏或加密")
            if any("超出限制" in issue for issue in issues):
                recommendations.append("调整数据值，确保在允许的范围内")
        return recommendations

    def _assess_bulk_severity(self, bulk_result):
        compliance_rate = bulk_result["compliance_rate"]
        if compliance_rate >= 0.95:
            return "excellent"
        elif compliance_rate >= 0.90:
            return "good"
        elif compliance_rate >= 0.80:
            return "acceptable"
        elif compliance_rate >= 0.70:
            return "concerning"
        else:
            return "critical"

    def _generate_bulk_recommendations(self, bulk_result):
        recommendations = []
        compliance_rate = bulk_result["compliance_rate"]
        if compliance_rate < 0.80:
            recommendations.append("🔴 合规率严重不足，建议立即审查数据质量和合规策略")
        elif compliance_rate < 0.90:
            recommendations.append("🟡 合规率需要提升，建议优化数据处理流程")
        if bulk_result["non_compliant_records"] > 0:
            recommendations.append(f"修正 {bulk_result['non_compliant_records']} 条不合规记录")
        return recommendations


class MockAuditLoggingManager:
    def __init__(self):
        self.audit_events = []
        self.logger = MockLogger()
        self._lock = MagicMock()

    def log_event(self, event):
        self.audit_events.append(event)
        return event.event_id

    def get_events(self, filters=None):
        events = self.audit_events.copy()
        if filters:
            # 简化的过滤逻辑
            if "event_type" in filters:
                events = [e for e in events if e.event_type == filters["event_type"]]
            if "user_id" in filters:
                events = [e for e in events if e.user_id == filters["user_id"]]
            if "severity" in filters:
                events = [e for e in events if e.severity == filters["severity"]]
        return events

    def get_high_risk_events(self):
        return [e for e in self.audit_events if e.severity in ["high", "critical"]]

    def generate_compliance_report(self, start_date=None, end_date=None):
        events = self.audit_events
        if start_date:
            events = [e for e in events if e.timestamp >= start_date]
        if end_date:
            events = [e for e in events if e.timestamp <= end_date]

        total_events = len(events)
        security_events = len([e for e in events if e.event_type == "security"])
        access_denied = len([e for e in events if e.result == "denied"])

        return {
            "total_events": total_events,
            "security_events": security_events,
            "access_denied_events": access_denied,
            "period_start": start_date.isoformat() if start_date else None,
            "period_end": end_date.isoformat() if end_date else None,
            "generated_at": datetime.now().isoformat()
        }


# 导入真实的类用于测试
from src.data.compliance.privacy_protector import PrivacyProtector
from src.data.compliance.data_policy_manager import DataPolicyManager
from src.data.compliance.compliance_checker import ComplianceChecker
from src.data.compliance.data_compliance_manager import DataComplianceManager


class TestPrivacyProtector:
    """测试隐私保护器"""

    def test_protect_with_standard_level(self):
        """测试标准级别隐私保护"""
        protector = PrivacyProtector()

        # 测试手机号脱敏
        assert protector.protect("13800138000") == "13*******00"

        # 测试邮箱脱敏
        assert protector.protect("user@example.com") == "u***@***.com"

        # 测试身份证号脱敏
        assert protector.protect("123456789012345678") == "123456****5678"

        # 测试信用卡号脱敏
        assert protector.protect("1234567890123456") == "1234****3456"

        # 测试银行账号脱敏
        assert protector.protect("1234567890123456789") == "1234****6789"

        # 测试地址脱敏
        assert protector.protect("北京市朝阳区建国门外大街1号") == "北京****1号"

        # 测试姓名脱敏
        assert protector.protect("张三") == "张*"

        # 测试短字符串
        assert protector.protect("abc") == "***"

        # 测试长字符串
        assert protector.protect("verylongstring") == "ve****ng"

    def test_protect_with_encrypted_level(self):
        """测试加密级别隐私保护"""
        protector = PrivacyProtector()

        original = "sensitive_data"
        encrypted = protector.protect(original, level="encrypted")

        # 验证是SHA256哈希
        assert len(encrypted) == 64
        assert encrypted.isalnum()
        assert encrypted == hashlib.sha256(original.encode()).hexdigest()

    def test_protect_with_none_level(self):
        """测试无保护级别"""
        protector = PrivacyProtector()

        data = "sensitive_data"
        protected = protector.protect(data, level="none")

        assert protected == data

    def test_protect_non_string_data(self):
        """测试非字符串数据保护"""
        protector = PrivacyProtector()

        # 测试数字
        assert protector.protect(12345) == 12345

        # 测试字典
        assert protector.protect({"key": "value"}) == {"key": "value"}

        # 测试列表
        assert protector.protect([1, 2, 3]) == [1, 2, 3]

    def test_protect_invalid_level(self):
        """测试无效保护级别"""
        protector = PrivacyProtector()

        # 无效级别应该使用默认的standard级别
        result = protector.protect("13800138000", level="invalid")
        assert result == "13*******00"  # standard级别的手机号脱敏结果


class TestMockPrivacyProtector:
    """测试Mock隐私保护器"""

    def test_mock_protect_standard(self):
        """测试Mock标准保护"""
        protector = MockPrivacyProtector()

        assert protector.protect("13800138000") == "13*******00"
        assert protector.protect("user@test.com") == "u***@***.com"
        assert protector.protect("张三") == "张*"

    def test_mock_protect_encrypted(self):
        """测试Mock加密保护"""
        protector = MockPrivacyProtector()

        result = protector.protect("test", level="encrypted")
        assert len(result) == 64
        assert result == hashlib.sha256("test".encode()).hexdigest()

    def test_mock_protect_none(self):
        """测试Mock无保护"""
        protector = MockPrivacyProtector()

        assert protector.protect("test", level="none") == "test"


class TestDataPolicyManager:
    """测试数据策略管理器"""

    def test_register_policy_success(self):
        """测试成功注册策略"""
        manager = DataPolicyManager()

        policy = {
            "name": "Test Policy",
            "required_fields": ["field1", "field2"],
            "field_types": {"field1": "string"},
            "enforcement_level": "strict"
        }

        success = manager.register_policy(policy)
        assert success is True
        assert policy["id"] in manager.policies

    def test_register_policy_invalid(self):
        """测试注册无效策略"""
        manager = DataPolicyManager()

        # 缺少必需字段
        invalid_policy = {"name": "Invalid Policy"}
        success = manager.register_policy(invalid_policy)
        assert success is False

    def test_get_policy(self):
        """测试获取策略"""
        manager = DataPolicyManager()

        policy = {
            "name": "Test Policy",
            "required_fields": ["field1"],
            "id": "test_policy"
        }

        manager.register_policy(policy)
        retrieved = manager.get_policy("test_policy")
        assert retrieved is not None
        assert retrieved["name"] == "Test Policy"

    def test_update_policy(self):
        """测试更新策略"""
        manager = DataPolicyManager()

        policy = {
            "name": "Test Policy",
            "required_fields": ["field1"],
            "id": "test_policy"
        }

        manager.register_policy(policy)
        success = manager.update_policy("test_policy", {"name": "Updated Policy"})
        assert success is True

        updated = manager.get_policy("test_policy")
        assert updated["name"] == "Updated Policy"

    def test_delete_policy(self):
        """测试删除策略"""
        manager = DataPolicyManager()

        policy = {
            "name": "Test Policy",
            "required_fields": ["field1"],
            "id": "test_policy"
        }

        manager.register_policy(policy)
        success = manager.delete_policy("test_policy")
        assert success is True
        assert manager.get_policy("test_policy") is None


class TestMockDataPolicyManager:
    """测试Mock数据策略管理器"""

    def test_mock_register_policy(self):
        """测试Mock注册策略"""
        manager = MockDataPolicyManager()

        policy = {
            "name": "Mock Policy",
            "required_fields": ["field1"]
        }

        success = manager.register_policy(policy)
        assert success is True
        assert len(manager.policies) == 1

    def test_mock_get_policy(self):
        """测试Mock获取策略"""
        manager = MockDataPolicyManager()

        policy = {
            "name": "Mock Policy",
            "required_fields": ["field1"],
            "id": "mock_policy"
        }

        manager.register_policy(policy)
        retrieved = manager.get_policy("mock_policy")
        assert retrieved["name"] == "Mock Policy"


class TestComplianceChecker:
    """测试合规校验器"""

    def test_check_compliant_data(self):
        """测试合规数据校验"""
        policy_manager = DataPolicyManager()
        checker = ComplianceChecker(policy_manager)

        # 注册策略
        policy = {
            "name": "User Policy",
            "required_fields": ["user_id", "email"],
            "field_types": {"user_id": "integer", "email": "string"},
            "id": "user_policy"
        }
        policy_manager.register_policy(policy)

        # 测试合规数据
        compliant_data = {
            "user_id": 123,
            "email": "test@example.com",
            "name": "Test User"
        }

        result = checker.check(compliant_data, "user_policy")
        assert result["compliance"] is True
        assert len(result["issues"]) == 0

    def test_check_non_compliant_data(self):
        """测试不合规数据校验"""
        policy_manager = DataPolicyManager()
        checker = ComplianceChecker(policy_manager)

        # 注册策略
        policy = {
            "name": "User Policy",
            "required_fields": ["user_id", "email"],
            "field_types": {"user_id": "integer", "email": "string"},
            "max_field_lengths": {"email": 10},
            "id": "user_policy"
        }
        policy_manager.register_policy(policy)

        # 测试不合规数据
        non_compliant_data = {
            "user_id": "not_an_integer",  # 类型错误
            "name": "Test User"  # 缺失email字段
        }

        result = checker.check(non_compliant_data, "user_policy")
        assert result["compliance"] is False
        assert len(result["issues"]) > 0

    def test_check_trading_compliance(self):
        """测试交易合规校验"""
        policy_manager = DataPolicyManager()
        checker = ComplianceChecker(policy_manager)

        # 合规交易数据
        compliant_trade = {
            "user_id": 123,
            "symbol": "AAPL",
            "quantity": 100,
            "price": 150.0,
            "trade_type": "buy",
            "amount": 15000.0,
            "timestamp": datetime.now().isoformat()
        }

        result = checker.check_trading_compliance(compliant_trade)
        assert result["compliance"] is True

        # 不合规交易数据
        non_compliant_trade = {
            "amount": -100,  # 负数金额
            "trade_type": "invalid_type",  # 无效交易类型
            "timestamp": datetime.now().isoformat()
        }

        result = checker.check_trading_compliance(non_compliant_trade)
        assert result["compliance"] is False

    def test_check_bulk_data(self):
        """测试批量数据校验"""
        policy_manager = DataPolicyManager()
        checker = ComplianceChecker(policy_manager)

        # 注册策略
        policy = {
            "name": "Bulk Policy",
            "required_fields": ["id"],
            "id": "bulk_policy"
        }
        policy_manager.register_policy(policy)

        # 批量数据
        data_list = [
            {"id": 1, "name": "Item 1"},
            {"id": 2, "name": "Item 2"},
            {"name": "Item 3"},  # 缺失id字段
        ]

        result = checker.check_bulk_data(data_list, "bulk_policy")
        assert result["total_records"] == 3
        assert result["compliant_records"] == 2
        assert result["non_compliant_records"] == 1
        assert result["compliance_rate"] == 2/3


class TestMockComplianceChecker:
    """测试Mock合规校验器"""

    def test_mock_check_compliance(self):
        """测试Mock合规校验"""
        policy_manager = MockDataPolicyManager()
        checker = MockComplianceChecker(policy_manager)

        policy = {
            "name": "Mock Policy",
            "required_fields": ["field1"],
            "id": "mock_policy"
        }
        policy_manager.register_policy(policy)

        data = {"field1": "value"}
        result = checker.check(data, "mock_policy")
        assert "compliance" in result
        assert "issues" in result

    def test_mock_check_trading_compliance(self):
        """测试Mock交易合规校验"""
        policy_manager = MockDataPolicyManager()
        checker = MockComplianceChecker(policy_manager)

        # 使用工作日且处于交易时段的固定时间戳，避免由于实时执行时间导致的脆弱失败
        safe_timestamp = datetime(2025, 5, 6, 10, 0, 0).isoformat()
        trade_data = {
            "amount": 1000,
            "trade_type": "buy",
            "timestamp": safe_timestamp
        }

        result = checker.check_trading_compliance(trade_data)
        assert result["compliance"] is True


class TestDataComplianceManager:
    """测试数据合规管理器"""

    def test_register_and_check_policy(self):
        """测试注册策略并校验"""
        manager = DataComplianceManager()

        policy = {
            "name": "Test Policy",
            "required_fields": ["user_id"],
            "field_types": {"user_id": "integer"},
            "id": "test_policy"
        }

        success = manager.register_policy(policy)
        assert success is True

        data = {"user_id": 123}
        result = manager.check_compliance(data, "test_policy")
        assert result["compliance"] is True

    def test_privacy_protection(self):
        """测试隐私保护功能"""
        manager = DataComplianceManager()

        # 测试数据脱敏
        protected = manager.protect_privacy("13800138000")
        assert protected == "13*******00"

        # 测试数据加密
        encrypted = manager.protect_privacy("secret", level="encrypted")
        assert len(encrypted) == 64

    def test_generate_compliance_report(self):
        """测试生成合规报告"""
        manager = DataComplianceManager()

        policy = {
            "name": "Report Policy",
            "required_fields": ["field1"],
            "id": "report_policy"
        }
        manager.register_policy(policy)

        data = {"field1": "value"}
        report = manager.generate_compliance_report(data, "report_policy")

        assert "policy_id" in report
        assert "compliance" in report
        assert "recommendations" in report

    def test_generate_bulk_compliance_report(self):
        """测试生成批量合规报告"""
        manager = DataComplianceManager()

        policy = {
            "name": "Bulk Report Policy",
            "required_fields": ["id"],
            "id": "bulk_report_policy"
        }
        manager.register_policy(policy)

        data_list = [
            {"id": 1},
            {"id": 2},
            {"name": "no_id"}  # 不合规
        ]

        report = manager.generate_bulk_compliance_report(data_list, "bulk_report_policy")

        assert report["total_records"] == 3
        assert report["compliant_records"] == 2
        assert "severity_assessment" in report

    def test_setup_default_policies(self):
        """测试设置默认策略"""
        manager = DataComplianceManager()

        # 初始状态下没有策略
        assert len(manager.policy_manager.list_policies()) == 0

        # 设置默认策略
        manager.setup_default_policies()

        policies = manager.policy_manager.list_policies()
        assert len(policies) >= 2  # 应该有用户数据策略和交易数据策略

    def test_audit_compliance_status(self):
        """测试审计合规状态"""
        manager = DataComplianceManager()

        # 设置默认策略
        manager.setup_default_policies()

        audit_result = manager.audit_compliance_status()

        assert "total_policies" in audit_result
        assert "active_policies" in audit_result
        assert "policies_by_category" in audit_result
        assert audit_result["total_policies"] >= 2


class TestMockDataComplianceManager:
    """测试Mock数据合规管理器"""

    def test_mock_privacy_protection(self):
        """测试Mock隐私保护"""
        manager = MockDataComplianceManager()

        protected = manager.protect_privacy("13800138000")
        assert protected == "13*******00"

    def test_mock_generate_report(self):
        """测试Mock生成报告"""
        manager = MockDataComplianceManager()

        policy = {
            "name": "Mock Report Policy",
            "required_fields": ["field1"],
            "id": "mock_report_policy"
        }
        manager.register_policy(policy)

        data = {"field1": "value"}
        report = manager.generate_compliance_report(data, "mock_report_policy")
        assert report["compliance"] is True


class TestPrivacyProtectionIntegration:
    """隐私保护集成测试"""

    def test_complete_privacy_workflow(self):
        """测试完整隐私保护工作流程"""
        protector = PrivacyProtector()

        # 测试各种数据类型的保护
        test_data = [
            ("13800138000", "13*******00"),  # 手机号
            ("user@test.com", "u***@***.com"),  # 邮箱
            ("123456789012345678", "123456****5678"),  # 身份证
            ("北京市朝阳区建国门外大街1号", "北京****1号"),  # 地址
            ("张三", "张*"),  # 姓名
            ("abc", "***"),  # 短字符串
            ("verylongstringdata", "ve****ta"),  # 长字符串
        ]

        for original, expected in test_data:
            protected = protector.protect(original)
            assert protected == expected, f"Failed for {original}"

    def test_compliance_and_privacy_integration(self):
        """测试合规校验和隐私保护集成"""
        manager = DataComplianceManager()

        # 设置默认策略
        manager.setup_default_policies()

        # 测试用户数据
        user_data = {
            "user_id": 123,
            "username": "testuser",
            "email": "user@example.com",
            "balance": 5000.0,
            "status": "active"
        }

        # 校验合规性
        compliance_result = manager.check_compliance(user_data, "user_data_policy")
        assert compliance_result["compliance"] is True

        # 保护隐私数据
        protected_email = manager.protect_privacy(user_data["email"])
        assert protected_email == "u***@***.com"

        # 验证保护后的数据仍然合规
        protected_user_data = user_data.copy()
        protected_user_data["email"] = protected_email
        protected_compliance = manager.check_compliance(protected_user_data, "user_data_policy")
        assert protected_compliance["compliance"] is True

    def test_bulk_privacy_protection(self):
        """测试批量隐私保护"""
        manager = DataComplianceManager()

        # 批量用户数据
        users_data = [
            {"email": "user1@test.com", "phone": "13800138001"},
            {"email": "user2@test.com", "phone": "13800138002"},
            {"email": "user3@test.com", "phone": "13800138003"},
        ]

        # 批量保护隐私
        protected_users = []
        for user in users_data:
            protected_user = user.copy()
            protected_user["email"] = manager.protect_privacy(user["email"])
            protected_user["phone"] = manager.protect_privacy(user["phone"])
            protected_users.append(protected_user)

        # 验证所有数据都被正确保护
        assert protected_users[0]["email"] == "u***@***.com"
        assert protected_users[0]["phone"] == "13*******01"
        assert protected_users[1]["email"] == "u***@***.com"
        assert protected_users[1]["phone"] == "13*******02"

    def test_audit_and_compliance_integration(self):
        """测试审计和合规集成"""
        manager = DataComplianceManager()

        # 设置策略并测试
        manager.setup_default_policies()

        # 执行合规审计
        audit_result = manager.audit_compliance_status()
        assert audit_result["total_policies"] >= 2

        # 测试交易合规
        trade_data = {
            "user_id": 123,
            "symbol": "AAPL",
            "quantity": 100,
            "price": 150.0,
            "trade_type": "buy",
            "amount": 15000.0,
            "timestamp": datetime.now().isoformat()
        }

        trade_compliance = manager.check_trading_compliance(trade_data)
        assert trade_compliance["compliance"] is True

        # 生成合规报告
        report = manager.generate_compliance_report(trade_data, "trade_data_policy")
        assert "recommendations" in report


class TestMockAuditLoggingManager:
    """测试Mock审计日志管理器"""

    def test_mock_log_event(self):
        """测试Mock记录审计事件"""
        manager = MockAuditLoggingManager()

        event = MockAuditEvent(
            event_id="test_event",
            event_type="access",
            severity="medium",
            timestamp=datetime.now(),
            user_id="user123",
            action="login",
            result="success"
        )

        event_id = manager.log_event(event)
        assert event_id == "test_event"
        assert len(manager.audit_events) == 1

    def test_mock_get_events_with_filters(self):
        """测试Mock获取过滤事件"""
        manager = MockAuditLoggingManager()

        # 添加多个事件
        events = [
            MockAuditEvent("e1", "access", "low", datetime.now(), "user1", action="login", result="success"),
            MockAuditEvent("e2", "security", "high", datetime.now(), "user2", action="failed_login", result="failure"),
            MockAuditEvent("e3", "access", "medium", datetime.now(), "user1", action="logout", result="success"),
        ]

        for event in events:
            manager.log_event(event)

        # 测试过滤
        access_events = manager.get_events({"event_type": "access"})
        assert len(access_events) == 2

        user1_events = manager.get_events({"user_id": "user1"})
        assert len(user1_events) == 2

    def test_mock_get_high_risk_events(self):
        """测试Mock获取高风险事件"""
        manager = MockAuditLoggingManager()

        # 添加高风险和低风险事件
        events = [
            MockAuditEvent("e1", "access", "low", datetime.now(), action="normal_access"),
            MockAuditEvent("e2", "security", "high", datetime.now(), action="suspicious_activity"),
            MockAuditEvent("e3", "security", "critical", datetime.now(), action="breach_attempt"),
        ]

        for event in events:
            manager.log_event(event)

        high_risk = manager.get_high_risk_events()
        assert len(high_risk) == 2

    def test_mock_generate_compliance_report(self):
        """测试Mock生成合规报告"""
        manager = MockAuditLoggingManager()

        # 添加测试事件
        events = [
            MockAuditEvent("e1", "access", "low", datetime.now(), action="login", result="success"),
            MockAuditEvent("e2", "security", "high", datetime.now(), action="login", result="denied"),
            MockAuditEvent("e3", "access", "low", datetime.now(), action="data_access", result="success"),
        ]

        for event in events:
            manager.log_event(event)

        report = manager.generate_compliance_report()
        assert report["total_events"] == 3
        assert report["security_events"] == 1
        assert report["access_denied_events"] == 1
