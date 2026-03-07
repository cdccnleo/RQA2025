from typing import List, Dict, Any
from datetime import datetime
from .data_policy_manager import DataPolicyManager
from .compliance_checker import ComplianceChecker
from .privacy_protector import PrivacyProtector
import logging


class DataComplianceManager:
    """数据合规管理主控 - 增强版"""

    def __init__(self):
        self.policy_manager = DataPolicyManager()
        self.compliance_checker = ComplianceChecker(self.policy_manager)
        self.privacy_protector = PrivacyProtector()
        self.logger = logging.getLogger(__name__)

    def register_policy(self, policy: Dict[str, Any]) -> bool:
        """注册合规策略"""
        return self.policy_manager.register_policy(policy)

    def check_compliance(self, data: Any, policy_id: str = None) -> Dict[str, Any]:
        """对数据进行合规性校验"""
        return self.compliance_checker.check(data, policy_id)

    def check_bulk_compliance(self, data_list: List[Dict[str, Any]], policy_id: str = None) -> Dict[str, Any]:
        """批量数据合规检查"""
        return self.compliance_checker.check_bulk_data(data_list, policy_id)

    def check_trading_compliance(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """专门的交易合规检查"""
        return self.compliance_checker.check_trading_compliance(trade_data)

    def protect_privacy(self, data: Any, level: str = "standard") -> Any:
        """对数据进行隐私保护（脱敏 / 加密）"""
        return self.privacy_protector.protect(data, level)

    def generate_compliance_report(self, data: Any, policy_id: str = None) -> Dict[str, Any]:
        """生成合规性报告"""
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

    def generate_bulk_compliance_report(self, data_list: List[Dict[str, Any]], policy_id: str = None) -> Dict[str, Any]:
        """生成批量合规报告"""
        bulk_result = self.check_bulk_compliance(data_list, policy_id)

        report = {
            "policy_id": policy_id,
            "total_records": bulk_result["total_records"],
            "compliant_records": bulk_result["compliant_records"],
            "non_compliant_records": bulk_result["non_compliant_records"],
            "compliance_rate": bulk_result["compliance_rate"],
            "sample_issues": bulk_result["all_issues"][:10],  # 只显示前10个问题
            "checked_at": bulk_result["checked_at"],
            "severity_assessment": self._assess_bulk_severity(bulk_result),
            "recommendations": self._generate_bulk_recommendations(bulk_result)
        }

        return report

    def setup_default_policies(self) -> None:
        """设置默认的合规策略"""
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

    def audit_compliance_status(self) -> Dict[str, Any]:
        """审计合规状态"""
        policies = self.policy_manager.list_policies()

        audit_result = {
            "total_policies": len(policies),
            "active_policies": len([p for p in policies.values() if p.get("active", True)]),
            "policies_by_category": {},
            "last_updated": datetime.now().isoformat(),
            "recommendations": []
        }

        # 按类别统计策略
        for policy in policies.values():
            category = policy.get("category", "uncategorized")
            if category not in audit_result["policies_by_category"]:
                audit_result["policies_by_category"][category] = 0
            audit_result["policies_by_category"][category] += 1

        # 生成审计建议
        if audit_result["total_policies"] == 0:
            audit_result["recommendations"].append("建议建立基础的合规策略")
        elif audit_result["total_policies"] < 3:
            audit_result["recommendations"].append("建议完善合规策略覆盖范围")

        return audit_result

    def _generate_recommendations(self, check_result: Dict[str, Any]) -> List[str]:
        """生成单个检查的建议"""
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

    def _assess_bulk_severity(self, bulk_result: Dict[str, Any]) -> str:
        """评估批量检查的严重程度"""
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

    def _generate_bulk_recommendations(self, bulk_result: Dict[str, Any]) -> List[str]:
        """生成批量检查的建议"""
        recommendations = []
        compliance_rate = bulk_result["compliance_rate"]

        if compliance_rate < 0.80:
            recommendations.append("🔴 合规率严重不足，建议立即审查数据质量和合规策略")
        elif compliance_rate < 0.90:
            recommendations.append("🟡 合规率需要提升，建议优化数据处理流程")

        if bulk_result["non_compliant_records"] > 0:
            recommendations.append(f"修正 {bulk_result['non_compliant_records']} 条不合规记录")

        return recommendations
