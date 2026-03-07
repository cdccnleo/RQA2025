#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 基础设施层日志系统 - AuditLogger使用示例

演示AuditLogger的审计日志记录功能，适用于安全审计、操作追踪、合规记录等场景。
"""

from infrastructure.logging import AuditLogger
import sys
import os
import time

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def authentication_audit_example():
    """身份验证审计示例"""
    print("=== 身份验证审计示例 ===\n")

    auth_logger = AuditLogger("security.auth", log_dir="logs/audit/auth")

    # 验证自动配置
    print(f"自动配置验证 - 分类: {auth_logger.category}, 格式: {auth_logger.format_type}")
    print()

    # 模拟登录审计
    login_events = [
        {"user_id": "user123", "ip": "192.168.1.100", "method": "password", "success": True},
        {"user_id": "user456", "ip": "10.0.0.50", "method": "password", "success": False},
        {"user_id": "admin", "ip": "192.168.1.200", "method": "certificate", "success": True},
        {"user_id": "user789", "ip": "203.0.113.1", "method": "password", "success": False},
    ]

    for event in login_events:
        timestamp = time.time()

        if event["success"]:
            auth_logger.info("用户登录成功",
                             user_id=event["user_id"],
                             ip_address=event["ip"],
                             login_method=event["method"],
                             timestamp=timestamp,
                             session_id=f"SESS-{int(timestamp)}",
                             user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        else:
            auth_logger.warning("用户登录失败",
                                user_id=event["user_id"],
                                ip_address=event["ip"],
                                login_method=event["method"],
                                timestamp=timestamp,
                                failure_reason="invalid_credentials",
                                attempt_count=3,
                                account_locked=False)

        time.sleep(0.1)

    print()


def authorization_audit_example():
    """授权审计示例"""
    print("=== 授权审计示例 ===\n")

    authz_logger = AuditLogger("security.authz", log_dir="logs/audit/authz")

    # 模拟权限检查
    authz_events = [
        {"user_id": "user123", "resource": "order", "action": "create", "allowed": True},
        {"user_id": "user456", "resource": "admin_panel", "action": "access", "allowed": False},
        {"user_id": "manager", "resource": "reports", "action": "view", "allowed": True},
        {"user_id": "user789", "resource": "payment", "action": "refund", "allowed": False},
    ]

    for event in authz_events:
        if event["allowed"]:
            authz_logger.info("权限检查通过",
                              user_id=event["user_id"],
                              resource=event["resource"],
                              action=event["action"],
                              timestamp=time.time(),
                              role="user",
                              policy_applied="default_policy",
                              decision_reason="role_based_access")
        else:
            authz_logger.warning("权限检查拒绝",
                                 user_id=event["user_id"],
                                 resource=event["resource"],
                                 action=event["action"],
                                 timestamp=time.time(),
                                 role="user",
                                 required_role="admin",
                                 policy_applied="strict_policy",
                                 decision_reason="insufficient_privileges")

        time.sleep(0.05)

    print()


def data_access_audit_example():
    """数据访问审计示例"""
    print("=== 数据访问审计示例 ===\n")

    data_logger = AuditLogger("security.data", log_dir="logs/audit/data")

    # 模拟数据访问
    data_events = [
        {"user_id": "analyst", "table": "user_orders", "operation": "SELECT", "record_count": 150},
        {"user_id": "admin", "table": "user_payments", "operation": "UPDATE", "record_count": 1},
        {"user_id": "user123", "table": "user_profile", "operation": "SELECT", "record_count": 1},
        {"user_id": "auditor", "table": "audit_logs", "operation": "SELECT", "record_count": 500},
    ]

    for event in data_events:
        data_logger.info("数据访问记录",
                         user_id=event["user_id"],
                         table_name=event["table"],
                         operation=event["operation"],
                         record_count=event["record_count"],
                         timestamp=time.time(),
                         query_id=f"QRY-{int(time.time()*1000)}",
                         execution_time="0.125s",
                         success=True)

        time.sleep(0.1)

    print()


def system_configuration_audit_example():
    """系统配置审计示例"""
    print("=== 系统配置审计示例 ===\n")

    config_logger = AuditLogger("system.config", log_dir="logs/audit/config")

    # 模拟配置变更
    config_changes = [
        {"component": "cache", "setting": "max_memory", "old_value": "512MB", "new_value": "1GB"},
        {"component": "database", "setting": "connection_pool_size", "old_value": "10", "new_value": "20"},
        {"component": "security", "setting": "password_policy",
            "old_value": "basic", "new_value": "strict"},
    ]

    for change in config_changes:
        config_logger.info("配置变更记录",
                           component=change["component"],
                           setting_name=change["setting"],
                           old_value=change["old_value"],
                           new_value=change["new_value"],
                           timestamp=time.time(),
                           changed_by="admin_user",
                           change_reason="performance_optimization",
                           requires_restart=False,
                           backup_created=True)

        time.sleep(0.2)

    print()


def compliance_reporting_example():
    """合规报告审计示例"""
    print("=== 合规报告审计示例 ===\n")

    compliance_logger = AuditLogger("compliance.report", log_dir="logs/audit/compliance")

    # 模拟合规检查
    compliance_events = [
        {"check_type": "data_retention", "status": "PASS",
            "details": "All data within retention period"},
        {"check_type": "access_control", "status": "WARNING",
            "details": "3 users have excessive permissions"},
        {"check_type": "encryption", "status": "PASS",
            "details": "All sensitive data properly encrypted"},
        {"check_type": "audit_trail", "status": "PASS", "details": "Complete audit trail maintained"},
    ]

    for event in compliance_events:
        if event["status"] == "PASS":
            compliance_logger.info("合规检查通过",
                                   check_type=event["check_type"],
                                   status=event["status"],
                                   details=event["details"],
                                   timestamp=time.time(),
                                   compliance_standard="GDPR",
                                   next_check_due="2025-10-23")
        elif event["status"] == "WARNING":
            compliance_logger.warning("合规检查警告",
                                      check_type=event["check_type"],
                                      status=event["status"],
                                      details=event["details"],
                                      timestamp=time.time(),
                                      severity="medium",
                                      remediation_required=True,
                                      assigned_to="security_team")

        time.sleep(0.15)

    print()


def security_incident_audit_example():
    """安全事件审计示例"""
    print("=== 安全事件审计示例 ===\n")

    security_logger = AuditLogger("security.incident", log_dir="logs/audit/incidents")

    # 模拟安全事件
    incidents = [
        {"type": "brute_force_attempt", "severity": "HIGH", "ip": "203.0.113.195", "blocked": True},
        {"type": "suspicious_activity", "severity": "MEDIUM",
            "user_id": "user999", "details": "Unusual login pattern"},
        {"type": "data_exfiltration", "severity": "CRITICAL",
            "user_id": "former_employee", "data_volume": "2.3GB"},
    ]

    for incident in incidents:
        if incident["severity"] == "CRITICAL":
            security_logger.critical("安全事件告警",
                                     incident_type=incident["type"],
                                     severity=incident["severity"],
                                     timestamp=time.time(),
                                     details=incident,
                                     immediate_action_required=True,
                                     notification_sent=True,
                                     incident_id=f"INC-{int(time.time())}")
        elif incident["severity"] == "HIGH":
            security_logger.error("高风险安全事件",
                                  incident_type=incident["type"],
                                  severity=incident["severity"],
                                  timestamp=time.time(),
                                  details=incident,
                                  automated_response="blocked",
                                  manual_review_required=True)
        else:
            security_logger.warning("安全事件警告",
                                    incident_type=incident["type"],
                                    severity=incident["severity"],
                                    timestamp=time.time(),
                                    details=incident,
                                    monitoring_increased=True)

        time.sleep(0.3)

    print()


def audit_log_analysis_example():
    """审计日志分析示例"""
    print("=== 审计日志分析示例 ===\n")

    analysis_logger = AuditLogger("audit.analysis", log_dir="logs/audit/analysis")

    # 模拟审计分析结果
    analysis_results = {
        "period": "2025-09-23",
        "total_events": 15420,
        "security_events": 23,
        "failed_logins": 156,
        "successful_logins": 8934,
        "data_access_events": 5234,
        "configuration_changes": 12,
        "compliance_score": 98.7
    }

    analysis_logger.info("审计日志分析报告",
                         analysis_period=analysis_results["period"],
                         total_events=analysis_results["total_events"],
                         security_events=analysis_results["security_events"],
                         failed_logins=analysis_results["failed_logins"],
                         successful_logins=analysis_results["successful_logins"],
                         data_access_events=analysis_results["data_access_events"],
                         configuration_changes=analysis_results["configuration_changes"],
                         compliance_score=analysis_results["compliance_score"],
                         timestamp=time.time(),
                         report_generated_by="automated_system",
                         review_required=analysis_results["security_events"] > 20)

    print()


def main():
    """主函数"""
    print("RQA2025 基础设施层日志系统 - AuditLogger使用示例")
    print("=" * 60)
    print()

    try:
        authentication_audit_example()
        authorization_audit_example()
        data_access_audit_example()
        system_configuration_audit_example()
        compliance_reporting_example()
        security_incident_audit_example()
        audit_log_analysis_example()

        print("🎉 所有审计日志示例执行完成！")
        print("\n审计日志文件位置:")
        print("- logs/audit/auth/       - 身份验证审计")
        print("- logs/audit/authz/      - 授权审计")
        print("- logs/audit/data/       - 数据访问审计")
        print("- logs/audit/config/     - 配置变更审计")
        print("- logs/audit/compliance/ - 合规审计")
        print("- logs/audit/incidents/  - 安全事件审计")
        print("- logs/audit/analysis/   - 审计分析报告")

        print("\n注意：审计日志采用JSON格式，便于分析和合规审查")

    except Exception as e:
        print(f"❌ 示例执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
