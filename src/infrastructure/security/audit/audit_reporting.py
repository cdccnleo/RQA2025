#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 审计报告生成器

专门负责审计报告的生成、格式化和导出
从AuditLoggingManager中分离出来，提高代码组织性
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, Counter
import csv
import pandas as pd


@dataclass
class ComplianceReport:
    """合规报告"""
    report_id: str
    report_type: str
    period_start: datetime
    period_end: datetime
    generated_at: datetime
    findings: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    compliance_score: float = 100.0
    risk_assessment: str = "low"
    metrics: Dict[str, Any] = field(default_factory=dict)
    total_events: int = 0


class AuditReportGenerator:
    """审计报告生成器"""

    def __init__(self):
        self._report_templates = {
            'summary': self._generate_summary_report,
            'security': self._generate_security_report,
            'compliance': self._generate_compliance_report,
            'user_activity': self._generate_user_activity_report,
            'resource_access': self._generate_resource_access_report,
            'risk_analysis': self._generate_risk_analysis_report
        }

    def generate_report(self, events: List['AuditEvent'],
                       report_type: str = "summary",
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None,
                       **kwargs) -> Dict[str, Any]:
        """生成审计报告"""
        # 过滤时间范围
        total_events = len(events)
        filtered_events = self._filter_events_by_time(events, start_time, end_time)

        if report_type not in self._report_templates:
            raise ValueError(f"Unsupported report type: {report_type}")

        report_func = self._report_templates[report_type]
        report = report_func(filtered_events, **kwargs)
        report.setdefault('filtered_events', len(filtered_events))
        report['total_events'] = total_events
        return report

    def generate_compliance_report(self, events: List['AuditEvent'],
                                 report_type: str = "compliance",
                                 days: int = 30) -> ComplianceReport:
        """生成合规报告"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        filtered_events = self._filter_events_by_time(events, start_time, end_time)

        findings = self._perform_compliance_checks(filtered_events, report_type)
        recommendations = self._generate_compliance_recommendations(findings)
        compliance_score = self._calculate_compliance_score(findings, filtered_events)
        risk_assessment = self._assess_compliance_risk(compliance_score)

        return ComplianceReport(
            report_id=f"compliance_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            report_type=report_type,
            period_start=start_time,
            period_end=end_time,
            generated_at=datetime.now(),
            findings=findings,
            recommendations=recommendations,
            compliance_score=compliance_score,
            risk_assessment=risk_assessment,
            metrics={
                "total_events": len(filtered_events),
                "findings_count": len(findings),
                "recommendation_count": len(recommendations),
            },
            total_events=len(filtered_events),
        )

    def export_report(self, report_data: Dict[str, Any],
                     format_type: str = "json",
                     output_path: Optional[Path] = None) -> str:
        """导出报告"""
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = Path(f"audit_report_{timestamp}.{format_type}")

        if format_type == "json":
            return self._export_json(report_data, output_path)
        elif format_type == "csv":
            return self._export_csv(report_data, output_path)
        elif format_type == "html":
            return self._export_html(report_data, output_path)
        elif format_type == "pdf":
            return self._export_pdf(report_data, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")

    def _generate_summary_report(self, events: List['AuditEvent'], **kwargs) -> Dict[str, Any]:
        """生成摘要报告"""
        total_events = len(events)

        # 事件类型统计
        event_types = Counter(event.event_type.value for event in events)

        # 严重程度统计
        severities = Counter(event.severity.value for event in events)

        # 用户活动统计
        user_activity = Counter(event.user_id for event in events if event.user_id)

        # 时间分布
        hourly_distribution = self._get_hourly_distribution(events)

        # 风险分数分布
        risk_distribution = self._get_risk_distribution(events)

        return {
            'report_type': 'summary',
            'generated_at': datetime.now().isoformat(),
            'total_events': total_events,
            'event_types': dict(event_types.most_common()),
            'severities': dict(severities.most_common()),
            'user_activity': dict(user_activity.most_common(10)),
            'hourly_distribution': hourly_distribution,
            'risk_distribution': risk_distribution,
            'period': {
                'start': min(e.timestamp for e in events).isoformat() if events else None,
                'end': max(e.timestamp for e in events).isoformat() if events else None
            }
        }

    def _generate_security_report(self, events: List['AuditEvent'], **kwargs) -> Dict[str, Any]:
        """生成安全报告"""
        security_events = [e for e in events if e.event_type.value == 'security']

        # 安全事件分析
        failed_logins = len([e for e in security_events if e.action == 'login' and e.result == 'failed'])
        successful_logins = len([e for e in security_events if e.action == 'login' and e.result == 'success'])

        # 高风险事件
        high_risk_events = [e for e in security_events if e.risk_score > 0.7]

        # 可疑IP地址
        suspicious_ips = self._analyze_suspicious_ips(security_events)

        return {
            'report_type': 'security',
            'generated_at': datetime.now().isoformat(),
            'total_events': len(events),
            'security_events_count': len(security_events),
            'login_attempts': {
                'successful': successful_logins,
                'failed': failed_logins,
                'success_rate': successful_logins / max(successful_logins + failed_logins, 1) * 100
            },
            'high_risk_events': len(high_risk_events),
            'suspicious_ips': suspicious_ips,
            'security_findings': self._analyze_security_findings(security_events)
        }

    def _generate_compliance_report(self, events: List['AuditEvent'], **kwargs) -> Dict[str, Any]:
        """生成合规报告"""
        compliance_events = [e for e in events if e.event_type.value == 'compliance']

        # 合规检查
        violations = len([e for e in compliance_events if e.result == 'violation'])
        compliances = len([e for e in compliance_events if e.result == 'compliant'])

        # 按合规类型分组
        compliance_by_type = defaultdict(int)
        for event in compliance_events:
            comp_type = event.details.get('compliance_type', 'unknown')
            compliance_by_type[comp_type] += 1

        return {
            'report_type': 'compliance',
            'generated_at': datetime.now().isoformat(),
            'total_events': len(events),
            'compliance_events_count': len(compliance_events),
            'violations': violations,
            'compliances': compliances,
            'compliance_rate': compliances / max(compliances + violations, 1) * 100,
            'compliance_by_type': dict(compliance_by_type)
        }

    def _generate_user_activity_report(self, events: List['AuditEvent'], **kwargs) -> Dict[str, Any]:
        """生成用户活动报告"""
        user_events = defaultdict(list)
        for event in events:
            if event.user_id:
                user_events[event.user_id].append(event)

        user_stats = {}
        for user_id, user_events_list in user_events.items():
            user_stats[user_id] = {
                'total_events': len(user_events_list),
                'event_types': Counter(e.event_type.value for e in user_events_list),
                'last_activity': max(e.timestamp for e in user_events_list),
                'risk_score_avg': sum(e.risk_score for e in user_events_list) / len(user_events_list),
                'high_risk_events': len([e for e in user_events_list if e.risk_score > 0.7])
            }

        return {
            'report_type': 'user_activity',
            'generated_at': datetime.now().isoformat(),
            'total_users': len(user_stats),
            'user_statistics': user_stats,
            'most_active_users': sorted(
                [(uid, stats['total_events']) for uid, stats in user_stats.items()],
                key=lambda x: x[1], reverse=True
            )[:10]
        }

    def _generate_resource_access_report(self, events: List['AuditEvent'], **kwargs) -> Dict[str, Any]:
        """生成资源访问报告"""
        resource_events = defaultdict(list)
        for event in events:
            if event.resource:
                resource_events[event.resource].append(event)

        resource_stats = {}
        for resource, res_events in resource_events.items():
            access_count = len(res_events)
            unique_users = len(set(e.user_id for e in res_events if e.user_id))
            failed_access = len([e for e in res_events if e.result == 'denied'])

            resource_stats[resource] = {
                'total_access': access_count,
                'unique_users': unique_users,
                'failed_access': failed_access,
                'success_rate': (access_count - failed_access) / max(access_count, 1) * 100,
                'last_access': max(e.timestamp for e in res_events).isoformat()
            }

        return {
            'report_type': 'resource_access',
            'generated_at': datetime.now().isoformat(),
            'total_resources': len(resource_stats),
            'resource_statistics': resource_stats,
            'resource_access': resource_stats,
            'most_accessed_resources': sorted(
                [(res, stats['total_access']) for res, stats in resource_stats.items()],
                key=lambda x: x[1], reverse=True
            )[:10],
            'top_resources': sorted(
                [(res, stats['total_access']) for res, stats in resource_stats.items()],
                key=lambda x: x[1], reverse=True
            )[:10],
        }

    def _generate_risk_analysis_report(self, events: List['AuditEvent'], **kwargs) -> Dict[str, Any]:
        """生成风险分析报告"""
        # 风险分数分布
        risk_levels = {
            'low': len([e for e in events if e.risk_score < 0.3]),
            'medium': len([e for e in events if 0.3 <= e.risk_score < 0.7]),
            'high': len([e for e in events if 0.7 <= e.risk_score < 0.9]),
            'critical': len([e for e in events if e.risk_score >= 0.9])
        }

        # 高风险事件详情
        high_risk_events = [e for e in events if e.risk_score >= 0.7]

        # 风险趋势分析
        risk_trend = self._analyze_risk_trend(events)

        return {
            'report_type': 'risk_analysis',
            'generated_at': datetime.now().isoformat(),
            'total_events': len(events),
            'risk_distribution': risk_levels,
            'high_risk_events_count': len(high_risk_events),
            'top_risk_events': [
                {
                    'event_id': e.event_id,
                    'user_id': e.user_id,
                    'resource': e.resource,
                    'risk_score': e.risk_score,
                    'severity': e.severity.value,
                    'timestamp': e.timestamp.isoformat()
                } for e in sorted(high_risk_events, key=lambda x: x.risk_score, reverse=True)[:10]
            ],
            'risk_trend': risk_trend
        }

    def _filter_events_by_time(self, events: List['AuditEvent'],
                             start_time: Optional[datetime] = None,
                             end_time: Optional[datetime] = None) -> List['AuditEvent']:
        """按时间过滤事件"""
        filtered_events = events.copy()

        if start_time:
            filtered_events = [e for e in filtered_events if e.timestamp >= start_time]

        if end_time:
            filtered_events = [e for e in filtered_events if e.timestamp <= end_time]

        return filtered_events

    def _get_hourly_distribution(self, events: List['AuditEvent']) -> Dict[str, int]:
        """获取小时分布"""
        hourly = defaultdict(int)
        for event in events:
            hour = event.timestamp.strftime('%H')
            hourly[hour] += 1
        return dict(hourly)

    def _get_risk_distribution(self, events: List['AuditEvent']) -> Dict[str, int]:
        """获取风险分布"""
        return {
            'low': len([e for e in events if e.risk_score < 0.3]),
            'medium': len([e for e in events if 0.3 <= e.risk_score < 0.6]),
            'high': len([e for e in events if 0.6 <= e.risk_score < 0.8]),
            'critical': len([e for e in events if e.risk_score >= 0.8]),
        }

    def _perform_compliance_checks(self, events: List['AuditEvent'], report_type: str) -> List[Dict[str, Any]]:
        """执行合规检查"""
        findings = []

        if report_type in ["general", "security"]:
            findings.extend(self._check_security_compliance(events))
            findings.extend(self._check_data_protection_compliance(events))

        if report_type in ["general", "audit"]:
            findings.extend(self._check_audit_compliance(events))

        return findings

    def _check_security_compliance(self, events: List['AuditEvent']) -> List[Dict[str, Any]]:
        """检查安全合规性"""
        findings = []

        failed_access = len([e for e in events if e.result == 'denied'])
        if failed_access > len(events) * 0.1:
            findings.append({
                'type': 'security',
                'severity': 'high',
                'description': f'访问失败率过高: {failed_access}/{len(events)}',
                'recommendation': '检查访问控制策略和用户权限配置'
            })

        high_risk_events = [e for e in events if e.risk_score > 0.8]
        if high_risk_events:
            findings.append({
                'type': 'security',
                'severity': 'high',
                'description': f'检测到 {len(high_risk_events)} 个高风险事件',
                'recommendation': '立即调查高风险事件并加强监控'
            })

        return findings

    def _check_data_protection_compliance(self, events: List['AuditEvent']) -> List[Dict[str, Any]]:
        """检查数据保护合规性"""
        findings = []

        sensitive_access = [e for e in events if 'sensitive' in str(e.resource).lower()]
        if sensitive_access:
            findings.append({
                'type': 'data_protection',
                'severity': 'medium',
                'description': f'检测到 {len(sensitive_access)} 次敏感数据访问',
                'recommendation': '确保敏感数据访问有适当的审计和控制'
            })

        return findings

    def _check_audit_compliance(self, events: List['AuditEvent']) -> List[Dict[str, Any]]:
        """检查审计合规性"""
        findings = []

        unique_resources = set(e.resource for e in events if e.resource)
        if len(unique_resources) < 5:
            findings.append({
                'type': 'audit',
                'severity': 'medium',
                'description': f'审计覆盖的资源类型较少: {len(unique_resources)}',
                'recommendation': '扩展审计规则以覆盖更多资源类型'
            })

        return findings

    def _generate_compliance_recommendations(self, findings: List[Dict[str, Any]]) -> List[str]:
        """生成合规建议"""
        recommendations = []

        if not findings:
            recommendations.append("合规状态良好，继续保持当前的安全实践")
            return recommendations

        high_severity_findings = [f for f in findings if f.get('severity') == 'high']
        if high_severity_findings:
            recommendations.append("优先处理高严重性问题，立即采取纠正措施")

        if len(findings) > 5:
            recommendations.append("审查和优化审计规则配置，减少误报")

        security_findings = [f for f in findings if f.get('type') == 'security']
        if security_findings:
            recommendations.append("加强安全监控和访问控制策略")

        data_protection_findings = [f for f in findings if f.get('type') == 'data_protection']
        if data_protection_findings:
            recommendations.append("加强敏感数据保护措施和访问控制")

        return recommendations

    def _calculate_compliance_score(self, findings: List[Dict[str, Any]], events: List['AuditEvent']) -> float:
        """计算合规分数"""
        if not events:
            return 100.0

        base_score = 100.0

        for finding in findings:
            severity = finding.get('severity', 'low')
            if severity == 'critical':
                base_score -= 30
            elif severity == 'high':
                base_score -= 20
            elif severity == 'medium':
                base_score -= 10
            elif severity == 'low':
                base_score -= 5

        return max(0.0, min(100.0, base_score))

    def _assess_compliance_risk(self, compliance_score: float) -> str:
        """评估合规风险"""
        if compliance_score >= 90:
            return "low"
        if compliance_score >= 80:
            return "medium"
        if compliance_score >= 60:
            return "high"
        return "critical"

    def _analyze_suspicious_ips(self, events: List['AuditEvent']) -> Dict[str, Any]:
        """分析可疑IP地址"""
        ip_events = defaultdict(list)
        for event in events:
            if event.ip_address:
                ip_events[event.ip_address].append(event)

        ip_counts = {ip: len(evts) for ip, evts in ip_events.items()}
        suspicious_ips_detail = {}
        for ip, ip_events_list in ip_events.items():
            failed_attempts = len([e for e in ip_events_list if e.result == 'failed'])
            if failed_attempts > 5:  # 失败次数超过阈值
                suspicious_ips_detail[ip] = {
                    'total_attempts': len(ip_events_list),
                    'failed_attempts': failed_attempts,
                    'last_attempt': max(e.timestamp for e in ip_events_list).isoformat()
                }

        return {
            'ip_counts': ip_counts,
            'suspicious_ips': suspicious_ips_detail,
        }

    def _analyze_security_findings(self, events: List['AuditEvent']) -> List[Dict[str, Any]]:
        """分析安全发现"""
        findings = []

        # 异常登录模式
        login_events = [e for e in events if e.action == 'login']
        if login_events:
            failed_rate = len([e for e in login_events if e.result == 'failed']) / len(login_events)
            if failed_rate > 0.2:
                findings.append({
                    'type': 'authentication',
                    'severity': 'high',
                    'description': f'登录失败率异常: {failed_rate:.1%}',
                    'recommendation': '检查认证系统和用户凭据'
                })

        return findings

    def _analyze_risk_trend(self, events: List['AuditEvent']) -> Dict[str, Any]:
        """分析风险趋势"""
        # 按天统计风险事件
        daily_risk = defaultdict(list)
        for event in events:
            day = event.timestamp.date()
            daily_risk[day].append(event.risk_score)

        trend_data = {}
        for day, scores in daily_risk.items():
            trend_data[day.isoformat()] = {
                'avg_risk': sum(scores) / len(scores),
                'max_risk': max(scores),
                'high_risk_count': len([s for s in scores if s > 0.7])
            }

        return {
            'trend': trend_data,
            'analysis': {
                'total_days': len(trend_data),
                'high_risk_days': len([d for d, v in trend_data.items() if v['high_risk_count'] > 0])
            }
        }

    def _export_json(self, report_data: Dict[str, Any], output_path: Path) -> str:
        """导出为JSON格式"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
        return str(output_path)

    def _export_csv(self, report_data: Dict[str, Any], output_path: Path) -> str:
        """导出为CSV格式"""
        # 简化的CSV导出，实际使用时需要根据报告类型定制
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Key', 'Value'])
            for key, value in report_data.items():
                if isinstance(value, (str, int, float)):
                    writer.writerow([key, value])
                else:
                    writer.writerow([key, json.dumps(value, ensure_ascii=False, default=str)])
        return str(output_path)

    def _export_html(self, report_data: Dict[str, Any], output_path: Path) -> str:
        """导出为HTML格式"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>审计报告 - {report_data.get('report_type', 'Unknown')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 10px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>审计报告</h1>
                <p>报告类型: {report_data.get('report_type', 'Unknown')}</p>
                <p>生成时间: {report_data.get('generated_at', 'Unknown')}</p>
            </div>
            <div class="section">
                <h2>报告内容</h2>
                <pre>{json.dumps(report_data, indent=2, ensure_ascii=False, default=str)}</pre>
            </div>
        </body>
        </html>
        """

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        return str(output_path)

    def _export_pdf(self, report_data: Dict[str, Any], output_path: Path) -> str:
        """导出为PDF格式"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            content_body = json.dumps(report_data, indent=2, ensure_ascii=False, default=str)
            padding = "\n".join(["-" * 80 for _ in range(5)])
            content = [
                "RQA2025 审计报告",
                f"生成时间: {datetime.now().isoformat()}",
                f"报告类型: {report_data.get('report_type', 'unknown')}",
                padding,
                content_body,
                padding,
            ]
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(content))
            return str(output_path)
        except Exception as e:
            logging.error(f"PDF导出失败: {e}")
            raise
