#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 审计管理器

负责审计日志和合规报告
分离了AuditLoggingManager的审计职责
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum
from src.infrastructure.security.core.types import (
    AuditEventParams, QueryFilterParams, ReportGenerationParams
)


class AuditManager:
    """
    审计管理器

    职责：专门管理审计日志、事件记录和合规检查
    提供高效的审计数据存储和查询功能
    """

    AuditEventParams = AuditEventParams
    QueryFilterParams = QueryFilterParams
    ReportGenerationParams = ReportGenerationParams

    def __init__(self, log_path: str = "data/security/audit"):
        import threading
        self.log_path = log_path
        self.events: List[Dict] = []
        self.event_buffer: List[Dict] = []
        self.buffer_size = 1000
        self.max_events = 100000  # 最大事件数量
        self._lock = threading.Lock()  # 线程锁

    def log_event(self, params: AuditEventParams) -> str:
        """
        记录审计事件

        Args:
            params: 审计事件参数

        Returns:
            事件ID
        """
        event = {
            'event_id': self._generate_event_id(),
            'event_type': params.event_type.value,
            'severity': params.severity.value,
            'timestamp': params.timestamp.isoformat(),
            'user_id': params.user_id,
            'session_id': params.session_id,
            'resource': params.resource,
            'action': params.action,
            'result': params.result,
            'details': params.details,
            'ip_address': params.ip_address,
            'user_agent': params.user_agent,
            'location': params.location,
            'risk_score': params.risk_score,
            'tags': list(params.tags)
        }

        # 添加到缓冲区（使用线程锁保护）
        with self._lock:
            self.event_buffer.append(event)

            # 如果缓冲区满，刷新到主存储
            if len(self.event_buffer) >= self.buffer_size:
                self._flush_buffer()

            # 检查是否需要清理旧事件
            if len(self.events) >= self.max_events:
                self._cleanup_old_events()

        return event['event_id']

    def query_events(self, params: QueryFilterParams) -> List[Dict]:
        """
        查询审计事件

        Args:
            params: 查询过滤参数

        Returns:
            匹配的事件列表
        """
        events = self.events + self.event_buffer

        # 应用过滤条件
        filtered_events = self._apply_filters(events, params)

        # 排序
        filtered_events.sort(key=lambda x: x['timestamp'], reverse=(params.sort_order == 'desc'))

        # 分页
        if params.offset > 0:
            filtered_events = filtered_events[params.offset:]
        if params.limit is not None and params.limit > 0:
            filtered_events = filtered_events[:params.limit]
        elif params.limit == 0:
            filtered_events = []

        return filtered_events

    def generate_security_report(
        self,
        params: Optional[ReportGenerationParams] = None,
        **kwargs: Any,
    ) -> Dict:
        """
        生成安全报告

        Args:
            params: 报告生成参数

        Returns:
            报告数据
        """
        if params is None:
            if kwargs:
                params = ReportGenerationParams(report_type="security", **kwargs)
            else:
                params = ReportGenerationParams(report_type="security")

        # 获取事件数据
        events = self.query_events(params.filters)

        # 生成报告统计
        stats = self._calculate_security_stats(events, params.time_range)

        # 如果需要分组
        grouped_data = None
        if params.group_by:
            if isinstance(params.group_by, (list, tuple)):
                group_fields = list(params.group_by)
            else:
                group_fields = sorted(list(params.group_by))
            grouped_raw = self._group_events(events, group_fields)
            grouped_data = self._structure_grouped_data(grouped_raw, group_fields)

        # 如果需要聚合
        aggregated_data = None
        if params.aggregation:
            aggregated_data = self._aggregate_events(events, params.aggregation)

        # 生成摘要
        summary = {
            'total_events': len(events),
            'risk_level': 'high' if stats.get('high_risk_events', 0) > 10 else 'medium' if stats.get('high_risk_events', 0) > 5 else 'low',
            'compliance_status': self._assess_compliance_status(stats),
            'recommendations': self._generate_security_recommendations(stats)
        }

        result = {
            'report_type': params.report_type,
            'generated_at': datetime.now().isoformat(),
            'time_range': params.time_range,
            'statistics': stats,
            'event_count': len(events),
            'summary': summary
        }

        if grouped_data is not None:
            result['grouped_data'] = grouped_data
        if aggregated_data is not None:
            result['aggregated_data'] = aggregated_data
        result['recommendations'] = summary['recommendations']

        return result

    def get_compliance_report(self, compliance_type: str = "general") -> Dict:
        """
        生成合规报告

        Args:
            compliance_type: 合规类型

        Returns:
            合规报告数据
        """
        # 获取最近的事件
        recent_events = self.query_events(QueryFilterParams(
            start_date=datetime.now() - timedelta(days=30)
        ))

        findings = self._perform_compliance_checks(recent_events, compliance_type)
        compliance_metrics = self._calculate_compliance_metrics(recent_events, compliance_type)
        compliance_score = self._calculate_compliance_score(findings, recent_events)
        compliance_metrics['compliance_score'] = compliance_score

        status = self._assess_compliance_status(compliance_metrics)
        recommendations = self._generate_compliance_recommendations(findings)

        return {
            'compliance_type': compliance_type,
            'report_date': datetime.now().isoformat(),
            'period': '30天',
            'metrics': compliance_metrics,
            'status': status,
            'findings': findings,
            'recommendations': recommendations,
            'total_events': len(recent_events),
        }

    def _flush_buffer(self) -> None:
        """刷新缓冲区到主存储"""
        if self.event_buffer:
            events_to_write = self.event_buffer.copy()
            self.events.extend(self.event_buffer)
            self.event_buffer.clear()

            if len(self.events) > self.max_events:
                self._cleanup_old_events()

            # 异步写入文件（简化实现）
            self._write_events_to_file(events_to_write)

    def _write_events_to_file(self, events_to_write: List[Dict] = None) -> None:
        """写入事件到文件"""
        import json
        from datetime import datetime
        from pathlib import Path

        if events_to_write is None:
            events_to_write = self.event_buffer

        if not events_to_write:
            return

        # 创建audit目录
        base_path = Path(self.log_path)
        audit_dir = base_path if not base_path.suffix else base_path.parent
        audit_dir.mkdir(parents=True, exist_ok=True)

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"audit_{timestamp}.json"
        filepath = audit_dir / filename

        # 写入事件
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(events_to_write, f, indent=2, ensure_ascii=False, default=str)
            logging.info(f"成功写入 {len(events_to_write)} 个审计事件到 {filepath}")
        except Exception as e:
            logging.error(f"写入审计事件失败: {e}")

    def _cleanup_old_events(self) -> None:
        """清理旧事件"""
        from pathlib import Path
        from datetime import datetime, timedelta

        # 保留最近的事件
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

        # 清理旧的审计文件（保留最近30天的文件）
        try:
            base_path = Path(self.log_path)
            audit_dir = base_path if not base_path.suffix else base_path.parent
            if audit_dir.exists():
                cutoff_date = datetime.now() - timedelta(days=30)
                for file_path in audit_dir.glob("audit_*.json"):
                    try:
                        # 从文件名提取日期
                        filename = file_path.name
                        date_str = filename.replace("audit_", "").replace(".json", "")
                        if len(date_str) >= 8:  # YYYYMMDD
                            file_date = datetime.strptime(date_str[:8], "%Y%m%d")
                            if file_date < cutoff_date:
                                file_path.unlink()
                                logging.info(f"删除旧审计文件: {file_path}")
                    except (ValueError, OSError) as e:
                        logging.warning(f"处理审计文件 {file_path} 时出错: {e}")
        except Exception as e:
            logging.error(f"清理旧审计文件失败: {e}")

    def _generate_event_id(self) -> str:
        """生成事件ID"""
        import uuid
        return f"evt_{uuid.uuid4().hex[:12]}"

    def _apply_filters(self, events: List[Dict], params: QueryFilterParams) -> List[Dict]:
        """应用过滤条件"""
        filtered = events

        severity_order = {
            'low': 0,
            'medium': 1,
            'high': 2,
            'critical': 3,
        }

        # 日期过滤
        if params.start_date:
            filtered = [e for e in filtered if e['timestamp'] >= params.start_date.isoformat()]
        if params.end_date:
            filtered = [e for e in filtered if e['timestamp'] <= params.end_date.isoformat()]

        if params.event_id:
            filtered = [e for e in filtered if e.get('event_id') == params.event_id]

        # 用户过滤
        if params.user_ids:
            filtered = [e for e in filtered if e.get('user_id') in params.user_ids]

        # 事件类型过滤
        if params.event_types:
            event_types = {et.value for et in params.event_types}
            filtered = [e for e in filtered if e.get('event_type') in event_types]

        # 严重程度过滤
        if params.severities:
            severities = {s.value for s in params.severities}
            filtered = [e for e in filtered if e.get('severity') in severities]
        if params.min_severity:
            min_rank = severity_order.get(params.min_severity.value, 0)
            filtered = [
                e for e in filtered
                if severity_order.get(e.get('severity', ''), -1) >= min_rank
            ]
        if params.max_severity:
            max_rank = severity_order.get(params.max_severity.value, 3)
            filtered = [
                e for e in filtered
                if severity_order.get(e.get('severity', ''), -1) <= max_rank
            ]

        # 资源过滤
        if params.resources:
            filtered = [e for e in filtered if e.get('resource') in params.resources]

        # 动作过滤
        if params.actions:
            filtered = [e for e in filtered if e.get('action') in params.actions]

        # 结果过滤
        if params.results:
            filtered = [e for e in filtered if e.get('result') in params.results]

        # 风险分数过滤
        if params.min_risk_score is not None:
            filtered = [e for e in filtered if e.get('risk_score', 0) >= params.min_risk_score]
        if params.max_risk_score is not None:
            filtered = [e for e in filtered if e.get('risk_score', 0) <= params.max_risk_score]

        if params.ip_addresses:
            filtered = [e for e in filtered if e.get('ip_address') in params.ip_addresses]
        if params.session_ids:
            filtered = [e for e in filtered if e.get('session_id') in params.session_ids]
        if params.locations:
            filtered = [e for e in filtered if e.get('location') in params.locations]
        if params.tags:
            filtered = [
                e for e in filtered
                if params.tags.intersection(set(e.get('tags', [])))
            ]
        if params.include_tags:
            filtered = [
                e for e in filtered
                if params.include_tags.issubset(set(e.get('tags', [])))
            ]
        if params.exclude_tags:
            filtered = [
                e for e in filtered
                if params.exclude_tags.isdisjoint(set(e.get('tags', [])))
            ]

        return filtered

    def _calculate_security_stats(self, events: List[Dict], time_range: Optional[Dict]) -> Dict:
        """计算安全统计"""
        stats = {
            'total_events': len(events),
            'events_by_type': defaultdict(int),
            'events_by_severity': defaultdict(int),
            'events_by_user': defaultdict(int),
            'high_risk_events': 0,
            'failed_events': 0,
            'access_denials': 0,
            'suspicious_activities': 0,
            'policy_violations': 0,
            'unique_users': set(),
            'unique_resources': set()
        }

        for event in events:
            stats['events_by_type'][event.get('event_type', 'unknown')] += 1
            stats['events_by_severity'][event.get('severity', 'unknown')] += 1
            stats['events_by_user'][event.get('user_id', 'unknown')] += 1

            if event.get('risk_score', 0) > 0.7:
                stats['high_risk_events'] += 1

            if event.get('result') in ['failure', 'denied', 'error']:
                stats['failed_events'] += 1

            # 计算访问拒绝次数
            if event.get('result') == 'denied':
                stats['access_denials'] = stats.get('access_denials', 0) + 1

            if event.get('user_id'):
                stats['unique_users'].add(event['user_id'])

            if event.get('resource'):
                stats['unique_resources'].add(event['resource'])

        stats['unique_users'] = len(stats['unique_users'])
        stats['unique_resources'] = len(stats['unique_resources'])

        # 计算成功率
        successful_authentications = sum(1 for event in events
                                       if event.get('details', {}).get('success', False))
        failed_authentications = sum(1 for event in events
                                   if event.get('result', '') == 'failure' or
                                      not event.get('details', {}).get('success', True))

        stats['successful_authentications'] = successful_authentications
        stats['failed_authentications'] = failed_authentications
        stats['success_rate'] = (successful_authentications / max(len(events), 1)) * 100

        return stats

    def _group_events(self, events: List[Dict], group_fields: List[str]) -> Dict:
        """分组事件，返回以元组为键的原始分组结构"""
        if not group_fields:
            return {}

        def normalize(field: str, value: Any) -> Any:
            if isinstance(value, Enum):
                return value.name
            if isinstance(value, str) and field in {"event_type", "severity"}:
                return value.upper()
            return value

        grouped: Dict[tuple, List[Dict]] = defaultdict(list)
        for event in events:
            key = tuple(normalize(field, event.get(field, 'unknown')) for field in group_fields)
            grouped[key].append(event)

        return dict(grouped)

    def _structure_grouped_data(self, grouped_raw: Dict[tuple, List[Dict]], fields: List[str]) -> Dict[str, Any]:
        """将原始分组结构转化为层级字典，便于报告展示"""
        if not grouped_raw:
            return {}

        def insert_node(tree: Dict[str, Any], key_parts: tuple, events: List[Dict]) -> None:
            node = tree
            for idx, part in enumerate(key_parts):
                part_key = str(part) if part is not None else "unknown"
                node = node.setdefault(part_key, {})
                node['count'] = node.get('count', 0) + len(events)
            node.setdefault('events', []).extend(events)

        tree: Dict[str, Any] = {}
        for key_tuple, events in grouped_raw.items():
            insert_node(tree, key_tuple, events)
        return tree

    def _generate_security_recommendations(self, stats: Dict) -> List[str]:
        """生成安全建议"""
        recommendations = []

        if stats.get('failed_authentications', 0) > 10:
            recommendations.append("建议加强身份验证安全措施")
        if stats.get('access_denials', 0) > 20:
            recommendations.append("建议审查访问控制策略")
        if stats.get('high_risk_events', 0) > 5:
            recommendations.append("建议监控高风险活动")
        if stats.get('success_rate', 100) < 95:
            recommendations.append("建议改善系统稳定性")

        return recommendations

    def _aggregate_events(self, events: List[Dict], aggregation: Dict) -> Dict:
        """聚合事件"""
        result = {}

        for agg_name, agg_config in aggregation.items():
            if isinstance(agg_config, (list, tuple, set)):
                fields = list(agg_config)
                counter: Dict[Any, int] = defaultdict(int)
                for event in events:
                    key_parts = []
                    for field in fields:
                        value = event.get(field, 'unknown')
                        if hasattr(value, "value"):
                            value = value.value
                        key_parts.append(str(value).upper())
                    key = tuple(key_parts)
                    counter[key[0] if len(key) == 1 else tuple(key)] += 1
                result[agg_name] = dict(counter)
                continue

            if isinstance(agg_config, str):
                if agg_config == 'count':
                    result[agg_name] = len(events)
                continue

            field_path = agg_config.get('field')
            operation = agg_config.get('operation', 'count')

            if field_path is None:
                # 简单计数
                result[agg_name] = len(events)
                continue

            # 解析嵌套字段路径
            def get_nested_value(obj, path):
                try:
                    for key in path.split('.'):
                        obj = obj[key]
                    return obj
                except (KeyError, TypeError):
                    return None

            values = []
            for event in events:
                value = get_nested_value(event, field_path)
                if value is not None:
                    values.append(value)

            if operation == 'count':
                result[agg_name] = len(values)
            elif operation == 'sum':
                result[agg_name] = sum(values) if values else 0
            elif operation == 'avg':
                result[agg_name] = sum(values) / len(values) if values else 0.0
            elif operation == 'max':
                result[agg_name] = max(values) if values else 0.0
            elif operation == 'min':
                result[agg_name] = min(values) if values else 0.0

        return result

    def _calculate_compliance_metrics(self, events: List[Dict], compliance_type: str) -> Dict:
        """计算合规指标"""
        total_events = len(events)
        metrics = {
            'total_events': total_events,
            'total_auditable_events': total_events,
            'successful_authentications': 0,
            'failed_authentications': 0,
            'authentication_events': 0,
            'security_events': 0,
            'access_denials': 0,
            'failed_logins': 0,
            'suspicious_activities': 0,
            'data_access_events': 0,
            'sensitive_data_accesses': 0,
            'policy_violations': 0,
            'config_changes': 0,
        }

        for event in events:
            event_type = event.get('event_type', '')
            result = event.get('result', '')
            action = event.get('action', '')
            details = event.get('details', {})

            if event_type == 'security':
                metrics['security_events'] += 1
                # 检查认证成功/失败
                if 'login' in action or 'auth' in action:
                    metrics['authentication_events'] += 1
                    if details.get('success', False):
                        metrics['successful_authentications'] += 1
                    elif result == 'failure' or not details.get('success', True):
                        metrics['failed_authentications'] += 1
            elif event_type == 'access' and result == 'denied':
                metrics['access_denials'] += 1
            elif 'login' in action and result == 'failure':
                metrics['failed_logins'] += 1
            elif event.get('risk_score', 0) > 0.8:
                metrics['suspicious_activities'] += 1
            elif event_type == 'data_operation':
                metrics['data_access_events'] += 1
                if details.get('sensitive_data') or 'sensitive' in str(event.get('resource', '')).lower():
                    metrics['sensitive_data_accesses'] += 1
            elif event_type == 'config_change':
                metrics['config_changes'] += 1

            if details.get('policy_violation'):
                metrics['policy_violations'] += 1

        return metrics

    def _perform_compliance_checks(self, events: List[Dict], compliance_type: str) -> List[Dict[str, Any]]:
        """针对指定事件执行合规检查，返回发现列表"""
        findings: List[Dict[str, Any]] = []

        if compliance_type in {"general", "security"}:
            failed_access = sum(1 for e in events if e.get('result') == 'denied')
            if failed_access > max(5, len(events) * 0.1):
                findings.append({
                    'type': 'security',
                    'severity': 'high',
                    'description': f'访问失败次数偏高: {failed_access}',
                    'recommendation': '审查访问控制策略并强化身份验证',
                })

            high_risk = [e for e in events if e.get('risk_score', 0) >= 0.8]
            if high_risk:
                findings.append({
                    'type': 'security',
                    'severity': 'high',
                    'description': f'检测到 {len(high_risk)} 条高风险事件',
                    'recommendation': '立即调查高风险事件并加强监控',
                })

        if compliance_type in {"general", "data"}:
            sensitive_access = [
                e for e in events
                if 'sensitive' in str(e.get('resource', '')).lower()
                or e.get('details', {}).get('sensitive_data')
            ]
            if sensitive_access:
                findings.append({
                    'type': 'data_protection',
                    'severity': 'medium',
                    'description': f'有 {len(sensitive_access)} 次敏感数据访问',
                    'recommendation': '确保敏感数据访问有严格的审批和审计流程',
                })

        if compliance_type in {"general", "audit"}:
            if not events:
                findings.append({
                    'type': 'audit',
                    'severity': 'low',
                    'description': '近30日内缺少审计事件记录',
                    'recommendation': '确认审计采集组件运行正常',
                })

        return findings

    def _generate_compliance_recommendations(self, findings: List[Dict[str, Any]]) -> List[str]:
        """根据发现生成合规建议"""
        if not findings:
            return ["合规状态良好，保持当前安全与审计策略"]

        recommendations = []
        for finding in findings:
            suggestion = finding.get('recommendation')
            if suggestion:
                recommendations.append(suggestion)

        if not recommendations:
            recommendations.append("针对发现的问题制定整改计划并跟踪落实")

        return recommendations

    def _calculate_compliance_score(self, findings: List[Dict[str, Any]], events: List[Dict]) -> float:
        """根据发现和事件总量计算合规得分"""
        if not events:
            return 100.0

        score = 100.0
        weight = {
            'critical': 30,
            'high': 20,
            'medium': 10,
            'low': 5,
        }

        for finding in findings:
            severity = finding.get('severity', 'low')
            deduction = weight.get(severity, 5)
            score -= deduction

        return max(0.0, min(100.0, score))

    def _assess_compliance_status(self, metrics: Dict) -> str:
        """评估合规状态"""
        failed_auth = metrics.get('failed_authentications', 0)
        if failed_auth >= 50:
            return 'non_compliant'

        risk_score = 0.0

        if failed_auth >= 25:
            risk_score += 0.4
        elif failed_auth >= 10:
            risk_score += 0.2

        access_denials = metrics.get('access_denials', 0)
        if access_denials >= 150:
            risk_score += 0.3
        elif access_denials >= 75:
            risk_score += 0.2

        suspicious = metrics.get('suspicious_activities', 0)
        if suspicious >= 15:
            risk_score += 0.4
        elif suspicious >= 5:
            risk_score += 0.2

        policy_violations = metrics.get('policy_violations', 0)
        if policy_violations >= 20:
            risk_score += 0.3
        elif policy_violations >= 10:
            risk_score += 0.2

        sensitive_accesses = metrics.get('sensitive_data_accesses', 0)
        if sensitive_accesses >= 50:
            risk_score += 0.3
        elif sensitive_accesses >= 20:
            risk_score += 0.2

        config_changes = metrics.get('config_changes', 0)
        if config_changes >= 25:
            risk_score += 0.2
        elif config_changes >= 10:
            risk_score += 0.1

        if risk_score >= 0.8:
            return 'non_compliant'
        if risk_score >= 0.4:
            return 'warning'
        return 'compliant'
