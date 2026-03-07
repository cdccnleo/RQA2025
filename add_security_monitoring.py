#!/usr/bin/env python3
# 添加安全监控辅助方法到PerformanceMonitor类

code_to_add = '''
    def _record_user_activity(self, user_id: str, operation: str, duration: float):
        """记录用户活动"""
        if user_id not in self.user_activity:
            self.user_activity[user_id] = {}

        if operation not in self.user_activity[user_id]:
            self.user_activity[user_id][operation] = []

        self.user_activity[user_id][operation].append({
            'timestamp': datetime.now(),
            'duration': duration
        })

        # 保持最近1000条记录
        if len(self.user_activity[user_id][operation]) > 1000:
            self.user_activity[user_id][operation] = self.user_activity[user_id][operation][-1000:]

    def _record_resource_access(self, resource: str, operation: str, duration: float):
        """记录资源访问"""
        if resource not in self.resource_access:
            self.resource_access[resource] = {}

        if operation not in self.resource_access[resource]:
            self.resource_access[resource][operation] = []

        self.resource_access[resource][operation].append({
            'timestamp': datetime.now(),
            'duration': duration
        })

        # 保持最近1000条记录
        if len(self.resource_access[resource][operation]) > 1000:
            self.resource_access[resource][operation] = self.resource_access[resource][operation][-1000:]

    def _get_user_activity_summary(self) -> Dict[str, Any]:
        """获取用户活动摘要"""
        summary = {}
        for user_id, operations in self.user_activity.items():
            user_summary = {}
            for op_name, records in operations.items():
                if records:
                    durations = [r['duration'] for r in records]
                    user_summary[op_name] = {
                        'total_operations': len(records),
                        'avg_duration': sum(durations) / len(durations),
                        'max_duration': max(durations),
                        'last_activity': records[-1]['timestamp'].isoformat()
                    }
            summary[user_id] = user_summary

        return summary

    def _get_resource_access_summary(self) -> Dict[str, Any]:
        """获取资源访问摘要"""
        summary = {}
        for resource, operations in self.resource_access.items():
            resource_summary = {}
            for op_name, records in operations.items():
                if records:
                    durations = [r['duration'] for r in records]
                    resource_summary[op_name] = {
                        'total_accesses': len(records),
                        'avg_duration': sum(durations) / len(durations),
                        'max_duration': max(durations),
                        'last_access': records[-1]['timestamp'].isoformat()
                    }
            summary[resource] = resource_summary

        return summary

    def _get_security_operation_trends(self) -> Dict[str, Any]:
        """获取安全操作趋势"""
        trends = {}
        current_time = datetime.now()

        # 分析最近1小时的趋势
        one_hour_ago = current_time - timedelta(hours=1)

        for operation_name, metrics in self._metrics.items():
            if 'security' in operation_name.lower() or operation_name in ['authenticate', 'authorize', 'audit']:
                # 计算最近1小时的操作次数
                recent_calls = 0
                if hasattr(metrics, 'call_times') and metrics.call_times:
                    for call_record in list(metrics.call_times):
                        if isinstance(call_record, dict):
                            call_time = call_record.get('timestamp', datetime.min)
                            if call_time > one_hour_ago:
                                recent_calls += 1
                        elif isinstance(call_record, (int, float)):
                            # 如果是简单的数值，假设是时间戳
                            if call_record > one_hour_ago.timestamp():
                                recent_calls += 1

                trends[operation_name] = {
                    'calls_per_hour': recent_calls,
                    'trend': 'increasing' if recent_calls > 5 else 'stable'
                }

        return trends

# 全局性能监控器实例
'''

# 读取现有文件
with open('src/infrastructure/security/monitoring/performance_monitor.py', 'r') as f:
    content = f.read()

# 替换插入点
old_marker = '# 全局性能监控器实例'
new_content = content.replace(old_marker, code_to_add)

# 写回文件
with open('src/infrastructure/security/monitoring/performance_monitor.py', 'w') as f:
    f.write(new_content)

print('✅ 已添加安全监控辅助方法到PerformanceMonitor类')
