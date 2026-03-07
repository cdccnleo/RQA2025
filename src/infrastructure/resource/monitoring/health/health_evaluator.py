
from ...models.alert_dataclasses import AlertPerformanceMetrics as PerformanceMetrics
from ..shared_interfaces import ILogger, StandardLogger
from .health_status import HealthStatus
from typing import Dict, List, Optional, Any, Tuple
"""
健康评估器

职责：专门负责评估系统健康状态
"""


class HealthEvaluator:
    """
    健康评估器

    职责：评估系统各方面的健康状态，生成健康评分和建议
    """

    def __init__(self, thresholds: Optional[Dict[str, float]] = None,
                 logger: Optional[ILogger] = None):
        self.logger = logger or StandardLogger(f"{self.__class__.__name__}")

        # 默认健康阈值
        self.thresholds = thresholds or {
            "cpu_critical": 90.0,
            "cpu_warning": 80.0,
            "memory_critical": 90.0,
            "memory_warning": 85.0,
            "disk_critical": 95.0,
            "disk_warning": 90.0,
            "alerts_critical": 10,
            "alerts_warning": 5,
            "test_success_rate_critical": 0.5,
            "test_success_rate_warning": 0.8
        }

    def evaluate_overall_health(self, metrics: Optional[PerformanceMetrics],
                                alert_stats: Optional[Dict[str, Any]] = None,
                                test_stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        评估整体系统健康状态

        Args:
            metrics: 性能指标
            alert_stats: 告警统计
            test_stats: 测试统计

        Returns:
            Dict[str, Any]: 健康评估结果
        """
        issues = []
        recommendations = []

        # 评估各项健康指标
        performance_score = self._assess_performance_health(metrics, issues, recommendations)
        alert_score = self._assess_alert_health(alert_stats, issues, recommendations)
        test_score = self._assess_test_health(test_stats, issues, recommendations)

        # 计算综合健康评分
        overall_score = (performance_score + alert_score + test_score) / 3

        # 确定整体健康状态
        overall_status = self._determine_health_status(overall_score)

        # 生成系统级评估
        system_issues, system_recommendations = self._assess_system_health(issues, recommendations)
        issues.extend(system_issues)
        recommendations.extend(system_recommendations)

        return {
            'overall_status': overall_status.value,
            'overall_score': overall_score,
            'performance_score': performance_score,
            'alert_score': alert_score,
            'test_score': test_score,
            'issues': issues,
            'recommendations': recommendations,
            'thresholds': self.thresholds.copy(),
            'evaluation_timestamp': metrics.timestamp if metrics else None
        }

    def _assess_performance_health(self, metrics: Optional[PerformanceMetrics],
                                   issues: List[str], recommendations: List[str]) -> float:
        """
        评估性能健康状态

        Returns:
            float: 性能健康评分 (0.0-1.0)
        """
        if not metrics:
            issues.append("无法获取性能指标")
            recommendations.append("检查性能监控组件是否正常工作")
            return 0.0

        score = 1.0
        issue_count = 0

        # 检查CPU健康
        cpu_score = self._check_cpu_health(metrics, issues, recommendations)
        score = min(score, cpu_score)
        if cpu_score < 1.0:
            issue_count += 1

        # 检查内存健康
        memory_score = self._check_memory_health(metrics, issues, recommendations)
        score = min(score, memory_score)
        if memory_score < 1.0:
            issue_count += 1

        # 检查磁盘健康
        disk_score = self._check_disk_health(metrics, issues, recommendations)
        score = min(score, disk_score)
        if disk_score < 1.0:
            issue_count += 1

        # 根据问题数量调整评分
        if issue_count > 1:
            score *= 0.8  # 多个问题时降低评分

        return score

    def _check_cpu_health(self, metrics: PerformanceMetrics,
                          issues: List[str], recommendations: List[str]) -> float:
        """检查CPU健康状态"""
        cpu_percent = metrics.cpu_percent

        if cpu_percent >= self.thresholds["cpu_critical"]:
            issues.append(f"CPU使用率严重过高: {cpu_percent}%")
            recommendations.append("立即检查CPU密集型进程并优化性能")
            return 0.2
        elif cpu_percent >= self.thresholds["cpu_warning"]:
            issues.append(f"CPU使用率较高: {cpu_percent}%")
            recommendations.append("监控CPU使用趋势，考虑优化相关进程")
            return 0.6

        return 1.0

    def _check_memory_health(self, metrics: PerformanceMetrics,
                             issues: List[str], recommendations: List[str]) -> float:
        """检查内存健康状态"""
        memory_percent = metrics.memory_percent

        if memory_percent >= self.thresholds["memory_critical"]:
            issues.append(f"内存使用率严重过高: {memory_percent}%")
            recommendations.append("立即检查内存泄漏并释放系统内存")
            return 0.2
        elif memory_percent >= self.thresholds["memory_warning"]:
            issues.append(f"内存使用率较高: {memory_percent}%")
            recommendations.append("监控内存使用趋势，考虑优化内存管理")
            return 0.6

        return 1.0

    def _check_disk_health(self, metrics: PerformanceMetrics,
                           issues: List[str], recommendations: List[str]) -> float:
        """检查磁盘健康状态"""
        disk_percent = metrics.disk_usage

        if disk_percent >= self.thresholds["disk_critical"]:
            issues.append(f"磁盘使用率严重过高: {disk_percent}%")
            recommendations.append("立即清理磁盘空间，删除不必要的文件")
            return 0.2
        elif disk_percent >= self.thresholds["disk_warning"]:
            issues.append(f"磁盘使用率较高: {disk_percent}%")
            recommendations.append("监控磁盘使用趋势，规划磁盘清理")
            return 0.6

        return 1.0

    def _assess_alert_health(self, alert_stats: Optional[Dict[str, Any]],
                             issues: List[str], recommendations: List[str]) -> float:
        """评估告警健康状态"""
        if not alert_stats:
            return 1.0  # 没有告警数据时假设健康

        active_alerts = alert_stats.get('active_count', 0)
        critical_alerts = alert_stats.get('critical_count', 0)

        if critical_alerts >= self.thresholds["alerts_critical"]:
            issues.append(f"严重告警数量过多: {critical_alerts}个")
            recommendations.append("立即处理所有严重告警，检查系统异常")
            return 0.1
        elif active_alerts >= self.thresholds["alerts_warning"]:
            issues.append(f"活跃告警数量较多: {active_alerts}个")
            recommendations.append("及时处理活跃告警，分析告警趋势")
            return 0.5

        return 1.0

    def _assess_test_health(self, test_stats: Optional[Dict[str, Any]],
                            issues: List[str], recommendations: List[str]) -> float:
        """评估测试健康状态"""
        if not test_stats:
            return 1.0  # 没有测试数据时假设健康

        success_rate = test_stats.get('success_rate', 1.0)
        failed_tests = test_stats.get('failed_count', 0)

        if success_rate <= self.thresholds["test_success_rate_critical"]:
            issues.append(f"测试成功率严重过低: {success_rate:.2%}")
            recommendations.append("立即修复测试失败，检查系统功能异常")
            return 0.1
        elif success_rate <= self.thresholds["test_success_rate_warning"]:
            issues.append(f"测试成功率较低: {success_rate:.2%}")
            recommendations.append("分析测试失败原因，提升测试覆盖率")
            return 0.5

        return 1.0

    def _assess_system_health(self, issues: List[str], recommendations: List[str]) -> Tuple[List[str], List[str]]:
        """评估系统级健康状态"""
        system_issues = []
        system_recommendations = []

        total_issues = len(issues)

        if total_issues >= 5:
            system_issues.append("系统健康问题严重，需要全面检查")
            system_recommendations.append("建议进行全面的系统健康诊断")
        elif total_issues >= 3:
            system_issues.append("系统存在多个健康问题")
            system_recommendations.append("建议优先处理关键健康问题")

        return system_issues, system_recommendations

    def _determine_health_status(self, health_score: float) -> HealthStatus:
        """根据健康评分确定健康状态"""
        if health_score >= 0.8:
            return HealthStatus.HEALTHY
        elif health_score >= 0.5:
            return HealthStatus.WARNING
        elif health_score > 0:
            return HealthStatus.CRITICAL
        else:
            return HealthStatus.UNKNOWN
