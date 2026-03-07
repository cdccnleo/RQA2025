#!/usr/bin/env python3
"""
RQA2025 基础设施层监控协调器

负责连续监控系统的生命周期管理和协调。
这是从ContinuousMonitoringSystem中拆分出来的核心协调组件。
"""

import threading
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class MonitoringCoordinator:
    """
    监控协调器

    负责监控系统的启动、停止、配置管理和生命周期协调。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化监控协调器

        Args:
            config: 监控配置
        """
        self.config = config or {
            'interval_seconds': 300,  # 5分钟监控间隔
            'max_history_items': 1000,
            'alert_thresholds': {
                'coverage_drop': 5,
                'performance_degradation': 10,
                'memory_usage_high': 80,
                'cpu_usage_high': 70,
            }
        }

        # 监控状态
        self.monitoring_active = False
        self.monitoring_thread = None
        self.start_time = None

        # 子组件
        self.metrics_collector = None
        self.alert_processor = None
        self.optimization_suggester = None
        self.data_manager = None

        # 监控统计
        self.monitoring_stats = {
            'cycles_completed': 0,
            'alerts_generated': 0,
            'suggestions_generated': 0,
            'errors_encountered': 0,
            'last_cycle_time': None
        }

        logger.info("监控协调器初始化完成")

    def set_components(self, metrics_collector=None, alert_processor=None,
                      optimization_suggester=None, data_manager=None):
        """
        设置子组件

        Args:
            metrics_collector: 指标收集器
            alert_processor: 告警处理器
            optimization_suggester: 优化建议器
            data_manager: 数据管理器
        """
        self.metrics_collector = metrics_collector
        self.alert_processor = alert_processor
        self.optimization_suggester = optimization_suggester
        self.data_manager = data_manager

        logger.info("监控协调器子组件设置完成")

    def start_monitoring(self) -> bool:
        """
        启动监控系统

        Returns:
            bool: 是否成功启动
        """
        if self.monitoring_active:
            logger.warning("监控系统已经在运行中")
            return False

        try:
            self.monitoring_active = True
            self.start_time = datetime.now()

            # 启动监控线程
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                name="MonitoringCoordinator",
                daemon=True
            )
            self.monitoring_thread.start()

            logger.info("监控系统已启动")
            return True

        except Exception as e:
            logger.error(f"启动监控系统失败: {e}")
            self.monitoring_active = False
            return False

    def stop_monitoring(self) -> bool:
        """
        停止监控系统

        Returns:
            bool: 是否成功停止
        """
        if not self.monitoring_active:
            logger.warning("监控系统未在运行")
            return False

        try:
            self.monitoring_active = False

            # 等待监控线程结束
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=10)

            logger.info("监控系统已停止")
            return True

        except Exception as e:
            logger.error(f"停止监控系统失败: {e}")
            return False

    def _monitoring_loop(self):
        """监控主循环"""
        logger.info("监控循环开始")

        while self.monitoring_active:
            try:
                cycle_start = time.time()

                # 执行监控周期
                self._perform_monitoring_cycle()

                # 更新统计
                self.monitoring_stats['cycles_completed'] += 1
                self.monitoring_stats['last_cycle_time'] = datetime.now()

                cycle_duration = time.time() - cycle_start
                logger.debug(".2f")

                # 等待下一个周期
                interval = self.config.get('interval_seconds', 300)
                time.sleep(min(interval, 300))  # 最多等待5分钟

            except Exception as e:
                self.monitoring_stats['errors_encountered'] += 1
                logger.error(f"监控循环异常: {e}")
                time.sleep(60)  # 出错时等待1分钟后重试

        logger.info("监控循环结束")

    def _perform_monitoring_cycle(self):
        """执行监控周期"""
        try:
            # 1. 收集指标
            if self.metrics_collector:
                metrics = self.metrics_collector.collect_all_metrics()
            else:
                metrics = {}
                logger.warning("指标收集器未设置")

            # 2. 处理告警
            if self.alert_processor:
                alerts = self.alert_processor.process_alerts(metrics)
                self.monitoring_stats['alerts_generated'] += len(alerts) if alerts else 0
            else:
                alerts = []
                logger.warning("告警处理器未设置")

            # 3. 生成优化建议
            if self.optimization_suggester:
                suggestions = self.optimization_suggester.generate_suggestions(metrics)
                self.monitoring_stats['suggestions_generated'] += len(suggestions) if suggestions else 0
            else:
                suggestions = []
                logger.warning("优化建议器未设置")

            # 4. 保存数据
            if self.data_manager:
                monitoring_data = {
                    'timestamp': datetime.now(),
                    'metrics': metrics,
                    'alerts': alerts,
                    'suggestions': suggestions,
                    'cycle_stats': self.monitoring_stats.copy()
                }
                self.data_manager.save_monitoring_data(monitoring_data)
            else:
                logger.warning("数据管理器未设置")

            logger.info(f"监控周期完成 - 指标:{len(metrics)}, 告警:{len(alerts)}, 建议:{len(suggestions)}")

        except Exception as e:
            logger.error(f"执行监控周期失败: {e}")
            self.monitoring_stats['errors_encountered'] += 1

    def update_config(self, new_config: Dict[str, Any]):
        """
        更新监控配置

        Args:
            new_config: 新配置
        """
        self.config.update(new_config)
        logger.info(f"监控配置已更新: {new_config}")

    def get_monitoring_status(self) -> Dict[str, Any]:
        """
        获取监控状态

        Returns:
            Dict[str, Any]: 监控状态信息
        """
        return {
            'active': self.monitoring_active,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            'config': self.config,
            'stats': self.monitoring_stats,
            'components_status': {
                'metrics_collector': self.metrics_collector is not None,
                'alert_processor': self.alert_processor is not None,
                'optimization_suggester': self.optimization_suggester is not None,
                'data_manager': self.data_manager is not None
            }
        }

    def get_health_status(self) -> Dict[str, Any]:
        """
        获取健康状态

        Returns:
            Dict[str, Any]: 健康状态信息
        """
        status = self.get_monitoring_status()

        # 检查健康条件
        issues = []

        if not status['active']:
            issues.append("监控系统未运行")

        if status['stats']['errors_encountered'] > 10:
            issues.append("错误次数过多")

        if not all(status['components_status'].values()):
            missing_components = [k for k, v in status['components_status'].items() if not v]
            issues.append(f"缺少组件: {', '.join(missing_components)}")

        # 计算健康评分
        health_score = 100
        if issues:
            health_score -= len(issues) * 20
            health_score = max(0, health_score)

        return {
            'status': 'healthy' if health_score >= 80 else 'warning' if health_score >= 50 else 'error',
            'health_score': health_score,
            'issues': issues,
            'details': status
        }

    def reset_stats(self):
        """重置监控统计"""
        self.monitoring_stats = {
            'cycles_completed': 0,
            'alerts_generated': 0,
            'suggestions_generated': 0,
            'errors_encountered': 0,
            'last_cycle_time': None
        }
        logger.info("监控统计已重置")

    def force_monitoring_cycle(self) -> Dict[str, Any]:
        """
        强制执行一次监控周期

        Returns:
            Dict[str, Any]: 执行结果
        """
        try:
            logger.info("强制执行监控周期")
            self._perform_monitoring_cycle()
            return {'success': True, 'message': '监控周期执行完成'}
        except Exception as e:
            logger.error(f"强制执行监控周期失败: {e}")
            return {'success': False, 'error': str(e)}


# 全局监控协调器实例
global_monitoring_coordinator = MonitoringCoordinator()
