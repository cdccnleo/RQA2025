#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
监控告警模块
负责系统监控数据的收集、分析和告警通知
"""

import time
import smtplib
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from src.infrastructure.utils.logger import get_logger
from src.infrastructure.config.config_manager import ConfigManager

logger = get_logger(__name__)

@dataclass
class AlertThreshold:
    """告警阈值配置"""
    metric_name: str
    warning: float
    critical: float
    duration: int  # 持续时间(秒)

@dataclass
class AlertRule:
    """告警规则"""
    name: str
    condition: str  # 条件表达式
    severity: str   # warning/critical
    notify_channels: List[str]  # 通知渠道
    cooldown: int    # 冷却时间(秒)

class AlertManager:
    def __init__(self, config: Dict[str, Any], config_manager: Optional[ConfigManager] = None):
        """
        初始化告警管理器
        :param config: 系统配置
        :param config_manager: 可选的配置管理器实例，用于测试时注入mock对象
        """
        self.config = config
        
        # 测试钩子：允许注入mock的ConfigManager
        if config_manager is not None:
            self.config_manager = config_manager
        else:
            self.config_manager = ConfigManager(config)
            
        self.alert_rules: List[AlertRule] = []
        self.alert_history: Dict[str, float] = {}  # 告警历史记录
        self.load_alert_rules()

        # 通知渠道配置
        self.notifiers = {
            'email': self._send_email,
            'sms': self._send_sms,
            'wechat': self._send_wechat,
            'slack': self._send_slack
        }

    def load_alert_rules(self) -> None:
        """加载告警规则配置"""
        try:
            rules_config = self.config_manager.get_config('alert_rules', [])
            for rule_config in rules_config:
                self.alert_rules.append(AlertRule(
                    name=rule_config['name'],
                    condition=rule_config['condition'],
                    severity=rule_config.get('severity', 'warning'),
                    notify_channels=rule_config.get('notify_channels', ['email']),
                    cooldown=rule_config.get('cooldown', 300)
                ))
            logger.info(f"加载了 {len(self.alert_rules)} 条告警规则")
        except Exception as e:
            logger.error(f"加载告警规则失败: {str(e)}")

    def check_metrics(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        检查指标数据并触发告警
        :param metrics: 指标数据字典
        :return: 触发的告警列表
        """
        triggered_alerts = []

        for rule in self.alert_rules:
            try:
                # 评估条件表达式
                if self._evaluate_condition(rule.condition, metrics):
                    # 检查冷却时间
                    last_alert_time = self.alert_history.get(rule.name, 0)
                    if time.time() - last_alert_time > rule.cooldown:
                        # 触发告警
                        alert = {
                            'rule': rule.name,
                            'severity': rule.severity,
                            'metrics': metrics,
                            'timestamp': time.time()
                        }
                        triggered_alerts.append(alert)

                        # 发送通知
                        self.notify(alert, rule.notify_channels)

                        # 更新历史记录
                        self.alert_history[rule.name] = time.time()
            except Exception as e:
                logger.error(f"检查告警规则 {rule.name} 失败: {str(e)}")

        return triggered_alerts

    def _evaluate_condition(self, condition: str, metrics: Dict[str, float]) -> bool:
        """
        评估条件表达式
        :param condition: 条件表达式字符串
        :param metrics: 指标数据
        :return: 是否满足条件
        """
        # 简单的表达式评估实现
        # 实际项目中应该使用更安全的表达式解析器
        try:
            # 替换指标变量
            for metric_name, value in metrics.items():
                condition = condition.replace(metric_name, str(value))

            # 评估表达式
            return eval(condition)
        except Exception as e:
            logger.error(f"评估条件表达式失败: {condition}, 错误: {str(e)}")
            return False

    def notify(self, alert: Dict[str, Any], channels: List[str]) -> bool:
        """
        发送告警通知
        :param alert: 告警信息
        :param channels: 通知渠道列表
        :return: 是否发送成功
        """
        success = True
        message = self._format_alert_message(alert)

        for channel in channels:
            if channel in self.notifiers:
                try:
                    self.notifiers[channel](message, alert['severity'])
                    logger.info(f"通过 {channel} 发送告警通知成功: {alert['rule']}")
                except Exception as e:
                    logger.error(f"通过 {channel} 发送告警通知失败: {str(e)}")
                    success = False
            else:
                logger.warning(f"未知的通知渠道: {channel}")
                success = False

        return success

    def _format_alert_message(self, alert: Dict[str, Any]) -> str:
        """格式化告警消息"""
        return f"""
        [告警] {alert['rule']}
        严重程度: {alert['severity']}
        触发时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert['timestamp']))}
        相关指标:
        {self._format_metrics(alert['metrics'])}
        """

    def _format_metrics(self, metrics: Dict[str, float]) -> str:
        """格式化指标数据"""
        return "\n".join([f"- {k}: {v}" for k, v in metrics.items()])

    def _send_email(self, message: str, severity: str) -> None:
        """发送邮件通知"""
        smtp_config = self.config_manager.get_config('smtp', {})

        with smtplib.SMTP(smtp_config.get('host', 'localhost'),
                         smtp_config.get('port', 25)) as server:
            server.login(smtp_config.get('user', ''),
                        smtp_config.get('password', ''))

            subject = f"[{severity.upper()}] 系统告警通知"
            msg = f"Subject: {subject}\n\n{message}"

            server.sendmail(
                smtp_config.get('from', 'alerts@rqa2025.com'),
                smtp_config.get('to', 'admin@rqa2025.com'),
                msg.encode('utf-8')
            )

    def _send_sms(self, message: str, severity: str) -> None:
        """发送短信通知"""
        sms_config = self.config_manager.get_config('sms', {})
        # 实际项目中应集成短信网关API
        logger.info(f"模拟发送短信到 {sms_config.get('phone', '')}: {message[:50]}...")

    def _send_wechat(self, message: str, severity: str) -> None:
        """发送企业微信通知"""
        wechat_config = self.config_manager.get_config('wechat', {})
        # 实际项目中应调用企业微信API
        logger.info(f"模拟发送企业微信通知到 {wechat_config.get('group', '')}: {message[:50]}...")

    def _send_slack(self, message: str, severity: str) -> None:
        """发送Slack通知"""
        slack_config = self.config_manager.get_config('slack', {})
        # 实际项目中应调用Slack Webhook
        logger.info(f"模拟发送Slack通知到 {slack_config.get('channel', '')}: {message[:50]}...")

    def start_monitoring(self, interval: int = 60) -> None:
        """
        启动监控循环
        :param interval: 监控间隔(秒)
        """
        logger.info(f"启动监控循环，间隔: {interval}秒")
        while True:
            try:
                # 获取最新指标数据
                metrics = self._collect_metrics()

                # 检查告警
                self.check_metrics(metrics)

                # 等待下一个周期
                time.sleep(interval)
            except Exception as e:
                logger.error(f"监控循环出错: {str(e)}")
                time.sleep(5)  # 出错后短暂等待

    def _collect_metrics(self) -> Dict[str, float]:
        """
        收集系统指标数据
        :return: 指标数据字典
        """
        # 实际项目中应从监控系统获取实时数据
        return {
            'cpu_usage': 65.2,
            'memory_usage': 78.5,
            'disk_usage': 45.7,
            'network_latency': 12.3,
            'order_queue_size': 125,
            'risk_check_latency': 5.7,
            'fpga_temperature': 48.2
        }

    def add_alert_rule(self, rule: AlertRule) -> None:
        """
        添加告警规则
        :param rule: 告警规则对象
        """
        self.alert_rules.append(rule)
        logger.info(f"添加告警规则: {rule.name}")

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """
        获取当前活跃告警
        :return: 活跃告警列表
        """
        # 实际项目中应从数据库或缓存中获取
        return []
