# 数据层功能增强分析报告（第三部分）

## 功能实现建议（续）

### 3. 监控告警（续）

#### 3.2 异常告警（续）

```python
class AlertManager:
    """告警管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化告警管理器
        
        Args:
            config: 配置信息，包含告警阈值、通知方式等
        """
        self.config = config
        self.alert_history = []
        self.alert_count = 0
        self.lock = threading.Lock()
        
        # 告警阈值配置
        self.thresholds = config.get('thresholds', {})
        
        # 通知配置
        self.notification_config = config.get('notification', {})
        
        # 告警级别
        self.alert_levels = {
            'info': 0,
            'warning': 1,
            'error': 2,
            'critical': 3
        }
        
        # 最小告警级别
        self.min_alert_level = self.alert_levels.get(
            config.get('min_alert_level', 'warning'),
            1  # 默认为warning级别
        )
    
    def check_threshold(
        self,
        metric_name: str,
        value: Union[int, float],
        level: str = 'warning',
        custom_message: Optional[str] = None
    ) -> bool:
        """
        检查指标是否超过阈值
        
        Args:
            metric_name: 指标名称
            value: 指标值
            level: 告警级别
            custom_message: 自定义告警消息
            
        Returns:
            bool: 是否触发告警
        """
        if metric_name not in self.thresholds:
            return False
        
        threshold = self.thresholds[metric_name]
        triggered = False
        
        if 'min' in threshold and value < threshold['min']:
            message = custom_message or f"{metric_name} is below minimum threshold: {value} < {threshold['min']}"
            triggered = True
        elif 'max' in threshold and value > threshold['max']:
            message = custom_message or f"{metric_name} exceeds maximum threshold: {value} > {threshold['max']}"
            triggered = True
        
        if triggered:
            self.alert(level, message, {
                'metric_name': metric_name,
                'value': value,
                'threshold': threshold
            })
        
        return triggered
    
    def alert(
        self,
        level: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        发送告警
        
        Args:
            level: 告警级别
            message: 告警消息
            details: 告警详情
            
        Returns:
            bool: 是否成功发送告警
        """
        # 检查告警级别
        alert_level = self.alert_levels.get(level.lower(), 0)
        if alert_level < self.min_alert_level:
            return False
        
        # 创建告警记录
        alert = {
            'id': self.alert_count,
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            'details': details or {}
        }
        
        # 记录告警
        with self.lock:
            self.alert_history.append(alert)
            self.alert_count += 1
        
        # 记录日志
        log_method = getattr(logger, level.lower(), logger.warning)
        log_method(f"ALERT: {message}")
        
        # 发送通知
        self._send_notification(alert)
        
        return True
    
    def _send_notification(self, alert: Dict[str, Any]) -> bool:
        """
        发送通知
        
        Args:
            alert: 告警信息
            
        Returns:
            bool: 是否成功发送通知
        """
        if not self.notification_config:
            return False
        
        # 检查告警级别是否需要通知
        alert_level = self.alert_levels.get(alert['level'].lower(), 0)
        notify_level = self.alert_levels.get(
            self.notification_config.get('min_level', 'error'),
            2  # 默认为error级别
        )
        
        if alert_level < notify_level:
            return False
        
        # 根据配置的通知方式发送通知
        methods = self.notification_config.get('methods', [])
        success = False
        
        for method in methods:
            if method['type'] == 'email':
                success = self._send_email_notification(alert, method) or success
            elif method['type'] == 'webhook':
                success = self._send_webhook_notification(alert, method) or success
            elif method['type'] == 'log':
                success = self._send_log_notification(alert, method) or success
        
        return success
    
    def _send_email_notification(
        self,
        alert: Dict[str, Any],
        config: Dict[str, Any]
    ) -> bool:
        """
        发送邮件通知
        
        Args:
            alert: 告警信息
            config: 邮件配置
            
        Returns:
            bool: 是否成功发送
        """
        try:
            smtp_server = config.get('smtp_server')
            smtp_port = config.get('smtp_port', 587)
            sender = config.get('sender')
            recipients = config.get('recipients', [])
            username = config.get('username')
            password = config.get('password')
            
            if not (smtp_server and sender and recipients):
                logger.error("Missing required email configuration")
                return False
            
            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = sender
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"[{alert['level'].upper()}] Data System Alert: {alert['message'][:50]}..."
            
            # 邮件内容
            body = f"""
            <html>
            <body>
                <h2>Data System Alert</h2>
                <p><strong>Level:</strong> {alert['level'].upper()}</p>
                <p><strong>Time:</strong> {alert['timestamp']}</p>
                <p><strong>Message:</strong> {alert['message']}</p>
                <h3>Details:</h3>
                <pre>{json.dumps(alert['details'], indent=2)}</pre>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            # 发送邮件
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                if username and password:
                    server.login(username, password)
                server.send_message(msg)
            
            return True
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False
    
    def _send_webhook_notification(
        self,
        alert: Dict[str, Any],
        config: Dict[str, Any]
    ) -> bool:
        """
        发送Webhook通知
        
        Args:
            alert: 告警信息
            config: Webhook配置
            
        Returns:
            bool: 是否成功发送
        """
        try:
            url = config.get('url')
            headers = config.get('headers', {})
            
            if not url:
                logger.error("Missing webhook URL")
                return False
            
            # 准备请求数据
            payload = {
                'alert': alert,
                'system': 'data_system',
                'timestamp': datetime.now().isoformat()
            }
            
            # 发送请求
            response = requests.post(url, json=payload, headers=headers, timeout=5)
            response.raise_for_status()
            
            return True
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            return False
    
    def _send_log_notification(
        self,
        alert: Dict[str, Any],
        config: Dict[str, Any]
    ) -> bool:
        """
        发送日志通知
        
        Args:
            alert: 告警信息
            config: 日志配置
            
        Returns:
            bool: 是否成功发送
        """
        try:
            log_file = config.get('file')
            
            if not log_file:
                logger.error("Missing log file path")
                return False
            
            # 写入日志文件
            with open(log_file, 'a') as f:
                f.write(f"{alert['timestamp']} [{alert['level'].upper()}] {alert['message']}\n")
                if alert['details']:
                    f.write(f"Details: {json.dumps(alert['details'])}\n")
                f.write("\n")
            
            return True
        except Exception as e:
            logger.error(f"Failed to send log notification: {e}")
            return False
    
    def get_alerts(
        self,
        level: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        获取告警历史
        
        Args:
            level: 告警级别
            start_time: 开始时间
            end_time: 结束时间
            limit: 返回数量限制
            
        Returns:
            List[Dict[str, Any]]: 告警历史
        """
        with self.lock:
            alerts = self.alert_history.copy()
        
        # 按级别筛选
        if level:
            level_value = self.alert_levels.get(level.lower(), -1)
            alerts = [a for a in alerts if self.alert_levels.get(a['level'].lower(), 0) >= level_value]
        
        # 按时间筛选
        if start_time:
            alerts = [a for a in alerts if a['timestamp'] >= start_time]
        
        if end_time:
            alerts = [a for a in alerts if a['timestamp'] <= end_time]
        
        # 按时间排序
        alerts.sort(key=lambda a: a['timestamp'], reverse=True)
        
        # 限制返回数量
        if limit and limit > 0:
            alerts = alerts[:limit]
        
        return alerts
    
    def clear_alerts(self):
        """清除所有告警历史"""
        with self.lock:
            self.alert_history = []
```

在 `DataManager` 中集成异常告警功能：

```python
def __init__(self, config: Dict[str, Any]):
    # ... 其他初始化代码 ...
    
    # 初始化告警管理器
    alert_config = config.get('alert_config', {})
    self.alert_manager = AlertManager(alert_config)

def check_data_thresholds(self, data_model: Optional[DataModel] = None) -> List[Dict[str, Any]]:
    """
    检查数据是否超过阈值
    
    Args:
        data_model: 数据模型，默认为当前模型
        
    Returns:
        List[Dict[str, Any]]: 触发的告警列表
    """
    if data_model is None:
        data_model = self.current_model
    
    if data_model is None:
        raise ValueError("No data model available")
    
    # 获取数据质量报告
    quality_report = self.check_data_quality(data_model)
    
    # 检查缺失值比例
    triggered_alerts = []
    for column, missing_ratio in quality_report['missing_values'].items():
        if self.alert_manager.check_threshold(
            f"missing_values.{column}",
            missing_ratio,
            level='warning',
            custom_message=f"Column '{column}' has high missing value ratio: {missing_ratio:.2%}"
        ):
            triggered_alerts.append({
                'metric': f"missing_values.{column}",
                'value': missing_ratio,
                'level': 'warning'
            })
    
    # 检查重复值比例
    duplicate_ratio = quality_report['duplicates']['duplicate_ratio']
    if self.alert_manager.check_threshold(
        "duplicates.ratio",
        duplicate_ratio,
        level='warning',
        custom_message=f"Data has high duplicate ratio: {duplicate_ratio:.2%}"
    ):
        triggered_alerts.append({
            'metric': "duplicates.ratio",
            'value': duplicate_ratio,
            'level': 'warning'
        })
    
    # 检查异常值比例
    if 'outliers' in quality_report:
        for column, outlier_info in quality_report['outliers'].items():
            outlier_ratio = outlier_info['outlier_ratio']
            if self.alert_manager.check_threshold(
                f"outliers.{column}",
                outlier_ratio,
                level='warning',
                custom_message=f"Column '{column}' has high outlier ratio: {outlier_ratio:.2%}"
            ):
                triggered_alerts.append({
                    'metric': f"outliers.{column}",
                    'value': outlier_ratio,
                    'level': 'warning'
                })
    
    return triggered_alerts

def alert(self, level: str, message: str, details: Optional[Dict[str, Any]] = None) -> bool:
    """
    发送告警
    
    Args:
        level: 告警级别
        message: 告警消息
        details: 告警详情
        
    Returns:
        bool: 是否成功发送告警
    """
    return self.alert_manager.alert(level, message, details)

def get_alerts(
    self,
    level: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    获取告警历史
    
    Args:
        level: 告警级别
        start_time: 开始时间
        end_time: 结束时间
        limit: 返回数量限制
        
    Returns:
        List[Dict[str, Any]]: 告警历史
    """
    return self.alert_manager.get_alerts(level, start_time, end_time, limit)
```

#### 3.3 数据质量报告

建议实现一个 `DataQualityReporter` 类，用于生成数据质量报告：

```python
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os