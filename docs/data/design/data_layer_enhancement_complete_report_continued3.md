# RQA2025 数据层功能增强完整报告（续3）

## 2. 功能分析（续）

### 2.3 监控告警（续）

#### 2.3.1 异常告警（续）

**核心代码示例**（续）：
```python
    def alert(
        self,
        level: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        发送告警（续）
        """
        if level not in self.config.get('alert_levels', {}):
            raise ValueError(f"Invalid alert level: {level}")
        
        alert_info = {
            'level': level,
            'message': message,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        
        # 记录告警历史
        self.alert_history.append(alert_info)
        
        # 根据级别配置发送通知
        level_config = self.config['alert_levels'][level]
        
        # 记录日志
        log_level = getattr(logging, level.upper(), logging.WARNING)
        logger.log(log_level, f"Alert: {message}")
        
        # 发送邮件通知
        if level_config.get('email', False):
            self._send_email_alert(alert_info)
        
        # 发送Webhook通知
        if level_config.get('webhook', False):
            self._send_webhook_alert(alert_info)
    
    def _send_email_alert(self, alert_info: Dict[str, Any]) -> None:
        """
        发送邮件告警
        
        Args:
            alert_info: 告警信息
        """
        email_config = self.config.get('email_config', {})
        if not email_config:
            logger.warning("Email configuration not found")
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = email_config['from']
            msg['To'] = ', '.join(email_config['to'])
            msg['Subject'] = f"[{alert_info['level'].upper()}] Alert: {alert_info['message']}"
            
            # 构建邮件内容
            body = f"""
            Alert Level: {alert_info['level']}
            Message: {alert_info['message']}
            Time: {alert_info['timestamp']}
            """
            
            if alert_info['details']:
                body += f"\nDetails:\n{json.dumps(alert_info['details'], indent=2)}"
            
            msg.attach(MIMEText(body, 'plain'))
            
            # 发送邮件
            with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
                if email_config.get('use_tls', False):
                    server.starttls()
                if 'username' in email_config and 'password' in email_config:
                    server.login(email_config['username'], email_config['password'])
                server.send_message(msg)
            
            logger.info("Email alert sent successfully")
        
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def _send_webhook_alert(self, alert_info: Dict[str, Any]) -> None:
        """
        发送Webhook告警
        
        Args:
            alert_info: 告警信息
        """
        webhook_config = self.config.get('webhook_config', {})
        if not webhook_config:
            logger.warning("Webhook configuration not found")
            return
        
        try:
            response = requests.post(
                webhook_config['url'],
                json=alert_info,
                headers=webhook_config.get('headers', {}),
                timeout=webhook_config.get('timeout', 5)
            )
            response.raise_for_status()
            logger.info("Webhook alert sent successfully")
        
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
    
    def get_alerts(
        self,
        level: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        获取告警历史
        
        Args:
            level: 告警级别过滤
            start_time: 开始时间（ISO格式）
            end_time: 结束时间（ISO格式）
            
        Returns:
            List[Dict[str, Any]]: 告警历史列表
        """
        alerts = self.alert_history
        
        if level:
            alerts = [a for a in alerts if a['level'] == level]
        
        if start_time:
            alerts = [a for a in alerts if a['timestamp'] >= start_time]
        
        if end_time:
            alerts = [a for a in alerts if a['timestamp'] <= end_time]
        
        return alerts
```

#### 2.3.2 性能监控

**现状分析**：
缺乏对数据加载和处理性能的监控机制，无法及时发现性能瓶颈和优化机会。

**实现建议**：
实现一个 `PerformanceMonitor` 类，用于监控数据加载和处理性能。该类将提供以下功能：

- 函数执行时间监控：使用装饰器监控函数执行时间
- 系统资源监控：监控 CPU、内存和磁盘使用情况
- 性能指标收集：收集和存储性能指标
- 性能报告生成：生成性能指标摘要报告

**核心代码示例**：
```python
import time
import psutil
import functools
from typing import Dict, List, Any, Optional, Union, Callable
import logging
from datetime import datetime
import threading

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        """初始化性能监控器"""
        self.metrics = []
        self.lock = threading.Lock()
    
    def monitor_execution_time(self, name: Optional[str] = None) -> Callable:
        """
        监控函数执行时间的装饰器
        
        Args:
            name: 监控名称，默认为函数名
            
        Returns:
            Callable: 装饰器函数
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    success = True
                except Exception as e:
                    success = False
                    raise e
                finally:
                    end_time = time.time()
                    execution_time = end_time - start_time
                    
                    metric = {
                        'name': name or func.__name__,
                        'type': 'execution_time',
                        'value': execution_time,
                        'success': success,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    with self.lock:
                        self.metrics.append(metric)
                
                return result
            
            return wrapper
        
        return decorator
    
    def monitor_system_resources(self, interval: float = 1.0) -> None:
        """
        监控系统资源使用情况
        
        Args:
            interval: 监控间隔（秒）
        """
        while True:
            try:
                # CPU使用率
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # 内存使用情况
                memory = psutil.virtual_memory()
                
                # 磁盘使用情况
                disk = psutil.disk_usage('/')
                
                metric = {
                    'type': 'system_resources',
                    'timestamp': datetime.now().isoformat(),
                    'cpu': {
                        'percent': cpu_percent
                    },
                    'memory': {
                        'total': memory.total,
                        'available': memory.available,
                        'percent': memory.percent,
                        'used': memory.used
                    },
                    'disk': {
                        'total': disk.total,
                        'used': disk.used,
                        'free': disk.free,
                        'percent': disk.percent
                    }
                }
                
                with self.lock:
                    self.metrics.append(metric)
                
            except Exception as e:
                logger.error(f"Failed to monitor system resources: {e}")
            
            time.sleep(interval)
    
    def start_resource_monitoring(self, interval: float = 1.0) -> None:
        """
        启动系统资源监控线程
        
        Args:
            interval: 监控间隔（秒）
        """
        thread = threading.Thread(
            target=self.monitor_system_resources,
            args=(interval,),
            daemon=True
        )
        thread.start()
    
    def get_metrics(
        self,
        metric_type: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        获取性能指标
        
        Args:
            metric_type: 指标类型过滤
            start_time: 开始时间（ISO格式）
            end_time: 结束时间（ISO格式）
            
        Returns:
            List[Dict[str, Any]]: 性能指标列表
        """
        with self.lock:
            metrics = self.metrics.copy()
        
        if metric_type:
            metrics = [m for m in metrics if m['type'] == metric_type]
        
        if start_time:
            metrics = [m for m in metrics if m['timestamp'] >= start_time]
        
        if end_time:
            metrics = [m for m in metrics if m['timestamp'] <= end_time]
        
        return metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """
        生成性能指标摘要报告
        
        Returns:
            Dict[str, Any]: 性能指标摘要
        """
        execution_times = [
            m for m in self.metrics
            if m['type'] == 'execution_time'
        ]
        
        system_resources = [
            m for m in self.metrics
            if m['type'] == 'system_resources'
        ]
        
        summary = {
            'execution_time': {
                'total_calls': len(execution_times),
                'success_rate': sum(1 for m in execution_times if m['success']) / len(execution_times) if execution_times else 0,
                'avg_time': sum(m['value'] for m in execution_times) / len(execution_times) if execution_times else 0,
                'min_time': min((m['value'] for m in execution_times), default=0),
                'max_time': max((m['value'] for m in execution_times), default=0)
            } if execution_times else None,
            
            'system_resources': {
                'cpu': {
                    'avg_percent': sum(m['cpu']['percent'] for m in system_resources) / len(system_resources) if system_resources else 0,
                    'max_percent': max((m['cpu']['percent'] for m in system_resources), default=0)
                },
                'memory': {
                    'avg_percent': sum(m['memory']['percent'] for m in system_resources) / len(system_resources) if system_resources else 0,
                    'max_percent': max((m['memory']['percent'] for m in system_resources), default=0)
                },
                'disk': {
                    'avg_percent': sum(m['disk']['percent'] for m in system_resources) / len(system_resources) if system_resources else 0,
                    'max_percent': max((m['disk']['percent'] for m in system_resources), default=0)
                }
            } if system_resources else None
        }
        
        return summary
```

#### 2.3.3 数据质量报告

**现状分析**：
缺乏生成数据质量报告的功能，无法直观地展示和分析数据质量状况。

**实现建议**：
实现一个 `DataQualityReporter` 类，用于生成数据质量报告。该类将提供以下功能：

- JSON报告：生成JSON格式的数据质量报告
- HTML报告：生成可视化的HTML格式报告
- Markdown报告：生成Markdown格式的数据质量报告
- 报告存储：将报告保存到指定目录

**核心代码示例**：
```python
import os
import json
import jinja2
import markdown
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DataQualityReporter:
    """数据质量报告生成器"""
    
    def __init__(self, report_dir: str = './reports'):
        """
        初始化数据质量报告生成器
        
        Args:
            report_dir: 报告目录
        """
        self.report_dir = report_dir
        os.makedirs(report_dir, exist_ok=True)
        
        # 加载HTML模板
        self.template_loader = jinja2.FileSystemLoader(searchpath="./templates")
        self.template_env = jinja2.Environment(loader=self.template_loader)
    
    def generate_report(
        self,
        quality_data: Dict[str, Any],
        format: str = 'html',
        filename: Optional[str] = None
    ) -> str:
        """
        生成数据质量报告
        
        Args:
            quality_data: 数据质量信息
            format: 报告格式（'json'、'html'、'markdown'）
            filename: 文件名，如果为None则自动生成
            
        Returns:
            str: 报告文件路径
        """
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"quality_report_{timestamp}"
        
        # 根据格式选择生成方法
        generate_methods = {
            'json': self._generate_json_report,
            'html': self._generate_html_report,
            'markdown': self._generate_markdown_report
        }
        
        if format not in generate_methods:
            raise ValueError(f"Unsupported report format: {format}")
        
        # 调用相应的生成方法
        return generate_methods[format](quality_data, filename)
    
    def _generate_json_report(
        self,
        quality_data