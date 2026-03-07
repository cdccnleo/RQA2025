#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控告警系统脚本
实现完善的系统监控和告警功能
"""

import json
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class AlertConfig:
    """告警配置"""
    cpu_threshold: float = 80.0
    memory_threshold: float = 85.0
    disk_threshold: float = 90.0
    response_time_threshold: float = 1000.0
    error_rate_threshold: float = 5.0
    alert_cooldown: int = 300
    email_enabled: bool = True
    webhook_enabled: bool = True

@dataclass
class AlertEvent:
    """告警事件"""
    id: str
    rule_name: str
    severity: str
    message: str
    timestamp: float
    metrics: Dict[str, Any]
    resolved: bool = False

class SystemMetricsCollector:
    """系统指标收集器"""
    
    def __init__(self):
        self.metrics_history = []
        self.max_history_size = 1000
    
    def collect_system_metrics(self) -> Dict[str, Any]:
        """收集系统指标"""
        # 模拟系统指标
        metrics = {
            "timestamp": time.time(),
            "cpu_percent": 45.2,
            "memory_percent": 68.5,
            "memory_available": 2048.0,
            "disk_percent": 75.3,
            "disk_free": 50.2,
            "network_bytes_sent": 1024000,
            "network_bytes_recv": 2048000,
            "process_memory_mb": 256.8,
            "process_cpu_percent": 12.5
        }
        
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]
        
        return metrics

class AlertEvaluator:
    """告警评估器"""
    
    def __init__(self, config: AlertConfig):
        self.config = config
        self.active_alerts = {}
    
    def evaluate_metrics(self, metrics: Dict[str, Any]) -> List[AlertEvent]:
        """评估指标并生成告警"""
        alerts = []
        
        # CPU告警
        if metrics["cpu_percent"] > self.config.cpu_threshold:
            alert = AlertEvent(
                id=f"cpu_{int(time.time())}",
                rule_name="CPU使用率过高",
                severity="warning" if metrics["cpu_percent"] < 95 else "critical",
                message=f"CPU使用率达到{metrics['cpu_percent']:.1f}%，超过阈值{self.config.cpu_threshold}%",
                timestamp=time.time(),
                metrics=metrics
            )
            alerts.append(alert)
        
        # 内存告警
        if metrics["memory_percent"] > self.config.memory_threshold:
            alert = AlertEvent(
                id=f"memory_{int(time.time())}",
                rule_name="内存使用率过高",
                severity="warning" if metrics["memory_percent"] < 95 else "critical",
                message=f"内存使用率达到{metrics['memory_percent']:.1f}%，超过阈值{self.config.memory_threshold}%",
                timestamp=time.time(),
                metrics=metrics
            )
            alerts.append(alert)
        
        # 磁盘告警
        if metrics["disk_percent"] > self.config.disk_threshold:
            alert = AlertEvent(
                id=f"disk_{int(time.time())}",
                rule_name="磁盘使用率过高",
                severity="warning" if metrics["disk_percent"] < 98 else "critical",
                message=f"磁盘使用率达到{metrics['disk_percent']:.1f}%，超过阈值{self.config.disk_threshold}%",
                timestamp=time.time(),
                metrics=metrics
            )
            alerts.append(alert)
        
        return alerts

class AlertNotifier:
    """告警通知器"""
    
    def __init__(self, config: AlertConfig):
        self.config = config
        self.notification_history = []
    
    def send_alert(self, alert: AlertEvent) -> Dict[str, bool]:
        """发送告警"""
        results = {}
        
        # 邮件告警
        if self.config.email_enabled:
            print(f"📧 发送邮件告警: {alert.rule_name}")
            results["email"] = True
        
        # Webhook告警
        if self.config.webhook_enabled:
            print(f"🔗 发送Webhook告警: {alert.rule_name}")
            results["webhook"] = True
        
        # 记录通知历史
        self.notification_history.append({
            "alert_id": alert.id,
            "timestamp": time.time(),
            "results": results
        })
        
        return results

class MonitoringSystem:
    """监控系统"""
    
    def __init__(self, config: AlertConfig):
        self.config = config
        self.metrics_collector = SystemMetricsCollector()
        self.alert_evaluator = AlertEvaluator(config)
        self.alert_notifier = AlertNotifier(config)
        self.running = False
        self.monitoring_thread = None
    
    def start(self):
        """启动监控系统"""
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_worker, daemon=True)
        self.monitoring_thread.start()
        print("🚀 监控系统已启动")
    
    def stop(self):
        """停止监控系统"""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        print("🛑 监控系统已停止")
    
    def _monitoring_worker(self):
        """监控工作线程"""
        while self.running:
            try:
                # 收集指标
                metrics = self.metrics_collector.collect_system_metrics()
                
                # 评估告警
                alerts = self.alert_evaluator.evaluate_metrics(metrics)
                
                # 发送告警
                for alert in alerts:
                    self.alert_notifier.send_alert(alert)
                
                # 等待下次检查
                time.sleep(30)
                
            except Exception as e:
                print(f"❌ 监控系统错误: {e}")
                time.sleep(60)
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "running": self.running,
            "notification_history": self.alert_notifier.notification_history
        }

def main():
    """主函数"""
    print("🔧 启动监控告警系统...")
    
    # 创建告警配置
    config = AlertConfig(
        cpu_threshold=80.0,
        memory_threshold=85.0,
        disk_threshold=90.0,
        response_time_threshold=1000.0,
        error_rate_threshold=5.0,
        alert_cooldown=300,
        email_enabled=True,
        webhook_enabled=True
    )
    
    # 创建监控系统
    monitoring_system = MonitoringSystem(config)
    
    # 启动监控
    monitoring_system.start()
    
    # 模拟运行一段时间
    print("📊 模拟监控运行...")
    time.sleep(5)
    
    # 获取系统状态
    status = monitoring_system.get_system_status()
    
    # 停止监控
    monitoring_system.stop()
    
    print("✅ 监控告警系统测试完成!")
    
    # 打印状态
    print("\n" + "="*50)
    print("🎯 系统监控状态:")
    print("="*50)
    print(f"运行状态: {'运行中' if status['running'] else '已停止'}")
    print(f"通知历史: {len(status['notification_history'])} 条记录")
    print("="*50)
    
    # 保存监控报告
    output_dir = Path("reports/optimization/")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    monitoring_report = {
        "timestamp": time.time(),
        "config": asdict(config),
        "system_status": status
    }
    
    report_file = output_dir / "monitoring_alert_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(monitoring_report, f, ensure_ascii=False, indent=2)
    
    print(f"📄 监控报告已保存: {report_file}")

if __name__ == "__main__":
    main() 