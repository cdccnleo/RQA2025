#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RQA2025 测试进度监控工具
实时跟踪测试执行进度和异常情况
"""

import time
from datetime import datetime
from typing import Dict, List
import pandas as pd
from dataclasses import dataclass
import smtplib
from email.mime.text import MIMEText
import socket

@dataclass
class TestCaseStatus:
    name: str
    test_type: str
    status: str  # 'pending', 'running', 'passed', 'failed'
    start_time: datetime
    end_time: datetime = None
    duration: float = None
    error_msg: str = None

class TestMonitor:
    def __init__(self, alert_config=None):
        """
        初始化测试监控器
        :param alert_config: 告警配置字典
        """
        self.test_cases: Dict[str, TestCaseStatus] = {}
        self.execution_history: List[TestCaseStatus] = []
        self.alert_config = alert_config or {
            "email": {
                "enabled": False,
                "smtp_server": "smtp.example.com",
                "smtp_port": 587,
                "username": "user@example.com",
                "password": "password",
                "recipients": ["team@example.com"]
            },
            "slack": {
                "enabled": False,
                "webhook_url": None
            }
        }

    def start_test_case(self, name: str, test_type: str):
        """记录测试用例开始"""
        if name in self.test_cases:
            raise ValueError(f"测试用例 '{name}' 已存在")

        test_case = TestCaseStatus(
            name=name,
            test_type=test_type,
            status="running",
            start_time=datetime.now()
        )
        self.test_cases[name] = test_case

    def end_test_case(self, name: str, status: str, error_msg: str = None):
        """记录测试用例结束"""
        if name not in self.test_cases:
            raise ValueError(f"测试用例 '{name}' 不存在")

        test_case = self.test_cases[name]
        test_case.status = status
        test_case.end_time = datetime.now()
        test_case.duration = (test_case.end_time - test_case.start_time).total_seconds()
        test_case.error_msg = error_msg

        # 添加到执行历史
        self.execution_history.append(test_case)

        # 检查是否需要发送告警
        if status == "failed":
            self._check_and_alert(test_case)

        # 从当前测试中移除
        del self.test_cases[name]

    def get_progress(self) -> Dict:
        """获取当前测试进度"""
        total = len(self.execution_history) + len(self.test_cases)
        if total == 0:
            return {
                "total": 0,
                "completed": 0,
                "running": 0,
                "passed": 0,
                "failed": 0,
                "progress": 0
            }

        completed = len(self.execution_history)
        running = len(self.test_cases)
        passed = sum(1 for tc in self.execution_history if tc.status == "passed")
        failed = completed - passed

        return {
            "total": total,
            "completed": completed,
            "running": running,
            "passed": passed,
            "failed": failed,
            "progress": round(completed / total * 100, 2)
        }

    def get_summary(self) -> pd.DataFrame:
        """获取测试执行摘要"""
        data = []
        for test_case in self.execution_history:
            data.append({
                "name": test_case.name,
                "type": test_case.test_type,
                "status": test_case.status,
                "start_time": test_case.start_time,
                "duration": test_case.duration,
                "error": test_case.error_msg
            })

        return pd.DataFrame(data)

    def get_failed_cases(self) -> List[TestCaseStatus]:
        """获取失败的测试用例"""
        return [tc for tc in self.execution_history if tc.status == "failed"]

    def _check_and_alert(self, test_case: TestCaseStatus):
        """检查并发送告警"""
        if self.alert_config["email"]["enabled"]:
            self._send_email_alert(test_case)

        if self.alert_config["slack"]["enabled"]:
            self._send_slack_alert(test_case)

    def _send_email_alert(self, test_case: TestCaseStatus):
        """发送邮件告警"""
        try:
            msg = MIMEText(
                f"""
                测试用例执行失败告警
                
                测试名称: {test_case.name}
                测试类型: {test_case.test_type}
                执行时间: {test_case.start_time}
                持续时间: {test_case.duration}秒
                错误信息:
                {test_case.error_msg}
                
                主机: {socket.gethostname()}
                """
            )

            msg["Subject"] = f"[RQA2025告警] 测试失败: {test_case.name}"
            msg["From"] = self.alert_config["email"]["username"]
            msg["To"] = ", ".join(self.alert_config["email"]["recipients"])

            with smtplib.SMTP(
                self.alert_config["email"]["smtp_server"],
                self.alert_config["email"]["smtp_port"]
            ) as server:
                server.starttls()
                server.login(
                    self.alert_config["email"]["username"],
                    self.alert_config["email"]["password"]
                )
                server.send_message(msg)

            print(f"📧 已发送邮件告警: {test_case.name}")
        except Exception as e:
            print(f"❌ 发送邮件告警失败: {str(e)}")

    def _send_slack_alert(self, test_case: TestCaseStatus):
        """发送Slack告警"""
        # 实际实现需要Slack Webhook集成
        print(f"⚠️ Slack告警(模拟): 测试失败 - {test_case.name}")

    def generate_report(self) -> str:
        """生成测试报告"""
        progress = self.get_progress()
        failed_cases = self.get_failed_cases()

        report = f"""
        RQA2025 测试执行报告
        =====================
        
        测试进度: {progress['progress']}%
        - 总计: {progress['total']}
        - 已完成: {progress['completed']}
        - 进行中: {progress['running']}
        - 通过: {progress['passed']}
        - 失败: {progress['failed']}
        
        """

        if failed_cases:
            report += "失败用例:\n"
            for case in failed_cases:
                report += f"- {case.name} ({case.test_type}): {case.error_msg}\n"

        return report


if __name__ == "__main__":
    # 示例用法
    monitor = TestMonitor()

    # 模拟测试执行
    test_cases = [
        ("熔断机制测试", "unit"),
        ("FPGA一致性测试", "unit"),
        ("交易全流程测试", "integration"),
        ("性能压力测试", "performance")
    ]

    for name, test_type in test_cases:
        monitor.start_test_case(name, test_type)
        print(f"开始测试: {name}")

        # 模拟测试执行时间
        time.sleep(1)

        # 随机设置测试结果
        import random
        if random.random() > 0.3:  # 70%通过率
            monitor.end_test_case(name, "passed")
            print(f"测试通过: {name}")
        else:
            error_msg = random.choice([
                "熔断阈值计算错误",
                "FPGA计算结果不一致",
                "订单执行超时",
                "性能不达标"
            ])
            monitor.end_test_case(name, "failed", error_msg)
            print(f"测试失败: {name} - {error_msg}")

    # 生成报告
    print("\n" + monitor.generate_report())
