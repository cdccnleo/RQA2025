#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
风险控制层PostgreSQL持久化测试

测试内容:
1. RiskCheckPersistence 测试
2. AlertPersistence 测试
3. RiskMetricPersistence 测试
4. RiskRulePersistence 测试
5. 性能测试
"""

import os
import sys
import json
import time
import tempfile
import shutil
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestRiskCheckPersistence(unittest.TestCase):
    """风险检查持久化测试"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_dir = os.path.join(self.temp_dir, "checks")
        os.makedirs(self.storage_dir, exist_ok=True)
    
    def tearDown(self):
        """测试后清理"""
        try:
            shutil.rmtree(self.temp_dir)
        except PermissionError:
            pass
    
    def test_check_save_and_load(self):
        """测试风险检查保存和加载"""
        from src.risk.persistence.risk_persistence import RiskCheckPersistence, RiskCheckData
        
        persistence = RiskCheckPersistence(storage_dir=self.storage_dir)
        
        self.assertFalse(persistence._use_postgresql)
        
        check = RiskCheckData(
            check_id="test_check_001",
            check_type="position",
            risk_level="medium",
            passed=False,
            score=0.45,
            symbol="000001.SZ",
            account_id="test_account",
            details={"position_ratio": 0.85},
            recommendations=["降低仓位至80%以下"]
        )
        
        result = persistence.save_check(check)
        self.assertTrue(result)
        
        file_path = os.path.join(self.storage_dir, "test_check_001.json")
        self.assertTrue(os.path.exists(file_path))
        
        loaded_check = persistence.get_check("test_check_001")
        self.assertIsNotNone(loaded_check)
        self.assertEqual(loaded_check.check_type, "position")
        self.assertEqual(loaded_check.risk_level, "medium")
        self.assertEqual(loaded_check.score, 0.45)
        self.assertEqual(loaded_check.symbol, "000001.SZ")
    
    def test_get_checks_by_symbol(self):
        """测试按标的获取风险检查记录"""
        from src.risk.persistence.risk_persistence import RiskCheckPersistence, RiskCheckData
        
        persistence = RiskCheckPersistence(storage_dir=self.storage_dir)
        
        for i in range(5):
            check = RiskCheckData(
                check_id=f"test_check_{i:03d}",
                check_type="market",
                risk_level="low" if i < 3 else "high",
                passed=i < 3,
                score=0.1 * (i + 1),
                symbol="000001.SZ" if i < 3 else "000002.SZ"
            )
            persistence.save_check(check)
        
        checks = persistence.get_checks_by_symbol("000001.SZ", limit=10)
        self.assertEqual(len(checks), 3)
        
        checks_002 = persistence.get_checks_by_symbol("000002.SZ", limit=10)
        self.assertEqual(len(checks_002), 2)
    
    def test_get_checks_by_level(self):
        """测试按风险等级获取检查记录"""
        from src.risk.persistence.risk_persistence import RiskCheckPersistence, RiskCheckData
        
        persistence = RiskCheckPersistence(storage_dir=self.storage_dir)
        
        levels = ["low", "medium", "high", "critical"]
        for i, level in enumerate(levels * 2):
            check = RiskCheckData(
                check_id=f"test_level_{i:03d}",
                check_type="position",
                risk_level=level,
                passed=level == "low"
            )
            persistence.save_check(check)
        
        high_checks = persistence.get_checks_by_level("high", limit=10)
        self.assertEqual(len(high_checks), 2)
        
        critical_checks = persistence.get_checks_by_level("critical", limit=10)
        self.assertEqual(len(critical_checks), 2)


class TestAlertPersistence(unittest.TestCase):
    """告警持久化测试"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_dir = os.path.join(self.temp_dir, "alerts")
        os.makedirs(self.storage_dir, exist_ok=True)
    
    def tearDown(self):
        """测试后清理"""
        try:
            shutil.rmtree(self.temp_dir)
        except PermissionError:
            pass
    
    def test_alert_save_and_load(self):
        """测试告警保存和加载"""
        from src.risk.persistence.risk_persistence import AlertPersistence, AlertData
        
        persistence = AlertPersistence(storage_dir=self.storage_dir)
        
        alert = AlertData(
            alert_id="test_alert_001",
            alert_type="risk_threshold",
            alert_level="warning",
            title="风险阈值告警",
            message="VaR超过预警阈值",
            status="active",
            rule_id="rule_var_limit",
            details={"var_value": 0.085, "threshold": 0.08}
        )
        
        result = persistence.save_alert(alert)
        self.assertTrue(result)
        
        loaded_alert = persistence.get_alert("test_alert_001")
        self.assertIsNotNone(loaded_alert)
        self.assertEqual(loaded_alert.alert_type, "risk_threshold")
        self.assertEqual(loaded_alert.alert_level, "warning")
        self.assertEqual(loaded_alert.status, "active")
    
    def test_alert_update(self):
        """测试告警更新"""
        from src.risk.persistence.risk_persistence import AlertPersistence, AlertData
        
        persistence = AlertPersistence(storage_dir=self.storage_dir)
        
        alert = AlertData(
            alert_id="test_alert_002",
            alert_type="position_limit",
            alert_level="error",
            title="持仓超限告警",
            message="持仓超过限额",
            status="active"
        )
        
        persistence.save_alert(alert)
        
        result = persistence.update_alert("test_alert_002", {
            "status": "acknowledged",
            "acknowledged_by": "admin",
            "acknowledged_at": datetime.now()
        })
        self.assertTrue(result)
        
        updated_alert = persistence.get_alert("test_alert_002")
        self.assertEqual(updated_alert.status, "acknowledged")
        self.assertEqual(updated_alert.acknowledged_by, "admin")
    
    def test_get_active_alerts(self):
        """测试获取活跃告警"""
        from src.risk.persistence.risk_persistence import AlertPersistence, AlertData
        
        persistence = AlertPersistence(storage_dir=self.storage_dir)
        
        for i in range(5):
            alert = AlertData(
                alert_id=f"test_active_{i:03d}",
                alert_type="risk_threshold",
                alert_level="warning",
                title=f"测试告警{i}",
                message=f"测试消息{i}",
                status="active" if i < 3 else "resolved"
            )
            persistence.save_alert(alert)
        
        active_alerts = persistence.get_active_alerts(limit=10)
        self.assertEqual(len(active_alerts), 3)
    
    def test_get_alerts_by_level(self):
        """测试按级别获取告警"""
        from src.risk.persistence.risk_persistence import AlertPersistence, AlertData
        
        persistence = AlertPersistence(storage_dir=self.storage_dir)
        
        levels = ["info", "warning", "error", "critical"]
        for i, level in enumerate(levels):
            alert = AlertData(
                alert_id=f"test_level_{i:03d}",
                alert_type="test",
                alert_level=level,
                title=f"测试告警{level}",
                message="测试消息"
            )
            persistence.save_alert(alert)
        
        critical_alerts = persistence.get_alerts_by_level("critical", limit=10)
        self.assertEqual(len(critical_alerts), 1)
        self.assertEqual(critical_alerts[0].alert_level, "critical")


class TestRiskMetricPersistence(unittest.TestCase):
    """风险指标持久化测试"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_dir = os.path.join(self.temp_dir, "metrics")
        os.makedirs(self.storage_dir, exist_ok=True)
    
    def tearDown(self):
        """测试后清理"""
        try:
            shutil.rmtree(self.temp_dir)
        except PermissionError:
            pass
    
    def test_metric_save_and_load(self):
        """测试风险指标保存和加载"""
        from src.risk.persistence.risk_persistence import RiskMetricPersistence, RiskMetricData
        
        persistence = RiskMetricPersistence(storage_dir=self.storage_dir)
        
        metric = RiskMetricData(
            metric_id="test_metric_001",
            metric_name="var_95",
            metric_type="market",
            value=0.0523,
            threshold_low=0.02,
            threshold_medium=0.05,
            threshold_high=0.08,
            risk_level="medium",
            calculation_method="historical",
            confidence_level=0.95
        )
        
        result = persistence.save_metric(metric)
        self.assertTrue(result)
        
        loaded_metric = persistence.get_metric("test_metric_001")
        self.assertIsNotNone(loaded_metric)
        self.assertEqual(loaded_metric.metric_name, "var_95")
        self.assertEqual(loaded_metric.value, 0.0523)
        self.assertEqual(loaded_metric.risk_level, "medium")
    
    def test_get_metrics_by_name(self):
        """测试按名称获取指标历史"""
        from src.risk.persistence.risk_persistence import RiskMetricPersistence, RiskMetricData
        
        persistence = RiskMetricPersistence(storage_dir=self.storage_dir)
        
        metric_names = ["var_95", "max_drawdown", "sharpe_ratio"]
        for name in metric_names:
            for i in range(3):
                metric = RiskMetricData(
                    metric_id=f"test_{name}_{i:03d}",
                    metric_name=name,
                    metric_type="market",
                    value=0.01 * (i + 1)
                )
                persistence.save_metric(metric)
        
        var_metrics = persistence.get_metrics_by_name("var_95", limit=10)
        self.assertEqual(len(var_metrics), 3)
        
        for m in var_metrics:
            self.assertEqual(m.metric_name, "var_95")
    
    def test_get_latest_metrics(self):
        """测试获取最新指标"""
        from src.risk.persistence.risk_persistence import RiskMetricPersistence, RiskMetricData
        
        persistence = RiskMetricPersistence(storage_dir=self.storage_dir)
        
        metric_names = ["var_95", "max_drawdown", "sharpe_ratio", "volatility"]
        for i, name in enumerate(metric_names):
            for j in range(3):
                metric = RiskMetricData(
                    metric_id=f"test_latest_{name}_{j:03d}",
                    metric_name=name,
                    metric_type="market",
                    value=0.01 * (j + 1)
                )
                time.sleep(0.01)  # 确保时间戳不同
                persistence.save_metric(metric)
        
        latest_metrics = persistence.get_latest_metrics(limit=10)
        self.assertGreaterEqual(len(latest_metrics), 4)


class TestRiskRulePersistence(unittest.TestCase):
    """风险规则持久化测试"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_dir = os.path.join(self.temp_dir, "rules")
        os.makedirs(self.storage_dir, exist_ok=True)
    
    def tearDown(self):
        """测试后清理"""
        try:
            shutil.rmtree(self.temp_dir)
        except PermissionError:
            pass
    
    def test_rule_save_and_load(self):
        """测试风险规则保存和加载"""
        from src.risk.persistence.risk_persistence import RiskRulePersistence, RiskRuleData
        
        persistence = RiskRulePersistence(storage_dir=self.storage_dir)
        
        rule = RiskRuleData(
            rule_id="test_rule_001",
            rule_name="VaR限额规则",
            rule_type="threshold",
            risk_type="market",
            conditions={"metric": "var_95", "threshold": 0.08, "operator": ">"},
            actions=["alert", "reduce_position"],
            alert_level="error",
            enabled=True,
            priority=10
        )
        
        result = persistence.save_rule(rule)
        self.assertTrue(result)
        
        loaded_rule = persistence.get_rule("test_rule_001")
        self.assertIsNotNone(loaded_rule)
        self.assertEqual(loaded_rule.rule_name, "VaR限额规则")
        self.assertEqual(loaded_rule.risk_type, "market")
        self.assertTrue(loaded_rule.enabled)
    
    def test_rule_update(self):
        """测试风险规则更新"""
        from src.risk.persistence.risk_persistence import RiskRulePersistence, RiskRuleData
        
        persistence = RiskRulePersistence(storage_dir=self.storage_dir)
        
        rule = RiskRuleData(
            rule_id="test_rule_002",
            rule_name="测试规则",
            rule_type="threshold",
            risk_type="liquidity",
            conditions={"threshold": 0.1},
            actions=["alert"],
            enabled=True
        )
        
        persistence.save_rule(rule)
        
        result = persistence.update_rule("test_rule_002", {
            "enabled": False,
            "priority": 50
        })
        self.assertTrue(result)
        
        updated_rule = persistence.get_rule("test_rule_002")
        self.assertFalse(updated_rule.enabled)
        self.assertEqual(updated_rule.priority, 50)
    
    def test_get_enabled_rules(self):
        """测试获取启用的规则"""
        from src.risk.persistence.risk_persistence import RiskRulePersistence, RiskRuleData
        
        persistence = RiskRulePersistence(storage_dir=self.storage_dir)
        
        for i in range(5):
            rule = RiskRuleData(
                rule_id=f"test_enabled_{i:03d}",
                rule_name=f"测试规则{i}",
                rule_type="threshold",
                risk_type="market",
                conditions={},
                actions=["alert"],
                enabled=i < 3,
                priority=i * 10
            )
            persistence.save_rule(rule)
        
        enabled_rules = persistence.get_enabled_rules()
        self.assertEqual(len(enabled_rules), 3)
        
        for rule in enabled_rules:
            self.assertTrue(rule.enabled)
    
    def test_delete_rule(self):
        """测试删除规则"""
        from src.risk.persistence.risk_persistence import RiskRulePersistence, RiskRuleData
        
        persistence = RiskRulePersistence(storage_dir=self.storage_dir)
        
        rule = RiskRuleData(
            rule_id="test_rule_delete",
            rule_name="待删除规则",
            rule_type="threshold",
            risk_type="market",
            conditions={},
            actions=["alert"]
        )
        
        persistence.save_rule(rule)
        
        result = persistence.delete_rule("test_rule_delete")
        self.assertTrue(result)
        
        deleted_rule = persistence.get_rule("test_rule_delete")
        self.assertIsNone(deleted_rule)


class TestPersistencePerformance(unittest.TestCase):
    """持久化性能测试"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """测试后清理"""
        try:
            shutil.rmtree(self.temp_dir)
        except PermissionError:
            pass
    
    def test_batch_save_performance(self):
        """测试批量保存性能"""
        from src.risk.persistence.risk_persistence import RiskCheckPersistence, RiskCheckData
        
        storage_dir = os.path.join(self.temp_dir, "checks")
        persistence = RiskCheckPersistence(storage_dir=storage_dir)
        
        start_time = time.time()
        
        for i in range(100):
            check = RiskCheckData(
                check_id=f"perf_check_{i:04d}",
                check_type="market",
                risk_level="low",
                passed=True,
                score=0.1
            )
            persistence.save_check(check)
        
        elapsed_time = time.time() - start_time
        
        self.assertLess(elapsed_time, 5.0, "批量保存100条风险检查应在5秒内完成")
        
        print(f"\n批量保存100条风险检查耗时: {elapsed_time:.3f}秒")
    
    def test_alert_batch_performance(self):
        """测试告警批量保存性能"""
        from src.risk.persistence.risk_persistence import AlertPersistence, AlertData
        
        storage_dir = os.path.join(self.temp_dir, "alerts")
        persistence = AlertPersistence(storage_dir=storage_dir)
        
        start_time = time.time()
        
        for i in range(100):
            alert = AlertData(
                alert_id=f"perf_alert_{i:04d}",
                alert_type="test",
                alert_level="warning",
                title=f"测试告警{i}",
                message="性能测试"
            )
            persistence.save_alert(alert)
        
        elapsed_time = time.time() - start_time
        
        self.assertLess(elapsed_time, 5.0, "批量保存100条告警应在5秒内完成")
        
        print(f"\n批量保存100条告警耗时: {elapsed_time:.3f}秒")


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """测试后清理"""
        try:
            shutil.rmtree(self.temp_dir)
        except PermissionError:
            pass
    
    def test_full_workflow(self):
        """测试完整工作流"""
        from src.risk.persistence.risk_persistence import (
            RiskCheckPersistence, RiskCheckData,
            AlertPersistence, AlertData,
            RiskMetricPersistence, RiskMetricData,
            RiskRulePersistence, RiskRuleData
        )
        
        # 1. 创建规则
        rule_persistence = RiskRulePersistence(
            storage_dir=os.path.join(self.temp_dir, "rules")
        )
        rule = RiskRuleData(
            rule_id="workflow_rule",
            rule_name="工作流测试规则",
            rule_type="threshold",
            risk_type="market",
            conditions={"var_95": 0.08},
            actions=["alert"],
            enabled=True
        )
        rule_persistence.save_rule(rule)
        
        # 2. 计算指标
        metric_persistence = RiskMetricPersistence(
            storage_dir=os.path.join(self.temp_dir, "metrics")
        )
        metric = RiskMetricData(
            metric_id="workflow_metric",
            metric_name="var_95",
            metric_type="market",
            value=0.092,
            threshold_high=0.08,
            risk_level="high"
        )
        metric_persistence.save_metric(metric)
        
        # 3. 执行检查
        check_persistence = RiskCheckPersistence(
            storage_dir=os.path.join(self.temp_dir, "checks")
        )
        check = RiskCheckData(
            check_id="workflow_check",
            check_type="var_check",
            risk_level="high",
            passed=False,
            score=0.92,
            details={"var_value": 0.092, "threshold": 0.08},
            recommendations=["降低仓位"]
        )
        check_persistence.save_check(check)
        
        # 4. 生成告警
        alert_persistence = AlertPersistence(
            storage_dir=os.path.join(self.temp_dir, "alerts")
        )
        alert = AlertData(
            alert_id="workflow_alert",
            alert_type="risk_threshold",
            alert_level="error",
            title="VaR超限告警",
            message="VaR(95%)=9.2%超过阈值8%",
            rule_id="workflow_rule"
        )
        alert_persistence.save_alert(alert)
        
        # 验证完整流程
        loaded_rule = rule_persistence.get_rule("workflow_rule")
        self.assertIsNotNone(loaded_rule)
        
        loaded_metric = metric_persistence.get_metric("workflow_metric")
        self.assertIsNotNone(loaded_metric)
        self.assertEqual(loaded_metric.risk_level, "high")
        
        loaded_check = check_persistence.get_check("workflow_check")
        self.assertIsNotNone(loaded_check)
        self.assertFalse(loaded_check.passed)
        
        loaded_alert = alert_persistence.get_alert("workflow_alert")
        self.assertIsNotNone(loaded_alert)
        self.assertEqual(loaded_alert.alert_level, "error")


if __name__ == "__main__":
    unittest.main(verbosity=2)
