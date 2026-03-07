#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RiskLogger 使用示例
演示风险监控系统的日志记录功能
"""

from infrastructure.logging import RiskLogger
import time
import random
import sys
import os
# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def simulate_risk_monitoring():
    """模拟风险监控日志记录"""

    # 创建风险Logger
    risk_logger = RiskLogger(
        name="risk.monitor",
        log_dir="logs/risk"
    )

    print("=== 风险Logger演示 ===")

    # 模拟风险检查
    risk_checks = [
        {
            "type": "position_limit",
            "portfolio": "TECH_FUND",
            "current_exposure": 950000,
            "limit": 1000000,
            "threshold": 0.95
        },
        {
            "type": "volatility_spike",
            "symbol": "TSLA",
            "current_volatility": 0.45,
            "normal_volatility": 0.25,
            "threshold": 0.40
        },
        {
            "type": "concentration_risk",
            "sector": "TECHNOLOGY",
            "exposure_percentage": 0.68,
            "limit": 0.50,
            "threshold": 0.60
        },
        {
            "type": "liquidity_risk",
            "asset": "CRYPTO_BASKET",
            "bid_ask_spread": 0.08,
            "normal_spread": 0.02,
            "threshold": 0.05
        }
    ]

    for check in risk_checks:
        # 模拟风险评估
        risk_level = random.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"])

        if risk_level == "CRITICAL":
            risk_logger.critical("严重风险警报",
                                 risk_type=check["type"],
                                 risk_level=risk_level,
                                 threshold=check.get("threshold"),
                                 current_value=check.get("current_exposure") or check.get("current_volatility") or check.get(
                                     "exposure_percentage") or check.get("bid_ask_spread"),
                                 limit=check.get("limit"),
                                 action="IMMEDIATE_INTERVENTION",
                                 timestamp=time.time()
                                 )
            print(f"🚨 严重风险: {check['type']}")
        elif risk_level == "HIGH":
            risk_logger.error("高风险警报",
                              risk_type=check["type"],
                              risk_level=risk_level,
                              threshold=check.get("threshold"),
                              current_value=check.get("current_exposure") or check.get("current_volatility") or check.get(
                                  "exposure_percentage") or check.get("bid_ask_spread"),
                              action="RISK_REDUCTION",
                              timestamp=time.time()
                              )
            print(f"⚠️ 高风险: {check['type']}")
        elif risk_level == "MEDIUM":
            risk_logger.warning("中等风险警报",
                                risk_type=check["type"],
                                risk_level=risk_level,
                                threshold=check.get("threshold"),
                                current_value=check.get("current_exposure") or check.get("current_volatility") or check.get(
                                    "exposure_percentage") or check.get("bid_ask_spread"),
                                action="MONITOR_CLOSELY",
                                timestamp=time.time()
                                )
            print(f"⚡ 中等风险: {check['type']}")
        else:
            risk_logger.info("风险检查通过",
                             risk_type=check["type"],
                             risk_level=risk_level,
                             current_value=check.get("current_exposure") or check.get("current_volatility") or check.get(
                                 "exposure_percentage") or check.get("bid_ask_spread"),
                             status="NORMAL",
                             timestamp=time.time()
                             )
            print(f"✅ 风险正常: {check['type']}")

        time.sleep(0.2)

    # 合规检查日志
    risk_logger.info("合规检查完成",
                     check_type="daily_compliance",
                     violations_found=0,
                     checks_passed=47,
                     total_checks=47,
                     timestamp=time.time()
                     )

    # 风险模型更新
    risk_logger.info("风险模型更新",
                     model_name="VaR_95",
                     update_reason="market_volatility_increase",
                     new_parameters={"confidence_level": 0.95,
                                     "time_horizon": 1, "method": "historical_simulation"},
                     validation_score=0.92,
                     timestamp=time.time()
                     )

    print("\n风险日志记录完成")
    print(f"Logger名称: {risk_logger.name}")
    print(f"日志级别: {risk_logger.level}")
    print(f"日志分类: {risk_logger.category}")
    print(f"日志目录: {risk_logger.log_dir}")


if __name__ == "__main__":
    simulate_risk_monitoring()
