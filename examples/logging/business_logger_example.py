#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 基础设施层日志系统 - BusinessLogger使用示例

演示BusinessLogger的业务日志记录功能，适用于订单、交易、用户管理等业务场景。
"""

from infrastructure.logging import BusinessLogger
import sys
import os
import time
import random

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def order_management_example():
    """订单管理业务日志示例"""
    print("=== 订单管理业务日志示例 ===\n")

    # 创建订单业务Logger
    order_logger = BusinessLogger("order.service", log_dir="logs/business/orders")

    # 验证自动配置
    print(f"自动配置验证 - 分类: {order_logger.category}, 格式: {order_logger.format_type}")
    print()

    # 模拟订单处理流程
    orders = [
        {"id": "ORD-2025-001", "customer_id": "CUST-001", "amount": 299.99, "items": 3},
        {"id": "ORD-2025-002", "customer_id": "CUST-002", "amount": 599.99, "items": 5},
        {"id": "ORD-2025-003", "customer_id": "CUST-001", "amount": 149.99, "items": 2},
    ]

    for order in orders:
        # 订单创建
        order_logger.info("订单创建开始",
                          order_id=order["id"],
                          customer_id=order["customer_id"],
                          amount=order["amount"],
                          item_count=order["items"])

        # 模拟处理时间
        time.sleep(0.1)

        # 随机成功或失败
        if random.choice([True, True, False]):  # 2/3成功率
            order_logger.info("订单创建成功",
                              order_id=order["id"],
                              processing_time="0.1s",
                              status="confirmed",
                              payment_required=True)
        else:
            order_logger.warning("订单创建失败 - 库存不足",
                                 order_id=order["id"],
                                 customer_id=order["customer_id"],
                                 reason="insufficient_inventory",
                                 suggested_retry=True)

    print()


def user_management_example():
    """用户管理业务日志示例"""
    print("=== 用户管理业务日志示例 ===\n")

    user_logger = BusinessLogger("user.service", log_dir="logs/business/users")

    # 模拟用户注册流程
    users = [
        {"username": "john_doe", "email": "john@example.com", "role": "customer"},
        {"username": "admin_user", "email": "admin@company.com", "role": "admin"},
        {"username": "jane_smith", "email": "jane@example.com", "role": "customer"},
    ]

    for user in users:
        user_logger.info("用户注册开始",
                         username=user["username"],
                         email=user["email"],
                         registration_method="web_form")

        time.sleep(0.05)

        if random.choice([True, False]):  # 50%成功率
            user_logger.info("用户注册成功",
                             username=user["username"],
                             user_id=f"USER-{random.randint(1000, 9999)}",
                             role=user["role"],
                             email_verified=False,
                             welcome_email_sent=True)
        else:
            user_logger.error("用户注册失败 - 用户名已存在",
                              username=user["username"],
                              email=user["email"],
                              error_code="USERNAME_EXISTS",
                              suggested_action="choose_different_username")

    print()


def payment_processing_example():
    """支付处理业务日志示例"""
    print("=== 支付处理业务日志示例 ===\n")

    payment_logger = BusinessLogger("payment.service", log_dir="logs/business/payments")

    # 模拟支付交易
    transactions = [
        {"id": "TXN-001", "order_id": "ORD-2025-001", "amount": 299.99, "method": "credit_card"},
        {"id": "TXN-002", "order_id": "ORD-2025-002", "amount": 599.99, "method": "paypal"},
        {"id": "TXN-003", "order_id": "ORD-2025-003", "amount": 149.99, "method": "bank_transfer"},
    ]

    for txn in transactions:
        payment_logger.info("支付处理开始",
                            transaction_id=txn["id"],
                            order_id=txn["order_id"],
                            amount=txn["amount"],
                            payment_method=txn["method"])

        time.sleep(0.2)

        if random.choice([True, True, True, False]):  # 75%成功率
            payment_logger.info("支付处理成功",
                                transaction_id=txn["id"],
                                order_id=txn["order_id"],
                                amount=txn["amount"],
                                processing_time="0.2s",
                                status="completed",
                                confirmation_code=f"CONF-{random.randint(100000, 999999)}")
        else:
            payment_logger.warning("支付处理失败",
                                   transaction_id=txn["id"],
                                   order_id=txn["order_id"],
                                   amount=txn["amount"],
                                   payment_method=txn["method"],
                                   error_code="CARD_DECLINED",
                                   retry_allowed=True,
                                   suggested_action="use_different_payment_method")

    print()


def inventory_management_example():
    """库存管理业务日志示例"""
    print("=== 库存管理业务日志示例 ===\n")

    inventory_logger = BusinessLogger("inventory.service", log_dir="logs/business/inventory")

    # 模拟库存操作
    operations = [
        {"product_id": "PROD-001", "operation": "stock_in", "quantity": 50, "warehouse": "WH-A"},
        {"product_id": "PROD-002", "operation": "stock_out", "quantity": 25, "warehouse": "WH-B"},
        {"product_id": "PROD-003", "operation": "adjustment", "quantity": -10, "warehouse": "WH-A"},
    ]

    for op in operations:
        if op["operation"] == "stock_in":
            inventory_logger.info("商品入库",
                                  product_id=op["product_id"],
                                  quantity=op["quantity"],
                                  warehouse=op["warehouse"],
                                  operation_type="stock_in",
                                  operator="system",
                                  reason="purchase_order")
        elif op["operation"] == "stock_out":
            inventory_logger.info("商品出库",
                                  product_id=op["product_id"],
                                  quantity=op["quantity"],
                                  warehouse=op["warehouse"],
                                  operation_type="stock_out",
                                  order_id=f"ORD-{random.randint(1000, 9999)}")
        elif op["operation"] == "adjustment":
            inventory_logger.warning("库存调整",
                                     product_id=op["product_id"],
                                     quantity_change=op["quantity"],
                                     warehouse=op["warehouse"],
                                     operation_type="adjustment",
                                     reason="damaged_goods",
                                     approved_by="manager",
                                     audit_required=True)

        time.sleep(0.1)

    print()


def business_metrics_example():
    """业务指标监控示例"""
    print("=== 业务指标监控示例 ===\n")

    metrics_logger = BusinessLogger("metrics.service", log_dir="logs/business/metrics")

    # 模拟业务指标收集
    metrics = {
        "orders_today": 145,
        "revenue_today": 45678.90,
        "conversion_rate": 3.2,
        "avg_order_value": 315.37,
        "customer_satisfaction": 4.7
    }

    metrics_logger.info("每日业务指标汇总",
                        period="2025-09-23",
                        orders_count=metrics["orders_today"],
                        total_revenue=metrics["revenue_today"],
                        conversion_rate=f"{metrics['conversion_rate']}%",
                        avg_order_value=metrics["avg_order_value"],
                        customer_satisfaction=metrics["customer_satisfaction"])

    # 性能指标
    if metrics["conversion_rate"] < 3.0:
        metrics_logger.warning("转化率偏低",
                               current_rate=metrics["conversion_rate"],
                               threshold=3.0,
                               degradation=f"{(3.0-metrics['conversion_rate'])/3.0*100:.1f}%",
                               investigation_required=True)

    if metrics["customer_satisfaction"] < 4.5:
        metrics_logger.warning("客户满意度预警",
                               current_score=metrics["customer_satisfaction"],
                               threshold=4.5,
                               action_required="send_survey",
                               follow_up_needed=True)

    print()


def main():
    """主函数"""
    print("RQA2025 基础设施层日志系统 - BusinessLogger使用示例")
    print("=" * 65)
    print()

    try:
        order_management_example()
        user_management_example()
        payment_processing_example()
        inventory_management_example()
        business_metrics_example()

        print("🎉 所有业务日志示例执行完成！")
        print("\n日志文件位置:")
        print("- logs/business/orders/     - 订单相关日志")
        print("- logs/business/users/      - 用户相关日志")
        print("- logs/business/payments/   - 支付相关日志")
        print("- logs/business/inventory/  - 库存相关日志")
        print("- logs/business/metrics/    - 指标相关日志")

    except Exception as e:
        print(f"❌ 示例执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
