# 微服务集成测试报告

生成时间: 2025-12-04 13:33:11

## 📊 测试概览

- **服务启动**: 0/3 个服务成功启动
- **契约测试**: 0/1 个通过
- **集成测试**: 0/1 个通过
- **容错测试**: 0/6 个通过
.1f.2f## 🔧 服务启动状态

- ❌ user-service
- ❌ order-service
- ❌ payment-service

## 📋 契约测试详情

| 测试名称 | 结果 | 时间 |
|----------|------|------|
| contract_order-service_user-service | ❌ | 10.18s |

## 🔗 集成测试详情

| 测试名称 | 结果 | 时间 |
|----------|------|------|
| complete_order_flow | ❌ | 10.17s |

## 🛡️ 容错测试详情

| 测试名称 | 结果 | 时间 |
|----------|------|------|
| fault_tolerance_user-service | ❌ | 37.68s |
| load_balancing_user-service | ❌ | 4.17s |
| fault_tolerance_order-service | ❌ | 37.61s |
| load_balancing_order-service | ❌ | 4.12s |
| fault_tolerance_payment-service | ❌ | 37.70s |
| load_balancing_payment-service | ❌ | 4.14s |
