# API 变更日志

本文档记录 RQA2025 API 的所有变更历史。

## [1.0.0] - 2026-03-29

### 新增

#### 认证模块
- ✅ JWT 认证机制
- ✅ Token 刷新功能
- ✅ 登录/登出接口

#### 策略管理
- ✅ 创建策略 (`POST /strategies`)
- ✅ 获取策略列表 (`GET /strategies`)
- ✅ 获取策略详情 (`GET /strategies/{id}`)
- ✅ 更新策略 (`PUT /strategies/{id}`)
- ✅ 删除策略 (`DELETE /strategies/{id}`)
- ✅ 执行回测 (`POST /strategies/{id}/backtest`)
- ✅ 获取回测结果 (`GET /backtests/{id}`)

#### 特征管理
- ✅ 创建特征 (`POST /features`)
- ✅ 获取特征列表 (`GET /features`)
- ✅ 获取特征详情 (`GET /features/{id}`)
- ✅ 计算特征 (`POST /features/{id}/compute`)

#### 模型管理
- ✅ 创建模型 (`POST /models`)
- ✅ 获取模型列表 (`GET /models`)
- ✅ 获取模型详情 (`GET /models/{id}`)
- ✅ 训练模型 (`POST /models/{id}/train`)
- ✅ 模型推理 (`POST /models/{id}/predict`)

#### 交易执行
- ✅ 下单 (`POST /orders`)
- ✅ 获取订单列表 (`GET /orders`)
- ✅ 获取订单详情 (`GET /orders/{id}`)
- ✅ 撤单 (`DELETE /orders/{id}`)

#### 监控指标
- ✅ 获取系统指标 (`GET /metrics/system`)
- ✅ 获取应用指标 (`GET /metrics/application`)
- ✅ 获取业务指标 (`GET /metrics/business`)

### 优化

- 统一的 API 响应格式
- 完善的错误码体系
- 详细的参数说明
- 完整的示例代码

### 文档

- 📚 API 参考文档
- 🚀 快速开始指南
- 💻 Python 示例代码
- 💻 JavaScript 示例代码

---

## 版本说明

### 版本号格式

我们使用 [语义化版本](https://semver.org/lang/zh-CN/) 规范：

```
主版本号.次版本号.修订号
```

- **主版本号**：不兼容的 API 修改
- **次版本号**：向下兼容的功能新增
- **修订号**：向下兼容的问题修正

### 版本支持策略

| 版本 | 状态 | 支持期限 |
|------|------|----------|
| 1.0.x | ✅ 活跃支持 | 2027-03-29 |

---

## 升级指南

### 从 0.x 升级到 1.0

1. 更新 API 基础 URL
2. 修改认证方式（新增 JWT 支持）
3. 更新请求/响应格式
4. 检查错误码变更

详细升级步骤请参考 [迁移指南](migration_guide.md)。

---

## 反馈与建议

如有任何问题或建议，请通过以下方式联系我们：

- 📧 邮箱：api-support@rqa2025.com
- 💬 社区：https://community.rqa2025.com
- 🐛 问题反馈：https://github.com/rqa2025/api/issues

---

*最后更新：2026-03-29*
