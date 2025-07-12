# 中国市场数据加载集成测试

## 测试概述

本测试套件验证中国市场数据加载的完整流程，包括：

1. 适配器注册和初始化
2. 数据加载流程
3. 中国市场特有验证规则
4. 缓存集成
5. 质量监控

## 测试环境准备

### 依赖安装

```bash
pip install pytest pytest-cov
```

### 测试数据

测试使用模拟数据，无需真实数据库连接。

## 运行测试

### 运行所有测试

```bash
pytest tests/integration/ -v
```

### 运行特定测试类

```bash
pytest tests/integration/test_china_integration.py::TestChinaIntegration -v
```

### 生成覆盖率报告

```bash
pytest --cov=src tests/integration/
```

## 测试用例

### 1. 适配器注册测试
- 验证中国市场适配器是否正确注册到数据加载器
- 检查适配器类型和实例

### 2. 数据加载流程测试
- 验证从配置到数据加载的完整流程
- 检查返回数据结构
- 验证中国市场特有字段

### 3. 验证规则测试
- 价格涨跌停检查
- 交易暂停验证
- 双源数据验证
- 监管合规检查

### 4. 错误处理测试
- 无效股票代码处理
- 数据验证失败场景
- 缓存失效场景

## 预期输出

测试通过时应显示所有测试用例通过：

```
============================= test session starts =============================
collected 3 items

tests/integration/test_china_integration.py::TestChinaIntegration::test_adapter_registration PASSED
tests/integration/test_china_integration.py::TestChinaIntegration::test_data_loading_flow PASSED
tests/integration/test_china_integration.py::TestChinaIntegration::test_validation_flow PASSED

============================== 3 passed in 0.12s =============================
```

## 维护说明

1. 添加新测试时：
   - 在`test_china_integration.py`中添加新的测试方法
   - 如需新的fixture，添加到`conftest.py`

2. 适配器更新时：
   - 确保更新对应的测试用例
   - 验证所有现有测试仍然通过
