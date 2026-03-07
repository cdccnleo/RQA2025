# Web-Static 管理页面模拟数据和硬编码问题修复总结

## 📋 检查概述

对 `web-static` 目录下的所有管理页面进行了全面检查，移除了所有模拟数据和硬编码问题，确保符合量化交易系统的严格要求。

## ✅ 修复的文件

### 1. strategy-conception.html ✅

#### 问题1: 硬编码股票代码和数据源
**位置**: 第628行

**修复前**:
```javascript
data_source: {source_type: 'yahoo', symbol: 'AAPL'},
```

**修复后**:
```javascript
data_source: {source_type: '', symbol: ''}, // 必须由用户配置，不能使用默认值
```

**改进**:
- ✅ 移除了硬编码的 `'AAPL'` 股票代码
- ✅ 移除了硬编码的 `'yahoo'` 数据源类型
- ✅ 改为从API动态获取可用的数据源配置
- ✅ 如果API不可用，使用空配置，强制用户手动配置

#### 问题2: 模拟收益计算
**位置**: 第902-905行

**修复前**:
```javascript
// 预期收益区间 (模拟计算)
const expectedReturn = complexityScore > 70 ? '15-25%' :
                      complexityScore > 40 ? '8-15%' : '3-8%';
```

**修复后**:
```javascript
// 预期收益区间 - 量化交易系统要求：不使用模拟数据
// 如果策略已保存且有回测结果，从API获取真实收益数据
const strategyId = document.getElementById('strategyId').value;
if (strategyId) {
    try {
        const response = await fetch(this.getApiBaseUrl(`/strategy/conceptions/${strategyId}`));
        if (response.ok) {
            const strategyData = await response.json();
            const backtestResult = strategyData.backtest_result;
            if (backtestResult && backtestResult.annual_return !== undefined) {
                const annualReturn = backtestResult.annual_return;
                const returnRange = `${(annualReturn * 0.8).toFixed(1)}%-${(annualReturn * 1.2).toFixed(1)}%`;
                document.getElementById('expectedReturn').textContent = returnRange;
                return;
            }
        }
    } catch (error) {
        console.warn('无法获取策略回测结果:', error);
    }
}

// 如果没有回测结果，显示"待回测"而非模拟数据
document.getElementById('expectedReturn').textContent = '待回测';
```

**改进**:
- ✅ 移除了基于复杂度的模拟收益计算
- ✅ 改为从API获取真实的回测结果
- ✅ 如果没有回测结果，显示"待回测"而非模拟数据
- ✅ 符合量化交易系统的零模拟数据要求

#### 其他改进
- ✅ 添加了 `getApiBaseUrl()` 方法，支持环境感知的API URL生成
- ✅ 将 `getDefaultNodeParams()` 改为异步方法，支持从API获取数据源配置
- ✅ 将 `autoAddRequiredNodes()` 和 `addNode()` 改为异步方法
- ✅ 将 `updateStrategyStats()` 改为异步方法，支持从API获取真实数据

### 2. index.html ✅

#### 问题: 模拟服务状态检查
**位置**: 第282-285行

**修复前**:
```javascript
// 模拟其他服务状态（实际应该调用相应API）
updateServiceStatus('db', '运行正常', 'green', '🟢');
updateServiceStatus('cache', '运行正常', 'green', '🟢');
updateServiceStatus('monitor', '运行正常', 'green', '🟢');
```

**修复后**:
```javascript
// 量化交易系统要求：不使用模拟数据，从真实API获取服务状态
// 检查数据库服务
try {
    const dbResponse = await fetch(getApiBaseUrl('/health/database'));
    if (dbResponse.ok) {
        const dbData = await dbResponse.json();
        const dbStatus = dbData.status === 'healthy' ? '运行正常' : '异常';
        const dbColor = dbData.status === 'healthy' ? 'green' : 'yellow';
        const dbIcon = dbData.status === 'healthy' ? '🟢' : '🟡';
        updateServiceStatus('db', dbStatus, dbColor, dbIcon);
    } else {
        updateServiceStatus('db', '检查失败', 'yellow', '🟡');
    }
} catch (error) {
    updateServiceStatus('db', '无法检查', 'gray', '⚪');
}

// 检查缓存服务
try {
    const cacheResponse = await fetch(getApiBaseUrl('/health/cache'));
    // ... 类似的处理逻辑
} catch (error) {
    updateServiceStatus('cache', '无法检查', 'gray', '⚪');
}

// 检查监控服务
try {
    const monitorResponse = await fetch(getApiBaseUrl('/health/monitoring'));
    // ... 类似的处理逻辑
} catch (error) {
    updateServiceStatus('monitor', '无法检查', 'gray', '⚪');
}
```

**改进**:
- ✅ 移除了所有硬编码的服务状态
- ✅ 改为从真实API获取数据库、缓存、监控服务的状态
- ✅ 添加了完整的错误处理
- ✅ 支持不同状态的显示（正常、异常、检查失败、无法检查）
- ✅ 添加了 `getApiBaseUrl()` 函数，支持环境感知的API URL生成

## 📊 修复统计

| 文件 | 问题类型 | 修复数量 | 状态 |
|------|---------|---------|------|
| strategy-conception.html | 硬编码股票代码 | 1 | ✅ |
| strategy-conception.html | 硬编码数据源 | 1 | ✅ |
| strategy-conception.html | 模拟收益计算 | 1 | ✅ |
| index.html | 模拟服务状态 | 3 | ✅ |

## 🎯 核心改进

### 1. 移除硬编码数据

- **股票代码**: 移除了 `'AAPL'` 等硬编码股票代码
- **数据源类型**: 移除了 `'yahoo'` 等硬编码数据源类型
- **服务状态**: 移除了硬编码的服务状态

### 2. 动态数据获取

- **数据源配置**: 从 `/api/v1/data/sources` API获取可用数据源
- **策略回测结果**: 从 `/api/v1/strategy/conceptions/{id}` API获取真实回测结果
- **服务健康状态**: 从 `/api/v1/health/*` API获取真实服务状态

### 3. 错误处理完善

- **API调用失败**: 显示"无法检查"或"检查失败"状态
- **数据缺失**: 显示"待回测"而非模拟数据
- **配置缺失**: 使用空配置，强制用户手动配置

### 4. 环境感知

- **API URL生成**: 支持本地开发和生产环境
- **自动检测**: 根据 `window.location.protocol` 和 `hostname` 自动选择API地址

## 📝 已检查的文件

| 文件 | 状态 | 说明 |
|------|------|------|
| dashboard.html | ✅ | 之前已修复，无模拟数据 |
| data-sources-config.html | ✅ | 之前已修复，无模拟数据 |
| strategy-backtest.html | ✅ | 之前已修复，无模拟数据 |
| trading-execution.html | ✅ | 之前已修复，无模拟数据 |
| strategy-conception.html | ✅ | 本次修复完成 |
| index.html | ✅ | 本次修复完成 |

## ✅ 完成状态

- ✅ 移除所有硬编码股票代码
- ✅ 移除所有硬编码数据源类型
- ✅ 移除所有模拟收益计算
- ✅ 移除所有模拟服务状态
- ✅ 实现动态数据获取
- ✅ 完善错误处理
- ✅ 支持环境感知的API调用

**所有 `web-static` 目录下的管理页面已完全符合量化交易系统的零模拟数据要求！** 🎯✨

