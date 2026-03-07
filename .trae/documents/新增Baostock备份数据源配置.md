# 修复Baostock数据源配置计划

## 问题分析

检查容器中的Baostock数据源配置，发现以下问题：

### 1. 编码问题

* 配置文件中中文字符显示为乱码（如"baostock\_a鑲℃暟鎹?"）

* 这可能导致系统无法正确识别和处理配置

### 2. 格式错误

* 数据源ID格式不正确，包含乱码

* 配置结构可能不完整

### 3. 配置问题

* 数据源被设置为`enabled: true`，但作为备份数据源应该设置为`false`

* 速率限制设置为"1娆?澶?"，可能不符合实际情况

## 修复计划

### 步骤1: 修复数据源配置

基于`akshare_stock_a`的正确配置结构，修复Baostock配置：

```json
{
  "id": "baostock_stock_a",
  "name": "Baostock A股数据",
  "type": "股票数据",
  "url": "http://www.baostock.com",
  "rate_limit": "100次/分钟",
  "enabled": false,
  "config": {
    "baostock_category": "A股",
    "default_days": 30,
    "adjust_type": "qfq",
    "enable_incremental": true,
    "description": "Baostock A股数据采集接口，作为AKShare的备份数据源，支持日线行情、股票基础信息等。",
    "baostock_function": "query_history_k_data_plus",
    "stock_pool_type": "custom",
    "custom_stocks": [
      {
        "code": "002837",
        "name": "英维克"
      }
    ],
    "strategy_config": {
      "strategy_id": "hf_trading",
      "pool_size": 100,
      "liquidity_threshold": 10000
    },
    "data_type_configs": {
      "realtime": {
        "enabled": false,
        "description": "实时行情数据"
      },
      "1min": {
        "enabled": false,
        "description": "1分钟K线数据"
      },
      "5min": {
        "enabled": false,
        "description": "5分钟K线数据"
      },
      "15min": {
        "enabled": false,
        "description": "15分钟K线数据"
      },
      "30min": {
        "enabled": false,
        "description": "30分钟K线数据"
      },
      "60min": {
        "enabled": false,
        "description": "60分钟K线数据"
      },
      "daily": {
        "enabled": true,
        "description": "日线数据"
      },
      "weekly": {
        "enabled": false,
        "description": "周线数据"
      },
      "monthly": {
        "enabled": false,
        "description": "月线数据"
      }
    },
    "data_types": [
      "daily"
    ]
  },
  "last_test": null,
  "status": "未测试"
}
```

### 步骤2: 更新配置文件

1. 读取容器中的完整配置文件
2. 替换错误的Baostock配置为正确的配置
3. 保存更新后的配置文件

### 步骤3: 验证配置修复

1. 重启应用服务
2. 检查API返回的数据源列表
3. 确认Baostock数据源配置正确

### 步骤4: 测试数据源功能

1. 验证数据源能够正确加载
2. 测试作为备份数据源的可用性
3. 确保与AKShare数据源的兼容性

## 技术要点

* **编码修复**: 确保配置文件使用正确的UTF-8编码

* **格式一致性**: 确保Baostock配置与AKShare配置结构一致

* **备份机制**: 将Baostock设置为禁用状态，作为AKShare的备份

* **功能完整性**: 确保配置包含所有必要的字段和设置

## 预期结果

* Baostock数据源配置正确，无编码错误

* 配置结构与AKShare保持一致

* 数据源默认为禁用状态，作为备份选项

* 系统能够正确加载和管理Baostock数据源

