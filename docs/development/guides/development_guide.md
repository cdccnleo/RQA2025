# RQA2025 开发指南

## 📋 概述

本文档为RQA2025项目的开发指南，包含代码规范、目录结构、开发流程等内容。

## 🏗️ 项目结构

### 目录组织原则

1. **按功能分层**: 数据层、特征层、模型层、交易层、基础设施层
2. **按模块分组**: 相关功能集中在同一目录下
3. **避免重复**: 相同功能不分散在多个目录
4. **命名规范**: 使用小写字母和下划线

### 核心目录说明

```
src/
├── acceleration/       # 硬件加速模块
├── trading/          # 交易核心模块
├── data/             # 数据处理模块
├── models/           # 模型管理模块
├── features/         # 特征工程模块
├── infrastructure/   # 基础设施模块
├── backtest/         # 回测模块
├── engine/           # 实时引擎模块
└── utils/            # 工具模块
```

## 📝 代码规范

### 1. 命名规范

#### 目录和文件命名
- 使用小写字母和下划线
- 避免单复数混用
- 功能描述准确

```python
# ✅ 正确
src/trading/order_manager.py
src/data/loaders/stock_loader.py
src/features/processors/technical_indicators.py

# ❌ 错误
src/Trading/OrderManager.py
src/data/loader/stockLoader.py
src/features/processors/technicalIndicators.py
```

#### 类命名
- 使用大驼峰命名法
- 避免缩写，使用完整单词

```python
# ✅ 正确
class OrderManager:
class StockDataLoader:
class TechnicalIndicatorProcessor:

# ❌ 错误
class orderManager:
class StockLoader:
class TechIndProcessor:
```

#### 函数和变量命名
- 使用小写字母和下划线
- 函数名应该是动词或动词短语
- 变量名应该是名词

```python
# ✅ 正确
def calculate_position_size():
def load_market_data():
def process_technical_indicators():

position_size = 1000
market_data = {}
technical_indicators = []

# ❌ 错误
def positionSize():
def loadMarketData():
def processTechInd():

PositionSize = 1000
MarketData = {}
techInd = []
```

### 2. 代码组织

#### 文件结构
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模块说明文档
描述模块的主要功能和用途
"""

# 标准库导入
import os
import sys
from typing import Dict, List, Optional

# 第三方库导入
import pandas as pd
import numpy as np

# 本地模块导入
from .base import BaseClass
from ..utils.helpers import helper_function

# 常量定义
DEFAULT_CONFIG = {
    "timeout": 30,
    "retries": 3
}

# 类定义
class MyClass(BaseClass):
    """类说明文档"""
    
    def __init__(self, config: Dict):
        """初始化方法"""
        self.config = config
        self._setup()
    
    def _setup(self):
        """私有方法，用于初始化设置"""
        pass
    
    def public_method(self) -> Dict:
        """公共方法"""
        pass

# 函数定义
def utility_function(param: str) -> str:
    """工具函数"""
    pass

# 主函数（如果适用）
if __name__ == "__main__":
    main()
```

#### 导入规范
```python
# ✅ 正确的导入顺序
# 1. 标准库
import os
import sys
from typing import Dict, List

# 2. 第三方库
import pandas as pd
import numpy as np

# 3. 本地模块
from .base import BaseClass
from ..utils.helpers import helper_function

# ❌ 错误的导入
import pandas as pd, numpy as np  # 一行多个导入
from .base import *  # 避免通配符导入
```

### 3. 文档规范

#### 类和函数文档
```python
class OrderManager:
    """订单管理器
    
    负责订单的创建、修改、查询和删除操作。
    支持多种订单类型和状态管理。
    
    Attributes:
        config (Dict): 配置参数
        orders (List): 订单列表
        max_orders (int): 最大订单数量
    """
    
    def create_order(self, symbol: str, quantity: int, price: float) -> Dict:
        """创建新订单
        
        Args:
            symbol (str): 交易标的代码
            quantity (int): 订单数量
            price (float): 订单价格
            
        Returns:
            Dict: 包含订单信息的字典
            
        Raises:
            ValueError: 当参数无效时
            OrderLimitExceeded: 当超过订单限制时
            
        Example:
            >>> manager = OrderManager()
            >>> order = manager.create_order("AAPL", 100, 150.0)
            >>> print(order["order_id"])
            "order_20250101_001"
        """
        pass
```

#### 模块文档
```python
"""
订单管理模块

本模块提供完整的订单管理功能，包括：
- 订单创建和修改
- 订单状态跟踪
- 订单历史查询
- 批量订单处理

主要组件：
- OrderManager: 订单管理器
- OrderValidator: 订单验证器
- OrderHistory: 订单历史记录

使用示例：
    from src.trading.order_manager import OrderManager
    
    manager = OrderManager()
    order = manager.create_order("AAPL", 100, 150.0)
"""
```

## 🔧 开发流程

### 1. 新功能开发

#### 步骤1: 创建功能分支
```bash
git checkout -b feature/new-feature-name
```

#### 步骤2: 创建测试文件
```python
# tests/unit/trading/test_new_feature.py
import pytest
from src.trading.new_feature import NewFeature

class TestNewFeature:
    """新功能测试用例"""
    
    def test_basic_functionality(self):
        """测试基本功能"""
        feature = NewFeature()
        result = feature.process()
        assert result is not None
    
    def test_edge_cases(self):
        """测试边界情况"""
        pass
```

#### 步骤3: 实现功能
```python
# src/trading/new_feature.py
class NewFeature:
    """新功能实现"""
    
    def __init__(self):
        self.config = {}
    
    def process(self) -> Dict:
        """处理逻辑"""
        return {"status": "success"}
```

#### 步骤4: 运行测试
```bash
python scripts/run_tests.py tests/unit/trading/test_new_feature.py
```

#### 步骤5: 提交代码
```bash
git add .
git commit -m "feat: 添加新功能

- 实现新功能的核心逻辑
- 添加完整的测试用例
- 更新相关文档"
```

### 2. 代码审查

#### 审查要点
1. **功能正确性**: 功能是否按预期工作
2. **代码质量**: 是否符合代码规范
3. **测试覆盖**: 是否有足够的测试用例
4. **文档完整性**: 是否有必要的文档
5. **性能影响**: 是否影响系统性能

#### 审查清单
- [ ] 代码符合命名规范
- [ ] 有适当的错误处理
- [ ] 有完整的文档字符串
- [ ] 有足够的测试用例
- [ ] 没有重复代码
- [ ] 性能影响可接受

### 3. 部署流程

#### 开发环境
```bash
# 安装依赖
pip install -r requirements.txt

# 运行测试
python scripts/run_tests.py

# 运行代码检查
python scripts/directory_structure_checker.py
```

#### 生产环境
```bash
# 构建镜像
docker build -t rqa2025:latest .

# 运行容器
docker run -d rqa2025:latest
```

## 🧪 测试规范

### 1. 测试类型

#### 单元测试
- 测试单个函数或类的方法
- 使用mock隔离外部依赖
- 覆盖正常和异常情况

```python
import pytest
from unittest.mock import Mock, patch

class TestOrderManager:
    def test_create_order_success(self):
        """测试成功创建订单"""
        manager = OrderManager()
        order = manager.create_order("AAPL", 100, 150.0)
        assert order["symbol"] == "AAPL"
        assert order["quantity"] == 100
    
    def test_create_order_invalid_symbol(self):
        """测试无效标的代码"""
        manager = OrderManager()
        with pytest.raises(ValueError):
            manager.create_order("", 100, 150.0)
```

#### 集成测试
- 测试多个模块的交互
- 使用真实的外部依赖
- 测试端到端流程

```python
class TestTradingIntegration:
    def test_order_execution_flow(self):
        """测试订单执行流程"""
        # 创建订单
        order = self.order_manager.create_order("AAPL", 100, 150.0)
        
        # 执行订单
        result = self.execution_engine.execute_order(order)
        
        # 验证结果
        assert result["status"] == "executed"
```

#### 性能测试
- 测试系统在高负载下的表现
- 测试响应时间和吞吐量
- 测试资源使用情况

```python
class TestPerformance:
    def test_order_processing_performance(self):
        """测试订单处理性能"""
        start_time = time.time()
        
        # 处理大量订单
        for i in range(1000):
            self.order_manager.create_order(f"STOCK{i}", 100, 100.0)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 验证性能要求
        assert processing_time < 1.0  # 1秒内处理1000个订单
```

### 2. 测试组织

#### 测试目录结构
```
tests/
├── unit/              # 单元测试
│   ├── trading/      # 交易模块测试
│   ├── data/         # 数据模块测试
│   └── models/       # 模型模块测试
├── integration/       # 集成测试
├── e2e/             # 端到端测试
└── performance/      # 性能测试
```

#### 测试文件命名
```
test_<module_name>.py
test_<class_name>.py
test_<function_name>.py
```

## 📊 监控和日志

### 1. 日志规范

#### 日志级别
```python
import logging

# DEBUG: 调试信息
logger.debug("Processing order: %s", order_id)

# INFO: 一般信息
logger.info("Order created successfully: %s", order_id)

# WARNING: 警告信息
logger.warning("Order partially filled: %s", order_id)

# ERROR: 错误信息
logger.error("Failed to create order: %s", error)

# CRITICAL: 严重错误
logger.critical("System failure: %s", error)
```

#### 日志格式
```python
import logging

# 配置日志格式
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
```

### 2. 监控指标

#### 业务指标
```python
# 订单相关指标
orders_created = Counter("orders_created_total", "Total orders created")
orders_executed = Counter("orders_executed_total", "Total orders executed")
order_processing_time = Histogram("order_processing_seconds", "Order processing time")

# 交易相关指标
trades_executed = Counter("trades_executed_total", "Total trades executed")
trade_volume = Gauge("trade_volume", "Current trade volume")
```

#### 系统指标
```python
# 性能指标
cpu_usage = Gauge("cpu_usage_percent", "CPU usage percentage")
memory_usage = Gauge("memory_usage_bytes", "Memory usage in bytes")
response_time = Histogram("response_time_seconds", "API response time")
```

## 🔒 安全规范

### 1. 数据安全

#### 敏感数据处理
```python
# ✅ 正确做法
import hashlib

def hash_password(password: str) -> str:
    """哈希密码"""
    return hashlib.sha256(password.encode()).hexdigest()

# ❌ 错误做法
def store_password(password: str):
    """直接存储明文密码"""
    database.store(password)  # 危险！
```

#### 配置管理
```python
# ✅ 正确做法
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

# ❌ 错误做法
API_KEY = "sk-1234567890abcdef"  # 硬编码密钥
```

### 2. 访问控制

#### 权限检查
```python
def execute_order(order: Dict, user: str) -> Dict:
    """执行订单（带权限检查）"""
    
    # 检查用户权限
    if not has_permission(user, "execute_orders"):
        raise PermissionError("用户没有执行订单的权限")
    
    # 检查订单限制
    if exceeds_limit(user, order):
        raise OrderLimitExceeded("超过订单限制")
    
    # 执行订单
    return process_order(order)
```

## 📚 文档维护

### 1. 文档类型

#### API文档
```python
"""
订单管理API

提供订单的创建、查询、修改和删除功能。

Endpoints:
    POST /api/orders - 创建订单
    GET /api/orders/{id} - 查询订单
    PUT /api/orders/{id} - 修改订单
    DELETE /api/orders/{id} - 删除订单
"""
```

#### 架构文档
```python
"""
系统架构说明

本系统采用分层架构设计：
- 表现层：API接口和用户界面
- 业务层：交易逻辑和业务规则
- 数据层：数据访问和存储
- 基础设施层：监控、日志、安全等
"""
```

### 2. 文档更新

#### 更新时机
- 添加新功能时
- 修改现有功能时
- 修复重要bug时
- 架构调整时

#### 更新内容
- API接口变更
- 配置参数变更
- 部署流程变更
- 依赖库变更

## 🚀 最佳实践

### 1. 代码质量

#### 代码审查
- 所有代码变更都需要审查
- 使用自动化工具检查代码质量
- 定期进行代码重构

#### 测试驱动开发
- 先写测试，再写代码
- 保持高测试覆盖率
- 定期运行完整测试套件

### 2. 性能优化

#### 性能监控
- 监控关键性能指标
- 设置性能告警
- 定期进行性能测试

#### 优化策略
- 使用缓存减少重复计算
- 优化数据库查询
- 使用异步处理提高并发

### 3. 错误处理

#### 异常处理
```python
def process_order(order: Dict) -> Dict:
    """处理订单"""
    try:
        # 验证订单
        validate_order(order)
        
        # 执行订单
        result = execute_order(order)
        
        # 记录日志
        logger.info("Order processed successfully: %s", order["id"])
        
        return result
        
    except ValidationError as e:
        logger.error("Order validation failed: %s", e)
        raise
        
    except ExecutionError as e:
        logger.error("Order execution failed: %s", e)
        raise
        
    except Exception as e:
        logger.critical("Unexpected error: %s", e)
        raise
```

#### 重试机制
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def api_call():
    """带重试机制的API调用"""
    response = requests.get("https://api.example.com/data")
    response.raise_for_status()
    return response.json()
```

## 📞 联系方式

如有问题或建议，请联系：
- 技术负责人: [联系方式]
- 项目文档: [文档链接]
- 问题反馈: [反馈渠道] 