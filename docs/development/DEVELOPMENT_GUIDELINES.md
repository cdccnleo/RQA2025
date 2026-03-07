# RQA2025 开发规范

## 📋 概述

本文档定义了RQA2025项目的开发规范和最佳实践，旨在确保代码质量、一致性和可维护性。

## 🎯 规范目标

1. **代码一致性** - 统一代码风格和结构
2. **质量保证** - 确保代码质量和可靠性
3. **可维护性** - 提高代码的可读性和维护性
4. **团队协作** - 规范开发流程和协作方式
5. **性能优化** - 确保系统性能和效率

## 📝 编码规范

### 1. Python编码规范

#### 1.1 基本格式
```python
# 正确示例
import os
import sys
from typing import List, Dict, Optional
import numpy as np

class ExampleClass:
    """类功能说明"""

    def __init__(self, param1: str, param2: Optional[int] = None):
        """初始化方法

        Args:
            param1: 必需参数说明
            param2: 可选参数说明
        """
        self.param1 = param1
        self.param2 = param2

    def example_method(self, data: List[Dict]) -> bool:
        """方法功能说明

        Args:
            data: 输入数据说明

        Returns:
            处理结果说明

        Raises:
            ValueError: 参数错误说明
        """
        if not data:
            raise ValueError("数据不能为空")

        # 方法实现
        result = self._process_data(data)
        return result > 0

    def _process_data(self, data: List[Dict]) -> float:
        """私有方法处理数据"""
        return sum(item.get('value', 0) for item in data)
```

#### 1.2 命名规范
```python
# 类名：PascalCase
class DataProcessor:
    pass

class HttpClient:
    pass

# 函数名和变量名：snake_case
def process_data():
    pass

def calculate_metrics():
    pass

user_name = "example"
config_file = "settings.yaml"

# 常量：UPPER_CASE
MAX_RETRY_COUNT = 3
DEFAULT_TIMEOUT = 30
CONFIG_PATH = "/etc/app/config"

# 模块名：snake_case
data_processor.py
http_client.py
utils.py
```

#### 1.3 导入规范
```python
# 标准库导入
import os
import sys
import json
from pathlib import Path

# 第三方库导入（空行分隔）
import numpy as np
import pandas as pd
import requests

# 本地模块导入（空行分隔）
from . import utils
from ..core import BaseProcessor
from ...config import settings
```

#### 1.4 类型注解
```python
from typing import List, Dict, Optional, Union, Any, Callable

# 函数类型注解
def process_data(data: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    pass

# 复杂类型注解
DataFrame = Dict[str, List[Union[str, int, float]]]
Callback = Callable[[str, Dict], bool]

def register_callback(name: str, callback: Callback) -> None:
    pass

# 类属性类型注解
class Processor:
    config: Dict[str, Any]
    is_active: bool = False

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
```

### 2. 文档规范

#### 2.1 文档字符串
```python
def complex_calculation(
    input_data: np.ndarray,
    parameters: Dict[str, float],
    options: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, Dict[str, float]]:
    """执行复杂计算

    该函数实现了基于输入数据和参数的复杂数值计算，
    支持多种配置选项和优化策略。

    Args:
        input_data: 输入数据数组，形状为 (N, M)
        parameters: 计算参数字典，包含以下键：
            - alpha: 权重系数，范围 [0, 1]
            - beta: 调节参数，默认值 1.0
            - gamma: 缩放因子，必须大于 0
        options: 可选配置参数：
            - method: 计算方法，'fast' 或 'accurate'
            - parallel: 是否启用并行计算
            - cache: 是否启用结果缓存

    Returns:
        tuple: 包含两个元素的元组
            - 计算结果数组，形状与输入相同
            - 统计信息字典，包含计算指标

    Raises:
        ValueError: 当输入参数无效时抛出
        RuntimeError: 当计算过程出错时抛出

    Examples:
        >>> data = np.random.rand(100, 50)
        >>> params = {'alpha': 0.5, 'beta': 1.0, 'gamma': 2.0}
        >>> result, stats = complex_calculation(data, params)
        >>> print(f"结果形状: {result.shape}")

    Note:
        - 计算过程可能需要较长时间，建议对大数据使用并行选项
        - 结果具有数值稳定性，但可能受浮点精度影响
        - 缓存功能仅在相同输入时生效

    See Also:
        - simple_calculation: 简化版本的计算函数
        - batch_calculation: 批量计算函数
    """
    pass
```

#### 2.2 代码注释
```python
# 好的注释示例
class RiskCalculator:
    def calculate_portfolio_risk(self, positions: List[Position]) -> float:
        """计算投资组合风险

        使用VaR（风险价值）方法计算组合整体风险。
        VaR表示在正常市场条件下，预期最大损失。
        """

        # 步骤1：提取持仓数据
        # 将持仓转换为权重向量和协方差矩阵
        weights = np.array([pos.weight for pos in positions])
        returns = np.array([pos.historical_returns for pos in positions])

        # 步骤2：计算协方差矩阵
        # 使用对数收益率计算资产间的相关性
        cov_matrix = np.cov(returns.T)

        # 步骤3：计算投资组合方差
        # 方差 = 权重^T * 协方差矩阵 * 权重
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))

        # 步骤4：转换为风险度量
        # 使用95%置信度的VaR计算
        confidence_level = 0.95
        portfolio_volatility = np.sqrt(portfolio_variance)
        var_95 = portfolio_volatility * np.sqrt(1 - confidence_level)

        return var_95
```

## 🏗️ 设计规范

### 3. 架构设计原则

#### 3.1 SOLID原则
```python
# 1. 单一职责原则 (Single Responsibility)
class UserService:
    """用户服务 - 仅负责用户相关操作"""
    def create_user(self, user_data: Dict) -> User:
        pass

    def update_user(self, user_id: str, user_data: Dict) -> User:
        pass

    def delete_user(self, user_id: str) -> bool:
        pass

class UserValidator:
    """用户验证 - 仅负责用户数据验证"""
    def validate_user_data(self, user_data: Dict) -> List[str]:
        pass

    def validate_user_permissions(self, user_id: str, action: str) -> bool:
        pass

# 2. 开闭原则 (Open/Closed)
class BaseProcessor(ABC):
    """基础处理器 - 对扩展开放，对修改关闭"""
    @abstractmethod
    def process(self, data: Any) -> Any:
        pass

class DataProcessor(BaseProcessor):
    """数据处理器"""
    def process(self, data: Any) -> Any:
        # 具体实现
        return processed_data

class ImageProcessor(BaseProcessor):
    """图像处理器"""
    def process(self, data: Any) -> Any:
        # 具体实现
        return processed_image

# 3. 里氏替换原则 (Liskov Substitution)
class Rectangle:
    """矩形"""
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    @property
    def area(self) -> float:
        return self.width * self.height

class Square(Rectangle):
    """正方形 - 正确实现"""
    def __init__(self, side: float):
        super().__init__(side, side)

    @property
    def width(self) -> float:
        return self._width

    @width.setter
    def width(self, value: float):
        self._width = value
        self._height = value  # 保持正方形性质

    @property
    def height(self) -> float:
        return self._height

    @height.setter
    def height(self, value: float):
        self._width = value
        self._height = value
```

#### 3.2 设计模式应用
```python
# 工厂模式
class ModelFactory:
    """模型工厂"""
    _registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str, model_class: Type) -> None:
        """注册模型类"""
        cls._registry[name] = model_class

    @classmethod
    def create(cls, name: str, **kwargs) -> Any:
        """创建模型实例"""
        if name not in cls._registry:
            raise ValueError(f"未知模型类型: {name}")
        return cls._registry[name](**kwargs)

# 策略模式
class TradingStrategy(ABC):
    """交易策略基类"""
    @abstractmethod
    def generate_signals(self, market_data: pd.DataFrame) -> pd.Series:
        pass

class MomentumStrategy(TradingStrategy):
    """动量策略"""
    def generate_signals(self, market_data: pd.DataFrame) -> pd.Series:
        # 实现动量策略逻辑
        pass

class MeanReversionStrategy(TradingStrategy):
    """均值回归策略"""
    def generate_signals(self, market_data: pd.DataFrame) -> pd.Series:
        # 实现均值回归策略逻辑
        pass

class TradingEngine:
    """交易引擎"""
    def __init__(self, strategy: TradingStrategy):
        self.strategy = strategy

    def execute_trading(self, market_data: pd.DataFrame) -> None:
        signals = self.strategy.generate_signals(market_data)
        # 执行交易逻辑
        pass
```

### 4. 错误处理规范

#### 4.1 异常处理
```python
# 正确异常处理
def load_configuration(config_path: str) -> Dict[str, Any]:
    """加载配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典

    Raises:
        FileNotFoundError: 配置文件不存在
        ValueError: 配置文件格式错误
        PermissionError: 没有读取权限
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        # 验证配置
        required_keys = ['database', 'api_key', 'timeout']
        missing_keys = [key for key in required_keys if key not in config_data]

        if missing_keys:
            raise ValueError(f"配置文件缺少必需键: {missing_keys}")

        return config_data

    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"配置文件格式错误: {e}")
    except PermissionError:
        raise PermissionError(f"没有读取权限: {config_path}")
    except Exception as e:
        # 记录详细错误信息
        logger.error(f"加载配置失败: {e}", exc_info=True)
        raise RuntimeError(f"配置加载失败: {e}")

# 使用示例
try:
    config = load_configuration("config.json")
    print("配置加载成功")
except (FileNotFoundError, ValueError, PermissionError) as e:
    print(f"配置错误: {e}")
    # 使用默认配置
    config = get_default_config()
except RuntimeError as e:
    print(f"系统错误: {e}")
    # 系统级错误处理
    sys.exit(1)
```

#### 4.2 日志记录
```python
import logging
from typing import Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class DataProcessor:
    """数据处理器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理数据"""
        try:
            self.logger.info(f"开始处理数据，形状: {data.shape}")

            # 验证输入
            if data.empty:
                self.logger.warning("输入数据为空")
                return data

            # 数据清洗
            cleaned_data = self._clean_data(data)
            self.logger.debug(f"数据清洗完成，剩余记录: {len(cleaned_data)}")

            # 数据转换
            transformed_data = self._transform_data(cleaned_data)
            self.logger.info(f"数据转换完成，输出形状: {transformed_data.shape}")

            return transformed_data

        except Exception as e:
            self.logger.error(f"数据处理失败: {e}", exc_info=True)
            raise

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据清洗"""
        # 记录清洗前的统计信息
        original_count = len(data)
        self.logger.debug(f"原始数据记录数: {original_count}")

        # 清洗逻辑
        cleaned = data.dropna()
        cleaned = cleaned[cleaned['value'] > 0]

        # 记录清洗结果
        cleaned_count = len(cleaned)
        removed_count = original_count - cleaned_count
        self.logger.debug(f"清洗完成，移除记录: {removed_count}")

        return cleaned

    def _transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据转换"""
        # 记录转换信息
        self.logger.debug("开始数据转换")

        # 转换逻辑
        transformed = data.copy()
        transformed['normalized_value'] = (
            transformed['value'] - transformed['value'].mean()
        ) / transformed['value'].std()

        self.logger.debug("数据转换完成")
        return transformed
```

## 🧪 测试规范

### 5. 单元测试

#### 5.1 测试结构
```python
import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

class TestDataProcessor(unittest.TestCase):
    """数据处理器测试"""

    def setUp(self):
        """测试前置设置"""
        self.config = {
            'clean_method': 'dropna',
            'normalize': True,
            'outlier_threshold': 3
        }
        self.processor = DataProcessor(self.config)

    def tearDown(self):
        """测试后置清理"""
        # 清理测试数据
        pass

    def test_init(self):
        """测试初始化"""
        self.assertIsNotNone(self.processor.config)
        self.assertEqual(self.processor.config['clean_method'], 'dropna')

    def test_process_empty_data(self):
        """测试处理空数据"""
        empty_data = pd.DataFrame()

        result = self.processor.process_data(empty_data)

        self.assertTrue(result.empty)
        # 验证日志记录
        # self.assertLogs('WARNING')

    def test_process_valid_data(self):
        """测试处理有效数据"""
        test_data = pd.DataFrame({
            'value': [1, 2, 3, 4, 5],
            'category': ['A', 'B', 'A', 'B', 'A']
        })

        result = self.processor.process_data(test_data)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 5)
        self.assertTrue('normalized_value' in result.columns)

    @patch('data_processor.external_api_call')
    def test_process_with_external_dependency(self, mock_api):
        """测试外部依赖"""
        # 配置模拟对象
        mock_api.return_value = {'status': 'success', 'data': [1, 2, 3]}

        test_data = pd.DataFrame({'value': [1, 2, 3]})
        result = self.processor.process_data(test_data)

        # 验证外部API被调用
        mock_api.assert_called_once()
        self.assertIsNotNone(result)

    def test_process_invalid_data(self):
        """测试处理无效数据"""
        invalid_data = pd.DataFrame({
            'value': ['invalid', 2, 3],
            'category': ['A', 'B', 'A']
        })

        with self.assertRaises(ValueError):
            self.processor.process_data(invalid_data)

    @unittest.skip("暂时跳过性能测试")
    def test_performance_large_dataset(self):
        """测试大数据集性能"""
        large_data = pd.DataFrame({
            'value': np.random.rand(10000),
            'category': np.random.choice(['A', 'B', 'C'], 10000)
        })

        import time
        start_time = time.time()

        result = self.processor.process_data(large_data)

        end_time = time.time()
        processing_time = end_time - start_time

        # 性能断言
        self.assertLess(processing_time, 1.0)  # 处理时间应小于1秒
        self.assertEqual(len(result), 10000)

# 测试套件
def create_test_suite():
    """创建测试套件"""
    suite = unittest.TestSuite()
    suite.addTest(TestDataProcessor('test_init'))
    suite.addTest(TestDataProcessor('test_process_empty_data'))
    suite.addTest(TestDataProcessor('test_process_valid_data'))
    return suite

if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)

    # 或者运行测试套件
    # runner = unittest.TextTestRunner(verbosity=2)
    # runner.run(create_test_suite())
```

#### 5.2 测试覆盖率
```python
# pytest配置文件 pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    --verbose
    --tb=short
    --strict-markers
    --cov=src
    --cov-report=html:htmlcov
    --cov-report=term-missing
    --cov-fail-under=80
    --durations=10

# 覆盖率要求
# - 单元测试覆盖率 >= 80%
# - 核心业务逻辑覆盖率 >= 90%
# - 错误处理覆盖率 >= 95%

# 覆盖率徽章配置
# README.md 中添加：
# [![Coverage Status](https://coveralls.io/repos/github/username/project/badge.svg?branch=main)](https://coveralls.io/github/username/project?branch=main)
```

### 6. 集成测试

#### 6.1 API测试
```python
import requests
import pytest
from fastapi.testclient import TestClient
from app.main import app

class TestAPI:
    """API集成测试"""

    @pytest.fixture
    def client(self):
        """测试客户端"""
        return TestClient(app)

    def test_health_check(self, client):
        """健康检查测试"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    def test_create_user(self, client):
        """创建用户测试"""
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "securepassword123"
        }

        response = client.post("/users", json=user_data)

        assert response.status_code == 201
        data = response.json()
        assert data["username"] == user_data["username"]
        assert data["email"] == user_data["email"]
        assert "id" in data
        assert "created_at" in data

    def test_get_user(self, client):
        """获取用户测试"""
        # 先创建用户
        user_data = {
            "username": "testuser2",
            "email": "test2@example.com",
            "password": "securepassword123"
        }
        create_response = client.post("/users", json=user_data)
        user_id = create_response.json()["id"]

        # 再获取用户
        response = client.get(f"/users/{user_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["id"] == user_id
        assert data["username"] == user_data["username"]

    def test_update_user(self, client):
        """更新用户测试"""
        # 创建用户
        user_data = {
            "username": "testuser3",
            "email": "test3@example.com",
            "password": "securepassword123"
        }
        create_response = client.post("/users", json=user_data)
        user_id = create_response.json()["id"]

        # 更新用户
        update_data = {
            "email": "updated@example.com"
        }
        response = client.put(f"/users/{user_id}", json=update_data)
        assert response.status_code == 200

        # 验证更新
        get_response = client.get(f"/users/{user_id}")
        updated_data = get_response.json()
        assert updated_data["email"] == update_data["email"]

    def test_delete_user(self, client):
        """删除用户测试"""
        # 创建用户
        user_data = {
            "username": "testuser4",
            "email": "test4@example.com",
            "password": "securepassword123"
        }
        create_response = client.post("/users", json=user_data)
        user_id = create_response.json()["id"]

        # 删除用户
        response = client.delete(f"/users/{user_id}")
        assert response.status_code == 204

        # 验证删除
        get_response = client.get(f"/users/{user_id}")
        assert get_response.status_code == 404

    def test_list_users(self, client):
        """列出用户测试"""
        response = client.get("/users")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 0

    def test_invalid_user_data(self, client):
        """无效用户数据测试"""
        invalid_data = {
            "username": "",  # 空用户名
            "email": "invalid-email",  # 无效邮箱
            "password": "123"  # 密码太短
        }

        response = client.post("/users", json=invalid_data)
        assert response.status_code == 422  # 验证错误

    def test_unauthorized_access(self, client):
        """未授权访问测试"""
        # 尝试访问需要认证的端点
        response = client.get("/users/profile")
        assert response.status_code == 401
```

## 🚀 性能优化规范

### 7. 性能优化原则

#### 7.1 代码优化
```python
# 避免在循环中重复计算
import math

# 不推荐
def calculate_distances_bad(points, center):
    distances = []
    for point in points:
        # 每次循环都计算距离
        distance = math.sqrt(
            (point[0] - center[0]) ** 2 +
            (point[1] - center[1]) ** 2
        )
        distances.append(distance)
    return distances

# 推荐
def calculate_distances_good(points, center):
    distances = []
    center_x, center_y = center  # 预先提取

    for point in points:
        point_x, point_y = point  # 预先提取
        # 避免重复属性访问
        distance = math.sqrt(
            (point_x - center_x) ** 2 +
            (point_y - center_y) ** 2
        )
        distances.append(distance)
    return distances

# 使用向量化操作
import numpy as np

def vectorized_distance_calculation(points, center):
    """向量化距离计算"""
    points = np.array(points)
    center = np.array(center)

    # 向量化操作，比循环更高效
    distances = np.sqrt(np.sum((points - center) ** 2, axis=1))
    return distances
```

#### 7.2 内存优化
```python
# 使用生成器避免大列表
def process_large_file_bad(file_path):
    """不推荐：一次性读取整个文件"""
    with open(file_path, 'r') as f:
        lines = f.readlines()  # 内存中保存所有行

    processed_lines = []
    for line in lines:
        processed_line = process_line(line)
        processed_lines.append(processed_line)

    return processed_lines

def process_large_file_good(file_path):
    """推荐：使用生成器"""
    def process_lines():
        with open(file_path, 'r') as f:
            for line in f:
                processed_line = process_line(line)
                yield processed_line

    return list(process_lines())  # 或者直接使用生成器

# 或者使用更简洁的生成器表达式
def process_large_file_best(file_path):
    """最佳实践：生成器表达式"""
    with open(file_path, 'r') as f:
        return [process_line(line) for line in f]

# 上下文管理器
class DatabaseConnection:
    """数据库连接上下文管理器"""
    def __init__(self, config):
        self.config = config
        self.connection = None

    def __enter__(self):
        # 建立连接
        self.connection = create_connection(self.config)
        return self.connection

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 确保连接被关闭
        if self.connection:
            self.connection.close()
        return False

# 使用示例
with DatabaseConnection(db_config) as conn:
    result = conn.execute("SELECT * FROM users")
    # 连接会自动关闭，即使发生异常
```

## 🔒 安全规范

### 8. 安全编码实践

#### 8.1 输入验证
```python
from typing import Dict, Any
import re

def validate_user_input(user_data: Dict[str, Any]) -> Dict[str, Any]:
    """用户输入验证"""
    validated_data = {}

    # 用户名验证
    username = user_data.get('username', '').strip()
    if not username:
        raise ValueError("用户名不能为空")
    if not re.match(r'^[a-zA-Z0-9_]{3,20}$', username):
        raise ValueError("用户名格式无效")
    validated_data['username'] = username

    # 邮箱验证
    email = user_data.get('email', '').strip().lower()
    if not email:
        raise ValueError("邮箱不能为空")
    if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
        raise ValueError("邮箱格式无效")
    validated_data['email'] = email

    # 密码验证
    password = user_data.get('password', '')
    if len(password) < 8:
        raise ValueError("密码长度至少8位")
    if not re.search(r'[A-Z]', password):
        raise ValueError("密码必须包含大写字母")
    if not re.search(r'[a-z]', password):
        raise ValueError("密码必须包含小写字母")
    if not re.search(r'[0-9]', password):
        raise ValueError("密码必须包含数字")
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        raise ValueError("密码必须包含特殊字符")

    # 密码哈希（不存储明文）
    import bcrypt
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    validated_data['password_hash'] = hashed_password.decode('utf-8')

    return validated_data
```

#### 8.2 SQL注入防护
```python
import sqlite3
from typing import List, Tuple, Optional

class SecureDatabase:
    """安全数据库操作"""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_user_by_id_safe(self, user_id: int) -> Optional[Dict]:
        """安全获取用户信息"""
        query = "SELECT id, username, email, created_at FROM users WHERE id = ?"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # 使用参数化查询
            cursor.execute(query, (user_id,))
            row = cursor.fetchone()

            if row:
                return dict(row)
            return None

    def get_users_by_role_safe(self, role: str, limit: int = 10) -> List[Dict]:
        """安全获取角色用户列表"""
        query = """
            SELECT id, username, email, created_at
            FROM users
            WHERE role = ?
            ORDER BY created_at DESC
            LIMIT ?
        """

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # 使用参数化查询，防止SQL注入
            cursor.execute(query, (role, limit))
            rows = cursor.fetchall()

            return [dict(row) for row in rows]

    def create_user_safe(self, user_data: Dict) -> int:
        """安全创建用户"""
        query = """
            INSERT INTO users (username, email, password_hash, created_at)
            VALUES (?, ?, ?, datetime('now'))
        """

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # 使用参数化查询
            cursor.execute(query, (
                user_data['username'],
                user_data['email'],
                user_data['password_hash']
            ))

            conn.commit()
            return cursor.lastrowid

# 不安全的示例（仅供参考，不要使用）
def get_user_by_id_unsafe(self, user_id: str) -> Optional[Dict]:
    """不安全的用户查询（SQL注入风险）"""
    query = f"SELECT * FROM users WHERE id = {user_id}"  # 危险！

    with sqlite3.connect(self.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query)  # SQL注入风险
        return cursor.fetchone()
```

## 📊 代码质量工具

### 9. 代码质量检查

#### 9.1 静态分析工具
```bash
# Black - 代码格式化
black --line-length 88 --target-version py38 src/

# isort - 导入排序
isort --profile black src/

# flake8 - 代码风格检查
flake8 --max-line-length 88 --extend-ignore E203,W503 src/

# mypy - 类型检查
mypy --ignore-missing-imports src/

# pylint - 代码质量检查
pylint --max-line-length 88 src/

# bandit - 安全检查
bandit -r src/

# radon - 复杂度分析
radon cc src/  # 圈复杂度
radon mi src/  # 可维护性指数
radon hal src/ # Halstead复杂度

# vulture - 死代码检测
vulture src/
```

#### 9.2 配置文件示例

```ini
# setup.cfg
[flake8]
max-line-length = 88
extend-ignore = E203, W503, E501
exclude = .git,__pycache__,build,dist
per-file-ignores =
    __init__.py:F401

[isort]
profile = black
line_length = 88
multi_line_output = 3
include_trailing_comma = True
force_grid_wrap = 0
use_parentheses = True
ensure_newline_before_comments = True

[mypy]
python_version = 3.8
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_incomplete_defs = True
check_untyped_defs = True
disallow_untyped_decorators = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
warn_unreachable = True
strict_equality = True

[pylint]
max-line-length = 88
disable = C0103,C0114,C0115,C0116,R0903,R0912,R0915
```

### 10. CI/CD集成

#### 10.1 GitHub Actions配置
```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-dev.txt

    - name: Run Black
      run: black --check --diff src/

    - name: Run isort
      run: isort --check-only --diff src/

    - name: Run flake8
      run: flake8 src/

    - name: Run mypy
      run: mypy src/

    - name: Run tests
      run: pytest --cov=src --cov-report=xml

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v2
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2

    - name: Run Bandit Security Check
      uses: jpetrucciani/bandit-check@1.6.0
      with:
        path: 'src'
        options: '-r -f json -o bandit-report.json'

    - name: Upload security report
      uses: github/codeql-action/upload-sarif@v1
      with:
        sarif_file: bandit-report.json
```

## 📚 相关文档

- [总体架构文档](../architecture/BUSINESS_PROCESS_DRIVEN_ARCHITECTURE.md)
- [代码规范](CODING_STANDARDS.md)
- [测试规范](TESTING_GUIDELINES.md)
- [安全规范](../security/SECURITY_GUIDELINES.md)
- [性能优化指南](PERFORMANCE_GUIDELINES.md)

## 🔄 更新历史

| 版本 | 日期 | 作者 | 主要变更 |
|------|------|------|---------|
| 1.0 | 2025-01-27 | 架构组 | 初始版本 |

---

**文档版本**: 1.0
**维护人员**: 架构组
**更新频率**: 按需更新
